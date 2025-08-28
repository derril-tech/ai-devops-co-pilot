"""
ArgoCD connector for deployment events
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID
import aiohttp
import backoff
import json

from .base import HTTPConnector, SignalProcessor, ConnectorError, ConnectorConfigError


logger = logging.getLogger(__name__)


class ArgoCDConnector(HTTPConnector):
    """ArgoCD connector for deployment events"""

    @property
    def connector_type(self) -> str:
        return "argocd"

    async def validate_config(self) -> bool:
        """Validate ArgoCD connector configuration"""
        required_fields = ["base_url"]
        for field in required_fields:
            if not self.get_config_value(field):
                raise ConnectorConfigError(f"Missing required field: {field}")

        # Validate authentication
        if not (self.get_config_value("token") or self.get_config_value("username") and self.get_config_value("password")):
            raise ConnectorConfigError("Authentication required: token or username/password")

        return True

    async def test_connection(self) -> bool:
        """Test connection to ArgoCD"""
        try:
            response = await self._make_request("GET", "/api/v1/version")
            return response.get("Version") is not None
        except Exception as e:
            logger.error(f"ArgoCD connection test failed: {e}")
            return False

    async def collect_data(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect deployment data from ArgoCD"""
        signals = []

        try:
            # Get applications
            applications = await self._get_applications()

            for app in applications:
                try:
                    app_signals = await self._collect_application_events(app, start_time, end_time)
                    signals.extend(app_signals)
                except Exception as e:
                    logger.error(f"Failed to collect events for app {app.get('metadata', {}).get('name')}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to collect ArgoCD data: {e}")

        return signals

    async def _get_applications(self) -> List[Dict[str, Any]]:
        """Get all ArgoCD applications"""
        try:
            response = await self._make_request("GET", "/api/v1/applications")
            return response.get("items", [])
        except Exception as e:
            logger.error(f"Failed to get applications: {e}")
            return []

    async def _collect_application_events(self, app: Dict[str, Any], start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect events for a specific application"""
        signals = []
        app_name = app.get("metadata", {}).get("name", "unknown")

        try:
            # Get application details
            app_detail = await self._get_application_detail(app_name)
            if not app_detail:
                return signals

            # Process sync status
            sync_signals = self._process_sync_status(app_detail, start_time, end_time)
            signals.extend(sync_signals)

            # Process health status
            health_signals = self._process_health_status(app_detail, start_time, end_time)
            signals.extend(health_signals)

            # Process resource tree
            resource_signals = await self._process_resource_tree(app_name, start_time, end_time)
            signals.extend(resource_signals)

        except Exception as e:
            logger.error(f"Failed to collect events for app {app_name}: {e}")

        return signals

    async def _get_application_detail(self, app_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed application information"""
        try:
            response = await self._make_request("GET", f"/api/v1/applications/{app_name}")
            return response
        except Exception as e:
            logger.error(f"Failed to get application detail for {app_name}: {e}")
            return None

    def _process_sync_status(self, app_detail: Dict[str, Any], start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Process application sync status into signals"""
        signals = []

        try:
            status = app_detail.get("status", {})
            sync = status.get("sync", {})

            # Sync status change events
            sync_status = sync.get("status")
            if sync_status:
                # Create a signal for current sync status
                # Note: ArgoCD doesn't provide historical sync status, so we create a point-in-time signal
                signal = SignalProcessor.create_signal_record(
                    org_id=self.org_id,
                    source=f"argocd:{self.connector_id}",
                    kind="event",
                    key=f"sync.{sync_status.lower()}",
                    value=1,
                    text=f"Application sync status: {sync_status}",
                    ts=datetime.utcnow(),  # Current time since no historical data
                    labels={
                        "application": app_detail.get("metadata", {}).get("name", "unknown"),
                        "namespace": app_detail.get("spec", {}).get("destination", {}).get("namespace", "unknown"),
                        "sync_status": sync_status,
                        "revision": sync.get("revision", "unknown")
                    },
                    meta={
                        "connector_type": "argocd",
                        "connector_id": str(self.connector_id),
                        "sync_info": sync
                    }
                )
                signals.append(signal)

        except Exception as e:
            logger.error(f"Failed to process sync status: {e}")

        return signals

    def _process_health_status(self, app_detail: Dict[str, Any], start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Process application health status into signals"""
        signals = []

        try:
            status = app_detail.get("status", {})
            health = status.get("health", {})

            health_status = health.get("status")
            if health_status:
                signal = SignalProcessor.create_signal_record(
                    org_id=self.org_id,
                    source=f"argocd:{self.connector_id}",
                    kind="event",
                    key=f"health.{health_status.lower()}",
                    value=1,
                    text=f"Application health status: {health_status}",
                    ts=datetime.utcnow(),
                    labels={
                        "application": app_detail.get("metadata", {}).get("name", "unknown"),
                        "namespace": app_detail.get("spec", {}).get("destination", {}).get("namespace", "unknown"),
                        "health_status": health_status,
                        "health_message": health.get("message", "")
                    },
                    meta={
                        "connector_type": "argocd",
                        "connector_id": str(self.connector_id),
                        "health_info": health
                    }
                )
                signals.append(signal)

        except Exception as e:
            logger.error(f"Failed to process health status: {e}")

        return signals

    async def _process_resource_tree(self, app_name: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Process application resource tree into signals"""
        signals = []

        try:
            # Get resource tree
            response = await self._make_request("GET", f"/api/v1/applications/{app_name}/resource-tree")
            nodes = response.get("nodes", [])

            for node in nodes:
                resource_signals = self._process_resource_node(node, app_name)
                signals.extend(resource_signals)

        except Exception as e:
            logger.error(f"Failed to process resource tree for {app_name}: {e}")

        return signals

    def _process_resource_node(self, node: Dict[str, Any], app_name: str) -> List[Dict[str, Any]]:
        """Process a resource tree node into signals"""
        signals = []

        try:
            # Create signal for resource status
            health_status = node.get("health", {}).get("status", "Unknown")
            sync_status = node.get("status", "Unknown")

            signal = SignalProcessor.create_signal_record(
                org_id=self.org_id,
                source=f"argocd:{self.connector_id}",
                kind="event",
                key=f"resource.{sync_status.lower()}",
                value=1,
                text=f"Resource {node.get('name', 'unknown')}: {sync_status}",
                ts=datetime.utcnow(),
                labels={
                    "application": app_name,
                    "resource_name": node.get("name", "unknown"),
                    "resource_kind": node.get("kind", "unknown"),
                    "namespace": node.get("namespace", "unknown"),
                    "sync_status": sync_status,
                    "health_status": health_status,
                    "resource_version": node.get("resourceVersion", "unknown")
                },
                meta={
                    "connector_type": "argocd",
                    "connector_id": str(self.connector_id),
                    "resource_info": node
                }
            )
            signals.append(signal)

        except Exception as e:
            logger.error(f"Failed to process resource node: {e}")

        return signals

    async def get_application_events(self, app_name: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get events for a specific application"""
        try:
            # Get application events (if available)
            response = await self._make_request("GET", f"/api/v1/applications/{app_name}/events")
            events = response.get("items", [])

            # Filter events by time range
            filtered_events = []
            for event in events:
                event_time = datetime.fromisoformat(event["lastTimestamp"].replace('Z', '+00:00'))
                if start_time <= event_time <= end_time:
                    filtered_events.append(event)

            return filtered_events

        except Exception as e:
            logger.error(f"Failed to get events for app {app_name}: {e}")
            return []

    async def trigger_sync(self, app_name: str, revision: str = None) -> Dict[str, Any]:
        """Trigger a sync operation for an application"""
        try:
            sync_request = {
                "revision": revision,
                "prune": True,
                "strategy": {
                    "hook": {
                        "force": False
                    }
                }
            }

            response = await self._make_request(
                "POST",
                f"/api/v1/applications/{app_name}/sync",
                json=sync_request
            )

            return response

        except Exception as e:
            logger.error(f"Failed to trigger sync for {app_name}: {e}")
            raise

    async def get_clusters(self) -> List[Dict[str, Any]]:
        """Get registered clusters"""
        try:
            response = await self._make_request("GET", "/api/v1/clusters")
            return response.get("items", [])
        except Exception as e:
            logger.error(f"Failed to get clusters: {e}")
            return []

    async def get_projects(self) -> List[Dict[str, Any]]:
        """Get ArgoCD projects"""
        try:
            response = await self._make_request("GET", "/api/v1/projects")
            return response.get("items", [])
        except Exception as e:
            logger.error(f"Failed to get projects: {e}")
            return []

    async def get_application_logs(self, app_name: str, pod_name: str = None, container_name: str = None,
                                  since_seconds: int = 3600) -> str:
        """Get application logs"""
        try:
            params = {
                "sinceSeconds": since_seconds
            }

            if pod_name:
                params["podName"] = pod_name
            if container_name:
                params["containerName"] = container_name

            response = await self._make_request(
                "GET",
                f"/api/v1/applications/{app_name}/logs",
                params=params
            )

            return response

        except Exception as e:
            logger.error(f"Failed to get logs for {app_name}: {e}")
            return ""
