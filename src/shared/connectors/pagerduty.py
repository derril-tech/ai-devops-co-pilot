"""
PagerDuty connector for incident data
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


class PagerDutyConnector(HTTPConnector):
    """PagerDuty connector for incident data"""

    @property
    def connector_type(self) -> str:
        return "pagerduty"

    async def validate_config(self) -> bool:
        """Validate PagerDuty connector configuration"""
        required_fields = ["token"]
        for field in required_fields:
            if not self.get_config_value(field):
                raise ConnectorConfigError(f"Missing required field: {field}")

        return True

    async def test_connection(self) -> bool:
        """Test connection to PagerDuty"""
        try:
            response = await self._make_request("GET", "/users")
            return response.get("users") is not None
        except Exception as e:
            logger.error(f"PagerDuty connection test failed: {e}")
            return False

    async def collect_data(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect incident data from PagerDuty"""
        signals = []

        try:
            # Collect incidents
            incident_signals = await self._collect_incidents(start_time, end_time)
            signals.extend(incident_signals)

            # Collect log entries (if enabled)
            if self.get_config_value("collect_log_entries", True):
                log_signals = await self._collect_log_entries(start_time, end_time)
                signals.extend(log_signals)

            # Collect maintenance windows (if enabled)
            if self.get_config_value("collect_maintenance", False):
                maintenance_signals = await self._collect_maintenance_windows(start_time, end_time)
                signals.extend(maintenance_signals)

        except Exception as e:
            logger.error(f"Failed to collect PagerDuty data: {e}")

        return signals

    async def _collect_incidents(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect incidents from PagerDuty"""
        signals = []

        try:
            # Build query parameters
            params = {
                "since": start_time.isoformat(),
                "until": end_time.isoformat(),
                "limit": self.get_config_value("limit", 100)
            }

            # Add additional filters
            if self.get_config_value("service_ids"):
                params["service_ids[]"] = self.get_config_value("service_ids")
            if self.get_config_value("team_ids"):
                params["team_ids[]"] = self.get_config_value("team_ids")

            response = await self._make_request("GET", "/incidents", params=params)
            incidents = response.get("incidents", [])

            for incident in incidents:
                incident_signals = self._process_incident(incident)
                signals.extend(incident_signals)

        except Exception as e:
            logger.error(f"Failed to collect incidents: {e}")

        return signals

    async def _collect_log_entries(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect log entries from PagerDuty"""
        signals = []

        try:
            # Get incidents first to get their IDs
            incidents_response = await self._make_request(
                "GET",
                "/incidents",
                {
                    "since": start_time.isoformat(),
                    "until": end_time.isoformat(),
                    "limit": 50
                }
            )

            incidents = incidents_response.get("incidents", [])

            for incident in incidents:
                try:
                    log_signals = await self._collect_incident_log_entries(incident["id"], start_time, end_time)
                    signals.extend(log_signals)
                except Exception as e:
                    logger.error(f"Failed to collect log entries for incident {incident['id']}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to collect log entries: {e}")

        return signals

    async def _collect_incident_log_entries(self, incident_id: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect log entries for a specific incident"""
        signals = []

        try:
            response = await self._make_request("GET", f"/incidents/{incident_id}/log_entries")
            log_entries = response.get("log_entries", [])

            for entry in log_entries:
                entry_time = datetime.fromisoformat(entry["created_at"].replace('Z', '+00:00'))

                if start_time <= entry_time <= end_time:
                    signal = self._process_log_entry(entry, incident_id)
                    if signal:
                        signals.append(signal)

        except Exception as e:
            logger.error(f"Failed to collect log entries for incident {incident_id}: {e}")

        return signals

    async def _collect_maintenance_windows(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect maintenance windows from PagerDuty"""
        signals = []

        try:
            response = await self._make_request("GET", "/maintenance_windows")
            maintenance_windows = response.get("maintenance_windows", [])

            for window in maintenance_windows:
                window_signals = self._process_maintenance_window(window, start_time, end_time)
                signals.extend(window_signals)

        except Exception as e:
            logger.error(f"Failed to collect maintenance windows: {e}")

        return signals

    def _process_incident(self, incident: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process an incident into signals"""
        signals = []

        try:
            created_at = datetime.fromisoformat(incident["created_at"].replace('Z', '+00:00'))

            # Incident creation signal
            signal = SignalProcessor.create_signal_record(
                org_id=self.org_id,
                source=f"pagerduty:{self.connector_id}",
                kind="event",
                key="incident.created",
                value=1,
                text=f"Incident: {incident['title']}",
                ts=created_at,
                labels={
                    "incident_id": incident["id"],
                    "incident_number": str(incident["incident_number"]),
                    "title": incident["title"],
                    "status": incident["status"],
                    "urgency": incident["urgency"],
                    "priority": incident.get("priority", {}).get("name", "unknown") if incident.get("priority") else "none",
                    "service_name": incident.get("service", {}).get("name", "unknown") if incident.get("service") else "unknown",
                    "escalation_policy": incident.get("escalation_policy", {}).get("name", "unknown") if incident.get("escalation_policy") else "unknown"
                },
                meta={
                    "connector_type": "pagerduty",
                    "connector_id": str(self.connector_id),
                    "incident_data": incident
                }
            )
            signals.append(signal)

            # Process status changes
            if incident.get("last_status_change_at"):
                status_changed_at = datetime.fromisoformat(incident["last_status_change_at"].replace('Z', '+00:00'))

                status_signal = SignalProcessor.create_signal_record(
                    org_id=self.org_id,
                    source=f"pagerduty:{self.connector_id}",
                    kind="event",
                    key=f"incident.{incident['status']}",
                    value=1,
                    text=f"Incident {incident['incident_number']} status: {incident['status']}",
                    ts=status_changed_at,
                    labels={
                        "incident_id": incident["id"],
                        "incident_number": str(incident["incident_number"]),
                        "status": incident["status"],
                        "previous_status": incident.get("last_status_change_by", "unknown")
                    },
                    meta={
                        "connector_type": "pagerduty",
                        "connector_id": str(self.connector_id),
                        "status_change": True
                    }
                )
                signals.append(status_signal)

        except Exception as e:
            logger.error(f"Failed to process incident: {e}")

        return signals

    def _process_log_entry(self, entry: Dict[str, Any], incident_id: str) -> Optional[Dict[str, Any]]:
        """Process a log entry into a signal"""
        try:
            created_at = datetime.fromisoformat(entry["created_at"].replace('Z', '+00:00'))

            signal = SignalProcessor.create_signal_record(
                org_id=self.org_id,
                source=f"pagerduty:{self.connector_id}",
                kind="event",
                key=f"log.{entry['type']}",
                value=1,
                text=entry.get("summary", ""),
                ts=created_at,
                labels={
                    "incident_id": incident_id,
                    "log_entry_id": entry["id"],
                    "type": entry["type"],
                    "channel": entry.get("channel", {}).get("type", "unknown") if entry.get("channel") else "unknown",
                    "agent_name": entry.get("agent", {}).get("name", "unknown") if entry.get("agent") else "unknown"
                },
                meta={
                    "connector_type": "pagerduty",
                    "connector_id": str(self.connector_id),
                    "log_entry_data": entry
                }
            )

            return signal

        except Exception as e:
            logger.error(f"Failed to process log entry: {e}")
            return None

    def _process_maintenance_window(self, window: Dict[str, Any], start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Process a maintenance window into signals"""
        signals = []

        try:
            start_time_window = datetime.fromisoformat(window["start_time"].replace('Z', '+00:00'))
            end_time_window = datetime.fromisoformat(window["end_time"].replace('Z', '+00:00'))

            # Only process if window overlaps with our collection period
            if not (start_time_window <= end_time and end_time_window >= start_time):
                return signals

            # Maintenance window start signal
            if start_time <= start_time_window <= end_time:
                start_signal = SignalProcessor.create_signal_record(
                    org_id=self.org_id,
                    source=f"pagerduty:{self.connector_id}",
                    kind="event",
                    key="maintenance.started",
                    value=1,
                    text=f"Maintenance: {window['summary']}",
                    ts=start_time_window,
                    labels={
                        "maintenance_id": window["id"],
                        "summary": window["summary"],
                        "status": "started",
                        "services_affected": str(len(window.get("services", [])))
                    },
                    meta={
                        "connector_type": "pagerduty",
                        "connector_id": str(self.connector_id),
                        "maintenance_data": window
                    }
                )
                signals.append(start_signal)

            # Maintenance window end signal
            if start_time <= end_time_window <= end_time:
                end_signal = SignalProcessor.create_signal_record(
                    org_id=self.org_id,
                    source=f"pagerduty:{self.connector_id}",
                    kind="event",
                    key="maintenance.ended",
                    value=1,
                    text=f"Maintenance ended: {window['summary']}",
                    ts=end_time_window,
                    labels={
                        "maintenance_id": window["id"],
                        "summary": window["summary"],
                        "status": "ended",
                        "services_affected": str(len(window.get("services", [])))
                    },
                    meta={
                        "connector_type": "pagerduty",
                        "connector_id": str(self.connector_id),
                        "maintenance_data": window
                    }
                )
                signals.append(end_signal)

        except Exception as e:
            logger.error(f"Failed to process maintenance window: {e}")

        return signals

    async def get_services(self) -> List[Dict[str, Any]]:
        """Get PagerDuty services"""
        try:
            response = await self._make_request("GET", "/services")
            return response.get("services", [])
        except Exception as e:
            logger.error(f"Failed to get services: {e}")
            return []

    async def get_teams(self) -> List[Dict[str, Any]]:
        """Get PagerDuty teams"""
        try:
            response = await self._make_request("GET", "/teams")
            return response.get("teams", [])
        except Exception as e:
            logger.error(f"Failed to get teams: {e}")
            return []

    async def get_escalation_policies(self) -> List[Dict[str, Any]]:
        """Get escalation policies"""
        try:
            response = await self._make_request("GET", "/escalation_policies")
            return response.get("escalation_policies", [])
        except Exception as e:
            logger.error(f"Failed to get escalation policies: {e}")
            return []

    async def create_incident(self, title: str, service_id: str, urgency: str = "high",
                             details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new incident"""
        try:
            incident_data = {
                "incident": {
                    "type": "incident",
                    "title": title,
                    "service": {
                        "id": service_id,
                        "type": "service"
                    },
                    "urgency": urgency,
                    "body": {
                        "type": "incident_body",
                        "details": details or {}
                    }
                }
            }

            response = await self._make_request("POST", "/incidents", json=incident_data)
            return response

        except Exception as e:
            logger.error(f"Failed to create incident: {e}")
            raise

    async def acknowledge_incident(self, incident_id: str) -> Dict[str, Any]:
        """Acknowledge an incident"""
        try:
            response = await self._make_request(
                "POST",
                f"/incidents/{incident_id}/merge",
                json={
                    "incidents": [{
                        "id": incident_id,
                        "type": "incident",
                        "status": "acknowledged"
                    }]
                }
            )
            return response

        except Exception as e:
            logger.error(f"Failed to acknowledge incident {incident_id}: {e}")
            raise

    async def resolve_incident(self, incident_id: str) -> Dict[str, Any]:
        """Resolve an incident"""
        try:
            response = await self._make_request(
                "PUT",
                f"/incidents/{incident_id}",
                json={
                    "incident": {
                        "id": incident_id,
                        "type": "incident",
                        "status": "resolved"
                    }
                }
            )
            return response

        except Exception as e:
            logger.error(f"Failed to resolve incident {incident_id}: {e}")
            raise
