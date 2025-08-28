"""
GitHub connector for PR and deployment events
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


class GitHubConnector(HTTPConnector):
    """GitHub connector for PR and deployment events"""

    @property
    def connector_type(self) -> str:
        return "github"

    async def validate_config(self) -> bool:
        """Validate GitHub connector configuration"""
        required_fields = ["base_url", "owner", "repo"]
        for field in required_fields:
            if not self.get_config_value(field):
                raise ConnectorConfigError(f"Missing required field: {field}")

        # Validate authentication
        if not (self.get_config_value("token") or self.get_config_value("username") and self.get_config_value("password")):
            raise ConnectorConfigError("Authentication required: token or username/password")

        return True

    async def test_connection(self) -> bool:
        """Test connection to GitHub"""
        try:
            owner = self.get_config_value("owner")
            repo = self.get_config_value("repo")
            response = await self._make_request("GET", f"/repos/{owner}/{repo}")
            return response.get("id") is not None
        except Exception as e:
            logger.error(f"GitHub connection test failed: {e}")
            return False

    async def collect_data(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect PR and deployment data from GitHub"""
        signals = []

        # Collect different types of events
        event_types = self.get_config_value("event_types", ["pull_requests", "deployments", "releases"])

        for event_type in event_types:
            try:
                if event_type == "pull_requests":
                    event_signals = await self._collect_pull_requests(start_time, end_time)
                elif event_type == "deployments":
                    event_signals = await self._collect_deployments(start_time, end_time)
                elif event_type == "releases":
                    event_signals = await self._collect_releases(start_time, end_time)
                else:
                    logger.warning(f"Unknown event type: {event_type}")
                    continue

                signals.extend(event_signals)

            except Exception as e:
                logger.error(f"Failed to collect {event_type}: {e}")
                continue

        return signals

    async def _collect_pull_requests(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect pull request events"""
        signals = []

        try:
            owner = self.get_config_value("owner")
            repo = self.get_config_value("repo")

            # Get PRs updated in the time range
            response = await self._make_request(
                "GET",
                f"/repos/{owner}/{repo}/pulls",
                {
                    "state": "all",
                    "sort": "updated",
                    "direction": "desc",
                    "per_page": 100
                }
            )

            for pr in response:
                pr_signals = self._process_pull_request(pr, start_time, end_time)
                signals.extend(pr_signals)

        except Exception as e:
            logger.error(f"Failed to collect pull requests: {e}")

        return signals

    async def _collect_deployments(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect deployment events"""
        signals = []

        try:
            owner = self.get_config_value("owner")
            repo = self.get_config_value("repo")

            # Get deployments
            response = await self._make_request(
                "GET",
                f"/repos/{owner}/{repo}/deployments",
                {"per_page": 100}
            )

            for deployment in response:
                deployment_signals = await self._process_deployment(deployment, start_time, end_time)
                signals.extend(deployment_signals)

        except Exception as e:
            logger.error(f"Failed to collect deployments: {e}")

        return signals

    async def _collect_releases(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect release events"""
        signals = []

        try:
            owner = self.get_config_value("owner")
            repo = self.get_config_value("repo")

            # Get releases
            response = await self._make_request(
                "GET",
                f"/repos/{owner}/{repo}/releases",
                {"per_page": 100}
            )

            for release in response:
                release_signals = self._process_release(release, start_time, end_time)
                signals.extend(release_signals)

        except Exception as e:
            logger.error(f"Failed to collect releases: {e}")

        return signals

    def _process_pull_request(self, pr: Dict[str, Any], start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Process a pull request into signals"""
        signals = []

        try:
            # Extract relevant timestamps
            created_at = datetime.fromisoformat(pr["created_at"].replace('Z', '+00:00'))
            updated_at = datetime.fromisoformat(pr["updated_at"].replace('Z', '+00:00'))
            merged_at = None
            if pr.get("merged_at"):
                merged_at = datetime.fromisoformat(pr["merged_at"].replace('Z', '+00:00'))

            # Create signals for different PR events
            events = []

            # PR created
            if start_time <= created_at <= end_time:
                events.append({
                    "timestamp": created_at,
                    "event_type": "pr_created",
                    "value": 1
                })

            # PR updated
            if start_time <= updated_at <= end_time:
                events.append({
                    "timestamp": updated_at,
                    "event_type": "pr_updated",
                    "value": 1
                })

            # PR merged
            if merged_at and start_time <= merged_at <= end_time:
                events.append({
                    "timestamp": merged_at,
                    "event_type": "pr_merged",
                    "value": 1
                })

            # Generate signals for each event
            for event in events:
                signal = SignalProcessor.create_signal_record(
                    org_id=self.org_id,
                    source=f"github:{self.connector_id}",
                    kind="event",
                    key=f"pr.{event['event_type']}",
                    value=event["value"],
                    text=f"PR #{pr['number']}: {pr['title']}",
                    ts=event["timestamp"],
                    labels={
                        "repo": f"{self.get_config_value('owner')}/{self.get_config_value('repo')}",
                        "pr_number": str(pr["number"]),
                        "pr_title": pr["title"],
                        "author": pr["user"]["login"] if pr.get("user") else "unknown",
                        "state": pr["state"],
                        "merged": str(pr.get("merged", False)),
                        "draft": str(pr.get("draft", False))
                    },
                    meta={
                        "connector_type": "github",
                        "connector_id": str(self.connector_id),
                        "event_type": event["event_type"],
                        "pr_data": pr
                    }
                )
                signals.append(signal)

        except Exception as e:
            logger.error(f"Failed to process pull request: {e}")

        return signals

    async def _process_deployment(self, deployment: Dict[str, Any], start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Process a deployment into signals"""
        signals = []

        try:
            created_at = datetime.fromisoformat(deployment["created_at"].replace('Z', '+00:00'))

            # Only process if within time range
            if not (start_time <= created_at <= end_time):
                return signals

            # Get deployment statuses
            deployment_id = deployment["id"]
            owner = self.get_config_value("owner")
            repo = self.get_config_value("repo")

            statuses_response = await self._make_request(
                "GET",
                f"/repos/{owner}/{repo}/deployments/{deployment_id}/statuses"
            )

            # Create deployment signal
            signal = SignalProcessor.create_signal_record(
                org_id=self.org_id,
                source=f"github:{self.connector_id}",
                kind="event",
                key="deployment.created",
                value=1,
                text=f"Deployment: {deployment.get('description', 'No description')}",
                ts=created_at,
                labels={
                    "repo": f"{owner}/{repo}",
                    "deployment_id": str(deployment_id),
                    "environment": deployment.get("environment", "production"),
                    "ref": deployment.get("ref", "unknown"),
                    "sha": deployment.get("sha", "unknown")[:7]
                },
                meta={
                    "connector_type": "github",
                    "connector_id": str(self.connector_id),
                    "deployment_data": deployment,
                    "statuses": statuses_response
                }
            )
            signals.append(signal)

            # Create status signals
            for status in statuses_response:
                status_created_at = datetime.fromisoformat(status["created_at"].replace('Z', '+00:00'))

                if start_time <= status_created_at <= end_time:
                    status_signal = SignalProcessor.create_signal_record(
                        org_id=self.org_id,
                        source=f"github:{self.connector_id}",
                        kind="event",
                        key=f"deployment.{status['state']}",
                        value=1,
                        text=f"Deployment {status['state']}: {status.get('description', '')}",
                        ts=status_created_at,
                        labels={
                            "repo": f"{owner}/{repo}",
                            "deployment_id": str(deployment_id),
                            "status_id": str(status["id"]),
                            "state": status["state"],
                            "environment": deployment.get("environment", "production")
                        },
                        meta={
                            "connector_type": "github",
                            "connector_id": str(self.connector_id),
                            "deployment_id": deployment_id,
                            "status_data": status
                        }
                    )
                    signals.append(status_signal)

        except Exception as e:
            logger.error(f"Failed to process deployment: {e}")

        return signals

    def _process_release(self, release: Dict[str, Any], start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Process a release into signals"""
        signals = []

        try:
            published_at = None
            if release.get("published_at"):
                published_at = datetime.fromisoformat(release["published_at"].replace('Z', '+00:00'))

            created_at = datetime.fromisoformat(release["created_at"].replace('Z', '+00:00'))

            # Use published_at if available, otherwise created_at
            event_time = published_at or created_at

            if not (start_time <= event_time <= end_time):
                return signals

            signal = SignalProcessor.create_signal_record(
                org_id=self.org_id,
                source=f"github:{self.connector_id}",
                kind="event",
                key="release.published" if published_at else "release.created",
                value=1,
                text=f"Release {release['tag_name']}: {release.get('name', 'No name')}",
                ts=event_time,
                labels={
                    "repo": f"{self.get_config_value('owner')}/{self.get_config_value('repo')}",
                    "release_id": str(release["id"]),
                    "tag_name": release["tag_name"],
                    "name": release.get("name", ""),
                    "author": release["author"]["login"] if release.get("author") else "unknown",
                    "prerelease": str(release.get("prerelease", False)),
                    "draft": str(release.get("draft", False))
                },
                meta={
                    "connector_type": "github",
                    "connector_id": str(self.connector_id),
                    "release_data": release
                }
            )
            signals.append(signal)

        except Exception as e:
            logger.error(f"Failed to process release: {e}")

        return signals

    async def get_webhooks(self) -> List[Dict[str, Any]]:
        """Get repository webhooks"""
        try:
            owner = self.get_config_value("owner")
            repo = self.get_config_value("repo")

            response = await self._make_request("GET", f"/repos/{owner}/{repo}/hooks")
            return response

        except Exception as e:
            logger.error(f"Failed to get webhooks: {e}")
            return []

    async def create_webhook(self, webhook_url: str, events: List[str] = None) -> Dict[str, Any]:
        """Create a webhook for real-time events"""
        try:
            if events is None:
                events = ["pull_request", "deployment", "release"]

            owner = self.get_config_value("owner")
            repo = self.get_config_value("repo")

            webhook_config = {
                "url": webhook_url,
                "content_type": "json",
                "secret": self.get_config_value("webhook_secret", "")
            }

            response = await self._make_request(
                "POST",
                f"/repos/{owner}/{repo}/hooks",
                json={
                    "name": "web",
                    "active": True,
                    "events": events,
                    "config": webhook_config
                }
            )

            return response

        except Exception as e:
            logger.error(f"Failed to create webhook: {e}")
            raise
