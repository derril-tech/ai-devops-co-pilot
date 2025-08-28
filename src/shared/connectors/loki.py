"""
Loki connector for logs ingestion
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


class LokiConnector(HTTPConnector):
    """Loki logs connector"""

    @property
    def connector_type(self) -> str:
        return "loki"

    async def validate_config(self) -> bool:
        """Validate Loki connector configuration"""
        required_fields = ["base_url"]
        for field in required_fields:
            if not self.get_config_value(field):
                raise ConnectorConfigError(f"Missing required field: {field}")

        # Validate URL format
        base_url = self.get_config_value("base_url")
        if not base_url.startswith(("http://", "https://")):
            raise ConnectorConfigError("base_url must start with http:// or https://")

        return True

    async def test_connection(self) -> bool:
        """Test connection to Loki"""
        try:
            response = await self._make_request("GET", "/ready")
            return response.get("status") == "ready"
        except Exception as e:
            logger.error(f"Loki connection test failed: {e}")
            return False

    async def collect_data(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect logs data from Loki"""
        signals = []

        # Get configured queries
        queries = self.get_config_value("queries", ["{job=~\".+\"}"])

        for query in queries:
            try:
                query_signals = await self._collect_query_logs(query, start_time, end_time)
                signals.extend(query_signals)
            except Exception as e:
                logger.error(f"Failed to collect logs for query {query}: {e}")
                continue

        return signals

    async def _collect_query_logs(self, query: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect logs for a specific query"""
        signals = []

        try:
            # Query Loki for log entries
            response = await self._make_request(
                "GET",
                "/loki/api/v1/query_range",
                {
                    "query": query,
                    "start": str(int(start_time.timestamp() * 1e9)),  # nanoseconds
                    "end": str(int(end_time.timestamp() * 1e9)),
                    "limit": self.get_config_value("limit", 5000)
                }
            )

            if response.get("status") != "success":
                logger.error(f"Loki query failed: {response}")
                return signals

            result = response.get("data", {}).get("result", [])
            for stream in result:
                stream_signals = self._parse_log_stream(stream, start_time, end_time)
                signals.extend(stream_signals)

        except Exception as e:
            logger.error(f"Log query failed for {query}: {e}")

        return signals

    def _parse_log_stream(self, stream: Dict[str, Any], start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Parse log stream into signal format"""
        signals = []

        try:
            labels = stream.get("stream", {})
            values = stream.get("values", [])

            for value in values:
                if len(value) != 2:
                    continue

                try:
                    timestamp_ns = int(value[0])
                    log_line = value[1]
                    log_timestamp = datetime.fromtimestamp(timestamp_ns / 1e9)
                except (ValueError, IndexError, OSError):
                    continue

                # Only include logs within our time range
                if not (start_time <= log_timestamp <= end_time):
                    continue

                # Extract log level and other metadata
                log_level = self._extract_log_level(log_line, labels)
                error_info = self._extract_error_info(log_line)

                # Create signal record
                signal = SignalProcessor.create_signal_record(
                    org_id=self.org_id,
                    source=f"loki:{self.connector_id}",
                    kind="log",
                    key=SignalProcessor.normalize_metric_name(labels.get("job", "unknown")),
                    value=None,
                    text=log_line,
                    ts=log_timestamp,
                    labels={
                        **labels,
                        "level": log_level,
                        "has_error": error_info["has_error"],
                        "error_type": error_info["error_type"]
                    },
                    meta={
                        "query_type": "range",
                        "connector_type": "loki",
                        "connector_id": str(self.connector_id),
                        "stream_labels": labels,
                        "log_level": log_level,
                        "error_info": error_info
                    }
                )

                signals.append(signal)

        except Exception as e:
            logger.error(f"Failed to parse log stream: {e}")

        return signals

    def _extract_log_level(self, log_line: str, labels: Dict[str, Any]) -> str:
        """Extract log level from log line or labels"""
        # Check labels first
        if "level" in labels:
            return labels["level"].lower()

        # Common log level patterns
        log_line_lower = log_line.lower()
        if any(word in log_line_lower for word in ["error", "err", "exception", "fail"]):
            return "error"
        elif any(word in log_line_lower for word in ["warn", "warning"]):
            return "warning"
        elif any(word in log_line_lower for word in ["info", "information"]):
            return "info"
        elif any(word in log_line_lower for word in ["debug", "trace"]):
            return "debug"

        return "unknown"

    def _extract_error_info(self, log_line: str) -> Dict[str, Any]:
        """Extract error information from log line"""
        error_info = {
            "has_error": False,
            "error_type": None,
            "stack_trace": False,
            "error_message": None
        }

        log_line_lower = log_line.lower()

        # Check for error indicators
        if any(word in log_line_lower for word in ["error", "err", "exception", "fail", "panic"]):
            error_info["has_error"] = True

            # Try to identify error type
            if "panic" in log_line_lower:
                error_info["error_type"] = "panic"
            elif "exception" in log_line_lower:
                error_info["error_type"] = "exception"
            elif "timeout" in log_line_lower:
                error_info["error_type"] = "timeout"
            elif "connection" in log_line_lower and ("refused" in log_line_lower or "failed" in log_line_lower):
                error_info["error_type"] = "connection_error"
            else:
                error_info["error_type"] = "generic_error"

        # Check for stack traces
        if any(indicator in log_line for indicator in [" at ", ".java:", ".go:", ".py:"]):
            error_info["stack_trace"] = True

        # Extract error message (simplified)
        if error_info["has_error"]:
            # Try to get a concise error message
            lines = log_line.split('\n')
            for line in lines:
                if any(word in line.lower() for word in ["error", "err", "exception", "fail"]):
                    error_info["error_message"] = line.strip()[:200]  # Limit length
                    break

        return error_info

    async def get_labels(self) -> Dict[str, List[str]]:
        """Get available label names and values"""
        try:
            response = await self._make_request("GET", "/loki/api/v1/labels")

            if response.get("status") != "success":
                return {}

            labels = {}
            for label_name in response.get("data", []):
                values = await self._get_label_values(label_name)
                labels[label_name] = values

            return labels

        except Exception as e:
            logger.error(f"Failed to get labels: {e}")
            return {}

    async def _get_label_values(self, label_name: str) -> List[str]:
        """Get values for a specific label"""
        try:
            response = await self._make_request(
                "GET",
                "/loki/api/v1/label/{}/values".format(label_name)
            )

            if response.get("status") == "success":
                return response.get("data", [])

        except Exception as e:
            logger.error(f"Failed to get values for label {label_name}: {e}")

        return []

    async def get_series(self, match: List[str], start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get series metadata for label matchers"""
        try:
            response = await self._make_request(
                "GET",
                "/loki/api/v1/series",
                {
                    "match": match,
                    "start": str(int(start_time.timestamp() * 1e9)),
                    "end": str(int(end_time.timestamp() * 1e9))
                }
            )

            if response.get("status") == "success":
                return response.get("data", [])

        except Exception as e:
            logger.error(f"Failed to get series: {e}")

        return []
