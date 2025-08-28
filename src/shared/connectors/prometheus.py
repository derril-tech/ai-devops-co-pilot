"""
Prometheus connector for metrics ingestion
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID
import aiohttp
import backoff

from .base import HTTPConnector, SignalProcessor, ConnectorError, ConnectorConfigError


logger = logging.getLogger(__name__)


class PrometheusConnector(HTTPConnector):
    """Prometheus metrics connector"""

    @property
    def connector_type(self) -> str:
        return "prometheus"

    async def validate_config(self) -> bool:
        """Validate Prometheus connector configuration"""
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
        """Test connection to Prometheus"""
        try:
            response = await self._make_request("GET", "/api/v1/query", {"query": "up"})
            return response.get("status") == "success"
        except Exception as e:
            logger.error(f"Prometheus connection test failed: {e}")
            return False

    async def collect_data(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect metrics data from Prometheus"""
        signals = []

        # Get configured metrics to collect
        metrics = self.get_config_value("metrics", ["up", "http_requests_total", "cpu_usage", "memory_usage"])

        for metric in metrics:
            try:
                metric_signals = await self._collect_metric(metric, start_time, end_time)
                signals.extend(metric_signals)
            except Exception as e:
                logger.error(f"Failed to collect metric {metric}: {e}")
                continue

        return signals

    async def _collect_metric(self, metric_name: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect specific metric data"""
        signals = []

        # Query for instant values
        instant_signals = await self._query_instant(metric_name, end_time)
        signals.extend(instant_signals)

        # Query for range values if configured
        if self.get_config_value("collect_range_data", True):
            range_signals = await self._query_range(metric_name, start_time, end_time)
            signals.extend(range_signals)

        return signals

    async def _query_instant(self, query: str, timestamp: datetime) -> List[Dict[str, Any]]:
        """Query instant metric values"""
        signals = []

        try:
            response = await self._make_request(
                "GET",
                "/api/v1/query",
                {
                    "query": query,
                    "time": timestamp.timestamp()
                }
            )

            if response.get("status") != "success":
                logger.error(f"Prometheus query failed: {response}")
                return signals

            result = response.get("data", {}).get("result", [])
            for item in result:
                signal = self._parse_metric_result(item, timestamp, "instant")
                if signal:
                    signals.append(signal)

        except Exception as e:
            logger.error(f"Instant query failed for {query}: {e}")

        return signals

    async def _query_range(self, query: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Query metric values over a time range"""
        signals = []

        try:
            # Use 5-minute step for range queries
            step_seconds = self.get_config_value("step_seconds", 300)

            response = await self._make_request(
                "GET",
                "/api/v1/query_range",
                {
                    "query": query,
                    "start": start_time.timestamp(),
                    "end": end_time.timestamp(),
                    "step": step_seconds
                }
            )

            if response.get("status") != "success":
                logger.error(f"Prometheus range query failed: {response}")
                return signals

            result = response.get("data", {}).get("result", [])
            for item in result:
                series_signals = self._parse_range_result(item, start_time, end_time)
                signals.extend(series_signals)

        except Exception as e:
            logger.error(f"Range query failed for {query}: {e}")

        return signals

    def _parse_metric_result(self, result: Dict[str, Any], timestamp: datetime, query_type: str) -> Optional[Dict[str, Any]]:
        """Parse metric result into signal format"""
        try:
            metric = result.get("metric", {})
            value = result.get("value")

            if not value or len(value) != 2:
                return None

            # Extract value and timestamp
            try:
                metric_value = float(value[1])
                metric_timestamp = datetime.fromtimestamp(float(value[0]))
            except (ValueError, IndexError):
                return None

            # Create signal record
            signal = SignalProcessor.create_signal_record(
                org_id=self.org_id,
                source=f"prometheus:{self.connector_id}",
                kind="metric",
                key=SignalProcessor.normalize_metric_name(metric.get("__name__", "unknown")),
                value=metric_value,
                text=None,
                ts=metric_timestamp,
                labels=metric,
                meta={
                    "query_type": query_type,
                    "connector_type": "prometheus",
                    "connector_id": str(self.connector_id)
                }
            )

            return signal

        except Exception as e:
            logger.error(f"Failed to parse metric result: {e}")
            return None

    def _parse_range_result(self, result: Dict[str, Any], start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Parse range query result into signal format"""
        signals = []

        try:
            metric = result.get("metric", {})
            values = result.get("values", [])

            for value in values:
                if len(value) != 2:
                    continue

                try:
                    metric_value = float(value[1])
                    metric_timestamp = datetime.fromtimestamp(float(value[0]))
                except (ValueError, IndexError):
                    continue

                # Only include values within our time range
                if not (start_time <= metric_timestamp <= end_time):
                    continue

                signal = SignalProcessor.create_signal_record(
                    org_id=self.org_id,
                    source=f"prometheus:{self.connector_id}",
                    kind="metric",
                    key=SignalProcessor.normalize_metric_name(metric.get("__name__", "unknown")),
                    value=metric_value,
                    text=None,
                    ts=metric_timestamp,
                    labels=metric,
                    meta={
                        "query_type": "range",
                        "connector_type": "prometheus",
                        "connector_id": str(self.connector_id)
                    }
                )

                signals.append(signal)

        except Exception as e:
            logger.error(f"Failed to parse range result: {e}")

        return signals

    async def get_metric_metadata(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific metric"""
        try:
            response = await self._make_request(
                "GET",
                "/api/v1/metadata",
                {"metric": metric_name}
            )

            if response.get("status") == "success":
                data = response.get("data", {})
                return data.get(metric_name, [{}])[0] if data.get(metric_name) else None

        except Exception as e:
            logger.error(f"Failed to get metadata for {metric_name}: {e}")

        return None

    async def discover_metrics(self, pattern: str = ".*") -> List[str]:
        """Discover available metrics matching pattern"""
        try:
            # Query for all metric names
            response = await self._make_request(
                "GET",
                "/api/v1/label/__name__/values"
            )

            if response.get("status") != "success":
                return []

            import re
            metrics = response.get("data", [])
            compiled_pattern = re.compile(pattern)

            return [metric for metric in metrics if compiled_pattern.match(metric)]

        except Exception as e:
            logger.error(f"Failed to discover metrics: {e}")
            return []

    async def get_targets_health(self) -> List[Dict[str, Any]]:
        """Get health status of Prometheus targets"""
        try:
            response = await self._make_request("GET", "/api/v1/targets")

            if response.get("status") != "success":
                return []

            targets = []
            for target in response.get("data", {}).get("activeTargets", []):
                targets.append({
                    "labels": target.get("labels", {}),
                    "health": target.get("health", "unknown"),
                    "lastError": target.get("lastError", ""),
                    "lastScrape": target.get("lastScrape", ""),
                    "scrapeUrl": target.get("scrapeUrl", "")
                })

            return targets

        except Exception as e:
            logger.error(f"Failed to get targets health: {e}")
            return []
