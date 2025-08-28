"""
Elasticsearch connector for logs ingestion
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


class ElasticsearchConnector(HTTPConnector):
    """Elasticsearch logs connector"""

    @property
    def connector_type(self) -> str:
        return "elasticsearch"

    async def validate_config(self) -> bool:
        """Validate Elasticsearch connector configuration"""
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
        """Test connection to Elasticsearch"""
        try:
            response = await self._make_request("GET", "/")
            return response.get("cluster_name") is not None
        except Exception as e:
            logger.error(f"Elasticsearch connection test failed: {e}")
            return False

    async def collect_data(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect logs data from Elasticsearch"""
        signals = []

        # Get configured queries
        queries = self.get_config_value("queries", [{"match_all": {}}])

        for query in queries:
            try:
                query_signals = await self._collect_query_logs(query, start_time, end_time)
                signals.extend(query_signals)
            except Exception as e:
                logger.error(f"Failed to collect logs for query {query}: {e}")
                continue

        return signals

    async def _collect_query_logs(self, query: Dict[str, Any], start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect logs for a specific query"""
        signals = []

        try:
            # Build Elasticsearch query
            es_query = self._build_search_query(query, start_time, end_time)

            # Search with scroll for large result sets
            response = await self._make_request(
                "POST",
                "/_search?scroll=1m",
                json=es_query
            )

            scroll_id = response.get("_scroll_id")
            hits = response.get("hits", {}).get("hits", [])

            # Process initial batch
            for hit in hits:
                signal = self._parse_log_hit(hit)
                if signal:
                    signals.append(signal)

            # Continue scrolling if there are more results
            while len(hits) > 0 and len(signals) < self.get_config_value("max_results", 10000):
                scroll_response = await self._make_request(
                    "POST",
                    "/_search/scroll",
                    json={
                        "scroll": "1m",
                        "scroll_id": scroll_id
                    }
                )

                scroll_id = scroll_response.get("_scroll_id")
                hits = scroll_response.get("hits", {}).get("hits", [])

                for hit in hits:
                    signal = self._parse_log_hit(hit)
                    if signal:
                        signals.append(signal)

        except Exception as e:
            logger.error(f"Log query failed for {query}: {e}")

        return signals

    def _build_search_query(self, query: Dict[str, Any], start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Build Elasticsearch search query"""
        # Get configured index pattern
        index_pattern = self.get_config_value("index_pattern", "*")

        # Build time range filter
        time_field = self.get_config_value("time_field", "@timestamp")
        range_filter = {
            "range": {
                time_field: {
                    "gte": start_time.isoformat(),
                    "lte": end_time.isoformat()
                }
            }
        }

        # Combine user query with time filter
        bool_query = {
            "bool": {
                "must": [query],
                "filter": [range_filter]
            }
        }

        return {
            "query": bool_query,
            "size": self.get_config_value("batch_size", 1000),
            "sort": [{time_field: "asc"}],
            "_source": self.get_config_value("source_fields", True)
        }

    def _parse_log_hit(self, hit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse Elasticsearch hit into signal format"""
        try:
            source = hit.get("_source", {})
            index = hit.get("_index", "")
            doc_id = hit.get("_id", "")

            # Extract timestamp
            time_field = self.get_config_value("time_field", "@timestamp")
            timestamp_str = source.get(time_field)

            if not timestamp_str:
                return None

            try:
                # Handle different timestamp formats
                if isinstance(timestamp_str, str):
                    if timestamp_str.endswith('Z'):
                        log_timestamp = datetime.fromisoformat(timestamp_str[:-1])
                    else:
                        log_timestamp = datetime.fromisoformat(timestamp_str)
                else:
                    # Assume it's a timestamp
                    log_timestamp = datetime.fromtimestamp(timestamp_str / 1000 if timestamp_str > 1e10 else timestamp_str)
            except (ValueError, OSError):
                return None

            # Extract log message
            message_field = self.get_config_value("message_field", "message")
            log_message = source.get(message_field, "")

            # Extract log level
            level_field = self.get_config_value("level_field", "level")
            log_level = source.get(level_field, "unknown")

            # Extract error information
            error_info = self._extract_error_info(log_message)

            # Create labels from source fields
            labels = {
                "index": index,
                "id": doc_id,
                "level": log_level
            }

            # Add other relevant fields as labels
            label_fields = self.get_config_value("label_fields", ["host", "service", "component", "namespace"])
            for field in label_fields:
                if field in source:
                    labels[field] = str(source[field])

            # Create signal record
            signal = SignalProcessor.create_signal_record(
                org_id=self.org_id,
                source=f"elasticsearch:{self.connector_id}",
                kind="log",
                key=index,  # Use index as the key
                value=None,
                text=log_message,
                ts=log_timestamp,
                labels=labels,
                meta={
                    "connector_type": "elasticsearch",
                    "connector_id": str(self.connector_id),
                    "index": index,
                    "document_id": doc_id,
                    "source": source,
                    "log_level": log_level,
                    "error_info": error_info
                }
            )

            return signal

        except Exception as e:
            logger.error(f"Failed to parse log hit: {e}")
            return None

    def _extract_error_info(self, log_message: str) -> Dict[str, Any]:
        """Extract error information from log message"""
        error_info = {
            "has_error": False,
            "error_type": None,
            "stack_trace": False,
            "error_message": None
        }

        if not log_message:
            return error_info

        message_lower = log_message.lower()

        # Check for error indicators
        if any(word in message_lower for word in ["error", "err", "exception", "fail", "fatal"]):
            error_info["has_error"] = True

            # Try to identify error type
            if "panic" in message_lower:
                error_info["error_type"] = "panic"
            elif "exception" in message_lower:
                error_info["error_type"] = "exception"
            elif "timeout" in message_lower:
                error_info["error_type"] = "timeout"
            elif "connection" in message_lower and ("refused" in message_lower or "failed" in message_lower):
                error_info["error_type"] = "connection_error"
            else:
                error_info["error_type"] = "generic_error"

        # Check for stack traces
        if any(indicator in log_message for indicator in [" at ", ".java:", ".go:", ".py:", "Traceback"]):
            error_info["stack_trace"] = True

        # Extract error message
        if error_info["has_error"]:
            error_info["error_message"] = log_message.strip()[:500]  # Limit length

        return error_info

    async def get_indices(self) -> List[str]:
        """Get available indices"""
        try:
            response = await self._make_request("GET", "/_cat/indices?format=json")

            indices = []
            for index_info in response:
                if not index_info.get("index", "").startswith("."):  # Skip system indices
                    indices.append(index_info["index"])

            return indices

        except Exception as e:
            logger.error(f"Failed to get indices: {e}")
            return []

    async def get_mapping(self, index: str) -> Dict[str, Any]:
        """Get mapping for an index"""
        try:
            response = await self._make_request("GET", f"/{index}/_mapping")

            return response.get(index, {})

        except Exception as e:
            logger.error(f"Failed to get mapping for index {index}: {e}")
            return {}

    async def get_cluster_health(self) -> Dict[str, Any]:
        """Get cluster health information"""
        try:
            response = await self._make_request("GET", "/_cluster/health")
            return response

        except Exception as e:
            logger.error(f"Failed to get cluster health: {e}")
            return {}
