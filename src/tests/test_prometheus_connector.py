"""
Tests for Prometheus connector
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

from ..shared.connectors.prometheus import PrometheusConnector


class TestPrometheusConnector:
    """Test cases for Prometheus connector"""

    @pytest.fixture
    def connector_config(self):
        """Sample connector configuration"""
        return {
            "base_url": "http://prometheus:9090",
            "timeout": 30,
            "metrics": ["up", "cpu_usage"],
            "step_seconds": 300
        }

    @pytest.fixture
    def connector(self, connector_config):
        """Create test connector instance"""
        org_id = uuid4()
        connector_id = uuid4()
        return PrometheusConnector(connector_id, org_id, connector_config)

    @pytest.mark.asyncio
    async def test_validate_config_valid(self, connector):
        """Test config validation with valid config"""
        assert await connector.validate_config() is True

    @pytest.mark.asyncio
    async def test_validate_config_missing_url(self):
        """Test config validation with missing URL"""
        config = {"timeout": 30}
        connector = PrometheusConnector(uuid4(), uuid4(), config)

        with pytest.raises(Exception):  # Should raise ConnectorConfigError
            await connector.validate_config()

    @pytest.mark.asyncio
    async def test_validate_config_invalid_url(self):
        """Test config validation with invalid URL"""
        config = {"base_url": "invalid-url"}
        connector = PrometheusConnector(uuid4(), uuid4(), config)

        with pytest.raises(Exception):  # Should raise ConnectorConfigError
            await connector.validate_config()

    @pytest.mark.asyncio
    async def test_parse_metric_result(self, connector):
        """Test parsing of metric result"""
        result = {
            "metric": {
                "__name__": "cpu_usage",
                "instance": "localhost:9090",
                "job": "prometheus"
            },
            "value": [1640995200.0, "0.85"]
        }

        timestamp = datetime.fromtimestamp(1640995200.0)
        signal = connector._parse_metric_result(result, timestamp, "instant")

        assert signal is not None
        assert signal["key"] == "cpu.usage"
        assert signal["value"] == 0.85
        assert signal["labels"]["instance"] == "localhost:9090"
        assert signal["meta"]["query_type"] == "instant"

    @pytest.mark.asyncio
    async def test_parse_range_result(self, connector):
        """Test parsing of range query result"""
        result = {
            "metric": {
                "__name__": "memory_usage",
                "instance": "localhost:9090"
            },
            "values": [
                [1640995200.0, "1024"],
                [1640995260.0, "1025"],
                [1640995320.0, "1026"]
            ]
        }

        start_time = datetime.fromtimestamp(1640995200.0)
        end_time = datetime.fromtimestamp(1640995320.0)

        signals = connector._parse_range_result(result, start_time, end_time)

        assert len(signals) == 3
        assert all(s["key"] == "memory.usage" for s in signals)
        assert all(s["meta"]["query_type"] == "range" for s in signals)

    @pytest.mark.asyncio
    async def test_normalize_metric_name(self):
        """Test metric name normalization"""
        from ..shared.connectors.base import SignalProcessor

        assert SignalProcessor.normalize_metric_name("cpu_usage") == "cpu.usage"
        assert SignalProcessor.normalize_metric_name("http_requests_total") == "http.requests.total"
        assert SignalProcessor.normalize_metric_name("__name__") == "name"

    @pytest.mark.asyncio
    async def test_create_signal_record(self):
        """Test signal record creation"""
        from ..shared.connectors.base import SignalProcessor

        org_id = uuid4()
        ts = datetime.utcnow()

        signal = SignalProcessor.create_signal_record(
            org_id=org_id,
            source="prometheus:test",
            kind="metric",
            key="cpu.usage",
            value=0.85,
            text=None,
            ts=ts,
            labels={"instance": "localhost"},
            meta={"test": True}
        )

        assert signal["org_id"] == str(org_id)
        assert signal["source"] == "prometheus:test"
        assert signal["kind"] == "metric"
        assert signal["key"] == "cpu.usage"
        assert signal["value"] == 0.85
        assert signal["labels"]["instance"] == "localhost"
        assert signal["meta"]["test"] is True


if __name__ == "__main__":
    pytest.main([__file__])
