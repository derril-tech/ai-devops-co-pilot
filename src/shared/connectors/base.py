"""
Base connector interface and utilities
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID
import aiohttp
import backoff

from ..models.api import ConnectorCreate, ConnectorResponse


logger = logging.getLogger(__name__)


class ConnectorError(Exception):
    """Base exception for connector errors"""
    pass


class ConnectorConfigError(ConnectorError):
    """Configuration error for connectors"""
    pass


class ConnectorConnectionError(ConnectorError):
    """Connection error for connectors"""
    pass


class ConnectorAuthError(ConnectorError):
    """Authentication error for connectors"""
    pass


class BaseConnector(ABC):
    """Abstract base class for all connectors"""

    def __init__(self, connector_id: UUID, org_id: UUID, config: Dict[str, Any]):
        self.connector_id = connector_id
        self.org_id = org_id
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def connector_type(self) -> str:
        """Return the connector type identifier"""
        pass

    @abstractmethod
    async def validate_config(self) -> bool:
        """Validate connector configuration"""
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """Test connection to external service"""
        pass

    @abstractmethod
    async def collect_data(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect data from external service"""
        pass

    async def initialize(self) -> None:
        """Initialize connector (called once on startup)"""
        pass

    async def cleanup(self) -> None:
        """Cleanup connector resources (called on shutdown)"""
        pass

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default"""
        return self.config.get(key, default)

    def require_config_value(self, key: str) -> Any:
        """Get required configuration value or raise error"""
        value = self.get_config_value(key)
        if value is None:
            raise ConnectorConfigError(f"Required configuration '{key}' is missing")
        return value


class HTTPConnector(BaseConnector):
    """Base class for HTTP-based connectors"""

    def __init__(self, connector_id: UUID, org_id: UUID, config: Dict[str, Any]):
        super().__init__(connector_id, org_id, config)
        self.base_url = self.require_config_value("base_url")
        self.timeout = self.get_config_value("timeout", 30)
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None:
        """Initialize HTTP session"""
        await super().initialize()

        # Configure authentication
        auth = None
        if self.get_config_value("username") and self.get_config_value("password"):
            auth = aiohttp.BasicAuth(
                self.get_config_value("username"),
                self.get_config_value("password")
            )

        # Configure headers
        headers = {}
        if self.get_config_value("api_key"):
            headers["Authorization"] = f"Bearer {self.get_config_value('api_key')}"
        elif self.get_config_value("token"):
            headers["Authorization"] = f"Token {self.get_config_value('token')}"

        # Create session
        self.session = aiohttp.ClientSession(
            base_url=self.base_url,
            auth=auth,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )

    async def cleanup(self) -> None:
        """Close HTTP session"""
        if self.session:
            await self.session.close()
        await super().cleanup()

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=60
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        if not self.session:
            raise ConnectorConnectionError("HTTP session not initialized")

        try:
            async with self.session.request(
                method,
                endpoint,
                params=params,
                data=data,
                json=json
            ) as response:
                if response.status == 401:
                    raise ConnectorAuthError("Authentication failed")
                elif response.status == 403:
                    raise ConnectorAuthError("Access forbidden")
                elif response.status >= 400:
                    text = await response.text()
                    raise ConnectorConnectionError(f"HTTP {response.status}: {text}")

                return await response.json()

        except aiohttp.ClientError as e:
            raise ConnectorConnectionError(f"HTTP request failed: {e}")

    async def test_connection(self) -> bool:
        """Test HTTP connection"""
        try:
            await self._make_request("GET", "/health")
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False


class SignalProcessor:
    """Utility for processing and normalizing signals"""

    @staticmethod
    def normalize_metric_name(name: str) -> str:
        """Normalize metric name to standard format"""
        return name.replace("__", ".").replace("_", ".").lower()

    @staticmethod
    def create_signal_record(
        org_id: UUID,
        source: str,
        kind: str,
        key: str,
        value: Optional[float],
        text: Optional[str],
        ts: datetime,
        labels: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create normalized signal record"""
        return {
            "org_id": str(org_id),
            "source": source,
            "kind": kind,
            "key": key,
            "value": value,
            "text": text,
            "ts": ts.isoformat(),
            "labels": labels or {},
            "meta": meta or {},
            "created_at": datetime.utcnow().isoformat()
        }

    @staticmethod
    def batch_signals(signals: List[Dict[str, Any]], batch_size: int = 1000) -> List[List[Dict[str, Any]]]:
        """Batch signals for efficient processing"""
        return [signals[i:i + batch_size] for i in range(0, len(signals), batch_size)]


class ConnectorRegistry:
    """Registry for managing connector instances"""

    def __init__(self):
        self.connectors: Dict[UUID, BaseConnector] = {}
        self.connector_classes: Dict[str, type] = {}

    def register_connector_class(self, connector_type: str, connector_class: type):
        """Register a connector class"""
        self.connector_classes[connector_type] = connector_class

    def create_connector(self, connector_data: ConnectorResponse) -> BaseConnector:
        """Create connector instance"""
        connector_class = self.connector_classes.get(connector_data.kind)
        if not connector_class:
            raise ConnectorError(f"Unknown connector type: {connector_data.kind}")

        connector = connector_class(
            connector_id=connector_data.id,
            org_id=connector_data.org_id,
            config=connector_data.config
        )

        self.connectors[connector_data.id] = connector
        return connector

    def get_connector(self, connector_id: UUID) -> Optional[BaseConnector]:
        """Get connector instance"""
        return self.connectors.get(connector_id)

    def remove_connector(self, connector_id: UUID) -> None:
        """Remove connector instance"""
        connector = self.connectors.pop(connector_id, None)
        if connector:
            asyncio.create_task(connector.cleanup())

    async def initialize_all(self) -> None:
        """Initialize all connectors"""
        for connector in self.connectors.values():
            try:
                await connector.initialize()
            except Exception as e:
                self.logger.error(f"Failed to initialize connector {connector.connector_id}: {e}")

    async def cleanup_all(self) -> None:
        """Cleanup all connectors"""
        for connector in self.connectors.values():
            try:
                await connector.cleanup()
            except Exception as e:
                self.logger.error(f"Failed to cleanup connector {connector.connector_id}: {e}")


# Global connector registry
connector_registry = ConnectorRegistry()
