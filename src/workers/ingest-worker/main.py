"""
Ingest worker for collecting data from external connectors
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID
import json

import nats
from pydantic import BaseModel

from ...shared.database.config import get_postgres_session, get_clickhouse_session
from ...shared.connectors.base import connector_registry
from ...shared.connectors.prometheus import PrometheusConnector
from ...shared.connectors.loki import LokiConnector
from ...shared.connectors.elasticsearch import ElasticsearchConnector
from ...shared.connectors.github import GitHubConnector
from ...shared.connectors.argocd import ArgoCDConnector
from ...shared.connectors.pagerduty import PagerDutyConnector
from ...shared.models.api import SignalCreate
from ...shared.models.base import Signal


logger = logging.getLogger(__name__)


class IngestConfig(BaseModel):
    """Ingest worker configuration"""
    batch_size: int = 1000
    collection_interval: int = 60  # seconds
    max_retries: int = 3
    backoff_factor: float = 2.0


class IngestWorker:
    """Worker for ingesting data from connectors"""

    def __init__(self, config: IngestConfig):
        self.config = config
        self.nc: Optional[nats.NATS] = None
        self.running = False
        self.tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """Start the ingest worker"""
        logger.info("Starting ingest worker")

        # Register connector classes
        connector_registry.register_connector_class("prometheus", PrometheusConnector)
        connector_registry.register_connector_class("loki", LokiConnector)
        connector_registry.register_connector_class("elasticsearch", ElasticsearchConnector)
        connector_registry.register_connector_class("github", GitHubConnector)
        connector_registry.register_connector_class("argocd", ArgoCDConnector)
        connector_registry.register_connector_class("pagerduty", PagerDutyConnector)

        # Connect to NATS
        self.nc = await nats.connect("nats://localhost:4222")

        # Initialize connectors
        await connector_registry.initialize_all()

        # Subscribe to ingestion topics
        await self._subscribe_topics()

        # Start collection loop
        self.running = True
        collection_task = asyncio.create_task(self._collection_loop())
        self.tasks.append(collection_task)

        logger.info("Ingest worker started successfully")

    async def stop(self) -> None:
        """Stop the ingest worker"""
        logger.info("Stopping ingest worker")
        self.running = False

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Close NATS connection
        if self.nc:
            await self.nc.close()

        # Cleanup connectors
        await connector_registry.cleanup_all()

        logger.info("Ingest worker stopped")

    async def _subscribe_topics(self) -> None:
        """Subscribe to NATS topics"""
        # Subscribe to signal ingestion requests
        await self.nc.subscribe("signal.ingest", cb=self._handle_ingest_request)

        # Subscribe to connector management
        await self.nc.subscribe("connector.create", cb=self._handle_connector_create)
        await self.nc.subscribe("connector.update", cb=self._handle_connector_update)
        await self.nc.subscribe("connector.delete", cb=self._handle_connector_delete)

    async def _collection_loop(self) -> None:
        """Main collection loop"""
        while self.running:
            try:
                await self._collect_from_all_connectors()
                await asyncio.sleep(self.config.collection_interval)
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying

    async def _collect_from_all_connectors(self) -> None:
        """Collect data from all active connectors"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)  # Collect last 5 minutes

        for connector_id, connector in connector_registry.connectors.items():
            try:
                logger.info(f"Collecting data from connector {connector_id}")

                # Collect data from connector
                signals_data = await connector.collect_data(start_time, end_time)

                if signals_data:
                    # Process and store signals
                    await self._process_signals(signals_data)

                    logger.info(f"Collected {len(signals_data)} signals from connector {connector_id}")
                else:
                    logger.debug(f"No data collected from connector {connector_id}")

            except Exception as e:
                logger.error(f"Failed to collect from connector {connector_id}: {e}")
                continue

    async def _process_signals(self, signals_data: List[Dict[str, Any]]) -> None:
        """Process and store collected signals"""
        try:
            # Convert to SignalCreate objects
            signals = []
            for data in signals_data:
                try:
                    signal = SignalCreate(**data)
                    signals.append(signal)
                except Exception as e:
                    logger.error(f"Failed to parse signal: {e}")
                    continue

            if not signals:
                return

            # Batch signals for efficient processing
            batch_size = self.config.batch_size
            for i in range(0, len(signals), batch_size):
                batch = signals[i:i + batch_size]
                await self._store_signal_batch(batch)

        except Exception as e:
            logger.error(f"Failed to process signals: {e}")

    async def _store_signal_batch(self, signals: List[SignalCreate]) -> None:
        """Store batch of signals to database"""
        try:
            async with get_postgres_session() as session:
                # Convert SignalCreate to database model
                db_signals = []
                for signal in signals:
                    db_signal = Signal(
                        org_id=signal.org_id,
                        source=signal.source,
                        kind=signal.kind,
                        ts=signal.ts,
                        key=signal.key,
                        value=str(signal.value) if signal.value is not None else None,
                        text=signal.text,
                        labels=json.dumps(signal.labels),
                        meta=json.dumps(signal.meta)
                    )
                    db_signals.append(db_signal)

                # Bulk insert
                session.add_all(db_signals)
                await session.commit()

                logger.debug(f"Stored {len(db_signals)} signals")

        except Exception as e:
            logger.error(f"Failed to store signal batch: {e}")
            raise

    async def _handle_ingest_request(self, msg) -> None:
        """Handle manual ingest request"""
        try:
            data = json.loads(msg.data.decode())
            connector_id = UUID(data["connector_id"])
            start_time = datetime.fromisoformat(data["start_time"])
            end_time = datetime.fromisoformat(data["end_time"])

            connector = connector_registry.get_connector(connector_id)
            if not connector:
                logger.error(f"Connector {connector_id} not found")
                return

            # Collect data from specific connector
            signals_data = await connector.collect_data(start_time, end_time)

            if signals_data:
                await self._process_signals(signals_data)
                logger.info(f"Manual ingest completed for connector {connector_id}")

        except Exception as e:
            logger.error(f"Failed to handle ingest request: {e}")

    async def _handle_connector_create(self, msg) -> None:
        """Handle connector creation"""
        try:
            data = json.loads(msg.data.decode())

            # Create connector instance
            from ...shared.models.api import ConnectorResponse
            connector_data = ConnectorResponse(**data)
            connector = connector_registry.create_connector(connector_data)

            # Initialize connector
            await connector.initialize()

            logger.info(f"Created connector {connector_data.id}")

        except Exception as e:
            logger.error(f"Failed to create connector: {e}")

    async def _handle_connector_update(self, msg) -> None:
        """Handle connector update"""
        try:
            data = json.loads(msg.data.decode())
            connector_id = UUID(data["id"])

            # Remove old connector
            connector_registry.remove_connector(connector_id)

            # Create new connector instance
            from ...shared.models.api import ConnectorResponse
            connector_data = ConnectorResponse(**data)
            connector = connector_registry.create_connector(connector_data)

            # Initialize new connector
            await connector.initialize()

            logger.info(f"Updated connector {connector_id}")

        except Exception as e:
            logger.error(f"Failed to update connector: {e}")

    async def _handle_connector_delete(self, msg) -> None:
        """Handle connector deletion"""
        try:
            data = json.loads(msg.data.decode())
            connector_id = UUID(data["id"])

            # Remove connector
            connector_registry.remove_connector(connector_id)

            logger.info(f"Deleted connector {connector_id}")

        except Exception as e:
            logger.error(f"Failed to delete connector: {e}")


async def main():
    """Main entry point"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    config = IngestConfig()

    # Create worker
    worker = IngestWorker(config)

    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info("Received signal, shutting down...")
        asyncio.create_task(worker.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start worker
        await worker.start()

        # Keep running until stopped
        while worker.running:
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Worker failed: {e}")
        sys.exit(1)
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
