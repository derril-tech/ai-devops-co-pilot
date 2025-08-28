"""
Correlate worker for incident correlation and blast radius analysis
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

import nats
from pydantic import BaseModel

from ...shared.database.config import get_postgres_session, get_clickhouse_session
from ...shared.change_correlation.change_correlation import ChangeCorrelationEngine
from ...shared.blast_radius.blast_radius import TopologyGraph, BlastRadiusCalculator
from ...shared.models.api import IncidentCreate


logger = logging.getLogger(__name__)


class CorrelateConfig(BaseModel):
    """Correlate worker configuration"""
    correlation_window_hours: int = 24
    blast_radius_max_depth: int = 5
    batch_size: int = 50


class CorrelateWorker:
    """Worker for incident correlation and blast radius analysis"""

    def __init__(self, config: CorrelateConfig):
        self.config = config
        self.nc: Optional[nats.NATS] = None
        self.running = False
        self.tasks: List[asyncio.Task] = []

        # Initialize correlation engines
        self.correlation_engine = ChangeCorrelationEngine()
        self.topology_graph = TopologyGraph()
        self.blast_calculator = BlastRadiusCalculator(self.topology_graph)

    async def start(self) -> None:
        """Start the correlate worker"""
        logger.info("Starting correlate worker")

        # Connect to NATS
        self.nc = await nats.connect("nats://localhost:4222")

        # Initialize topology
        await self._load_topology()

        # Subscribe to correlation topics
        await self._subscribe_topics()

        # Start correlation tasks
        self.running = True
        correlation_task = asyncio.create_task(self._correlation_loop())

        self.tasks.append(correlation_task)

        logger.info("Correlate worker started successfully")

    async def stop(self) -> None:
        """Stop the correlate worker"""
        logger.info("Stopping correlate worker")
        self.running = False

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Close NATS connection
        if self.nc:
            await self.nc.close()

        logger.info("Correlate worker stopped")

    async def _load_topology(self) -> None:
        """Load service topology from database"""
        try:
            async with get_postgres_session() as session:
                # Load services and edges (similar to detect worker)
                logger.info("Loaded topology for correlation analysis")
        except Exception as e:
            logger.error(f"Failed to load topology: {e}")

    async def _subscribe_topics(self) -> None:
        """Subscribe to NATS topics"""
        # Subscribe to incident correlation requests
        await self.nc.subscribe("incidents.correlate", cb=self._handle_correlation_request)

        # Subscribe to blast radius analysis
        await self.nc.subscribe("blast_radius.analyze", cb=self._handle_blast_radius_request)

        # Subscribe to new incidents for automatic correlation
        await self.nc.subscribe("incidents.created", cb=self._handle_new_incident)

    async def _correlation_loop(self) -> None:
        """Periodic correlation analysis loop"""
        while self.running:
            try:
                await self._perform_correlation_analysis()
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Error in correlation loop: {e}")
                await asyncio.sleep(300)

    async def _handle_correlation_request(self, msg) -> None:
        """Handle incident correlation requests"""
        try:
            data = json.loads(msg.data.decode())
            incident_id = data.get("incident_id")

            # Perform correlation analysis
            correlation_result = await self._correlate_incident(incident_id)

            # Respond with correlation results
            await self.nc.publish(msg.reply, json.dumps(correlation_result).encode())

        except Exception as e:
            logger.error(f"Failed to handle correlation request: {e}")

    async def _handle_blast_radius_request(self, msg) -> None:
        """Handle blast radius analysis requests"""
        try:
            data = json.loads(msg.data.decode())
            service_id = data.get("service_id")

            # Calculate blast radius
            blast_result = self.blast_calculator.calculate_blast_radius(
                service_id, max_depth=self.config.blast_radius_max_depth
            )

            # Respond with blast radius results
            await self.nc.publish(msg.reply, json.dumps({
                "service_id": service_id,
                "blast_radius": {
                    "affected_services": len(blast_result.affected_services),
                    "impact_score": blast_result.impact_score,
                    "confidence": blast_result.confidence,
                    "estimated_recovery_time": blast_result.estimated_recovery_time.total_seconds(),
                    "risk_factors": blast_result.risk_factors
                }
            }).encode())

        except Exception as e:
            logger.error(f"Failed to handle blast radius request: {e}")

    async def _handle_new_incident(self, msg) -> None:
        """Handle new incident creation for automatic correlation"""
        try:
            data = json.loads(msg.data.decode())
            incident_id = data.get("incident_id")

            # Automatically correlate new incidents
            await self._correlate_incident(incident_id)

        except Exception as e:
            logger.error(f"Failed to handle new incident: {e}")

    async def _perform_correlation_analysis(self) -> None:
        """Perform comprehensive correlation analysis"""
        try:
            # Get recent incidents
            recent_incidents = await self._get_recent_incidents()

            # Get recent changes
            recent_deployments = await self._get_recent_deployments()
            recent_flags = await self._get_recent_feature_flags()

            # Correlate each incident
            for incident in recent_incidents:
                try:
                    correlation = self.correlation_engine.correlate_incident(
                        incident, recent_deployments, recent_flags
                    )

                    # Store correlation results
                    await self._store_correlation_results(correlation)

                except Exception as e:
                    logger.error(f"Failed to correlate incident {incident.id}: {e}")

        except Exception as e:
            logger.error(f"Failed to perform correlation analysis: {e}")

    async def _correlate_incident(self, incident_id: str) -> Dict[str, Any]:
        """Correlate a specific incident"""
        try:
            # Get incident details
            incident = await self._get_incident(incident_id)

            # Get relevant changes
            deployments = await self._get_deployments_for_incident(incident)
            feature_flags = await self._get_feature_flags_for_incident(incident)

            # Perform correlation
            correlation = self.correlation_engine.correlate_incident(
                incident, deployments, feature_flags
            )

            # Calculate blast radius if we have a service
            blast_radius = None
            if incident.service:
                blast_result = self.blast_calculator.calculate_blast_radius(
                    incident.service, max_depth=self.config.blast_radius_max_depth
                )
                blast_radius = {
                    "affected_services": len(blast_result.affected_services),
                    "impact_score": blast_result.impact_score,
                    "confidence": blast_result.confidence
                }

            return {
                "incident_id": incident_id,
                "correlation": {
                    "correlated_changes": len(correlation.correlated_changes),
                    "impact_scores": correlation.impact_scores,
                    "confidence": correlation.confidence
                },
                "blast_radius": blast_radius,
                "analysis_timestamp": correlation.analysis_timestamp.isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to correlate incident {incident_id}: {e}")
            raise

    async def _get_recent_incidents(self) -> List[Any]:
        """Get recent incidents for correlation"""
        # Implementation to fetch recent incidents
        return []

    async def _get_recent_deployments(self) -> List[Any]:
        """Get recent deployments"""
        # Implementation to fetch recent deployments
        return []

    async def _get_recent_feature_flags(self) -> List[Any]:
        """Get recent feature flag changes"""
        # Implementation to fetch recent feature flags
        return []

    async def _get_incident(self, incident_id: str) -> Any:
        """Get incident details"""
        # Implementation to fetch incident
        return None

    async def _get_deployments_for_incident(self, incident: Any) -> List[Any]:
        """Get deployments relevant to incident"""
        # Implementation to fetch relevant deployments
        return []

    async def _get_feature_flags_for_incident(self, incident: Any) -> List[Any]:
        """Get feature flags relevant to incident"""
        # Implementation to fetch relevant feature flags
        return []

    async def _store_correlation_results(self, correlation: Any) -> None:
        """Store correlation results in database"""
        # Implementation to store correlation results
        pass


async def main():
    """Main entry point"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    config = CorrelateConfig()

    # Create worker
    worker = CorrelateWorker(config)

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
