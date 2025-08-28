"""
Detect worker for anomaly detection and incident correlation
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
from ...shared.anomaly_detection.stl_decomposition import SeasonalBaseline
from ...shared.anomaly_detection.esd_spike_detection import ESDSpikeDetector
from ...shared.anomaly_detection.red_use_monitor import ServiceHealthMonitor
from ...shared.log_clustering.log_clustering import LogClusteringEngine
from ...shared.change_correlation.change_correlation import ChangeCorrelationEngine
from ...shared.blast_radius.blast_radius import TopologyGraph, BlastRadiusCalculator
from ...shared.models.api import IncidentCreate


logger = logging.getLogger(__name__)


class DetectConfig(BaseModel):
    """Detect worker configuration"""
    anomaly_window_hours: int = 24
    health_check_interval: int = 300  # 5 minutes
    log_cluster_batch_size: int = 1000
    correlation_time_window_hours: int = 24
    blast_radius_max_depth: int = 5


class DetectWorker:
    """Worker for detecting anomalies and incidents"""

    def __init__(self, config: DetectConfig):
        self.config = config
        self.nc: Optional[nats.NATS] = None
        self.running = False
        self.tasks: List[asyncio.Task] = []

        # Initialize detection engines
        self.seasonal_baseline = SeasonalBaseline()
        self.spike_detector = ESDSpikeDetector(max_anomalies=10)
        self.health_monitor = ServiceHealthMonitor(window_minutes=5)
        self.log_clusterer = LogClusteringEngine()
        self.correlation_engine = ChangeCorrelationEngine()

        # Topology graph for blast radius
        self.topology_graph = TopologyGraph()
        self.blast_calculator = BlastRadiusCalculator(self.topology_graph)

    async def start(self) -> None:
        """Start the detect worker"""
        logger.info("Starting detect worker")

        # Connect to NATS
        self.nc = await nats.connect("nats://localhost:4222")

        # Initialize topology
        await self._load_topology()

        # Subscribe to detection topics
        await self._subscribe_topics()

        # Start detection loops
        self.running = True
        health_task = asyncio.create_task(self._health_check_loop())
        anomaly_task = asyncio.create_task(self._anomaly_detection_loop())
        log_task = asyncio.create_task(self._log_clustering_loop())

        self.tasks.extend([health_task, anomaly_task, log_task])

        logger.info("Detect worker started successfully")

    async def stop(self) -> None:
        """Stop the detect worker"""
        logger.info("Stopping detect worker")
        self.running = False

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Close NATS connection
        if self.nc:
            await self.nc.close()

        logger.info("Detect worker stopped")

    async def _load_topology(self) -> None:
        """Load service topology from database"""
        try:
            async with get_postgres_session() as session:
                # Load services
                result = await session.execute("SELECT * FROM topologies")
                for row in result.fetchall():
                    # Create ServiceNode from database row
                    from ...shared.blast_radius.blast_radius import ServiceNode
                    service = ServiceNode(
                        id=row.id,
                        name=row.node,
                        type=row.type,
                        attributes=row.attrs or {},
                        health_score=1.0,
                        criticality=0.5
                    )
                    self.topology_graph.add_service(service)

                # Load topology edges
                result = await session.execute("SELECT * FROM topology_edges")
                for row in result.fetchall():
                    from ...shared.blast_radius.blast_radius import ServiceEdge
                    edge = ServiceEdge(
                        source=row.source_node,
                        target=row.target_node,
                        relationship=row.relationship,
                        weight=1.0,
                        attributes=row.attrs or {}
                    )
                    self.topology_graph.add_edge(edge)

                logger.info(f"Loaded topology with {len(self.topology_graph.services)} services")

        except Exception as e:
            logger.error(f"Failed to load topology: {e}")

    async def _subscribe_topics(self) -> None:
        """Subscribe to NATS topics"""
        # Subscribe to signal processing
        await self.nc.subscribe("signals.metric", cb=self._handle_metric_signals)
        await self.nc.subscribe("signals.log", cb=self._handle_log_signals)

        # Subscribe to health check requests
        await self.nc.subscribe("health.check", cb=self._handle_health_check)

        # Subscribe to incident analysis requests
        await self.nc.subscribe("incidents.analyze", cb=self._handle_incident_analysis)

    async def _health_check_loop(self) -> None:
        """Periodic health check loop"""
        while self.running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _anomaly_detection_loop(self) -> None:
        """Periodic anomaly detection loop"""
        while self.running:
            try:
                await self._detect_anomalies()
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Error in anomaly detection loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying

    async def _log_clustering_loop(self) -> None:
        """Periodic log clustering loop"""
        while self.running:
            try:
                await self._cluster_logs()
                await asyncio.sleep(1800)  # Run every 30 minutes
            except Exception as e:
                logger.error(f"Error in log clustering loop: {e}")
                await asyncio.sleep(300)

    async def _perform_health_checks(self) -> None:
        """Perform comprehensive health checks"""
        try:
            # Get recent metrics from ClickHouse
            async with get_clickhouse_session() as session:
                # Query for recent metrics by service
                services_data = await self._get_recent_service_metrics(session)

            # Assess health for each service
            health_assessments = self.health_monitor.batch_assess_services(services_data)

            # Publish health status updates
            for assessment in health_assessments:
                await self._publish_health_alert(assessment)

                # Create incident if health is critical
                if assessment.overall_status.value == "critical":
                    await self._create_incident_from_health_check(assessment)

        except Exception as e:
            logger.error(f"Failed to perform health checks: {e}")

    async def _detect_anomalies(self) -> None:
        """Detect anomalies in time series data"""
        try:
            async with get_clickhouse_session() as session:
                # Get time series data for anomaly detection
                time_series_data = await self._get_time_series_for_anomaly_detection(session)

            anomalies_found = []

            for series_name, values, timestamps in time_series_data:
                try:
                    # Perform STL decomposition
                    baseline = self.seasonal_baseline.calculate_baseline(values, timestamps)

                    # Detect spikes
                    spike_results = self.spike_detector.detect_spikes(values, timestamps)

                    # Check for anomalies
                    for i, result in enumerate(spike_results):
                        if result.is_spike:
                            anomaly = {
                                "series": series_name,
                                "timestamp": timestamps[i],
                                "value": values[i],
                                "spike_score": result.spike_score,
                                "confidence": result.confidence
                            }
                            anomalies_found.append(anomaly)

                except Exception as e:
                    logger.error(f"Failed to detect anomalies in {series_name}: {e}")

            # Create incidents for significant anomalies
            for anomaly in anomalies_found:
                if anomaly["confidence"] > 0.8:
                    await self._create_anomaly_incident(anomaly)

        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")

    async def _cluster_logs(self) -> None:
        """Cluster recent logs for pattern detection"""
        try:
            async with get_postgres_session() as session:
                # Get recent unprocessed logs
                recent_logs = await self._get_recent_logs(session)

            if recent_logs:
                # Extract log messages and metadata
                log_messages = [log["message"] for log in recent_logs]
                log_ids = [log["id"] for log in recent_logs]
                timestamps = [datetime.fromisoformat(log["timestamp"]) for log in recent_logs]

                # Perform clustering
                clustering_result = self.log_clusterer.cluster_logs(
                    log_messages, log_ids, timestamps
                )

                # Process clustering results
                await self._process_clustering_results(clustering_result)

                logger.info(f"Clustered {len(log_messages)} logs into {len(clustering_result.clusters)} groups")

        except Exception as e:
            logger.error(f"Failed to cluster logs: {e}")

    async def _handle_metric_signals(self, msg) -> None:
        """Handle incoming metric signals"""
        try:
            data = json.loads(msg.data.decode())
            signals = data.get("signals", [])

            # Process signals for real-time anomaly detection
            for signal in signals:
                await self._process_realtime_signal(signal)

        except Exception as e:
            logger.error(f"Failed to handle metric signals: {e}")

    async def _handle_log_signals(self, msg) -> None:
        """Handle incoming log signals"""
        try:
            data = json.loads(msg.data.decode())
            signals = data.get("signals", [])

            # Process logs for real-time clustering
            for signal in signals:
                await self._process_realtime_log(signal)

        except Exception as e:
            logger.error(f"Failed to handle log signals: {e}")

    async def _handle_health_check(self, msg) -> None:
        """Handle health check requests"""
        try:
            data = json.loads(msg.data.decode())
            service_name = data.get("service")

            # Perform targeted health check
            health_status = await self._get_service_health(service_name)

            # Respond with health status
            await self.nc.publish(msg.reply, json.dumps(health_status).encode())

        except Exception as e:
            logger.error(f"Failed to handle health check: {e}")

    async def _handle_incident_analysis(self, msg) -> None:
        """Handle incident analysis requests"""
        try:
            data = json.loads(msg.data.decode())
            incident_id = data.get("incident_id")

            # Perform comprehensive incident analysis
            analysis_result = await self._analyze_incident(incident_id)

            # Respond with analysis
            await self.nc.publish(msg.reply, json.dumps(analysis_result).encode())

        except Exception as e:
            logger.error(f"Failed to handle incident analysis: {e}")

    async def _get_recent_service_metrics(self, session) -> Dict[str, Dict[str, Any]]:
        """Get recent metrics for all services"""
        # This would query ClickHouse for recent metrics
        # Placeholder implementation
        return {}

    async def _get_time_series_for_anomaly_detection(self, session) -> List[Tuple[str, List[float], List[datetime]]]:
        """Get time series data for anomaly detection"""
        # This would query ClickHouse for time series data
        # Placeholder implementation
        return []

    async def _get_recent_logs(self, session) -> List[Dict[str, Any]]:
        """Get recent unprocessed logs"""
        # This would query PostgreSQL for recent logs
        # Placeholder implementation
        return []

    async def _create_incident_from_health_check(self, assessment) -> None:
        """Create incident from health check failure"""
        # Implementation for creating incidents from health checks
        pass

    async def _create_anomaly_incident(self, anomaly) -> None:
        """Create incident from anomaly detection"""
        # Implementation for creating incidents from anomalies
        pass

    async def _process_clustering_results(self, result) -> None:
        """Process log clustering results"""
        # Implementation for processing clustering results
        pass

    async def _process_realtime_signal(self, signal) -> None:
        """Process real-time signal for anomaly detection"""
        # Implementation for real-time signal processing
        pass

    async def _process_realtime_log(self, signal) -> None:
        """Process real-time log for clustering"""
        # Implementation for real-time log processing
        pass

    async def _get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Get health status for a specific service"""
        # Implementation for service health check
        return {"status": "healthy"}

    async def _analyze_incident(self, incident_id: str) -> Dict[str, Any]:
        """Analyze incident with full correlation and blast radius"""
        # Implementation for comprehensive incident analysis
        return {"analysis": "complete"}

    async def _publish_health_alert(self, assessment) -> None:
        """Publish health alert to NATS"""
        # Implementation for publishing health alerts
        pass


async def main():
    """Main entry point"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    config = DetectConfig()

    # Create worker
    worker = DetectWorker(config)

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
