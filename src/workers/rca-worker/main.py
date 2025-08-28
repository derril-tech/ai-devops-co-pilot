"""
RCA (Root Cause Analysis) worker for incident analysis
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
from ...shared.rca.hypothesis_builder import HypothesisBuilder
from ...shared.rca.evidence_cards import EvidenceCardManager, EvidenceType, EvidenceStrength
from ...shared.rag.knowledge_base import KnowledgeBase
from ...shared.rag.cited_answers import AnswerGenerator, CitedAnswer
from ...shared.anomaly_detection.red_use_monitor import ServiceHealthMonitor


logger = logging.getLogger(__name__)


class RCAConfig(BaseModel):
    """RCA worker configuration"""
    hypothesis_timeout: int = 300  # 5 minutes
    evidence_collection_window: timedelta = timedelta(hours=24)
    knowledge_base_search_limit: int = 10
    min_confidence_threshold: float = 0.3
    batch_size: int = 10


class RCAWorker:
    """Worker for Root Cause Analysis"""

    def __init__(self, config: RCAConfig):
        self.config = config
        self.nc: Optional[nats.NATS] = None
        self.running = False
        self.tasks: List[asyncio.Task] = []

        # Initialize RCA components
        self.hypothesis_builder = HypothesisBuilder()
        self.evidence_manager = EvidenceCardManager()
        self.knowledge_base = KnowledgeBase()
        self.answer_generator = AnswerGenerator()
        self.health_monitor = ServiceHealthMonitor()

    async def start(self) -> None:
        """Start the RCA worker"""
        logger.info("Starting RCA worker")

        # Connect to NATS
        self.nc = await nats.connect("nats://localhost:4222")

        # Load knowledge base
        await self._load_knowledge_base()

        # Subscribe to RCA topics
        await self._subscribe_topics()

        # Start analysis tasks
        self.running = True
        analysis_task = asyncio.create_task(self._analysis_loop())

        self.tasks.append(analysis_task)

        logger.info("RCA worker started successfully")

    async def stop(self) -> None:
        """Stop the RCA worker"""
        logger.info("Stopping RCA worker")
        self.running = False

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Close NATS connection
        if self.nc:
            await self.nc.close()

        logger.info("RCA worker stopped")

    async def _load_knowledge_base(self) -> None:
        """Load knowledge base documents"""
        try:
            async with get_postgres_session() as session:
                # Load documents from database (placeholder implementation)
                logger.info("Knowledge base loaded")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")

    async def _subscribe_topics(self) -> None:
        """Subscribe to NATS topics"""
        # Subscribe to incident RCA requests
        await self.nc.subscribe("incidents.analyze", cb=self._handle_incident_analysis)

        # Subscribe to hypothesis generation
        await self.nc.subscribe("rca.generate_hypotheses", cb=self._handle_hypothesis_generation)

        # Subscribe to evidence collection
        await self.nc.subscribe("rca.collect_evidence", cb=self._handle_evidence_collection)

        # Subscribe to knowledge base queries
        await self.nc.subscribe("rca.knowledge_query", cb=self._handle_knowledge_query)

    async def _analysis_loop(self) -> None:
        """Periodic analysis loop"""
        while self.running:
            try:
                await self._perform_periodic_analysis()
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(300)

    async def _handle_incident_analysis(self, msg) -> None:
        """Handle incident analysis requests"""
        try:
            data = json.loads(msg.data.decode())
            incident_id = data.get("incident_id")

            # Perform comprehensive RCA
            analysis_result = await self._perform_root_cause_analysis(incident_id)

            # Respond with analysis
            await self.nc.publish(msg.reply, json.dumps(analysis_result).encode())

        except Exception as e:
            logger.error(f"Failed to handle incident analysis: {e}")

    async def _handle_hypothesis_generation(self, msg) -> None:
        """Handle hypothesis generation requests"""
        try:
            data = json.loads(msg.data.decode())
            incident_id = data.get("incident_id")
            context = data.get("context", {})

            # Generate hypotheses
            hypotheses = await self.hypothesis_builder.generate_hypotheses(incident_id, context)

            # Respond with hypotheses
            await self.nc.publish(msg.reply, json.dumps({
                "incident_id": incident_id,
                "hypotheses": [
                    {
                        "id": h.id,
                        "type": h.type.value,
                        "title": h.title,
                        "confidence": h.confidence,
                        "confidence_level": h.confidence_level.value
                    }
                    for h in hypotheses
                ]
            }).encode())

        except Exception as e:
            logger.error(f"Failed to handle hypothesis generation: {e}")

    async def _handle_evidence_collection(self, msg) -> None:
        """Handle evidence collection requests"""
        try:
            data = json.loads(msg.data.decode())
            service = data.get("service")
            time_window = timedelta(hours=data.get("hours", 24))

            # Collect evidence
            evidence = self.evidence_manager.get_evidence_for_service(service, time_window)

            # Respond with evidence
            await self.nc.publish(msg.reply, json.dumps({
                "service": service,
                "evidence_count": len(evidence),
                "evidence": [
                    {
                        "id": e.id,
                        "type": e.type.value,
                        "title": e.title,
                        "strength": e.strength.value,
                        "confidence": e.confidence,
                        "timestamp": e.timestamp.isoformat()
                    }
                    for e in evidence
                ]
            }).encode())

        except Exception as e:
            logger.error(f"Failed to handle evidence collection: {e}")

    async def _handle_knowledge_query(self, msg) -> None:
        """Handle knowledge base queries"""
        try:
            data = json.loads(msg.data.decode())
            query = data.get("query")

            # Search knowledge base
            search_results = self.knowledge_base.search_similar(query, top_k=5)

            # Generate cited answer
            cited_answer = self.answer_generator.generate_cited_answer(query, search_results)

            # Respond with answer
            await self.nc.publish(msg.reply, json.dumps({
                "query": query,
                "answer": cited_answer.answer,
                "confidence": cited_answer.confidence,
                "citations": len(cited_answer.citations),
                "reasoning": cited_answer.reasoning
            }).encode())

        except Exception as e:
            logger.error(f"Failed to handle knowledge query: {e}")

    async def _perform_root_cause_analysis(self, incident_id: str) -> Dict[str, Any]:
        """Perform comprehensive root cause analysis"""
        try:
            # Get incident details
            incident = await self._get_incident_details(incident_id)

            # Gather context data
            context = await self._gather_incident_context(incident)

            # Generate hypotheses
            hypotheses = await self.hypothesis_builder.generate_hypotheses(incident_id, context)

            # Collect evidence
            evidence = await self._collect_incident_evidence(incident, context)

            # Query knowledge base
            knowledge_results = await self._query_knowledge_base(incident)

            # Generate cited answer for RCA
            cited_answer = self._generate_rca_answer(incident, hypotheses, evidence, knowledge_results)

            return {
                "incident_id": incident_id,
                "hypotheses": [
                    {
                        "id": h.id,
                        "type": h.type.value,
                        "title": h.title,
                        "confidence": h.confidence,
                        "confidence_level": h.confidence_level.value,
                        "evidence_count": len(h.supporting_evidence)
                    }
                    for h in hypotheses
                ],
                "evidence_count": len(evidence),
                "knowledge_sources": len(knowledge_results),
                "cited_answer": {
                    "answer": cited_answer.answer,
                    "confidence": cited_answer.confidence,
                    "citations": cited_answer.citation_count
                },
                "analysis_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to perform RCA for incident {incident_id}: {e}")
            return {"error": str(e)}

    async def _get_incident_details(self, incident_id: str) -> Dict[str, Any]:
        """Get incident details from database"""
        # Placeholder implementation
        return {
            "id": incident_id,
            "title": "Sample Incident",
            "service": "api-gateway",
            "timestamp": datetime.now(),
            "severity": "high"
        }

    async def _gather_incident_context(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """Gather context data for incident analysis"""
        service = incident.get("service", "unknown")
        incident_time = incident.get("timestamp", datetime.now())

        # Get metrics around incident time
        metrics = await self._get_metrics_around_time(service, incident_time)

        # Get logs around incident time
        logs = await self._get_logs_around_time(service, incident_time)

        # Get recent changes
        changes = await self._get_recent_changes(service, incident_time)

        return {
            "metrics": metrics,
            "logs": logs,
            "changes": changes,
            "incident": incident
        }

    async def _collect_incident_evidence(self, incident: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect evidence for incident"""
        service = incident.get("service", "unknown")
        evidence = self.evidence_manager.get_evidence_for_service(service, self.config.evidence_collection_window)

        return [
            {
                "id": e.id,
                "type": e.type.value,
                "title": e.title,
                "strength": e.strength.value,
                "confidence": e.confidence
            }
            for e in evidence
        ]

    async def _query_knowledge_base(self, incident: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query knowledge base for relevant information"""
        query = f"incident troubleshooting {incident.get('service', 'unknown')} {incident.get('title', '')}"
        search_results = self.knowledge_base.search_similar(query, top_k=5)

        return [
            {
                "title": result.document.title,
                "similarity": result.similarity_score,
                "anchors": result.anchors
            }
            for result in search_results
        ]

    def _generate_rca_answer(self, incident: Dict[str, Any], hypotheses: List[Any],
                           evidence: List[Dict[str, Any]], knowledge_results: List[Dict[str, Any]]) -> CitedAnswer:
        """Generate cited answer for RCA"""
        # Create a mock retrieval result for demonstration
        from ...shared.rag.knowledge_base import RetrievalResult, DocumentChunk, Document

        mock_document = Document(
            id="rca_guide",
            title="Incident Response Guide",
            content="This guide provides comprehensive incident response procedures...",
            type="runbook",
            source="internal"
        )

        mock_chunk = DocumentChunk(
            id="rca_chunk",
            document_id="rca_guide",
            content="When analyzing incidents, consider multiple hypotheses and gather evidence from multiple sources.",
            embedding=[],
            chunk_index=0,
            total_chunks=1
        )

        mock_result = RetrievalResult(
            document=mock_document,
            chunk=mock_chunk,
            similarity_score=0.8,
            matched_text="consider multiple hypotheses",
            context="When analyzing incidents...",
            anchors=["Root Cause Analysis"]
        )

        return self.answer_generator.generate_cited_answer(
            f"What is the root cause of {incident.get('title', 'this incident')}?",
            [mock_result]
        )

    async def _get_metrics_around_time(self, service: str, incident_time: datetime) -> List[Dict[str, Any]]:
        """Get metrics around incident time"""
        # Placeholder implementation
        return []

    async def _get_logs_around_time(self, service: str, incident_time: datetime) -> List[Dict[str, Any]]:
        """Get logs around incident time"""
        # Placeholder implementation
        return []

    async def _get_recent_changes(self, service: str, incident_time: datetime) -> Dict[str, Any]:
        """Get recent changes for service"""
        # Placeholder implementation
        return {"deployments": [], "feature_flags": []}

    async def _perform_periodic_analysis(self) -> None:
        """Perform periodic analysis tasks"""
        try:
            # Clean up old evidence
            cleaned = self.evidence_manager.cleanup_old_evidence()
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} old evidence cards")

            # Update knowledge base statistics
            stats = self.knowledge_base.get_document_stats()
            logger.info(f"Knowledge base stats: {stats}")

        except Exception as e:
            logger.error(f"Failed to perform periodic analysis: {e}")


async def main():
    """Main entry point"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    config = RCAConfig()

    # Create worker
    worker = RCAWorker(config)

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
