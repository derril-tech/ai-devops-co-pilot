"""
RAG (Retrieval-Augmented Generation) worker for knowledge base operations
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

import nats
from pydantic import BaseModel

from ...shared.database.config import get_postgres_session, get_clickhouse_session
from ...shared.rag.knowledge_base import KnowledgeBase, Document, DocumentType
from ...shared.rag.cited_answers import AnswerGenerator, CitedAnswer


logger = logging.getLogger(__name__)


class RAGConfig(BaseModel):
    """RAG worker configuration"""
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_search_results: int = 10
    similarity_threshold: float = 0.3


class RAGWorker:
    """Worker for RAG operations"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.nc: Optional[nats.NATS] = None
        self.running = False
        self.tasks: List[asyncio.Task] = []

        # Initialize RAG components
        self.knowledge_base = KnowledgeBase()
        self.answer_generator = AnswerGenerator()

    async def start(self) -> None:
        """Start the RAG worker"""
        logger.info("Starting RAG worker")

        # Connect to NATS
        self.nc = await nats.connect("nats://localhost:4222")

        # Load knowledge base
        await self._load_knowledge_base()

        # Subscribe to RAG topics
        await self._subscribe_topics()

        # Start maintenance tasks
        self.running = True
        maintenance_task = asyncio.create_task(self._maintenance_loop())

        self.tasks.append(maintenance_task)

        logger.info("RAG worker started successfully")

    async def stop(self) -> None:
        """Stop the RAG worker"""
        logger.info("Stopping RAG worker")
        self.running = False

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Close NATS connection
        if self.nc:
            await self.nc.close()

        logger.info("RAG worker stopped")

    async def _load_knowledge_base(self) -> None:
        """Load knowledge base from database"""
        try:
            async with get_postgres_session() as session:
                # Load documents (placeholder implementation)
                documents = await self._fetch_documents_from_db(session)

                for doc_data in documents:
                    document = Document(
                        id=doc_data["id"],
                        title=doc_data["title"],
                        content=doc_data["content"],
                        type=DocumentType(doc_data["type"]),
                        source=doc_data["source"],
                        tags=doc_data.get("tags", [])
                    )
                    self.knowledge_base.add_document(document)

                logger.info(f"Loaded {len(documents)} documents into knowledge base")

        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")

    async def _subscribe_topics(self) -> None:
        """Subscribe to NATS topics"""
        # Subscribe to knowledge queries
        await self.nc.subscribe("rag.query", cb=self._handle_knowledge_query)

        # Subscribe to document ingestion
        await self.nc.subscribe("rag.ingest", cb=self._handle_document_ingestion)

        # Subscribe to document search
        await self.nc.subscribe("rag.search", cb=self._handle_document_search)

        # Subscribe to cited answer generation
        await self.nc.subscribe("rag.generate_answer", cb=self._handle_answer_generation)

    async def _maintenance_loop(self) -> None:
        """Periodic maintenance loop"""
        while self.running:
            try:
                await self._perform_maintenance()
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(300)

    async def _handle_knowledge_query(self, msg) -> None:
        """Handle knowledge base queries"""
        try:
            data = json.loads(msg.data.decode())
            query = data.get("query")
            top_k = data.get("top_k", self.config.max_search_results)
            document_types = data.get("document_types")

            # Search knowledge base
            search_results = self.knowledge_base.search_similar(
                query,
                top_k=top_k,
                document_types=document_types
            )

            # Format results
            results = [
                {
                    "document_id": result.document.id,
                    "title": result.document.title,
                    "similarity": result.similarity_score,
                    "matched_text": result.matched_text,
                    "anchors": result.anchors,
                    "content": result.chunk.content[:500]  # Limit content length
                }
                for result in search_results
            ]

            # Respond with results
            await self.nc.publish(msg.reply, json.dumps({
                "query": query,
                "results": results,
                "count": len(results)
            }).encode())

        except Exception as e:
            logger.error(f"Failed to handle knowledge query: {e}")

    async def _handle_document_ingestion(self, msg) -> None:
        """Handle document ingestion requests"""
        try:
            data = json.loads(msg.data.decode())

            # Create document
            document = Document(
                id=data.get("id", f"doc_{datetime.now().timestamp()}"),
                title=data["title"],
                content=data["content"],
                type=DocumentType(data.get("type", "runbook")),
                source=data.get("source", "unknown"),
                tags=data.get("tags", [])
            )

            # Add to knowledge base
            chunks_added = self.knowledge_base.add_document(document)

            # Store in database
            await self._store_document_in_db(document)

            # Respond with success
            await self.nc.publish(msg.reply, json.dumps({
                "document_id": document.id,
                "chunks_added": chunks_added,
                "status": "ingested"
            }).encode())

            logger.info(f"Ingested document: {document.title}")

        except Exception as e:
            logger.error(f"Failed to handle document ingestion: {e}")

    async def _handle_document_search(self, msg) -> None:
        """Handle document search requests"""
        try:
            data = json.loads(msg.data.decode())
            query = data.get("query")
            filters = data.get("filters", {})

            # Apply filters
            document_types = filters.get("types")
            tags = filters.get("tags")

            # Search with filters
            results = self.knowledge_base.search_similar(
                query,
                top_k=self.config.max_search_results,
                document_types=document_types
            )

            # Apply tag filter if specified
            if tags:
                results = [r for r in results if any(tag in r.document.tags for tag in tags)]

            # Format results
            search_response = {
                "query": query,
                "filters": filters,
                "results": [
                    {
                        "document": {
                            "id": r.document.id,
                            "title": r.document.title,
                            "type": r.document.type.value,
                            "tags": r.document.tags,
                            "source": r.document.source
                        },
                        "chunk": {
                            "content": r.chunk.content,
                            "metadata": r.chunk.metadata
                        },
                        "similarity": r.similarity_score,
                        "matched_text": r.matched_text,
                        "anchors": r.anchors
                    }
                    for r in results
                ],
                "total_found": len(results)
            }

            await self.nc.publish(msg.reply, json.dumps(search_response).encode())

        except Exception as e:
            logger.error(f"Failed to handle document search: {e}")

    async def _handle_answer_generation(self, msg) -> None:
        """Handle cited answer generation requests"""
        try:
            data = json.loads(msg.data.decode())
            question = data.get("question")
            context_docs = data.get("context_documents", [])

            # Convert context documents to retrieval results
            retrieval_results = []
            for doc_data in context_docs:
                # This is a simplified conversion - in practice, you'd reconstruct proper objects
                doc_id = doc_data.get("document_id")
                if doc_id in self.knowledge_base.documents:
                    document = self.knowledge_base.documents[doc_id]
                    # Find a relevant chunk
                    chunk = next(iter(self.knowledge_base.chunks.values()))
                    # Create mock retrieval result
                    from ...shared.rag.knowledge_base import RetrievalResult
                    result = RetrievalResult(
                        document=document,
                        chunk=chunk,
                        similarity_score=0.8,
                        matched_text="",
                        context="",
                        anchors=[]
                    )
                    retrieval_results.append(result)

            # Generate cited answer
            cited_answer = self.answer_generator.generate_cited_answer(question, retrieval_results)

            # Respond with answer
            await self.nc.publish(msg.reply, json.dumps({
                "question": question,
                "answer": cited_answer.answer,
                "confidence": cited_answer.confidence,
                "citations": cited_answer.citation_count,
                "reasoning": cited_answer.reasoning
            }).encode())

        except Exception as e:
            logger.error(f"Failed to handle answer generation: {e}")

    async def _fetch_documents_from_db(self, session) -> List[Dict[str, Any]]:
        """Fetch documents from database"""
        # Placeholder implementation
        return []

    async def _store_document_in_db(self, document: Document) -> None:
        """Store document in database"""
        # Placeholder implementation
        pass

    async def _perform_maintenance(self) -> None:
        """Perform maintenance tasks"""
        try:
            # Get knowledge base statistics
            stats = self.knowledge_base.get_document_stats()
            logger.info(f"Knowledge base stats: {stats}")

            # Clean up any temporary data
            # This could include removing old embeddings, updating indexes, etc.

        except Exception as e:
            logger.error(f"Failed to perform maintenance: {e}")


class KnowledgeIngestionPipeline:
    """Pipeline for ingesting knowledge from various sources"""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base

    async def ingest_from_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Ingest knowledge from files"""
        results = {
            "total_files": len(file_paths),
            "successful": 0,
            "failed": 0,
            "documents_created": []
        }

        for file_path in file_paths:
            try:
                document = await self._process_file(file_path)
                if document:
                    chunks_added = self.knowledge_base.add_document(document)
                    results["documents_created"].append({
                        "id": document.id,
                        "title": document.title,
                        "chunks": chunks_added
                    })
                    results["successful"] += 1
                else:
                    results["failed"] += 1

            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                results["failed"] += 1

        return results

    async def ingest_from_web(self, urls: List[str]) -> Dict[str, Any]:
        """Ingest knowledge from web URLs"""
        results = {
            "total_urls": len(urls),
            "successful": 0,
            "failed": 0,
            "documents_created": []
        }

        for url in urls:
            try:
                document = await self._process_url(url)
                if document:
                    chunks_added = self.knowledge_base.add_document(document)
                    results["documents_created"].append({
                        "id": document.id,
                        "title": document.title,
                        "chunks": chunks_added
                    })
                    results["successful"] += 1
                else:
                    results["failed"] += 1

            except Exception as e:
                logger.error(f"Failed to process URL {url}: {e}")
                results["failed"] += 1

        return results

    async def _process_file(self, file_path: str) -> Optional[Document]:
        """Process a file into a document"""
        # Placeholder implementation
        # In practice, this would read different file formats (PDF, DOCX, MD, etc.)
        return None

    async def _process_url(self, url: str) -> Optional[Document]:
        """Process a URL into a document"""
        # Placeholder implementation
        # In practice, this would fetch and parse web content
        return None


async def main():
    """Main entry point"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    config = RAGConfig()

    # Create worker
    worker = RAGWorker(config)

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
