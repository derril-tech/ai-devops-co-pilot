"""
Knowledge base system with RAG capabilities for documentation and runbooks
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import hashlib
from pathlib import Path

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Types of documents in the knowledge base"""
    RUNBOOK = "runbook"
    TROUBLESHOOTING_GUIDE = "troubleshooting_guide"
    ADR = "adr"  # Architecture Decision Record
    API_DOCUMENTATION = "api_documentation"
    DEPLOYMENT_GUIDE = "deployment_guide"
    MONITORING_GUIDE = "monitoring_guide"
    SECURITY_GUIDE = "security_guide"
    PERFORMANCE_GUIDE = "performance_guide"
    INCIDENT_POSTMORTEM = "incident_postmortem"
    CHANGE_MANAGEMENT = "change_management"


class DocumentStatus(Enum):
    """Status of documents in the knowledge base"""
    ACTIVE = "active"
    DRAFT = "draft"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


@dataclass
class Document:
    """Document in the knowledge base"""
    id: str
    title: str
    content: str
    type: DocumentType
    source: str  # URL, file path, or external system
    status: DocumentStatus = DocumentStatus.ACTIVE
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    author: Optional[str] = None
    last_reviewed: Optional[datetime] = None


@dataclass
class DocumentChunk:
    """Chunk of a document for retrieval"""
    id: str
    document_id: str
    content: str
    embedding: np.ndarray
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RetrievalResult:
    """Result of document retrieval"""
    document: Document
    chunk: DocumentChunk
    similarity_score: float
    matched_text: str
    context: str
    anchors: List[str]  # Section anchors/headings


@dataclass
class RAGResponse:
    """RAG system response with citations"""
    answer: str
    confidence: float
    sources: List[RetrievalResult]
    reasoning: str
    generated_at: datetime = field(default_factory=datetime.now)


class DocumentProcessor:
    """Processes documents into chunks for embedding"""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_document(self, document: Document) -> List[DocumentChunk]:
        """Process document into chunks"""
        chunks = []

        # Split content into sections
        sections = self._split_into_sections(document.content)

        chunk_index = 0

        for section in sections:
            # Split section into chunks
            section_chunks = self._chunk_text(section['content'], section['heading'])

            for chunk_content in section_chunks:
                chunk_id = f"{document.id}_chunk_{chunk_index}"

                chunk = DocumentChunk(
                    id=chunk_id,
                    document_id=document.id,
                    content=chunk_content,
                    embedding=np.array([]),  # Will be set during embedding
                    chunk_index=chunk_index,
                    total_chunks=len(section_chunks),
                    metadata={
                        'section': section['heading'],
                        'document_title': document.title,
                        'document_type': document.type.value,
                        'tags': document.tags
                    }
                )

                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _split_into_sections(self, content: str) -> List[Dict[str, str]]:
        """Split document content into sections based on headings"""
        sections = []
        lines = content.split('\n')

        current_section = {'heading': 'Introduction', 'content': ''}
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line is a heading (starts with #)
            if line.startswith('#'):
                # Save previous section
                if current_content:
                    current_section['content'] = '\n'.join(current_content)
                    sections.append(current_section.copy())

                # Start new section
                heading_level = len(line) - len(line.lstrip('#'))
                heading_text = line.lstrip('#').strip()
                current_section = {'heading': heading_text, 'content': ''}
                current_content = []
            else:
                current_content.append(line)

        # Add final section
        if current_content:
            current_section['content'] = '\n'.join(current_content)
            sections.append(current_section)

        return sections

    def _chunk_text(self, text: str, section_heading: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Find word boundary
            if end < len(text):
                while end > start and text[end] not in ' \t\n':
                    end -= 1
                if end == start:  # No word boundary found
                    end = start + self.chunk_size

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = max(start + 1, end - self.overlap)

        return chunks


class EmbeddingManager:
    """Manages document embeddings for similarity search"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Lazy load the embedding model"""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        self._load_model()
        embedding = self.model.encode(text, show_progress_bar=False)
        return np.array(embedding)

    def batch_generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        self._load_model()
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [np.array(emb) for emb in embeddings]

    def calculate_similarity(self, query_embedding: np.ndarray,
                           document_embeddings: List[np.ndarray]) -> List[float]:
        """Calculate cosine similarity between query and documents"""
        if not document_embeddings:
            return []

        # Stack embeddings into matrix
        doc_matrix = np.vstack(document_embeddings)

        # Calculate similarities
        similarities = cosine_similarity([query_embedding], doc_matrix)[0]

        return similarities.tolist()


class KnowledgeBase:
    """Main knowledge base with RAG capabilities"""

    def __init__(self, embedding_manager: EmbeddingManager = None):
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, DocumentChunk] = {}
        self.document_chunks: Dict[str, List[str]] = {}  # document_id -> chunk_ids

        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.processor = DocumentProcessor()

    def add_document(self, document: Document) -> int:
        """Add document to knowledge base"""
        # Store document
        self.documents[document.id] = document

        # Process into chunks
        chunks = self.processor.process_document(document)

        # Generate embeddings for chunks
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_manager.batch_generate_embeddings(chunk_texts)

        # Store chunks with embeddings
        chunk_ids = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            self.chunks[chunk.id] = chunk
            chunk_ids.append(chunk.id)

        self.document_chunks[document.id] = chunk_ids

        logger.info(f"Added document '{document.title}' with {len(chunks)} chunks")
        return len(chunks)

    def search_similar(self, query: str, top_k: int = 5,
                      document_types: List[DocumentType] = None,
                      tags: List[str] = None) -> List[RetrievalResult]:
        """Search for documents similar to query"""
        if not self.chunks:
            return []

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embedding(query)

        # Filter chunks by criteria
        candidate_chunks = []
        for chunk_id, chunk in self.chunks.items():
            # Filter by document type
            if document_types:
                doc = self.documents.get(chunk.document_id)
                if doc and doc.type not in document_types:
                    continue

            # Filter by tags
            if tags:
                doc = self.documents.get(chunk.document_id)
                if doc and not any(tag in doc.tags for tag in tags):
                    continue

            candidate_chunks.append(chunk)

        if not candidate_chunks:
            return []

        # Calculate similarities
        chunk_embeddings = [chunk.embedding for chunk in candidate_chunks]
        similarities = self.embedding_manager.calculate_similarity(query_embedding, chunk_embeddings)

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = candidate_chunks[idx]
            document = self.documents[chunk.document_id]
            similarity = similarities[idx]

            # Find matched text (simple keyword matching)
            matched_text = self._find_matched_text(query, chunk.content)

            # Extract context
            context = self._extract_context(chunk.content, matched_text)

            # Find anchors (section headings)
            anchors = self._find_anchors(chunk)

            result = RetrievalResult(
                document=document,
                chunk=chunk,
                similarity_score=similarity,
                matched_text=matched_text,
                context=context,
                anchors=anchors
            )

            results.append(result)

        return results

    def generate_answer(self, query: str, context_results: List[RetrievalResult]) -> RAGResponse:
        """Generate answer using retrieved context"""
        if not context_results:
            return RAGResponse(
                answer="No relevant information found in the knowledge base.",
                confidence=0.0,
                sources=[],
                reasoning="No matching documents found"
            )

        # Combine relevant information from top results
        relevant_info = []
        sources = []

        for result in context_results[:3]:  # Use top 3 results
            if result.similarity_score > 0.3:  # Only use reasonably similar results
                info = {
                    'title': result.document.title,
                    'content': result.chunk.content,
                    'similarity': result.similarity_score,
                    'anchors': result.anchors
                }
                relevant_info.append(info)
                sources.append(result)

        # Generate answer based on retrieved information
        answer = self._synthesize_answer(query, relevant_info)

        # Calculate confidence based on similarity scores and number of sources
        avg_similarity = np.mean([r.similarity_score for r in sources])
        source_count = len(sources)
        confidence = min(avg_similarity * (1 + source_count * 0.1), 0.95)

        reasoning = f"Answer synthesized from {source_count} relevant documents with average similarity {avg_similarity:.2f}"

        return RAGResponse(
            answer=answer,
            confidence=confidence,
            sources=sources,
            reasoning=reasoning
        )

    def _find_matched_text(self, query: str, content: str) -> str:
        """Find text in content that matches the query"""
        query_words = set(query.lower().split())
        content_lower = content.lower()

        # Find sentences containing query words
        sentences = re.split(r'[.!?]+', content)

        best_match = ""
        max_matches = 0

        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            matches = len(query_words.intersection(sentence_words))

            if matches > max_matches:
                max_matches = matches
                best_match = sentence.strip()

        return best_match if best_match else content[:200] + "..."

    def _extract_context(self, content: str, matched_text: str) -> str:
        """Extract context around matched text"""
        if matched_text in content:
            start = max(0, content.find(matched_text) - 100)
            end = min(len(content), content.find(matched_text) + len(matched_text) + 100)
            return content[start:end]
        return content[:300] + "..."

    def _find_anchors(self, chunk: DocumentChunk) -> List[str]:
        """Find section anchors/headings in chunk"""
        anchors = []

        # Look for section heading in metadata
        if 'section' in chunk.metadata:
            anchors.append(chunk.metadata['section'])

        # Look for headings in content
        lines = chunk.content.split('\n')
        for line in lines[:5]:  # Check first few lines
            if line.strip().startswith('#'):
                heading = line.strip('#').strip()
                if heading:
                    anchors.append(heading)

        return anchors

    def _synthesize_answer(self, query: str, relevant_info: List[Dict[str, Any]]) -> str:
        """Synthesize answer from relevant information"""
        if not relevant_info:
            return "No relevant information found."

        # Simple synthesis - combine information from top sources
        answer_parts = []

        for info in relevant_info[:2]:  # Use top 2 sources
            title = info['title']
            content = info['content'][:300]  # Limit content length

            part = f"According to {title}: {content}"
            answer_parts.append(part)

        if len(relevant_info) > 1:
            answer = " ".join(answer_parts)
        else:
            answer = answer_parts[0] if answer_parts else "Information found but unable to synthesize answer."

        return answer + "\n\nPlease refer to the cited sources for complete information."

    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        if not self.documents:
            return {"total_documents": 0}

        total_chunks = sum(len(chunks) for chunks in self.document_chunks.values())

        # Count document types
        type_counts = {}
        for doc in self.documents.values():
            type_counts[doc.type.value] = type_counts.get(doc.type.value, 0) + 1

        # Count tags
        all_tags = []
        for doc in self.documents.values():
            all_tags.extend(doc.tags)

        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "total_documents": len(self.documents),
            "total_chunks": total_chunks,
            "document_types": type_counts,
            "tags": tag_counts,
            "avg_chunks_per_document": total_chunks / len(self.documents) if self.documents else 0
        }

    def remove_document(self, document_id: str) -> bool:
        """Remove document from knowledge base"""
        if document_id not in self.documents:
            return False

        # Remove chunks
        if document_id in self.document_chunks:
            for chunk_id in self.document_chunks[document_id]:
                if chunk_id in self.chunks:
                    del self.chunks[chunk_id]
            del self.document_chunks[document_id]

        # Remove document
        del self.documents[document_id]

        logger.info(f"Removed document {document_id}")
        return True

    def update_document(self, document: Document) -> int:
        """Update existing document"""
        if document.id not in self.documents:
            raise ValueError(f"Document {document.id} not found")

        # Remove old version
        self.remove_document(document.id)

        # Add new version
        return self.add_document(document)

    def find_related_documents(self, document_id: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Find documents related to the given document"""
        if document_id not in self.documents:
            return []

        target_doc = self.documents[document_id]

        # Use document title and tags as query
        query_parts = [target_doc.title]
        query_parts.extend(target_doc.tags)
        query = " ".join(query_parts)

        # Search for similar documents
        results = self.search_similar(query, top_k=top_k + 1)  # +1 to exclude self

        related = []
        for result in results:
            if result.document.id != document_id:  # Exclude self
                related.append((result.document, result.similarity_score))

        return related[:top_k]
