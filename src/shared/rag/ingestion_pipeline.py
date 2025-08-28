"""
Knowledge ingestion pipeline for processing and indexing documents
"""
import asyncio
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import json

from .knowledge_base import KnowledgeBase, Document, DocumentType


logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents into the knowledge base"""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self.supported_formats = {
            '.md': self._process_markdown,
            '.txt': self._process_text,
            '.json': self._process_json,
            '.html': self._process_html
        }

    async def ingest_directory(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
        """Ingest all documents from a directory"""
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory does not exist: {directory_path}")

        results = {
            "total_files": 0,
            "processed": 0,
            "failed": 0,
            "documents_created": [],
            "errors": []
        }

        # Find all files
        pattern = "**/*" if recursive else "*"
        files = list(directory.glob(pattern))

        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                results["total_files"] += 1

                try:
                    document = await self.ingest_file(str(file_path))
                    if document:
                        results["documents_created"].append({
                            "id": document.id,
                            "title": document.title,
                            "source": document.source,
                            "type": document.type.value
                        })
                        results["processed"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append(f"Failed to process {file_path}")

                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"Error processing {file_path}: {str(e)}")

        logger.info(f"Ingested {results['processed']} documents from {directory_path}")
        return results

    async def ingest_file(self, file_path: str) -> Optional[Document]:
        """Ingest a single file"""
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        file_extension = file_path_obj.suffix.lower()

        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")

        processor = self.supported_formats[file_extension]
        document = await processor(file_path)

        if document:
            chunks_added = self.knowledge_base.add_document(document)
            logger.info(f"Ingested document: {document.title} ({chunks_added} chunks)")

        return document

    async def _process_markdown(self, file_path: str) -> Optional[Document]:
        """Process Markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract title from first heading
            lines = content.split('\n')
            title = "Untitled Document"

            for line in lines:
                if line.strip().startswith('# '):
                    title = line.strip()[2:].strip()
                    break

            # Determine document type based on content and filename
            doc_type = self._infer_document_type(content, file_path)

            document = Document(
                id=f"doc_{Path(file_path).stem}_{datetime.now().timestamp()}",
                title=title,
                content=content,
                type=doc_type,
                source=file_path,
                tags=self._extract_tags(content)
            )

            return document

        except Exception as e:
            logger.error(f"Failed to process markdown file {file_path}: {e}")
            return None

    async def _process_text(self, file_path: str) -> Optional[Document]:
        """Process plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            title = Path(file_path).stem.replace('_', ' ').title()

            document = Document(
                id=f"doc_{Path(file_path).stem}_{datetime.now().timestamp()}",
                title=title,
                content=content,
                type=DocumentType.RUNBOOK,  # Default type
                source=file_path
            )

            return document

        except Exception as e:
            logger.error(f"Failed to process text file {file_path}: {e}")
            return None

    async def _process_json(self, file_path: str) -> Optional[Document]:
        """Process JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict):
                title = data.get('title', Path(file_path).stem)
                content = data.get('content', json.dumps(data, indent=2))
                doc_type = DocumentType(data.get('type', 'runbook'))

                document = Document(
                    id=f"doc_{Path(file_path).stem}_{datetime.now().timestamp()}",
                    title=title,
                    content=content,
                    type=doc_type,
                    source=file_path,
                    tags=data.get('tags', [])
                )

                return document

        except Exception as e:
            logger.error(f"Failed to process JSON file {file_path}: {e}")

        return None

    async def _process_html(self, file_path: str) -> Optional[Document]:
        """Process HTML file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple HTML parsing to extract title and text
            title = "Untitled HTML Document"

            # Look for title tag
            import re
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()

            # Remove HTML tags for content
            import re
            clean_content = re.sub(r'<[^>]+>', '', content)
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()

            document = Document(
                id=f"doc_{Path(file_path).stem}_{datetime.now().timestamp()}",
                title=title,
                content=clean_content,
                type=DocumentType.TROUBLESHOOTING_GUIDE,
                source=file_path
            )

            return document

        except Exception as e:
            logger.error(f"Failed to process HTML file {file_path}: {e}")
            return None

    def _infer_document_type(self, content: str, file_path: str) -> DocumentType:
        """Infer document type from content and filename"""
        content_lower = content.lower()
        filename = Path(file_path).name.lower()

        # Check filename patterns
        if 'runbook' in filename or 'playbook' in filename:
            return DocumentType.RUNBOOK
        elif 'adr' in filename or 'decision' in filename:
            return DocumentType.ADR
        elif 'api' in filename:
            return DocumentType.API_DOCUMENTATION
        elif 'deploy' in filename:
            return DocumentType.DEPLOYMENT_GUIDE

        # Check content patterns
        if 'architecture decision' in content_lower:
            return DocumentType.ADR
        elif 'api' in content_lower and ('endpoint' in content_lower or 'request' in content_lower):
            return DocumentType.API_DOCUMENTATION
        elif 'troubleshoot' in content_lower or 'incident' in content_lower:
            return DocumentType.TROUBLESHOOTING_GUIDE
        elif 'deploy' in content_lower or 'release' in content_lower:
            return DocumentType.DEPLOYMENT_GUIDE
        elif 'security' in content_lower or 'vulnerability' in content_lower:
            return DocumentType.SECURITY_GUIDE
        elif 'performance' in content_lower or 'optimization' in content_lower:
            return DocumentType.PERFORMANCE_GUIDE

        return DocumentType.RUNBOOK  # Default

    def _extract_tags(self, content: str) -> List[str]:
        """Extract tags from document content"""
        tags = []

        # Look for common tag patterns
        content_lower = content.lower()

        # Technology tags
        technologies = ['kubernetes', 'docker', 'aws', 'gcp', 'azure', 'python', 'javascript', 'java', 'go']
        for tech in technologies:
            if tech in content_lower:
                tags.append(tech)

        # Domain tags
        domains = ['monitoring', 'logging', 'security', 'performance', 'deployment', 'database']
        for domain in domains:
            if domain in content_lower:
                tags.append(domain)

        return list(set(tags))  # Remove duplicates

    async def create_sample_documents(self) -> List[Document]:
        """Create sample documents for testing"""
        documents = []

        # Sample runbook
        runbook = Document(
            id="sample_runbook_001",
            title="Database Connection Troubleshooting",
            content="""
# Database Connection Troubleshooting Runbook

## Problem
Application cannot connect to database

## Initial Assessment
1. Check database service status
2. Verify connection string
3. Test network connectivity
4. Review recent changes

## Solutions

### Solution 1: Restart Database Service
If the database service is down:
```bash
sudo systemctl restart postgresql
```

### Solution 2: Check Connection Limits
Verify connection limits haven't been exceeded:
```sql
SELECT count(*) FROM pg_stat_activity;
```

### Solution 3: Network Issues
Check firewall and network configuration:
```bash
telnet database-host 5432
```

## Prevention
- Monitor connection pools
- Set up alerts for connection limit warnings
- Implement circuit breakers
""",
            type=DocumentType.RUNBOOK,
            source="internal",
            tags=["database", "postgresql", "troubleshooting", "networking"]
        )

        # Sample ADR
        adr = Document(
            id="sample_adr_001",
            title="Decision: Microservices Architecture",
            content="""
# Architecture Decision Record: Microservices Migration

## Status
Accepted

## Context
The monolithic application has grown too large and complex to maintain efficiently.

## Decision
We will migrate to a microservices architecture using the following approach:

1. Domain-driven design for service boundaries
2. Event-driven communication between services
3. Centralized configuration management
4. Distributed tracing for observability

## Consequences

### Positive
- Improved scalability
- Technology diversity
- Easier deployment
- Better fault isolation

### Negative
- Increased complexity
- Distributed system challenges
- Eventual consistency issues
- Operational overhead

## Alternatives Considered
- Modular monolith
- Serverless architecture
- Stay with monolithic approach

## Implementation Plan
Phase 1: Extract user service
Phase 2: Extract order service
Phase 3: Implement event streaming
""",
            type=DocumentType.ADR,
            source="internal",
            tags=["architecture", "microservices", "migration"]
        )

        documents.extend([runbook, adr])
        return documents

    async def bulk_ingest(self, documents: List[Document]) -> Dict[str, Any]:
        """Bulk ingest multiple documents"""
        results = {
            "total_documents": len(documents),
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "errors": []
        }

        for document in documents:
            try:
                chunks_added = self.knowledge_base.add_document(document)
                results["successful"] += 1
                results["total_chunks"] += chunks_added

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Failed to ingest {document.title}: {str(e)}")

        logger.info(f"Bulk ingestion completed: {results['successful']} successful, {results['failed']} failed")
        return results
