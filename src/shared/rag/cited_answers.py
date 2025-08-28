"""
Cited answers system with document anchors and references
"""
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import re

from .knowledge_base import RetrievalResult, Document


logger = logging.getLogger(__name__)


class CitationStyle(Enum):
    """Citation styles for references"""
    IEEE = "ieee"
    APA = "apa"
    MLA = "mla"
    HARVARD = "harvard"
    CHICAGO = "chicago"
    SIMPLE = "simple"


@dataclass
class Citation:
    """Citation for a document reference"""
    document_id: str
    document_title: str
    anchors: List[str]
    page_numbers: Optional[str] = None
    section_numbers: Optional[str] = None
    citation_text: str = ""
    url: Optional[str] = None
    accessed_date: datetime = field(default_factory=datetime.now)


@dataclass
class CitedAnswer:
    """Answer with citations and references"""
    question: str
    answer: str
    citations: List[Citation]
    confidence: float
    reasoning: str
    generated_at: datetime = field(default_factory=datetime.now)
    citation_style: CitationStyle = CitationStyle.SIMPLE

    @property
    def citation_count(self) -> int:
        """Number of citations"""
        return len(self.citations)

    @property
    def unique_documents(self) -> int:
        """Number of unique documents cited"""
        return len(set(c.document_id for c in self.citations))

    def format_citations(self, style: CitationStyle = None) -> str:
        """Format citations in the specified style"""
        if style is None:
            style = self.citation_style

        formatter = CitationFormatter()
        return formatter.format_citations(self.citations, style)

    def get_citation_summary(self) -> Dict[str, Any]:
        """Get summary of citations"""
        return {
            "total_citations": self.citation_count,
            "unique_documents": self.unique_documents,
            "citation_style": self.citation_style.value,
            "documents": [
                {
                    "id": c.document_id,
                    "title": c.document_title,
                    "anchors": c.anchors,
                    "citation_text": c.citation_text
                }
                for c in self.citations
            ]
        }


class CitationFormatter:
    """Formats citations in different styles"""

    def format_citations(self, citations: List[Citation], style: CitationStyle) -> str:
        """Format list of citations"""
        formatted_citations = []

        for i, citation in enumerate(citations, 1):
            formatted = self.format_single_citation(citation, style, i)
            formatted_citations.append(formatted)

        return "\n".join(formatted_citations)

    def format_single_citation(self, citation: Citation, style: CitationStyle, index: int) -> str:
        """Format a single citation"""
        if style == CitationStyle.SIMPLE:
            return self._format_simple(citation, index)
        elif style == CitationStyle.IEEE:
            return self._format_ieee(citation, index)
        elif style == CitationStyle.APA:
            return self._format_apa(citation, index)
        elif style == CitationStyle.MLA:
            return self._format_mla(citation, index)
        elif style == CitationStyle.HARVARD:
            return self._format_harvard(citation, index)
        elif style == CitationStyle.CHICAGO:
            return self._format_chicago(citation, index)
        else:
            return self._format_simple(citation, index)

    def _format_simple(self, citation: Citation, index: int) -> str:
        """Format citation in simple style"""
        anchors_text = ", ".join(citation.anchors) if citation.anchors else ""
        anchor_part = f" (Section: {anchors_text})" if anchors_text else ""

        return f"[{index}] {citation.document_title}{anchor_part}"

    def _format_ieee(self, citation: Citation, index: int) -> str:
        """Format citation in IEEE style"""
        year = citation.accessed_date.year
        anchors_text = ", ".join(citation.anchors) if citation.anchors else ""

        if citation.url:
            return f"[{index}] \"{citation.document_title},\" {year}. [Online]. Available: {citation.url}"
        else:
            source = citation.url or "Internal Documentation"
            return f"[{index}] \"{citation.document_title},\" {source}, {year}"

    def _format_apa(self, citation: Citation, index: int) -> str:
        """Format citation in APA style"""
        year = citation.accessed_date.year

        if citation.url:
            return f"{citation.document_title}. ({year}). Retrieved from {citation.url}"
        else:
            return f"{citation.document_title}. ({year})."

    def _format_mla(self, citation: Citation, index: int) -> str:
        """Format citation in MLA style"""
        year = citation.accessed_date.year

        if citation.url:
            return f"\"{citation.document_title}.\" {year}. {citation.url}."
        else:
            return f"\"{citation.document_title}.\" {year}."

    def _format_harvard(self, citation: Citation, index: int) -> str:
        """Format citation in Harvard style"""
        year = citation.accessed_date.year

        if citation.url:
            return f"{citation.document_title} ({year}) {citation.url} (accessed {citation.accessed_date.strftime('%d %B %Y')})"
        else:
            return f"{citation.document_title} ({year})"

    def _format_chicago(self, citation: Citation, index: int) -> str:
        """Format citation in Chicago style"""
        year = citation.accessed_date.year

        if citation.url:
            return f"\"{citation.document_title}.\" Accessed {citation.accessed_date.strftime('%B %d, %Y')}. {citation.url}."
        else:
            return f"\"{citation.document_title}.\" {year}."


class AnswerGenerator:
    """Generates cited answers from retrieved documents"""

    def __init__(self, citation_style: CitationStyle = CitationStyle.SIMPLE):
        self.citation_style = citation_style

    def generate_cited_answer(self, question: str, retrieval_results: List[RetrievalResult],
                            confidence_threshold: float = 0.3) -> CitedAnswer:
        """
        Generate an answer with citations from retrieved documents

        Args:
            question: The question being answered
            retrieval_results: Results from document retrieval
            confidence_threshold: Minimum similarity score to include

        Returns:
            CitedAnswer with citations
        """
        # Filter results by confidence
        relevant_results = [r for r in retrieval_results if r.similarity_score >= confidence_threshold]

        if not relevant_results:
            return CitedAnswer(
                question=question,
                answer="No relevant information found in the knowledge base.",
                citations=[],
                confidence=0.0,
                reasoning="No documents met the confidence threshold"
            )

        # Generate answer from relevant results
        answer_text = self._synthesize_answer(question, relevant_results)

        # Create citations
        citations = self._create_citations(relevant_results)

        # Calculate overall confidence
        avg_confidence = sum(r.similarity_score for r in relevant_results) / len(relevant_results)
        source_count = len(relevant_results)
        overall_confidence = min(avg_confidence * (1 + source_count * 0.1), 0.95)

        # Generate reasoning
        reasoning = self._generate_reasoning(relevant_results, overall_confidence)

        return CitedAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            confidence=overall_confidence,
            reasoning=reasoning,
            citation_style=self.citation_style
        )

    def _synthesize_answer(self, question: str, results: List[RetrievalResult]) -> str:
        """Synthesize answer from retrieval results"""
        if not results:
            return "No information available."

        # Simple synthesis approach
        answer_parts = []

        # Use the most relevant result as the primary answer
        primary_result = max(results, key=lambda r: r.similarity_score)

        # Extract relevant information from the primary result
        primary_content = primary_result.chunk.content
        primary_title = primary_result.document.title

        # Create answer based on question type
        if question.lower().startswith(('what', 'how', 'why')):
            answer_parts.append(f"According to {primary_title}: {primary_content[:500]}")
        else:
            answer_parts.append(primary_content[:500])

        # Add information from other relevant sources
        other_results = [r for r in results if r != primary_result][:2]  # Limit to 2 additional sources

        for result in other_results:
            if result.similarity_score > 0.5:
                additional_info = result.chunk.content[:200]
                answer_parts.append(f"Additional information from {result.document.title}: {additional_info}")

        # Combine answer parts
        full_answer = " ".join(answer_parts)

        # Add citation reference
        citation_refs = [f"[{i+1}]" for i in range(len(results))]
        full_answer += f"\n\nReferences: {', '.join(citation_refs)}"

        return full_answer

    def _create_citations(self, results: List[RetrievalResult]) -> List[Citation]:
        """Create citations from retrieval results"""
        citations = []

        for i, result in enumerate(results):
            citation = Citation(
                document_id=result.document.id,
                document_title=result.document.title,
                anchors=result.anchors,
                citation_text=self._generate_citation_text(result, i + 1),
                url=result.document.source if result.document.source.startswith(('http', 'https')) else None
            )
            citations.append(citation)

        return citations

    def _generate_citation_text(self, result: RetrievalResult, index: int) -> str:
        """Generate citation text for a result"""
        title = result.document.title
        anchors = ", ".join(result.anchors) if result.anchors else ""

        if anchors:
            return f"{title} (Section: {anchors})"
        else:
            return title

    def _generate_reasoning(self, results: List[RetrievalResult], confidence: float) -> str:
        """Generate reasoning for the answer"""
        source_count = len(results)
        avg_similarity = sum(r.similarity_score for r in results) / source_count

        reasoning_parts = [
            f"Answer synthesized from {source_count} relevant document{'s' if source_count != 1 else ''}",
            ".2f",
            ".2f"
        ]

        if source_count > 1:
            doc_types = set(r.document.type.value for r in results)
            reasoning_parts.append(f" including {', '.join(doc_types)}")

        return "".join(reasoning_parts)


class AnswerValidator:
    """Validates cited answers for quality and correctness"""

    def __init__(self):
        pass

    def validate_answer(self, answer: CitedAnswer) -> Dict[str, Any]:
        """Validate the quality of a cited answer"""
        validation_results = {
            "is_valid": True,
            "issues": [],
            "score": 0.0,
            "recommendations": []
        }

        # Check answer length
        if len(answer.answer) < 50:
            validation_results["issues"].append("Answer is too short")
            validation_results["is_valid"] = False

        # Check citation count
        if answer.citation_count == 0:
            validation_results["issues"].append("No citations provided")
            validation_results["is_valid"] = False

        # Check confidence
        if answer.confidence < 0.3:
            validation_results["issues"].append("Low confidence in answer")
            validation_results["recommendations"].append("Consider reviewing source documents")

        # Check for hallucinated content (simplified check)
        hallucination_score = self._check_for_hallucinations(answer)
        if hallucination_score > 0.7:
            validation_results["issues"].append("Potential hallucinated content detected")
            validation_results["recommendations"].append("Verify answer against source documents")

        # Calculate overall score
        base_score = answer.confidence
        citation_bonus = min(answer.citation_count * 0.1, 0.3)  # Max 0.3 bonus for citations
        validation_results["score"] = base_score + citation_bonus

        if validation_results["issues"]:
            validation_results["score"] *= 0.8  # Reduce score for issues

        return validation_results

    def _check_for_hallucinations(self, answer: CitedAnswer) -> float:
        """Check for potential hallucinations in the answer"""
        # Simplified hallucination detection
        # In a real implementation, this would use more sophisticated NLP techniques

        answer_words = set(answer.answer.lower().split())
        source_words = set()

        for citation in answer.citations:
            # This is a placeholder - in reality, you'd need access to the full document content
            # For now, we'll use a simple heuristic
            source_words.update(citation.document_title.lower().split())

        # Calculate overlap
        if not source_words:
            return 0.5  # Neutral score if no source words

        overlap = len(answer_words.intersection(source_words)) / len(answer_words)

        # Low overlap might indicate hallucination
        hallucination_score = max(0, 1 - overlap)

        return hallucination_score


class AnswerExplainer:
    """Explains cited answers with detailed reasoning"""

    def __init__(self):
        pass

    def explain_answer(self, answer: CitedAnswer) -> Dict[str, Any]:
        """Provide detailed explanation of the answer"""
        explanation = {
            "summary": self._generate_summary(answer),
            "reasoning_steps": self._generate_reasoning_steps(answer),
            "evidence_mapping": self._map_evidence_to_answer(answer),
            "confidence_factors": self._analyze_confidence_factors(answer),
            "alternative_views": self._identify_alternative_views(answer)
        }

        return explanation

    def _generate_summary(self, answer: CitedAnswer) -> str:
        """Generate a summary of the answer"""
        return f"This answer addresses '{answer.question}' with {answer.citation_count} citations from {answer.unique_documents} unique documents. The answer has {answer.confidence:.1%} confidence."

    def _generate_reasoning_steps(self, answer: CitedAnswer) -> List[str]:
        """Generate step-by-step reasoning"""
        steps = [
            f"Question analysis: Understanding the intent behind '{answer.question}'",
            f"Document retrieval: Found {len(answer.citations)} relevant documents",
            "Information synthesis: Combining information from multiple sources",
            "Answer generation: Creating coherent response with citations",
            f"Confidence assessment: Calculated {answer.confidence:.1%} confidence based on source relevance"
        ]

        return steps

    def _map_evidence_to_answer(self, answer: CitedAnswer) -> Dict[str, Any]:
        """Map evidence to parts of the answer"""
        # Simplified mapping - in reality, this would use more sophisticated NLP
        mapping = {}

        answer_sentences = answer.answer.split('. ')

        for i, sentence in enumerate(answer_sentences):
            relevant_citations = []

            for citation in answer.citations:
                # Simple keyword matching
                sentence_words = set(sentence.lower().split())
                title_words = set(citation.document_title.lower().split())

                if sentence_words.intersection(title_words):
                    relevant_citations.append(citation.document_id)

            if relevant_citations:
                mapping[f"sentence_{i+1}"] = {
                    "text": sentence,
                    "supporting_citations": relevant_citations
                }

        return mapping

    def _analyze_confidence_factors(self, answer: CitedAnswer) -> Dict[str, Any]:
        """Analyze factors contributing to answer confidence"""
        factors = {
            "source_quality": self._assess_source_quality(answer),
            "information_recency": self._assess_information_recency(answer),
            "consensus_level": self._assess_consensus_level(answer),
            "completeness": self._assess_answer_completeness(answer)
        }

        return factors

    def _assess_source_quality(self, answer: CitedAnswer) -> float:
        """Assess the quality of sources used"""
        # Simplified assessment based on document types
        quality_score = 0.0

        for citation in answer.citations:
            # This is a placeholder - in reality, you'd have quality scores for different document types
            if "runbook" in citation.document_title.lower():
                quality_score += 0.9
            elif "guide" in citation.document_title.lower():
                quality_score += 0.7
            else:
                quality_score += 0.5

        return quality_score / len(answer.citations) if answer.citations else 0.0

    def _assess_information_recency(self, answer: CitedAnswer) -> float:
        """Assess how recent the information is"""
        # Placeholder - in reality, you'd check document update dates
        return 0.8

    def _assess_consensus_level(self, answer: CitedAnswer) -> float:
        """Assess level of consensus among sources"""
        if len(answer.citations) <= 1:
            return 0.5

        # Simplified consensus calculation
        return min(1.0, len(answer.citations) * 0.2)

    def _assess_answer_completeness(self, answer: CitedAnswer) -> float:
        """Assess how complete the answer is"""
        # Simple length-based assessment
        word_count = len(answer.answer.split())
        completeness = min(1.0, word_count / 200)  # Expect ~200 words for complete answer

        return completeness

    def _identify_alternative_views(self, answer: CitedAnswer) -> List[str]:
        """Identify alternative viewpoints or approaches"""
        alternatives = []

        if answer.citation_count < 3:
            alternatives.append("Consider reviewing additional documentation for comprehensive coverage")

        if answer.confidence < 0.7:
            alternatives.append("Low confidence suggests exploring alternative sources or consulting experts")

        # Add domain-specific alternatives based on question content
        question_lower = answer.question.lower()
        if "troubleshoot" in question_lower:
            alternatives.append("Consider checking system logs and metrics for additional diagnostic information")
        elif "deploy" in question_lower:
            alternatives.append("Review deployment pipeline and rollback procedures")

        return alternatives
