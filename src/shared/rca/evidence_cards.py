"""
Evidence cards system for structured RCA evidence presentation
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class EvidenceType(Enum):
    """Types of evidence"""
    METRIC_ANOMALY = "metric_anomaly"
    LOG_PATTERN = "log_pattern"
    TRACE_SPAN = "trace_span"
    DEPLOYMENT_CHANGE = "deployment_change"
    CONFIGURATION_CHANGE = "configuration_change"
    RESOURCE_UTILIZATION = "resource_utilization"
    EXTERNAL_DEPENDENCY = "external_dependency"
    NETWORK_LATENCY = "network_latency"
    ERROR_RATE_SPIKE = "error_rate_spike"
    PERFORMANCE_DEGRADATION = "performance_degradation"


class EvidenceStrength(Enum):
    """Strength levels for evidence"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class EvidenceCard:
    """Structured evidence card for RCA"""
    id: str
    type: EvidenceType
    title: str
    description: str
    strength: EvidenceStrength
    confidence: float
    timestamp: datetime
    time_window: timedelta
    source: str
    service: str
    data: Dict[str, Any] = field(default_factory=dict)
    related_hypotheses: List[str] = field(default_factory=list)
    counterexamples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def evidence_score(self) -> float:
        """Calculate overall evidence score"""
        strength_weights = {
            EvidenceStrength.WEAK: 0.25,
            EvidenceStrength.MODERATE: 0.5,
            EvidenceStrength.STRONG: 0.75,
            EvidenceStrength.VERY_STRONG: 1.0
        }

        base_score = strength_weights.get(self.strength, 0.5)
        confidence_adjustment = (self.confidence - 0.5) * 0.2  # ±0.1 adjustment

        return max(0.0, min(1.0, base_score + confidence_adjustment))

    @property
    def is_recent(self) -> bool:
        """Check if evidence is recent (within last hour)"""
        return (datetime.now() - self.timestamp) < timedelta(hours=1)

    @property
    def has_counterexamples(self) -> bool:
        """Check if evidence has counterexamples"""
        return len(self.counterexamples) > 0


@dataclass
class EvidenceRelationship:
    """Relationship between evidence cards"""
    source_evidence_id: str
    target_evidence_id: str
    relationship_type: str  # 'supports', 'contradicts', 'correlates', 'causes'
    strength: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvidenceCardManager:
    """Manager for evidence cards and relationships"""

    def __init__(self):
        self.evidence_cards: Dict[str, EvidenceCard] = {}
        self.relationships: List[EvidenceRelationship] = []
        self.evidence_index: Dict[str, List[str]] = {}  # service -> evidence_ids

    def create_evidence_card(self, evidence_data: Dict[str, Any]) -> EvidenceCard:
        """Create a new evidence card from data"""
        try:
            # Validate required fields
            required_fields = ['type', 'title', 'description', 'timestamp', 'source', 'service']
            for field in required_fields:
                if field not in evidence_data:
                    raise ValueError(f"Missing required field: {field}")

            # Create evidence card
            card = EvidenceCard(
                id=evidence_data.get('id', f"evidence_{datetime.now().timestamp()}"),
                type=EvidenceType(evidence_data['type']),
                title=evidence_data['title'],
                description=evidence_data['description'],
                strength=EvidenceStrength(evidence_data.get('strength', 'moderate')),
                confidence=evidence_data.get('confidence', 0.8),
                timestamp=evidence_data['timestamp'],
                time_window=evidence_data.get('time_window', timedelta(hours=1)),
                source=evidence_data['source'],
                service=evidence_data['service'],
                data=evidence_data.get('data', {}),
                related_hypotheses=evidence_data.get('related_hypotheses', []),
                counterexamples=evidence_data.get('counterexamples', []),
                metadata=evidence_data.get('metadata', {})
            )

            # Store card
            self.evidence_cards[card.id] = card

            # Update index
            if card.service not in self.evidence_index:
                self.evidence_index[card.service] = []
            self.evidence_index[card.service].append(card.id)

            logger.info(f"Created evidence card: {card.id} for service {card.service}")
            return card

        except Exception as e:
            logger.error(f"Failed to create evidence card: {e}")
            raise

    def get_evidence_for_service(self, service: str, time_window: Optional[timedelta] = None) -> List[EvidenceCard]:
        """Get evidence cards for a specific service"""
        if service not in self.evidence_index:
            return []

        evidence_ids = self.evidence_index[service]
        cards = [self.evidence_cards[eid] for eid in evidence_ids if eid in self.evidence_cards]

        # Filter by time window if specified
        if time_window:
            cutoff_time = datetime.now() - time_window
            cards = [card for card in cards if card.timestamp >= cutoff_time]

        # Sort by evidence score (descending)
        cards.sort(key=lambda x: x.evidence_score, reverse=True)

        return cards

    def get_evidence_for_hypothesis(self, hypothesis_id: str) -> List[EvidenceCard]:
        """Get evidence cards related to a hypothesis"""
        related_cards = []

        for card in self.evidence_cards.values():
            if hypothesis_id in card.related_hypotheses:
                related_cards.append(card)

        # Sort by relevance and strength
        related_cards.sort(key=lambda x: (x.evidence_score, x.confidence), reverse=True)

        return related_cards

    def add_relationship(self, source_id: str, target_id: str, relationship_type: str,
                        strength: float, description: str, metadata: Dict[str, Any] = None) -> None:
        """Add relationship between evidence cards"""
        if source_id not in self.evidence_cards or target_id not in self.evidence_cards:
            raise ValueError("Source or target evidence card not found")

        relationship = EvidenceRelationship(
            source_evidence_id=source_id,
            target_evidence_id=target_id,
            relationship_type=relationship_type,
            strength=strength,
            description=description,
            metadata=metadata or {}
        )

        self.relationships.append(relationship)

        logger.info(f"Added {relationship_type} relationship between {source_id} and {target_id}")

    def get_related_evidence(self, evidence_id: str) -> List[Tuple[EvidenceCard, EvidenceRelationship]]:
        """Get evidence cards related to the given evidence"""
        related = []

        for relationship in self.relationships:
            if relationship.source_evidence_id == evidence_id:
                target_card = self.evidence_cards.get(relationship.target_evidence_id)
                if target_card:
                    related.append((target_card, relationship))
            elif relationship.target_evidence_id == evidence_id:
                source_card = self.evidence_cards.get(relationship.source_evidence_id)
                if source_card:
                    related.append((source_card, relationship))

        return related

    def find_conflicting_evidence(self, evidence_id: str) -> List[EvidenceCard]:
        """Find evidence that conflicts with the given evidence"""
        conflicting = []

        for relationship in self.relationships:
            if (relationship.source_evidence_id == evidence_id or
                relationship.target_evidence_id == evidence_id):
                if relationship.relationship_type == 'contradicts':
                    # Find the other evidence in the relationship
                    other_id = (relationship.target_evidence_id
                              if relationship.source_evidence_id == evidence_id
                              else relationship.source_evidence_id)

                    other_card = self.evidence_cards.get(other_id)
                    if other_card:
                        conflicting.append(other_card)

        return conflicting

    def calculate_evidence_consensus(self, evidence_ids: List[str]) -> Dict[str, Any]:
        """Calculate consensus among multiple evidence cards"""
        if not evidence_ids:
            return {"consensus_score": 0, "agreement_level": "none"}

        cards = [self.evidence_cards.get(eid) for eid in evidence_ids]
        cards = [card for card in cards if card is not None]

        if not cards:
            return {"consensus_score": 0, "agreement_level": "none"}

        # Calculate average evidence score
        avg_score = sum(card.evidence_score for card in cards) / len(cards)

        # Check for conflicting evidence
        conflicting_pairs = 0
        total_pairs = len(cards) * (len(cards) - 1) / 2

        for i, card1 in enumerate(cards):
            for card2 in cards[i+1:]:
                # Check if these cards contradict each other
                for rel in self.relationships:
                    if ((rel.source_evidence_id == card1.id and rel.target_evidence_id == card2.id) or
                        (rel.source_evidence_id == card2.id and rel.target_evidence_id == card1.id)):
                        if rel.relationship_type == 'contradicts':
                            conflicting_pairs += 1
                            break

        conflict_ratio = conflicting_pairs / total_pairs if total_pairs > 0 else 0
        consensus_score = avg_score * (1 - conflict_ratio)

        # Determine agreement level
        if consensus_score >= 0.8:
            agreement_level = "strong"
        elif consensus_score >= 0.6:
            agreement_level = "moderate"
        elif consensus_score >= 0.4:
            agreement_level = "weak"
        else:
            agreement_level = "conflicting"

        return {
            "consensus_score": consensus_score,
            "agreement_level": agreement_level,
            "average_evidence_score": avg_score,
            "conflict_ratio": conflict_ratio,
            "total_evidence": len(cards)
        }

    def generate_evidence_summary(self, service: str, time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Generate summary of evidence for a service"""
        cards = self.get_evidence_for_service(service, time_window)

        if not cards:
            return {
                "service": service,
                "total_evidence": 0,
                "evidence_types": {},
                "strength_distribution": {},
                "average_confidence": 0,
                "time_window": time_window.total_seconds()
            }

        # Count evidence types
        evidence_types = {}
        for card in cards:
            evidence_types[card.type.value] = evidence_types.get(card.type.value, 0) + 1

        # Count strength distribution
        strength_dist = {}
        for card in cards:
            strength_dist[card.strength.value] = strength_dist.get(card.strength.value, 0) + 1

        # Calculate average confidence
        avg_confidence = sum(card.confidence for card in cards) / len(cards)

        # Get strongest evidence
        strongest_evidence = max(cards, key=lambda x: x.evidence_score) if cards else None

        return {
            "service": service,
            "total_evidence": len(cards),
            "evidence_types": evidence_types,
            "strength_distribution": strength_dist,
            "average_confidence": avg_confidence,
            "strongest_evidence": {
                "id": strongest_evidence.id,
                "title": strongest_evidence.title,
                "score": strongest_evidence.evidence_score
            } if strongest_evidence else None,
            "time_window": time_window.total_seconds(),
            "recent_evidence_count": sum(1 for card in cards if card.is_recent)
        }

    def cleanup_old_evidence(self, max_age: timedelta = timedelta(days=7)) -> int:
        """Clean up old evidence cards"""
        cutoff_time = datetime.now() - max_age
        old_evidence_ids = []

        for evidence_id, card in self.evidence_cards.items():
            if card.timestamp < cutoff_time:
                old_evidence_ids.append(evidence_id)

        # Remove old evidence
        for evidence_id in old_evidence_ids:
            del self.evidence_cards[evidence_id]

            # Remove from index
            for service, evidence_list in self.evidence_index.items():
                if evidence_id in evidence_list:
                    evidence_list.remove(evidence_id)

        # Remove old relationships
        self.relationships = [
            rel for rel in self.relationships
            if rel.source_evidence_id in self.evidence_cards
            and rel.target_evidence_id in self.evidence_cards
        ]

        logger.info(f"Cleaned up {len(old_evidence_ids)} old evidence cards")
        return len(old_evidence_ids)

    def export_evidence_cards(self, service: Optional[str] = None) -> List[Dict[str, Any]]:
        """Export evidence cards as dictionaries"""
        cards_to_export = []

        if service:
            cards_to_export = self.get_evidence_for_service(service)
        else:
            cards_to_export = list(self.evidence_cards.values())

        return [
            {
                "id": card.id,
                "type": card.type.value,
                "title": card.title,
                "description": card.description,
                "strength": card.strength.value,
                "confidence": card.confidence,
                "timestamp": card.timestamp.isoformat(),
                "time_window_seconds": card.time_window.total_seconds(),
                "source": card.source,
                "service": card.service,
                "data": card.data,
                "related_hypotheses": card.related_hypotheses,
                "counterexamples": card.counterexamples,
                "metadata": card.metadata,
                "evidence_score": card.evidence_score,
                "created_at": card.created_at.isoformat()
            }
            for card in cards_to_export
        ]

    def import_evidence_cards(self, cards_data: List[Dict[str, Any]]) -> int:
        """Import evidence cards from dictionaries"""
        imported_count = 0

        for card_data in cards_data:
            try:
                # Convert string timestamps back to datetime
                card_data['timestamp'] = datetime.fromisoformat(card_data['timestamp'])
                card_data['created_at'] = datetime.fromisoformat(card_data['created_at'])
                card_data['time_window'] = timedelta(seconds=card_data['time_window_seconds'])
                del card_data['time_window_seconds']

                # Convert enum values back
                card_data['type'] = EvidenceType(card_data['type'])
                card_data['strength'] = EvidenceStrength(card_data['strength'])

                # Remove computed fields
                card_data.pop('evidence_score', None)

                self.create_evidence_card(card_data)
                imported_count += 1

            except Exception as e:
                logger.error(f"Failed to import evidence card: {e}")
                continue

        logger.info(f"Imported {imported_count} evidence cards")
        return imported_count
