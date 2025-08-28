"""
Hypothesis builder for Root Cause Analysis (RCA)
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import re

from ..anomaly_detection.stl_decomposition import SeasonalBaseline
from ..anomaly_detection.esd_spike_detection import ESDSpikeDetector
from ..change_correlation.change_correlation import ChangeCorrelationEngine
from ..log_clustering.log_clustering import LogClusteringEngine
from ..database.config import get_postgres_session


logger = logging.getLogger(__name__)


class HypothesisType(Enum):
    """Types of hypotheses that can be generated"""
    DEPLOYMENT_RELATED = "deployment_related"
    CONFIGURATION_CHANGE = "configuration_change"
    RESOURCE_CONSTRAINT = "resource_constraint"
    EXTERNAL_DEPENDENCY = "external_dependency"
    CODE_BUG = "code_bug"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    LOAD_SPIKE = "load_spike"
    DATA_CORRUPTION = "data_corruption"
    NETWORK_ISSUE = "network_issue"
    HUMAN_ERROR = "human_error"


class ConfidenceLevel(Enum):
    """Confidence levels for hypotheses"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class Evidence:
    """Evidence supporting or contradicting a hypothesis"""
    id: str
    type: str  # 'supporting', 'contradicting', 'neutral'
    source: str  # 'metrics', 'logs', 'traces', 'changes', 'topology'
    description: str
    timestamp: datetime
    confidence: float
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    """Root cause hypothesis"""
    id: str
    type: HypothesisType
    title: str
    description: str
    confidence: float
    confidence_level: ConfidenceLevel
    supporting_evidence: List[Evidence] = field(default_factory=list)
    contradicting_evidence: List[Evidence] = field(default_factory=list)
    related_services: List[str] = field(default_factory=list)
    time_window: timedelta
    generated_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HypothesisEvaluation:
    """Evaluation of hypothesis quality"""
    hypothesis_id: str
    precision_score: float  # How accurate the hypothesis is
    recall_score: float     # How complete the hypothesis is
    evidence_quality: float # Quality of supporting evidence
    overall_score: float
    evaluation_timestamp: datetime


class HypothesisBuilder:
    """Builds and evaluates root cause hypotheses"""

    def __init__(self):
        self.seasonal_baseline = SeasonalBaseline()
        self.spike_detector = ESDSpikeDetector(max_anomalies=5)
        self.correlation_engine = ChangeCorrelationEngine()
        self.log_clusterer = LogClusteringEngine()

        # Hypothesis templates
        self.templates = self._load_hypothesis_templates()

    def _load_hypothesis_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load hypothesis generation templates"""
        return {
            "deployment_related": {
                "title_template": "Issue caused by recent deployment of {service} version {version}",
                "description_template": "Deployment of {service} v{version} at {time} correlates with incident start",
                "evidence_types": ["deployment", "correlation", "metrics"]
            },
            "resource_constraint": {
                "title_template": "Resource constraint in {service} - {resource} utilization at {percentage}%",
                "description_template": "{resource} utilization spiked to {percentage}% around incident time",
                "evidence_types": ["metrics", "resource", "health"]
            },
            "external_dependency": {
                "title_template": "Issue in external dependency {dependency}",
                "description_template": "Calls to {dependency} showing increased latency/errors",
                "evidence_types": ["metrics", "external", "topology"]
            },
            "load_spike": {
                "title_template": "Unexpected load spike on {service}",
                "description_template": "Request rate increased {percentage}% above baseline",
                "evidence_types": ["metrics", "baseline", "seasonal"]
            },
            "network_issue": {
                "title_template": "Network connectivity issue affecting {service}",
                "description_template": "Network errors detected between {source} and {destination}",
                "evidence_types": ["network", "metrics", "logs"]
            }
        }

    async def generate_hypotheses(self, incident_id: str, context: Dict[str, Any]) -> List[Hypothesis]:
        """
        Generate hypotheses for an incident

        Args:
            incident_id: ID of the incident
            context: Context data including metrics, logs, changes, etc.

        Returns:
            List of generated hypotheses
        """
        hypotheses = []

        try:
            # Extract context data
            metrics = context.get("metrics", [])
            logs = context.get("logs", [])
            changes = context.get("changes", {})
            topology = context.get("topology", {})
            incident_data = context.get("incident", {})

            # Generate different types of hypotheses
            deployment_hypotheses = await self._generate_deployment_hypotheses(
                incident_data, changes.get("deployments", [])
            )
            hypotheses.extend(deployment_hypotheses)

            resource_hypotheses = await self._generate_resource_hypotheses(
                incident_data, metrics
            )
            hypotheses.extend(resource_hypotheses)

            dependency_hypotheses = await self._generate_dependency_hypotheses(
                incident_data, metrics, topology
            )
            hypotheses.extend(dependency_hypotheses)

            load_hypotheses = await self._generate_load_hypotheses(
                incident_data, metrics
            )
            hypotheses.extend(load_hypotheses)

            # Evaluate and rank hypotheses
            evaluated_hypotheses = await self._evaluate_hypotheses(hypotheses, context)

            # Sort by confidence
            evaluated_hypotheses.sort(key=lambda h: h.confidence, reverse=True)

            logger.info(f"Generated {len(evaluated_hypotheses)} hypotheses for incident {incident_id}")

            return evaluated_hypotheses[:10]  # Return top 10

        except Exception as e:
            logger.error(f"Failed to generate hypotheses for incident {incident_id}: {e}")
            return []

    async def _generate_deployment_hypotheses(self, incident: Dict[str, Any],
                                           deployments: List[Dict[str, Any]]) -> List[Hypothesis]:
        """Generate deployment-related hypotheses"""
        hypotheses = []

        incident_time = incident.get("timestamp", datetime.now())
        service = incident.get("service", "unknown")

        for deployment in deployments:
            # Check if deployment is recent and related to the service
            deployment_time = deployment.get("timestamp", datetime.min)
            deployment_service = deployment.get("service", "unknown")

            time_diff = abs((incident_time - deployment_time).total_seconds())
            service_match = service == deployment_service

            if time_diff < 3600 and service_match:  # Within 1 hour
                template = self.templates["deployment_related"]

                hypothesis = Hypothesis(
                    id=f"deployment_{deployment['id']}_{incident_time.timestamp()}",
                    type=HypothesisType.DEPLOYMENT_RELATED,
                    title=template["title_template"].format(
                        service=deployment_service,
                        version=deployment.get("version", "unknown")
                    ),
                    description=template["description_template"].format(
                        service=deployment_service,
                        version=deployment.get("version", "unknown"),
                        time=deployment_time.isoformat()
                    ),
                    confidence=self._calculate_deployment_confidence(time_diff, deployment),
                    confidence_level=ConfidenceLevel.HIGH,
                    time_window=timedelta(hours=1),
                    generated_at=datetime.now(),
                    updated_at=datetime.now(),
                    metadata={
                        "deployment_id": deployment["id"],
                        "deployment_time": deployment_time.isoformat(),
                        "time_diff_seconds": time_diff
                    }
                )

                hypotheses.append(hypothesis)

        return hypotheses

    async def _generate_resource_hypotheses(self, incident: Dict[str, Any],
                                          metrics: List[Dict[str, Any]]) -> List[Hypothesis]:
        """Generate resource constraint hypotheses"""
        hypotheses = []

        service = incident.get("service", "unknown")
        incident_time = incident.get("timestamp", datetime.now())

        # Look for resource metrics around incident time
        resource_metrics = [m for m in metrics if m.get("service") == service]
        resource_metrics = [m for m in resource_metrics
                          if abs((m["timestamp"] - incident_time).total_seconds()) < 300]  # 5 min window

        # Check for high utilization
        for metric in resource_metrics:
            metric_name = metric.get("name", "")
            value = metric.get("value", 0)

            if "cpu" in metric_name.lower() and value > 80:
                hypothesis = self._create_resource_hypothesis(
                    "cpu", value, incident_time, service
                )
                hypotheses.append(hypothesis)

            elif "memory" in metric_name.lower() and value > 90:
                hypothesis = self._create_resource_hypothesis(
                    "memory", value, incident_time, service
                )
                hypotheses.append(hypothesis)

        return hypotheses

    async def _generate_dependency_hypotheses(self, incident: Dict[str, Any],
                                           metrics: List[Dict[str, Any]],
                                           topology: Dict[str, Any]) -> List[Hypothesis]:
        """Generate external dependency hypotheses"""
        hypotheses = []

        service = incident.get("service", "unknown")

        # Look for dependency-related metrics
        dependency_metrics = [m for m in metrics if "dependency" in m.get("name", "").lower()]
        dependency_metrics = [m for m in dependency_metrics if m.get("service") == service]

        # Check for dependency failures
        for metric in dependency_metrics:
            if metric.get("name", "").lower().endswith("error") and metric.get("value", 0) > 0:
                dependency_name = self._extract_dependency_name(metric)

                hypothesis = Hypothesis(
                    id=f"dependency_{dependency_name}_{metric['timestamp'].timestamp()}",
                    type=HypothesisType.EXTERNAL_DEPENDENCY,
                    title=f"Issue in external dependency {dependency_name}",
                    description=f"Calls to {dependency_name} showing errors around incident time",
                    confidence=0.7,
                    confidence_level=ConfidenceLevel.MEDIUM,
                    time_window=timedelta(minutes=10),
                    generated_at=datetime.now(),
                    updated_at=datetime.now(),
                    metadata={
                        "dependency": dependency_name,
                        "error_count": metric.get("value", 0)
                    }
                )

                hypotheses.append(hypothesis)

        return hypotheses

    async def _generate_load_hypotheses(self, incident: Dict[str, Any],
                                      metrics: List[Dict[str, Any]]) -> List[Hypothesis]:
        """Generate load spike hypotheses"""
        hypotheses = []

        service = incident.get("service", "unknown")
        incident_time = incident.get("timestamp", datetime.now())

        # Look for request rate metrics
        request_metrics = [m for m in metrics
                          if "request" in m.get("name", "").lower() and m.get("service") == service]

        for metric in request_metrics:
            # Check if this represents a spike
            spike_result = self.spike_detector.detect_single_value(
                metric["value"],
                [m["value"] for m in request_metrics if m != metric]
            )

            if spike_result.is_spike:
                percentage_increase = self._calculate_percentage_increase(metric, request_metrics)

                hypothesis = Hypothesis(
                    id=f"load_spike_{service}_{metric['timestamp'].timestamp()}",
                    type=HypothesisType.LOAD_SPIKE,
                    title=f"Unexpected load spike on {service}",
                    description=f"Request rate increased {percentage_increase:.1f}% above baseline",
                    confidence=spike_result.confidence,
                    confidence_level=ConfidenceLevel.MEDIUM,
                    time_window=timedelta(minutes=5),
                    generated_at=datetime.now(),
                    updated_at=datetime.now(),
                    metadata={
                        "spike_score": spike_result.spike_score,
                        "percentage_increase": percentage_increase,
                        "metric_value": metric["value"]
                    }
                )

                hypotheses.append(hypothesis)

        return hypotheses

    def _create_resource_hypothesis(self, resource: str, utilization: float,
                                  incident_time: datetime, service: str) -> Hypothesis:
        """Create a resource constraint hypothesis"""
        template = self.templates["resource_constraint"]

        return Hypothesis(
            id=f"resource_{resource}_{service}_{incident_time.timestamp()}",
            type=HypothesisType.RESOURCE_CONSTRAINT,
            title=template["title_template"].format(
                service=service,
                resource=resource,
                percentage=utilization
            ),
            description=template["description_template"].format(
                resource=resource,
                percentage=utilization
            ),
            confidence=min(utilization / 100, 0.9),  # Higher utilization = higher confidence
            confidence_level=ConfidenceLevel.HIGH if utilization > 90 else ConfidenceLevel.MEDIUM,
            time_window=timedelta(minutes=10),
            generated_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={
                "resource": resource,
                "utilization": utilization,
                "service": service
            }
        )

    def _calculate_deployment_confidence(self, time_diff: float, deployment: Dict[str, Any]) -> float:
        """Calculate confidence for deployment hypothesis"""
        # Recent deployments are more likely to be the cause
        time_factor = max(0, 1 - (time_diff / 3600))  # Decay over 1 hour

        # Failed deployments are more suspicious
        success_factor = 0.6 if deployment.get("success", True) else 1.0

        # Major version changes are more risky
        version = deployment.get("version", "")
        version_factor = 0.8 if self._is_major_version_change(version) else 0.6

        return (time_factor * 0.5 + success_factor * 0.3 + version_factor * 0.2)

    def _calculate_percentage_increase(self, current_metric: Dict[str, Any],
                                     historical_metrics: List[Dict[str, Any]]) -> float:
        """Calculate percentage increase from baseline"""
        if not historical_metrics:
            return 0.0

        current_value = current_metric["value"]
        baseline_values = [m["value"] for m in historical_metrics]

        if not baseline_values:
            return 0.0

        baseline_avg = sum(baseline_values) / len(baseline_values)

        if baseline_avg == 0:
            return 0.0 if current_value == 0 else float('inf')

        return ((current_value - baseline_avg) / baseline_avg) * 100

    def _extract_dependency_name(self, metric: Dict[str, Any]) -> str:
        """Extract dependency name from metric"""
        metric_name = metric.get("name", "")
        # Try to extract dependency name from metric name
        match = re.search(r'dependency[._]([^._]+)', metric_name, re.IGNORECASE)
        if match:
            return match.group(1)

        # Fallback to service name or unknown
        return metric.get("labels", {}).get("dependency", "unknown")

    def _is_major_version_change(self, version: str) -> bool:
        """Check if version change is major"""
        if not version:
            return False

        # Simple check for major version changes (1.x -> 2.x)
        parts = version.split('.')
        if len(parts) >= 2:
            try:
                major = int(parts[0])
                return major > 1  # Assume anything > 1 is a major change
            except ValueError:
                pass

        return False

    async def _evaluate_hypotheses(self, hypotheses: List[Hypothesis],
                                 context: Dict[str, Any]) -> List[Hypothesis]:
        """Evaluate and refine hypotheses with additional evidence"""
        evaluated_hypotheses = []

        for hypothesis in hypotheses:
            # Gather supporting and contradicting evidence
            supporting_evidence = await self._gather_supporting_evidence(hypothesis, context)
            contradicting_evidence = await self._gather_contradicting_evidence(hypothesis, context)

            # Update hypothesis with evidence
            hypothesis.supporting_evidence = supporting_evidence
            hypothesis.contradicting_evidence = contradicting_evidence

            # Recalculate confidence based on evidence
            hypothesis.confidence = self._recalculate_confidence(hypothesis)
            hypothesis.confidence_level = self._calculate_confidence_level(hypothesis.confidence)

            evaluated_hypotheses.append(hypothesis)

        return evaluated_hypotheses

    async def _gather_supporting_evidence(self, hypothesis: Hypothesis,
                                        context: Dict[str, Any]) -> List[Evidence]:
        """Gather evidence supporting the hypothesis"""
        evidence = []

        # Implementation would gather specific evidence based on hypothesis type
        # This is a simplified version

        if hypothesis.type == HypothesisType.DEPLOYMENT_RELATED:
            # Look for deployment-related evidence
            evidence.append(Evidence(
                id=f"evidence_deployment_{hypothesis.id}",
                type="supporting",
                source="changes",
                description="Recent deployment correlates with incident timing",
                timestamp=hypothesis.generated_at,
                confidence=0.8,
                data=hypothesis.metadata
            ))

        elif hypothesis.type == HypothesisType.RESOURCE_CONSTRAINT:
            # Look for resource usage evidence
            evidence.append(Evidence(
                id=f"evidence_resource_{hypothesis.id}",
                type="supporting",
                source="metrics",
                description="High resource utilization detected",
                timestamp=hypothesis.generated_at,
                confidence=0.9,
                data=hypothesis.metadata
            ))

        return evidence

    async def _gather_contradicting_evidence(self, hypothesis: Hypothesis,
                                           context: Dict[str, Any]) -> List[Evidence]:
        """Gather evidence contradicting the hypothesis"""
        # Simplified implementation
        return []

    def _recalculate_confidence(self, hypothesis: Hypothesis) -> float:
        """Recalculate hypothesis confidence based on evidence"""
        base_confidence = hypothesis.confidence

        # Adjust based on evidence quality and quantity
        supporting_weight = len(hypothesis.supporting_evidence) * 0.1
        contradicting_weight = len(hypothesis.contradicting_evidence) * -0.15

        new_confidence = base_confidence + supporting_weight + contradicting_weight
        return max(0.0, min(1.0, new_confidence))

    def _calculate_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level"""
        if confidence >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.6:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.4:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    async def refine_hypothesis(self, hypothesis: Hypothesis,
                               new_evidence: List[Evidence]) -> Hypothesis:
        """Refine hypothesis with new evidence"""
        # Add new evidence
        for evidence in new_evidence:
            if evidence.type == "supporting":
                hypothesis.supporting_evidence.append(evidence)
            elif evidence.type == "contradicting":
                hypothesis.contradicting_evidence.append(evidence)

        # Recalculate confidence
        hypothesis.confidence = self._recalculate_confidence(hypothesis)
        hypothesis.confidence_level = self._calculate_confidence_level(hypothesis.confidence)
        hypothesis.updated_at = datetime.now()

        return hypothesis
