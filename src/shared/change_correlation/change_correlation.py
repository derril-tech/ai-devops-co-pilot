"""
Change correlation engine for linking deployments and feature flags to incidents
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class Deployment:
    """Deployment event"""
    id: str
    service: str
    version: str
    timestamp: datetime
    duration: Optional[timedelta]
    success: bool
    metadata: Dict[str, Any]


@dataclass
class FeatureFlag:
    """Feature flag change"""
    id: str
    name: str
    action: str  # 'enabled', 'disabled', 'modified'
    timestamp: datetime
    service: str
    value: Any
    metadata: Dict[str, Any]


@dataclass
class Incident:
    """Incident data"""
    id: str
    title: str
    timestamp: datetime
    service: str
    severity: str
    resolved_at: Optional[datetime]
    metadata: Dict[str, Any]


@dataclass
class CorrelationResult:
    """Result of change correlation analysis"""
    incident_id: str
    correlated_changes: List[Dict[str, Any]]
    impact_scores: Dict[str, float]
    confidence: float
    time_window: timedelta
    analysis_timestamp: datetime


class ChangeCorrelationEngine:
    """Engine for correlating changes with incidents"""

    def __init__(self, max_time_window: timedelta = timedelta(hours=24)):
        """
        Initialize change correlation engine

        Args:
            max_time_window: Maximum time window to consider for correlations
        """
        self.max_time_window = max_time_window

        # Historical correlation data for learning
        self.correlation_history: List[CorrelationResult] = []

    def correlate_incident(self, incident: Incident, deployments: List[Deployment],
                          feature_flags: List[FeatureFlag]) -> CorrelationResult:
        """
        Correlate an incident with recent changes

        Args:
            incident: The incident to analyze
            deployments: Recent deployments
            feature_flags: Recent feature flag changes

        Returns:
            CorrelationResult with correlated changes and impact scores
        """
        correlated_changes = []
        impact_scores = {}

        # Filter changes within time window
        time_window_start = incident.timestamp - self.max_time_window
        relevant_deployments = [
            d for d in deployments
            if time_window_start <= d.timestamp <= incident.timestamp
        ]
        relevant_flags = [
            f for f in feature_flags
            if time_window_start <= f.timestamp <= incident.timestamp
        ]

        # Correlate deployments
        for deployment in relevant_deployments:
            correlation = self._correlate_deployment(incident, deployment)
            if correlation:
                correlated_changes.append(correlation)
                impact_scores[deployment.id] = correlation['impact_score']

        # Correlate feature flags
        for flag in relevant_flags:
            correlation = self._correlate_feature_flag(incident, flag)
            if correlation:
                correlated_changes.append(correlation)
                impact_scores[flag.id] = correlation['impact_score']

        # Calculate overall confidence
        confidence = self._calculate_confidence(correlated_changes, incident)

        result = CorrelationResult(
            incident_id=incident.id,
            correlated_changes=correlated_changes,
            impact_scores=impact_scores,
            confidence=confidence,
            time_window=self.max_time_window,
            analysis_timestamp=datetime.now()
        )

        # Store for learning
        self.correlation_history.append(result)

        return result

    def _correlate_deployment(self, incident: Incident, deployment: Deployment) -> Optional[Dict[str, Any]]:
        """Correlate a deployment with an incident"""
        time_diff = incident.timestamp - deployment.timestamp

        # Only consider deployments within reasonable time window
        if time_diff > self.max_time_window or time_diff < timedelta(minutes=1):
            return None

        # Calculate impact score based on multiple factors
        impact_score = self._calculate_deployment_impact(incident, deployment, time_diff)

        # Only include if impact score is significant
        if impact_score < 0.1:
            return None

        return {
            'type': 'deployment',
            'change_id': deployment.id,
            'service': deployment.service,
            'timestamp': deployment.timestamp,
            'time_diff_minutes': time_diff.total_seconds() / 60,
            'impact_score': impact_score,
            'details': {
                'version': deployment.version,
                'success': deployment.success,
                'duration': deployment.duration.total_seconds() if deployment.duration else None
            }
        }

    def _correlate_feature_flag(self, incident: Incident, flag: FeatureFlag) -> Optional[Dict[str, Any]]:
        """Correlate a feature flag change with an incident"""
        time_diff = incident.timestamp - flag.timestamp

        # Only consider flag changes within reasonable time window
        if time_diff > self.max_time_window or time_diff < timedelta(minutes=1):
            return None

        # Calculate impact score
        impact_score = self._calculate_flag_impact(incident, flag, time_diff)

        # Only include if impact score is significant
        if impact_score < 0.1:
            return None

        return {
            'type': 'feature_flag',
            'change_id': flag.id,
            'service': flag.service,
            'timestamp': flag.timestamp,
            'time_diff_minutes': time_diff.total_seconds() / 60,
            'impact_score': impact_score,
            'details': {
                'flag_name': flag.name,
                'action': flag.action,
                'value': flag.value
            }
        }

    def _calculate_deployment_impact(self, incident: Incident, deployment: Deployment,
                                   time_diff: timedelta) -> float:
        """Calculate impact score for a deployment"""
        score = 0.0

        # Time proximity factor (exponential decay)
        hours_diff = time_diff.total_seconds() / 3600
        time_factor = np.exp(-hours_diff / 2)  # Half-life of 2 hours

        # Service match factor
        service_match = 1.0 if incident.service == deployment.service else 0.5

        # Deployment success factor
        success_factor = 0.3 if not deployment.success else 1.0

        # Severity factor
        severity_weights = {
            'low': 0.5,
            'medium': 0.7,
            'high': 0.9,
            'critical': 1.0
        }
        severity_factor = severity_weights.get(incident.severity, 0.5)

        # Historical correlation factor
        historical_factor = self._get_historical_correlation_factor(deployment.service)

        score = (time_factor * 0.4 +
                service_match * 0.3 +
                success_factor * 0.2 +
                severity_factor * 0.1) * historical_factor

        return min(score, 1.0)

    def _calculate_flag_impact(self, incident: Incident, flag: FeatureFlag,
                             time_diff: timedelta) -> float:
        """Calculate impact score for a feature flag change"""
        score = 0.0

        # Time proximity factor
        hours_diff = time_diff.total_seconds() / 3600
        time_factor = np.exp(-hours_diff / 1)  # Half-life of 1 hour (flags have quicker impact)

        # Service match factor
        service_match = 1.0 if incident.service == flag.service else 0.3

        # Action type factor (enabling flags are riskier than disabling)
        action_weights = {
            'enabled': 0.9,
            'modified': 0.7,
            'disabled': 0.4
        }
        action_factor = action_weights.get(flag.action, 0.5)

        # Flag type factor (some flags are more risky)
        flag_name = flag.name.lower()
        if any(keyword in flag_name for keyword in ['experiment', 'beta', 'test']):
            risk_factor = 0.8
        else:
            risk_factor = 0.6

        score = (time_factor * 0.4 +
                service_match * 0.3 +
                action_factor * 0.2 +
                risk_factor * 0.1)

        return min(score, 1.0)

    def _calculate_confidence(self, correlated_changes: List[Dict[str, Any]], incident: Incident) -> float:
        """Calculate overall confidence in the correlation analysis"""
        if not correlated_changes:
            return 0.0

        # Base confidence on number and quality of correlations
        num_changes = len(correlated_changes)
        avg_impact = np.mean([change['impact_score'] for change in correlated_changes])

        # Confidence increases with number of correlated changes
        count_factor = min(num_changes / 3, 1.0)  # Cap at 3 changes

        # High impact changes increase confidence
        impact_factor = avg_impact

        # Severity increases confidence in correlations
        severity_weights = {'low': 0.7, 'medium': 0.8, 'high': 0.9, 'critical': 1.0}
        severity_factor = severity_weights.get(incident.severity, 0.8)

        confidence = (count_factor * 0.4 + impact_factor * 0.4 + severity_factor * 0.2)

        return min(confidence, 1.0)

    def _get_historical_correlation_factor(self, service: str) -> float:
        """Get historical correlation factor for a service"""
        if not self.correlation_history:
            return 1.0

        # Look at recent correlations for this service
        recent_correlations = [
            corr for corr in self.correlation_history[-50:]  # Last 50 correlations
            if any(change.get('service') == service for change in corr.correlated_changes)
        ]

        if not recent_correlations:
            return 1.0

        # Calculate average confidence for this service
        avg_confidence = np.mean([corr.confidence for corr in recent_correlations])

        # Adjust factor based on historical success
        if avg_confidence > 0.7:
            return 1.2  # Increase likelihood
        elif avg_confidence < 0.3:
            return 0.8  # Decrease likelihood

        return 1.0

    def batch_correlate_incidents(self, incidents: List[Incident],
                                deployments: List[Deployment],
                                feature_flags: List[FeatureFlag]) -> List[CorrelationResult]:
        """Correlate multiple incidents with changes"""
        results = []

        for incident in incidents:
            result = self.correlate_incident(incident, deployments, feature_flags)
            results.append(result)

        return results

    def get_correlation_summary(self, results: List[CorrelationResult]) -> Dict[str, Any]:
        """Generate summary of correlation analysis"""
        if not results:
            return {"total_incidents": 0}

        total_incidents = len(results)
        correlated_incidents = sum(1 for r in results if r.correlated_changes)
        avg_confidence = np.mean([r.confidence for r in results])

        # Count change types
        deployment_count = 0
        flag_count = 0

        for result in results:
            for change in result.correlated_changes:
                if change['type'] == 'deployment':
                    deployment_count += 1
                elif change['type'] == 'feature_flag':
                    flag_count += 1

        return {
            "total_incidents": total_incidents,
            "correlated_incidents": correlated_incidents,
            "correlation_rate": correlated_incidents / total_incidents if total_incidents > 0 else 0,
            "average_confidence": avg_confidence,
            "total_changes": deployment_count + flag_count,
            "deployment_changes": deployment_count,
            "feature_flag_changes": flag_count
        }

    def get_service_risk_profile(self, service: str, days: int = 30) -> Dict[str, Any]:
        """Get risk profile for a service based on historical correlations"""
        cutoff_date = datetime.now() - timedelta(days=days)

        relevant_results = [
            result for result in self.correlation_history
            if result.analysis_timestamp >= cutoff_date and
            any(change.get('service') == service for change in result.correlated_changes)
        ]

        if not relevant_results:
            return {"service": service, "risk_level": "unknown", "correlation_count": 0}

        # Analyze correlation patterns
        total_incidents = len(relevant_results)
        high_confidence = sum(1 for r in relevant_results if r.confidence > 0.8)
        avg_changes_per_incident = np.mean([
            len(r.correlated_changes) for r in relevant_results
        ])

        # Determine risk level
        if high_confidence / total_incidents > 0.7:
            risk_level = "high"
        elif high_confidence / total_incidents > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "service": service,
            "risk_level": risk_level,
            "correlation_count": total_incidents,
            "high_confidence_rate": high_confidence / total_incidents if total_incidents > 0 else 0,
            "avg_changes_per_incident": avg_changes_per_incident
        }
