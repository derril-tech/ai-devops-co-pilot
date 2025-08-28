"""
DORA (DevOps Research and Assessment) metrics implementation
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from collections import defaultdict

from ..database.config import get_postgres_session, get_clickhouse_session


logger = logging.getLogger(__name__)


class DORAMetric(Enum):
    """DORA metric types"""
    DEPLOYMENT_FREQUENCY = "deployment_frequency"
    LEAD_TIME_FOR_CHANGES = "lead_time_for_changes"
    CHANGE_FAILURE_RATE = "change_failure_rate"
    TIME_TO_RESTORE_SERVICE = "time_to_restore_service"


class DORARating(Enum):
    """DORA performance ratings"""
    ELITE = "elite"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class DeploymentEvent:
    """Deployment event data"""
    id: str
    service_name: str
    environment: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "success"  # success, failure, rollback
    commit_sha: str = ""
    triggered_by: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IncidentEvent:
    """Incident event data"""
    id: str
    title: str
    severity: str
    start_time: datetime
    end_time: Optional[datetime] = None
    affected_services: List[str] = field(default_factory=list)
    root_cause: str = ""
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DORAScorecard:
    """DORA metrics scorecard"""
    service_name: str
    time_period: str
    calculated_at: datetime

    # Core metrics
    deployment_frequency: float = 0.0  # deployments per day
    lead_time_for_changes: float = 0.0  # minutes from commit to deploy
    change_failure_rate: float = 0.0  # percentage of failed deployments
    time_to_restore_service: float = 0.0  # minutes to recover from incidents

    # Additional metrics
    deployment_success_rate: float = 0.0
    mean_time_between_failures: float = 0.0
    incident_count: int = 0
    deployment_count: int = 0

    # Rating
    overall_rating: DORARating = DORARating.LOW

    # Trends
    trends: Dict[str, Any] = field(default_factory=dict)

    # Benchmarks
    benchmarks: Dict[str, Any] = field(default_factory=dict)


class DORAMetrics:
    """DORA metrics calculator and analyzer"""

    def __init__(self):
        self.deployment_events: List[DeploymentEvent] = []
        self.incident_events: List[IncidentEvent] = []
        self.scorecards: Dict[str, DORAScorecard] = {}

    async def calculate_deployment_frequency(self, service_name: str,
                                          start_date: datetime,
                                          end_date: datetime) -> float:
        """
        Calculate deployment frequency (deployments per day)

        Args:
            service_name: Name of the service
            start_date: Start of calculation period
            end_date: End of calculation period

        Returns:
            Deployments per day
        """
        deployments = await self._get_deployments_for_service(
            service_name, start_date, end_date
        )

        days = (end_date - start_date).days
        if days == 0:
            days = 1

        return len(deployments) / days

    async def calculate_lead_time_for_changes(self, service_name: str,
                                            start_date: datetime,
                                            end_date: datetime) -> float:
        """
        Calculate lead time for changes (minutes from commit to deploy)

        Args:
            service_name: Name of the service
            start_date: Start of calculation period
            end_date: End of calculation period

        Returns:
            Average lead time in minutes
        """
        deployments = await self._get_deployments_for_service(
            service_name, start_date, end_date
        )

        if not deployments:
            return 0.0

        lead_times = []
        for deployment in deployments:
            # This would calculate actual lead time from commit to deployment
            # For now, use a mock calculation
            lead_time = await self._calculate_lead_time(deployment)
            lead_times.append(lead_time)

        return statistics.mean(lead_times) if lead_times else 0.0

    async def calculate_change_failure_rate(self, service_name: str,
                                          start_date: datetime,
                                          end_date: datetime) -> float:
        """
        Calculate change failure rate (percentage of failed deployments)

        Args:
            service_name: Name of the service
            start_date: Start of calculation period
            end_date: End of calculation period

        Returns:
            Failure rate as percentage (0.0 to 1.0)
        """
        deployments = await self._get_deployments_for_service(
            service_name, start_date, end_date
        )

        if not deployments:
            return 0.0

        failed_deployments = [d for d in deployments if d.status in ["failure", "rollback"]]
        return len(failed_deployments) / len(deployments)

    async def calculate_time_to_restore_service(self, service_name: str,
                                              start_date: datetime,
                                              end_date: datetime) -> float:
        """
        Calculate time to restore service (minutes to recover from incidents)

        Args:
            service_name: Name of the service
            start_date: Start of calculation period
            end_date: End of calculation period

        Returns:
            Average time to restore in minutes
        """
        incidents = await self._get_incidents_for_service(
            service_name, start_date, end_date
        )

        if not incidents:
            return 0.0

        restore_times = []
        for incident in incidents:
            if incident.end_time and incident.start_time:
                restore_time = (incident.end_time - incident.start_time).total_seconds() / 60
                restore_times.append(restore_time)

        return statistics.mean(restore_times) if restore_times else 0.0

    async def generate_scorecard(self, service_name: str,
                               time_period_days: int = 30) -> DORAScorecard:
        """
        Generate complete DORA scorecard for a service

        Args:
            service_name: Name of the service
            time_period_days: Number of days to analyze

        Returns:
            Complete DORA scorecard
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period_days)

        # Calculate core metrics
        deployment_freq = await self.calculate_deployment_frequency(
            service_name, start_date, end_date
        )

        lead_time = await self.calculate_lead_time_for_changes(
            service_name, start_date, end_date
        )

        failure_rate = await self.calculate_change_failure_rate(
            service_name, start_date, end_date
        )

        restore_time = await self.calculate_time_to_restore_service(
            service_name, start_date, end_date
        )

        # Get additional metrics
        deployments = await self._get_deployments_for_service(
            service_name, start_date, end_date
        )

        incidents = await self._get_incidents_for_service(
            service_name, start_date, end_date
        )

        # Calculate additional metrics
        deployment_success_rate = 1.0 - failure_rate
        mtbf = await self._calculate_mtbf(service_name, start_date, end_date)

        # Determine overall rating
        rating = self._calculate_overall_rating(
            deployment_freq, lead_time, failure_rate, restore_time
        )

        # Calculate trends
        trends = await self._calculate_trends(service_name, time_period_days)

        # Get benchmarks
        benchmarks = await self._get_benchmarks(service_name)

        scorecard = DORAScorecard(
            service_name=service_name,
            time_period=f"{time_period_days} days",
            calculated_at=datetime.now(),
            deployment_frequency=deployment_freq,
            lead_time_for_changes=lead_time,
            change_failure_rate=failure_rate,
            time_to_restore_service=restore_time,
            deployment_success_rate=deployment_success_rate,
            mean_time_between_failures=mtbf,
            incident_count=len(incidents),
            deployment_count=len(deployments),
            overall_rating=rating,
            trends=trends,
            benchmarks=benchmarks
        )

        # Cache the scorecard
        self.scorecards[f"{service_name}_{time_period_days}d"] = scorecard

        return scorecard

    def _calculate_overall_rating(self, deployment_freq: float, lead_time: float,
                                failure_rate: float, restore_time: float) -> DORARating:
        """Calculate overall DORA rating based on the four metrics"""

        # Elite criteria
        if (deployment_freq >= 1/7 and  # Multiple deploys per day (on-demand)
            lead_time <= 60 and         # Less than 1 hour
            failure_rate <= 0.15 and    # Less than 15%
            restore_time <= 60):        # Less than 1 hour
            return DORARating.ELITE

        # High criteria
        elif (deployment_freq >= 1/30 and  # Deploy daily
              lead_time <= 1440 and       # Less than 1 day
              failure_rate <= 0.30 and    # Less than 30%
              restore_time <= 1440):      # Less than 1 day
            return DORARating.HIGH

        # Medium criteria
        elif (deployment_freq >= 1/90 and  # Deploy weekly
              lead_time <= 10080 and      # Less than 1 week
              failure_rate <= 0.45 and    # Less than 45%
              restore_time <= 10080):     # Less than 1 week
            return DORARating.MEDIUM

        # Low criteria (everything else)
        else:
            return DORARating.LOW

    async def _calculate_trends(self, service_name: str,
                               time_period_days: int) -> Dict[str, Any]:
        """Calculate metric trends over time"""
        trends = {}

        # Calculate metrics for different periods
        periods = [7, 14, 30]  # 1 week, 2 weeks, 1 month

        for period in periods:
            if period <= time_period_days:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=period)

                metrics = await self._calculate_metrics_for_period(
                    service_name, start_date, end_date
                )

                trends[f"{period}d"] = metrics

        return trends

    async def _calculate_metrics_for_period(self, service_name: str,
                                          start_date: datetime,
                                          end_date: datetime) -> Dict[str, float]:
        """Calculate all metrics for a specific period"""
        deployment_freq = await self.calculate_deployment_frequency(
            service_name, start_date, end_date
        )

        lead_time = await self.calculate_lead_time_for_changes(
            service_name, start_date, end_date
        )

        failure_rate = await self.calculate_change_failure_rate(
            service_name, start_date, end_date
        )

        restore_time = await self.calculate_time_to_restore_service(
            service_name, start_date, end_date
        )

        return {
            "deployment_frequency": deployment_freq,
            "lead_time_for_changes": lead_time,
            "change_failure_rate": failure_rate,
            "time_to_restore_service": restore_time
        }

    async def _get_benchmarks(self, service_name: str) -> Dict[str, Any]:
        """Get industry benchmarks for comparison"""
        # This would typically query industry benchmark data
        # For now, return static benchmarks based on DORA research

        return {
            "elite_thresholds": {
                "deployment_frequency": 1/7,  # Multiple per day
                "lead_time_for_changes": 60,  # < 1 hour
                "change_failure_rate": 0.15,  # < 15%
                "time_to_restore_service": 60  # < 1 hour
            },
            "high_thresholds": {
                "deployment_frequency": 1/30,  # Daily
                "lead_time_for_changes": 1440,  # < 1 day
                "change_failure_rate": 0.30,    # < 30%
                "time_to_restore_service": 1440  # < 1 day
            },
            "industry_averages": {
                "deployment_frequency": 1/90,   # Weekly
                "lead_time_for_changes": 10080,  # < 1 week
                "change_failure_rate": 0.45,     # < 45%
                "time_to_restore_service": 10080  # < 1 week
            }
        }

    async def get_metrics_for_incident(self, incident_id: str,
                                     days_before: int = 7,
                                     days_after: int = 1) -> Dict[str, Any]:
        """
        Get DORA metrics specifically for an incident context

        Args:
            incident_id: ID of the incident
            days_before: Days before incident to analyze
            days_after: Days after incident to analyze

        Returns:
            DORA metrics for the incident period
        """
        # Get incident details
        incident = await self._get_incident_by_id(incident_id)
        if not incident:
            return {"error": f"Incident {incident_id} not found"}

        # Calculate metrics before incident
        before_start = incident.start_time - timedelta(days=days_before)
        before_end = incident.start_time

        before_metrics = await self._calculate_metrics_for_period(
            incident.affected_services[0] if incident.affected_services else "unknown",
            before_start, before_end
        )

        # Calculate metrics after incident
        after_start = incident.end_time or incident.start_time
        after_end = after_start + timedelta(days=days_after)

        after_metrics = await self._calculate_metrics_for_period(
            incident.affected_services[0] if incident.affected_services else "unknown",
            after_start, after_end
        )

        return {
            "incident_id": incident_id,
            "before_incident": before_metrics,
            "after_incident": after_metrics,
            "comparison": self._compare_metrics(before_metrics, after_metrics),
            "incident_impact": self._assess_incident_impact(incident, before_metrics, after_metrics)
        }

    def _compare_metrics(self, before: Dict[str, float],
                        after: Dict[str, float]) -> Dict[str, Any]:
        """Compare metrics before and after an event"""
        comparison = {}

        for metric_name in before.keys():
            before_value = before.get(metric_name, 0)
            after_value = after.get(metric_name, 0)

            if before_value != 0:
                change_percent = ((after_value - before_value) / before_value) * 100
            else:
                change_percent = 0 if after_value == 0 else float('inf')

            comparison[metric_name] = {
                "before": before_value,
                "after": after_value,
                "change_percent": change_percent,
                "improved": self._is_improved(metric_name, before_value, after_value)
            }

        return comparison

    def _is_improved(self, metric_name: str, before: float, after: float) -> bool:
        """Check if a metric value represents an improvement"""
        # For these metrics, lower values are better
        lower_better = ["lead_time_for_changes", "change_failure_rate", "time_to_restore_service"]

        if metric_name in lower_better:
            return after < before
        else:
            # Higher is better for deployment frequency
            return after > before

    def _assess_incident_impact(self, incident: IncidentEvent,
                              before_metrics: Dict[str, float],
                              after_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Assess the impact of an incident on DORA metrics"""
        impact = {
            "severity": incident.severity,
            "duration_minutes": (
                (incident.end_time - incident.start_time).total_seconds() / 60
                if incident.end_time else 0
            ),
            "metrics_impact": {},
            "recovery_status": "unknown"
        }

        # Assess impact on each metric
        for metric_name, before_value in before_metrics.items():
            after_value = after_metrics.get(metric_name, 0)
            change_percent = abs(((after_value - before_value) / before_value) * 100) if before_value != 0 else 0

            if change_percent > 50:  # 50% change
                impact_level = "severe"
            elif change_percent > 25:  # 25% change
                impact_level = "significant"
            elif change_percent > 10:  # 10% change
                impact_level = "moderate"
            else:
                impact_level = "minimal"

            impact["metrics_impact"][metric_name] = {
                "impact_level": impact_level,
                "change_percent": change_percent
            }

        # Determine recovery status
        if incident.end_time:
            impact["recovery_status"] = "recovered"
        else:
            impact["recovery_status"] = "ongoing"

        return impact

    async def _calculate_mtbf(self, service_name: str,
                            start_date: datetime,
                            end_date: datetime) -> float:
        """Calculate Mean Time Between Failures"""
        incidents = await self._get_incidents_for_service(
            service_name, start_date, end_date
        )

        if len(incidents) < 2:
            # If fewer than 2 incidents, use time since start of period
            period_days = (end_date - start_date).days
            return period_days * 24 * 60  # Convert to minutes

        # Calculate time between incidents
        incident_times = sorted([inc.start_time for inc in incidents])
        time_differences = []

        for i in range(1, len(incident_times)):
            diff_minutes = (incident_times[i] - incident_times[i-1]).total_seconds() / 60
            time_differences.append(diff_minutes)

        return statistics.mean(time_differences) if time_differences else 0.0

    # Data access methods (would be implemented with actual database queries)

    async def _get_deployments_for_service(self, service_name: str,
                                         start_date: datetime,
                                         end_date: datetime) -> List[DeploymentEvent]:
        """Get deployment events for a service within a time period"""
        # This would query the deployment database
        # For now, return mock data
        return [
            DeploymentEvent(
                id=f"deploy_{i}",
                service_name=service_name,
                environment="production",
                start_time=start_date + timedelta(hours=i*24),
                end_time=start_date + timedelta(hours=i*24, minutes=30),
                status="success" if i % 10 != 0 else "failure"
            )
            for i in range(10)
        ]

    async def _get_incidents_for_service(self, service_name: str,
                                       start_date: datetime,
                                       end_date: datetime) -> List[IncidentEvent]:
        """Get incident events for a service within a time period"""
        # This would query the incident database
        # For now, return mock data
        return [
            IncidentEvent(
                id=f"incident_{i}",
                title=f"Incident {i}",
                severity="high" if i % 3 == 0 else "medium",
                start_time=start_date + timedelta(hours=i*48),
                end_time=start_date + timedelta(hours=i*48, minutes=120),
                affected_services=[service_name]
            )
            for i in range(3)
        ]

    async def _get_incident_by_id(self, incident_id: str) -> Optional[IncidentEvent]:
        """Get incident details by ID"""
        # This would query the incident database
        # For now, return mock data
        return IncidentEvent(
            id=incident_id,
            title=f"Incident {incident_id}",
            severity="high",
            start_time=datetime.now() - timedelta(hours=2),
            end_time=datetime.now(),
            affected_services=["api-service"]
        )

    async def _calculate_lead_time(self, deployment: DeploymentEvent) -> float:
        """Calculate lead time for a deployment"""
        # This would calculate actual lead time from commit to deployment
        # For now, return mock data
        import random
        return random.uniform(30, 1440)  # 30 minutes to 24 hours

    def export_scorecard(self, scorecard: DORAScorecard, format: str = "json") -> str:
        """Export DORA scorecard in specified format"""
        data = {
            "service_name": scorecard.service_name,
            "time_period": scorecard.time_period,
            "calculated_at": scorecard.calculated_at.isoformat(),
            "metrics": {
                "deployment_frequency": scorecard.deployment_frequency,
                "lead_time_for_changes": scorecard.lead_time_for_changes,
                "change_failure_rate": scorecard.change_failure_rate,
                "time_to_restore_service": scorecard.time_to_restore_service
            },
            "additional_metrics": {
                "deployment_success_rate": scorecard.deployment_success_rate,
                "mean_time_between_failures": scorecard.mean_time_between_failures,
                "incident_count": scorecard.incident_count,
                "deployment_count": scorecard.deployment_count
            },
            "rating": scorecard.overall_rating.value,
            "trends": scorecard.trends,
            "benchmarks": scorecard.benchmarks
        }

        if format == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def get_service_comparison(self, service_names: List[str],
                                   time_period_days: int = 30) -> Dict[str, Any]:
        """Compare DORA metrics across multiple services"""
        service_scorecards = {}

        for service_name in service_names:
            scorecard = await self.generate_scorecard(service_name, time_period_days)
            service_scorecards[service_name] = scorecard

        # Calculate rankings
        rankings = self._calculate_rankings(service_scorecards)

        return {
            "services": {name: self.export_scorecard(scorecard, "json")
                        for name, scorecard in service_scorecards.items()},
            "rankings": rankings,
            "comparison_summary": self._generate_comparison_summary(service_scorecards)
        }

    def _calculate_rankings(self, scorecards: Dict[str, DORAScorecard]) -> Dict[str, Any]:
        """Calculate rankings across services"""
        rankings = {
            "overall": {},
            "deployment_frequency": {},
            "lead_time_for_changes": {},
            "change_failure_rate": {},
            "time_to_restore_service": {}
        }

        # Sort services by each metric
        for metric in ["deployment_frequency", "lead_time_for_changes", "change_failure_rate", "time_to_restore_service"]:
            sorted_services = sorted(
                scorecards.keys(),
                key=lambda s: getattr(scorecards[s], metric),
                reverse=(metric == "deployment_frequency")  # Higher is better for deployment freq
            )

            for rank, service in enumerate(sorted_services, 1):
                rankings[metric][service] = rank

        # Overall ranking based on DORA rating
        rating_order = {DORARating.ELITE: 4, DORARating.HIGH: 3, DORARating.MEDIUM: 2, DORARating.LOW: 1}
        sorted_overall = sorted(
            scorecards.keys(),
            key=lambda s: rating_order[scorecards[s].overall_rating],
            reverse=True
        )

        for rank, service in enumerate(sorted_overall, 1):
            rankings["overall"][service] = rank

        return rankings

    def _generate_comparison_summary(self, scorecards: Dict[str, DORAScorecard]) -> Dict[str, Any]:
        """Generate comparison summary across services"""
        summary = {
            "total_services": len(scorecards),
            "rating_distribution": {},
            "best_performers": {},
            "improvement_opportunities": {}
        }

        # Rating distribution
        for scorecard in scorecards.values():
            rating = scorecard.overall_rating.value
            summary["rating_distribution"][rating] = summary["rating_distribution"].get(rating, 0) + 1

        # Best performers for each metric
        metrics = ["deployment_frequency", "lead_time_for_changes", "change_failure_rate", "time_to_restore_service"]
        for metric in metrics:
            if metric == "deployment_frequency":
                # Higher is better
                best_service = max(scorecards.keys(), key=lambda s: getattr(scorecards[s], metric))
            else:
                # Lower is better
                best_service = min(scorecards.keys(), key=lambda s: getattr(scorecards[s], metric))

            summary["best_performers"][metric] = {
                "service": best_service,
                "value": getattr(scorecards[best_service], metric)
            }

        return summary
