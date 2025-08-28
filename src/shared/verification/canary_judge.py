"""
Canary judge for evaluating remediation effectiveness
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
from ..metrics.slo_metrics import SLOMetrics


logger = logging.getLogger(__name__)


class CanaryDecision(Enum):
    """Decision from canary judge"""
    SUCCESS = "success"
    FAILURE = "failure"
    INCONCLUSIVE = "inconclusive"
    ROLLBACK = "rollback"


class MetricDirection(Enum):
    """Direction in which metric improvement is expected"""
    INCREASE = "increase"  # Higher is better (e.g., success rate)
    DECREASE = "decrease"  # Lower is better (e.g., error rate)
    STABLE = "stable"     # Should remain stable (e.g., throughput)


@dataclass
class CanaryMetric:
    """Metric definition for canary evaluation"""
    name: str
    query: str
    direction: MetricDirection
    threshold: float  # Percentage improvement/degradation threshold
    window_minutes: int = 30
    description: str = ""


@dataclass
class CanaryResult:
    """Result of canary evaluation"""
    decision: CanaryDecision
    confidence: float
    metrics_improved: List[str]
    metrics_degraded: List[str]
    metrics_stable: List[str]
    analysis: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CanaryExperiment:
    """Canary experiment configuration"""
    id: str
    name: str
    description: str
    incident_id: str
    fix_id: str
    start_time: datetime
    baseline_window: timedelta
    canary_window: timedelta
    metrics: List[CanaryMetric]
    status: str = "running"
    results: Optional[CanaryResult] = None


class CanaryJudge:
    """Intelligent canary judge for remediation effectiveness"""

    def __init__(self):
        self.experiments: Dict[str, CanaryExperiment] = {}
        self.baseline_metrics: Dict[str, Dict[str, List[float]]] = {}
        self.default_metrics = self._get_default_metrics()

    def _get_default_metrics(self) -> List[CanaryMetric]:
        """Get default metrics for canary evaluation"""
        return [
            CanaryMetric(
                name="error_rate",
                query="rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])",
                direction=MetricDirection.DECREASE,
                threshold=0.20,  # 20% improvement expected
                window_minutes=30,
                description="HTTP error rate percentage"
            ),
            CanaryMetric(
                name="latency_p95",
                query="histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                direction=MetricDirection.DECREASE,
                threshold=0.15,  # 15% improvement expected
                window_minutes=30,
                description="95th percentile request latency"
            ),
            CanaryMetric(
                name="success_rate",
                query="rate(http_requests_total{status!~\"5..\"}[5m]) / rate(http_requests_total[5m])",
                direction=MetricDirection.INCREASE,
                threshold=0.10,  # 10% improvement expected
                window_minutes=30,
                description="HTTP success rate percentage"
            ),
            CanaryMetric(
                name="throughput",
                query="rate(http_requests_total[5m])",
                direction=MetricDirection.STABLE,
                threshold=0.05,  # 5% deviation allowed
                window_minutes=30,
                description="Request throughput"
            ),
            CanaryMetric(
                name="cpu_usage",
                query="rate(process_cpu_user_seconds_total[5m])",
                direction=MetricDirection.STABLE,
                threshold=0.10,  # 10% deviation allowed
                window_minutes=30,
                description="CPU usage rate"
            ),
            CanaryMetric(
                name="memory_usage",
                query="process_resident_memory_bytes / process_virtual_memory_bytes",
                direction=MetricDirection.STABLE,
                threshold=0.15,  # 15% deviation allowed
                window_minutes=30,
                description="Memory usage ratio"
            )
        ]

    async def start_canary_experiment(self, incident_id: str, fix_id: str,
                                     experiment_name: str = None,
                                     custom_metrics: List[CanaryMetric] = None,
                                     baseline_window_hours: int = 24,
                                     canary_window_hours: int = 2) -> str:
        """
        Start a canary experiment to evaluate remediation effectiveness

        Args:
            incident_id: ID of the incident being remediated
            fix_id: ID of the applied fix
            experiment_name: Optional name for the experiment
            custom_metrics: Custom metrics to evaluate
            baseline_window_hours: Hours to look back for baseline
            canary_window_hours: Hours to run canary evaluation

        Returns:
            Experiment ID
        """
        experiment_id = f"canary_{incident_id}_{fix_id}_{int(datetime.now().timestamp())}"

        if experiment_name is None:
            experiment_name = f"Canary for fix {fix_id}"

        # Use custom metrics or defaults
        metrics = custom_metrics if custom_metrics else self.default_metrics

        experiment = CanaryExperiment(
            id=experiment_id,
            name=experiment_name,
            description=f"Evaluating effectiveness of fix {fix_id} for incident {incident_id}",
            incident_id=incident_id,
            fix_id=fix_id,
            start_time=datetime.now(),
            baseline_window=timedelta(hours=baseline_window_hours),
            canary_window=timedelta(hours=canary_window_hours),
            metrics=metrics
        )

        self.experiments[experiment_id] = experiment

        # Start baseline data collection
        await self._collect_baseline_data(experiment_id)

        logger.info(f"Started canary experiment {experiment_id} for fix {fix_id}")

        return experiment_id

    async def evaluate_canary(self, experiment_id: str) -> CanaryResult:
        """
        Evaluate the canary experiment and make a decision

        Args:
            experiment_id: ID of the experiment to evaluate

        Returns:
            CanaryResult with decision and analysis
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]

        # Collect current metrics
        current_metrics = await self._collect_current_metrics(experiment)

        # Compare with baseline
        comparison_results = self._compare_with_baseline(experiment_id, current_metrics)

        # Make decision
        decision, confidence = self._make_decision(comparison_results, experiment)

        # Categorize metrics
        improved, degraded, stable = self._categorize_metrics(comparison_results, experiment)

        # Create analysis
        analysis = {
            "baseline_window": experiment.baseline_window.total_seconds(),
            "canary_window": experiment.canary_window.total_seconds(),
            "comparison_results": comparison_results,
            "decision_factors": self._analyze_decision_factors(comparison_results),
            "risk_assessment": self._assess_risk(comparison_results)
        }

        result = CanaryResult(
            decision=decision,
            confidence=confidence,
            metrics_improved=improved,
            metrics_degraded=degraded,
            metrics_stable=stable,
            analysis=analysis
        )

        experiment.results = result
        experiment.status = "completed"

        logger.info(f"Canary experiment {experiment_id} completed with decision: {decision.value}")

        return result

    async def _collect_baseline_data(self, experiment_id: str) -> None:
        """Collect baseline metrics data"""
        experiment = self.experiments[experiment_id]
        baseline_start = experiment.start_time - experiment.baseline_window

        self.baseline_metrics[experiment_id] = {}

        for metric in experiment.metrics:
            try:
                # Query historical metrics data
                baseline_data = await self._query_metric_data(
                    metric.query,
                    baseline_start,
                    experiment.start_time
                )

                self.baseline_metrics[experiment_id][metric.name] = baseline_data

                logger.debug(f"Collected baseline data for {metric.name}: {len(baseline_data)} points")

            except Exception as e:
                logger.error(f"Failed to collect baseline for {metric.name}: {e}")
                self.baseline_metrics[experiment_id][metric.name] = []

    async def _collect_current_metrics(self, experiment: CanaryExperiment) -> Dict[str, float]:
        """Collect current metrics values"""
        current_metrics = {}

        for metric in experiment.metrics:
            try:
                # Query current metric value
                value = await self._query_current_metric(metric.query)
                current_metrics[metric.name] = value

            except Exception as e:
                logger.error(f"Failed to collect current value for {metric.name}: {e}")
                current_metrics[metric.name] = 0.0

        return current_metrics

    def _compare_with_baseline(self, experiment_id: str,
                              current_metrics: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Compare current metrics with baseline"""
        comparison_results = {}
        baseline_data = self.baseline_metrics.get(experiment_id, {})
        experiment = self.experiments[experiment_id]

        for metric in experiment.metrics:
            baseline_values = baseline_data.get(metric.name, [])
            current_value = current_metrics.get(metric.name, 0.0)

            if not baseline_values:
                comparison_results[metric.name] = {
                    "error": "No baseline data available",
                    "current": current_value,
                    "baseline_avg": None,
                    "change_percent": None
                }
                continue

            baseline_avg = statistics.mean(baseline_values)
            baseline_std = statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0

            if baseline_avg == 0:
                change_percent = float('inf') if current_value > 0 else 0.0
            else:
                change_percent = (current_value - baseline_avg) / baseline_avg

            # Statistical significance (simple z-score)
            z_score = abs(current_value - baseline_avg) / baseline_std if baseline_std > 0 else 0

            comparison_results[metric.name] = {
                "current": current_value,
                "baseline_avg": baseline_avg,
                "baseline_std": baseline_std,
                "change_percent": change_percent,
                "z_score": z_score,
                "direction": metric.direction.value,
                "threshold": metric.threshold,
                "significant": z_score > 2.0  # 95% confidence
            }

        return comparison_results

    def _make_decision(self, comparison_results: Dict[str, Dict[str, Any]],
                      experiment: CanaryExperiment) -> Tuple[CanaryDecision, float]:
        """Make canary decision based on comparison results"""
        improved_count = 0
        degraded_count = 0
        total_metrics = len(comparison_results)

        for metric_name, result in comparison_results.items():
            if "error" in result:
                continue

            change_percent = result["change_percent"]
            threshold = result["threshold"]
            direction = result["direction"]

            if abs(change_percent) < threshold:
                # Within acceptable range
                continue

            if direction == MetricDirection.INCREASE.value and change_percent > threshold:
                improved_count += 1
            elif direction == MetricDirection.DECREASE.value and change_percent < -threshold:
                improved_count += 1
            elif direction == MetricDirection.STABLE.value and abs(change_percent) > threshold:
                degraded_count += 1
            else:
                degraded_count += 1

        # Decision logic
        if degraded_count == 0 and improved_count > 0:
            decision = CanaryDecision.SUCCESS
            confidence = min(0.95, improved_count / total_metrics)
        elif degraded_count > improved_count:
            decision = CanaryDecision.FAILURE
            confidence = min(0.90, degraded_count / total_metrics)
        elif degraded_count > 0 and improved_count == 0:
            decision = CanaryDecision.ROLLBACK
            confidence = 0.85
        else:
            decision = CanaryDecision.INCONCLUSIVE
            confidence = 0.50

        return decision, confidence

    def _categorize_metrics(self, comparison_results: Dict[str, Dict[str, Any]],
                           experiment: CanaryExperiment) -> Tuple[List[str], List[str], List[str]]:
        """Categorize metrics as improved, degraded, or stable"""
        improved = []
        degraded = []
        stable = []

        for metric_name, result in comparison_results.items():
            if "error" in result:
                continue

            change_percent = result["change_percent"]
            threshold = result["threshold"]

            if abs(change_percent) < threshold:
                stable.append(metric_name)
            elif result["direction"] in [MetricDirection.INCREASE.value, MetricDirection.DECREASE.value]:
                # Check if change is in desired direction
                metric = next(m for m in experiment.metrics if m.name == metric_name)
                desired_direction = metric.direction == MetricDirection.INCREASE

                if (desired_direction and change_percent > 0) or (not desired_direction and change_percent < 0):
                    improved.append(metric_name)
                else:
                    degraded.append(metric_name)
            else:
                # Stable metric that changed too much
                degraded.append(metric_name)

        return improved, degraded, stable

    def _analyze_decision_factors(self, comparison_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze factors that influenced the decision"""
        significant_changes = []
        anomalies = []

        for metric_name, result in comparison_results.items():
            if "error" in result:
                continue

            if result["significant"]:
                significant_changes.append({
                    "metric": metric_name,
                    "change_percent": result["change_percent"],
                    "z_score": result["z_score"]
                })

            # Check for anomalies (3+ standard deviations)
            if result["z_score"] > 3.0:
                anomalies.append({
                    "metric": metric_name,
                    "z_score": result["z_score"],
                    "severity": "critical" if result["z_score"] > 5.0 else "high"
                })

        return {
            "significant_changes": significant_changes,
            "anomalies": anomalies,
            "total_metrics_analyzed": len(comparison_results)
        }

    def _assess_risk(self, comparison_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Assess risk level of the canary results"""
        risk_score = 0
        risk_factors = []

        for metric_name, result in comparison_results.items():
            if "error" in result:
                risk_score += 0.5
                risk_factors.append(f"Missing data for {metric_name}")
                continue

            change_percent = abs(result["change_percent"])
            z_score = result["z_score"]

            # Risk based on change magnitude
            if change_percent > 0.5:  # 50% change
                risk_score += 1.0
                risk_factors.append(f"Large change in {metric_name} ({change_percent:.1%})")
            elif z_score > 3.0:
                risk_score += 0.8
                risk_factors.append(f"Statistical anomaly in {metric_name} (z={z_score:.1f})")

        # Normalize risk score
        risk_score = min(risk_score / len(comparison_results), 1.0) if comparison_results else 0

        risk_level = "low" if risk_score < 0.3 else "medium" if risk_score < 0.7 else "high"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors
        }

    async def _query_metric_data(self, query: str, start_time: datetime,
                                end_time: datetime) -> List[float]:
        """Query metric data from monitoring system"""
        # This would integrate with Prometheus/ClickHouse
        # For now, return mock data
        import random
        return [random.uniform(0.1, 1.0) for _ in range(20)]

    async def _query_current_metric(self, query: str) -> float:
        """Query current metric value"""
        # This would integrate with Prometheus/ClickHouse
        # For now, return mock data
        import random
        return random.uniform(0.1, 1.0)

    def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a canary experiment"""
        if experiment_id not in self.experiments:
            return None

        experiment = self.experiments[experiment_id]
        elapsed = datetime.now() - experiment.start_time
        progress = min(elapsed.total_seconds() / experiment.canary_window.total_seconds(), 1.0)

        return {
            "id": experiment.id,
            "name": experiment.name,
            "status": experiment.status,
            "progress": progress,
            "start_time": experiment.start_time.isoformat(),
            "canary_window_hours": experiment.canary_window.total_seconds() / 3600,
            "results": experiment.results.__dict__ if experiment.results else None
        }

    def get_active_experiments(self) -> List[Dict[str, Any]]:
        """Get all active canary experiments"""
        active = [exp for exp in self.experiments.values() if exp.status == "running"]
        return [self.get_experiment_status(exp.id) for exp in active]

    def get_experiment_results(self, experiment_id: str) -> Optional[CanaryResult]:
        """Get results of a completed experiment"""
        if experiment_id not in self.experiments:
            return None

        return self.experiments[experiment_id].results
