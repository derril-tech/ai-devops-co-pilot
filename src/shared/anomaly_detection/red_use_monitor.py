"""
RED (Rate, Error, Duration) and USE (Utilization, Saturation, Errors) method implementations
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class REDMetrics:
    """RED method metrics"""
    rate: float  # Requests per second
    error_rate: float  # Error rate (0-1)
    duration_p50: float  # 50th percentile response time
    duration_p95: float  # 95th percentile response time
    duration_p99: float  # 99th percentile response time

    @property
    def health_score(self) -> float:
        """Calculate overall health score (0-1, higher is better)"""
        # Normalize each metric to 0-1 scale
        rate_score = min(1.0, self.rate / 100.0)  # Cap at 100 RPS
        error_score = max(0.0, 1.0 - self.error_rate * 5)  # Penalize errors heavily
        duration_score = max(0.0, 1.0 - (self.duration_p95 / 5000.0))  # Penalize slow responses

        return (rate_score + error_score + duration_score) / 3.0


@dataclass
class USEMetrics:
    """USE method metrics"""
    utilization: float  # Utilization percentage (0-100)
    saturation: float   # Saturation level (0-1)
    errors: int        # Error count

    @property
    def health_score(self) -> float:
        """Calculate overall health score (0-1, higher is better)"""
        # Utilization: optimal is 70-80%, penalize over/under utilization
        if self.utilization < 50:
            util_score = self.utilization / 50.0
        elif self.utilization <= 80:
            util_score = 1.0
        else:
            util_score = max(0.0, 2.0 - self.utilization / 40.0)

        # Saturation: lower is better
        saturation_score = max(0.0, 1.0 - self.saturation)

        # Errors: fewer is better
        error_score = max(0.0, 1.0 - min(1.0, self.errors / 10.0))

        return (util_score + saturation_score + error_score) / 3.0


@dataclass
class ServiceHealth:
    """Combined service health assessment"""
    service_name: str
    red_metrics: REDMetrics
    use_metrics: USEMetrics
    overall_status: HealthStatus
    timestamp: datetime
    alerts: List[str]

    @property
    def health_score(self) -> float:
        """Combined health score"""
        return (self.red_metrics.health_score + self.use_metrics.health_score) / 2.0


class REDMonitor:
    """RED method monitor for service health"""

    def __init__(self, window_minutes: int = 5):
        self.window_minutes = window_minutes

    def calculate_red_metrics(self, request_data: List[Dict[str, Any]],
                            time_window: timedelta = None) -> REDMetrics:
        """
        Calculate RED metrics from request data

        Args:
            request_data: List of request records with timestamps, durations, and status codes
            time_window: Time window to analyze (defaults to window_minutes)

        Returns:
            REDMetrics object
        """
        if not request_data:
            return REDMetrics(0, 0, 0, 0, 0)

        if time_window is None:
            time_window = timedelta(minutes=self.window_minutes)

        # Filter data by time window
        cutoff_time = datetime.now() - time_window
        filtered_data = [
            req for req in request_data
            if req['timestamp'] >= cutoff_time
        ]

        if not filtered_data:
            return REDMetrics(0, 0, 0, 0, 0)

        # Calculate rate (requests per second)
        total_requests = len(filtered_data)
        rate = total_requests / time_window.total_seconds()

        # Calculate error rate
        error_requests = sum(1 for req in filtered_data if req.get('status_code', 200) >= 400)
        error_rate = error_requests / total_requests if total_requests > 0 else 0

        # Calculate duration percentiles
        durations = [req.get('duration_ms', 0) for req in filtered_data if req.get('duration_ms') is not None]

        if durations:
            duration_p50 = np.percentile(durations, 50)
            duration_p95 = np.percentile(durations, 95)
            duration_p99 = np.percentile(durations, 99)
        else:
            duration_p50 = duration_p95 = duration_p99 = 0

        return REDMetrics(rate, error_rate, duration_p50, duration_p95, duration_p99)

    def assess_red_health(self, metrics: REDMetrics) -> Tuple[HealthStatus, List[str]]:
        """Assess health status based on RED metrics"""
        alerts = []

        # Rate assessment
        if metrics.rate > 1000:
            alerts.append(f"Very high request rate: {metrics.rate:.1f} RPS")
        elif metrics.rate > 500:
            alerts.append(f"High request rate: {metrics.rate:.1f} RPS")

        # Error rate assessment
        if metrics.error_rate > 0.1:  # 10%
            alerts.append(f"High error rate: {metrics.error_rate:.1%}")
        elif metrics.error_rate > 0.05:  # 5%
            alerts.append(f"Elevated error rate: {metrics.error_rate:.1%}")

        # Duration assessment
        if metrics.duration_p95 > 10000:  # 10 seconds
            alerts.append(f"Very slow responses: P95 = {metrics.duration_p95:.0f}ms")
        elif metrics.duration_p95 > 5000:  # 5 seconds
            alerts.append(f"Slow responses: P95 = {metrics.duration_p95:.0f}ms")

        # Determine overall status
        if metrics.error_rate > 0.1 or metrics.duration_p95 > 10000:
            status = HealthStatus.CRITICAL
        elif metrics.error_rate > 0.05 or metrics.duration_p95 > 5000 or metrics.rate > 1000:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY

        return status, alerts


class USEMonitor:
    """USE method monitor for resource utilization"""

    def __init__(self):
        pass

    def calculate_use_metrics(self, resource_data: Dict[str, Any]) -> USEMetrics:
        """
        Calculate USE metrics from resource data

        Args:
            resource_data: Dictionary containing utilization, saturation, and error data

        Returns:
            USEMetrics object
        """
        utilization = resource_data.get('utilization', 0)
        saturation = resource_data.get('saturation', 0)
        errors = resource_data.get('errors', 0)

        return USEMetrics(utilization, saturation, errors)

    def assess_use_health(self, metrics: USEMetrics) -> Tuple[HealthStatus, List[str]]:
        """Assess health status based on USE metrics"""
        alerts = []

        # Utilization assessment
        if metrics.utilization > 90:
            alerts.append(f"Critical utilization: {metrics.utilization:.1f}%")
        elif metrics.utilization > 80:
            alerts.append(f"High utilization: {metrics.utilization:.1f}%")

        # Saturation assessment
        if metrics.saturation > 0.9:
            alerts.append(f"Critical saturation: {metrics.saturation:.1%}")
        elif metrics.saturation > 0.7:
            alerts.append(f"High saturation: {metrics.saturation:.1%}")

        # Error assessment
        if metrics.errors > 100:
            alerts.append(f"High error count: {metrics.errors}")
        elif metrics.errors > 10:
            alerts.append(f"Elevated errors: {metrics.errors}")

        # Determine overall status
        if metrics.utilization > 90 or metrics.saturation > 0.9 or metrics.errors > 100:
            status = HealthStatus.CRITICAL
        elif metrics.utilization > 80 or metrics.saturation > 0.7 or metrics.errors > 10:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY

        return status, alerts


class ServiceHealthMonitor:
    """Combined RED/USE service health monitor"""

    def __init__(self, window_minutes: int = 5):
        self.red_monitor = REDMonitor(window_minutes)
        self.use_monitor = USEMonitor()

    def assess_service_health(self, service_name: str, request_data: List[Dict[str, Any]],
                            resource_data: Dict[str, Any]) -> ServiceHealth:
        """
        Assess overall service health using RED and USE methods

        Args:
            service_name: Name of the service
            request_data: Request telemetry data
            resource_data: Resource utilization data

        Returns:
            ServiceHealth assessment
        """
        # Calculate RED metrics
        red_metrics = self.red_monitor.calculate_red_metrics(request_data)
        red_status, red_alerts = self.red_monitor.assess_red_health(red_metrics)

        # Calculate USE metrics
        use_metrics = self.use_monitor.calculate_use_metrics(resource_data)
        use_status, use_alerts = self.use_monitor.assess_use_health(use_metrics)

        # Combine alerts
        all_alerts = red_alerts + use_alerts

        # Determine overall status (worst of RED and USE)
        if red_status == HealthStatus.CRITICAL or use_status == HealthStatus.CRITICAL:
            overall_status = HealthStatus.CRITICAL
        elif red_status == HealthStatus.WARNING or use_status == HealthStatus.WARNING:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY

        return ServiceHealth(
            service_name=service_name,
            red_metrics=red_metrics,
            use_metrics=use_metrics,
            overall_status=overall_status,
            timestamp=datetime.now(),
            alerts=all_alerts
        )

    def batch_assess_services(self, services_data: Dict[str, Dict[str, Any]]) -> List[ServiceHealth]:
        """
        Assess health for multiple services

        Args:
            services_data: Dictionary mapping service names to their data

        Returns:
            List of ServiceHealth assessments
        """
        assessments = []

        for service_name, data in services_data.items():
            request_data = data.get('requests', [])
            resource_data = data.get('resources', {})

            assessment = self.assess_service_health(service_name, request_data, resource_data)
            assessments.append(assessment)

        return assessments

    def get_health_summary(self, assessments: List[ServiceHealth]) -> Dict[str, Any]:
        """Generate summary of service health across all assessed services"""
        if not assessments:
            return {"total_services": 0, "healthy": 0, "warning": 0, "critical": 0}

        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.UNKNOWN: 0
        }

        total_alerts = 0
        avg_health_score = 0.0

        for assessment in assessments:
            status_counts[assessment.overall_status] += 1
            total_alerts += len(assessment.alerts)
            avg_health_score += assessment.health_score

        avg_health_score /= len(assessments)

        return {
            "total_services": len(assessments),
            "healthy": status_counts[HealthStatus.HEALTHY],
            "warning": status_counts[HealthStatus.WARNING],
            "critical": status_counts[HealthStatus.CRITICAL],
            "unknown": status_counts[HealthStatus.UNKNOWN],
            "total_alerts": total_alerts,
            "average_health_score": avg_health_score,
            "health_distribution": {
                status.value: count for status, count in status_counts.items()
            }
        }
