"""
SLO (Service Level Objective) metrics and burn rate calculations
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


class SLOStatus(Enum):
    """SLO status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    BREACHED = "breached"
    UNKNOWN = "unknown"


class SLOType(Enum):
    """Types of SLOs"""
    AVAILABILITY = "availability"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CUSTOM = "custom"


@dataclass
class SLODefinition:
    """SLO definition"""
    id: str
    name: str
    description: str
    service_name: str
    slo_type: SLOType
    target_value: float  # Target percentage (e.g., 99.9 for 99.9% availability)
    window_days: int = 30  # Rolling window in days
    query: str = ""  # PromQL or similar query
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SLOMeasurement:
    """SLO measurement data point"""
    slo_id: str
    timestamp: datetime
    actual_value: float
    target_value: float
    compliance_percentage: float
    error_budget_remaining: float  # Percentage remaining


@dataclass
class BurnRateData:
    """Burn rate calculation data"""
    slo_id: str
    time_window: str  # e.g., "1h", "6h", "1d"
    burn_rate: float  # How fast error budget is being consumed
    projected_exhaustion: Optional[datetime] = None
    severity: str = "low"  # low, medium, high, critical
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SLODashboard:
    """SLO dashboard data"""
    service_name: str
    time_period: str
    generated_at: datetime

    # SLO status summary
    slo_status: Dict[str, SLOStatus] = field(default_factory=dict)
    overall_health: SLOStatus = SLOStatus.UNKNOWN

    # Burn rate data
    burn_rates: Dict[str, List[BurnRateData]] = field(default_factory=dict)

    # Error budget status
    error_budgets: Dict[str, float] = field(default_factory=dict)

    # Historical data
    historical_data: Dict[str, List[SLOMeasurement]] = field(default_factory=dict)

    # Alerts and recommendations
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class SLOMetrics:
    """SLO metrics calculator and burn rate analyzer"""

    def __init__(self):
        self.slo_definitions: Dict[str, SLODefinition] = {}
        self.measurements: Dict[str, List[SLOMeasurement]] = {}
        self.burn_rate_cache: Dict[str, List[BurnRateData]] = {}

        # Default SLO thresholds
        self.burn_rate_thresholds = {
            "low": 1.0,      # Normal consumption
            "medium": 2.0,   # 2x normal
            "high": 5.0,     # 5x normal
            "critical": 10.0  # 10x normal
        }

    async def define_slo(self, name: str, description: str, service_name: str,
                        slo_type: SLOType, target_value: float,
                        window_days: int = 30, query: str = "") -> SLODefinition:
        """
        Define a new SLO

        Args:
            name: SLO name
            description: SLO description
            service_name: Service this SLO applies to
            slo_type: Type of SLO
            target_value: Target value (e.g., 99.9 for 99.9%)
            window_days: Rolling window in days
            query: Query to calculate the metric

        Returns:
            SLODefinition object
        """
        slo_id = f"slo_{service_name}_{slo_type.value}_{int(datetime.now().timestamp())}"

        slo = SLODefinition(
            id=slo_id,
            name=name,
            description=description,
            service_name=service_name,
            slo_type=slo_type,
            target_value=target_value,
            window_days=window_days,
            query=query
        )

        self.slo_definitions[slo_id] = slo
        self.measurements[slo_id] = []

        logger.info(f"Defined SLO {slo_id} for {service_name}")

        return slo

    async def record_measurement(self, slo_id: str, actual_value: float,
                               timestamp: Optional[datetime] = None) -> SLOMeasurement:
        """
        Record an SLO measurement

        Args:
            slo_id: ID of the SLO
            actual_value: Actual measured value
            timestamp: Timestamp of measurement

        Returns:
            SLOMeasurement object
        """
        if slo_id not in self.slo_definitions:
            raise ValueError(f"SLO {slo_id} not found")

        slo = self.slo_definitions[slo_id]

        if timestamp is None:
            timestamp = datetime.now()

        # Calculate compliance percentage
        compliance_percentage = self._calculate_compliance(actual_value, slo.target_value, slo.slo_type)

        # Calculate error budget remaining
        error_budget_remaining = self._calculate_error_budget_remaining(slo_id, compliance_percentage)

        measurement = SLOMeasurement(
            slo_id=slo_id,
            timestamp=timestamp,
            actual_value=actual_value,
            target_value=slo.target_value,
            compliance_percentage=compliance_percentage,
            error_budget_remaining=error_budget_remaining
        )

        self.measurements[slo_id].append(measurement)

        # Clean old measurements (keep last 90 days)
        cutoff_date = datetime.now() - timedelta(days=90)
        self.measurements[slo_id] = [
            m for m in self.measurements[slo_id]
            if m.timestamp >= cutoff_date
        ]

        return measurement

    def _calculate_compliance(self, actual_value: float, target_value: float,
                            slo_type: SLOType) -> float:
        """Calculate compliance percentage"""
        if slo_type in [SLOType.AVAILABILITY, SLOType.ERROR_RATE]:
            # For these metrics, actual_value is already a percentage (0-100)
            # target_value is also a percentage
            return min(100.0, max(0.0, actual_value))

        elif slo_type == SLOType.LATENCY:
            # For latency, we want actual_value <= target_value for compliance
            # Convert to percentage: if actual <= target, 100%, else 0%
            # In practice, this would be more nuanced with percentile calculations
            return 100.0 if actual_value <= target_value else 0.0

        elif slo_type == SLOType.THROUGHPUT:
            # For throughput, we want actual_value >= target_value
            return 100.0 if actual_value >= target_value else 0.0

        else:
            # Default calculation
            return min(100.0, max(0.0, actual_value))

    def _calculate_error_budget_remaining(self, slo_id: str, compliance_percentage: float) -> float:
        """Calculate remaining error budget percentage"""
        slo = self.slo_definitions[slo_id]

        # Error budget is 100% - target%
        # e.g., for 99.9% SLO, error budget is 0.1%
        error_budget_total = 100.0 - slo.target_value

        # Calculate consumed error budget based on current compliance
        # This is a simplified calculation
        target_compliance = slo.target_value

        if compliance_percentage >= target_compliance:
            # Meeting or exceeding target
            return 100.0  # Full budget remaining
        else:
            # Below target, calculate consumed budget
            deficit = target_compliance - compliance_percentage
            consumed_percentage = (deficit / error_budget_total) * 100
            return max(0.0, 100.0 - consumed_percentage)

    async def calculate_burn_rate(self, slo_id: str, time_window_hours: int = 1) -> BurnRateData:
        """
        Calculate burn rate for an SLO

        Args:
            slo_id: ID of the SLO
            time_window_hours: Time window for burn rate calculation

        Returns:
            BurnRateData object
        """
        if slo_id not in self.slo_definitions:
            raise ValueError(f"SLO {slo_id} not found")

        slo = self.slo_definitions[slo_id]
        measurements = self.measurements.get(slo_id, [])

        if not measurements:
            return BurnRateData(
                slo_id=slo_id,
                time_window=f"{time_window_hours}h",
                burn_rate=0.0,
                severity="low"
            )

        # Get measurements in the time window
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        window_measurements = [m for m in measurements if m.timestamp >= cutoff_time]

        if not window_measurements:
            return BurnRateData(
                slo_id=slo_id,
                time_window=f"{time_window_hours}h",
                burn_rate=0.0,
                severity="low"
            )

        # Calculate burn rate
        # Burn rate is how fast error budget is being consumed relative to time
        error_budget_consumed = []

        for measurement in window_measurements:
            budget_remaining = measurement.error_budget_remaining
            budget_consumed = 100.0 - budget_remaining
            error_budget_consumed.append(budget_consumed)

        if error_budget_consumed:
            avg_consumption = statistics.mean(error_budget_consumed)
            # Burn rate = consumption per hour
            burn_rate = avg_consumption / time_window_hours
        else:
            burn_rate = 0.0

        # Determine severity
        severity = "low"
        if burn_rate >= self.burn_rate_thresholds["critical"]:
            severity = "critical"
        elif burn_rate >= self.burn_rate_thresholds["high"]:
            severity = "high"
        elif burn_rate >= self.burn_rate_thresholds["medium"]:
            severity = "medium"

        # Calculate projected exhaustion
        projected_exhaustion = None
        if burn_rate > 0:
            hours_to_exhaustion = (100.0 - window_measurements[-1].error_budget_remaining) / burn_rate
            projected_exhaustion = datetime.now() + timedelta(hours=hours_to_exhaustion)

        burn_data = BurnRateData(
            slo_id=slo_id,
            time_window=f"{time_window_hours}h",
            burn_rate=burn_rate,
            projected_exhaustion=projected_exhaustion,
            severity=severity
        )

        # Cache burn rate data
        if slo_id not in self.burn_rate_cache:
            self.burn_rate_cache[slo_id] = []
        self.burn_rate_cache[slo_id].append(burn_data)

        # Keep only recent burn rate data (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self.burn_rate_cache[slo_id] = [
            b for b in self.burn_rate_cache[slo_id]
            if b.timestamp >= cutoff
        ]

        return burn_data

    async def get_slo_status(self, slo_id: str) -> SLOStatus:
        """Get current status of an SLO"""
        if slo_id not in self.slo_definitions:
            return SLOStatus.UNKNOWN

        measurements = self.measurements.get(slo_id, [])
        if not measurements:
            return SLOStatus.UNKNOWN

        # Get latest measurement
        latest = max(measurements, key=lambda m: m.timestamp)

        # Check error budget
        if latest.error_budget_remaining <= 0:
            return SLOStatus.BREACHED
        elif latest.error_budget_remaining <= 10:  # Less than 10% remaining
            return SLOStatus.WARNING
        else:
            return SLOStatus.HEALTHY

    async def generate_slo_dashboard(self, service_name: str,
                                   time_period_days: int = 7) -> SLODashboard:
        """
        Generate SLO dashboard for a service

        Args:
            service_name: Name of the service
            time_period_days: Time period for dashboard data

        Returns:
            SLODashboard object
        """
        # Get SLOs for this service
        service_slos = [
            slo for slo in self.slo_definitions.values()
            if slo.service_name == service_name
        ]

        dashboard = SLODashboard(
            service_name=service_name,
            time_period=f"{time_period_days} days",
            generated_at=datetime.now()
        )

        # Calculate status for each SLO
        for slo in service_slos:
            status = await self.get_slo_status(slo.id)
            dashboard.slo_status[slo.id] = status

            # Get burn rates for different time windows
            burn_rates = []
            for window_hours in [1, 6, 24]:
                burn_rate = await self.calculate_burn_rate(slo.id, window_hours)
                burn_rates.append(burn_rate)

            dashboard.burn_rates[slo.id] = burn_rates

            # Get latest error budget
            measurements = self.measurements.get(slo.id, [])
            if measurements:
                latest = max(measurements, key=lambda m: m.timestamp)
                dashboard.error_budgets[slo.id] = latest.error_budget_remaining

            # Get historical data
            cutoff_date = datetime.now() - timedelta(days=time_period_days)
            historical = [m for m in measurements if m.timestamp >= cutoff_date]
            dashboard.historical_data[slo.id] = historical

        # Determine overall health
        dashboard.overall_health = self._calculate_overall_health(dashboard.slo_status)

        # Generate alerts and recommendations
        dashboard.alerts = await self._generate_alerts(service_name, dashboard)
        dashboard.recommendations = await self._generate_recommendations(service_name, dashboard)

        return dashboard

    def _calculate_overall_health(self, slo_status: Dict[str, SLOStatus]) -> SLOStatus:
        """Calculate overall health across all SLOs"""
        if not slo_status:
            return SLOStatus.UNKNOWN

        statuses = list(slo_status.values())

        if SLOStatus.BREACHED in statuses:
            return SLOStatus.BREACHED
        elif SLOStatus.WARNING in statuses:
            return SLOStatus.WARNING
        elif SLOStatus.HEALTHY in statuses:
            return SLOStatus.HEALTHY
        else:
            return SLOStatus.UNKNOWN

    async def _generate_alerts(self, service_name: str,
                             dashboard: SLODashboard) -> List[Dict[str, Any]]:
        """Generate alerts based on dashboard data"""
        alerts = []

        for slo_id, status in dashboard.slo_status.items():
            slo = self.slo_definitions.get(slo_id)
            if not slo:
                continue

            # Check for breached SLOs
            if status == SLOStatus.BREACHED:
                alerts.append({
                    "type": "slo_breach",
                    "severity": "critical",
                    "slo_id": slo_id,
                    "slo_name": slo.name,
                    "message": f"SLO '{slo.name}' has been breached",
                    "details": {
                        "target": slo.target_value,
                        "current_budget": dashboard.error_budgets.get(slo_id, 0)
                    }
                })

            # Check for high burn rates
            burn_rates = dashboard.burn_rates.get(slo_id, [])
            for burn_rate in burn_rates:
                if burn_rate.severity in ["high", "critical"]:
                    alerts.append({
                        "type": "high_burn_rate",
                        "severity": burn_rate.severity,
                        "slo_id": slo_id,
                        "slo_name": slo.name,
                        "message": f"High error budget burn rate detected for '{slo.name}'",
                        "details": {
                            "burn_rate": burn_rate.burn_rate,
                            "time_window": burn_rate.time_window,
                            "projected_exhaustion": burn_rate.projected_exhaustion.isoformat() if burn_rate.projected_exhaustion else None
                        }
                    })

        return alerts

    async def _generate_recommendations(self, service_name: str,
                                     dashboard: SLODashboard) -> List[str]:
        """Generate recommendations based on dashboard data"""
        recommendations = []

        # Overall health recommendations
        if dashboard.overall_health == SLOStatus.BREACHED:
            recommendations.append("Immediate action required: One or more SLOs are breached")
            recommendations.append("Consider implementing emergency measures to restore service levels")

        elif dashboard.overall_health == SLOStatus.WARNING:
            recommendations.append("Monitor closely: SLO error budgets are running low")
            recommendations.append("Prepare contingency plans for potential SLO breaches")

        # Burn rate recommendations
        high_burn_slos = []
        for slo_id, burn_rates in dashboard.burn_rates.items():
            for burn_rate in burn_rates:
                if burn_rate.severity == "high":
                    slo = self.slo_definitions.get(slo_id)
                    if slo:
                        high_burn_slos.append(slo.name)

        if high_burn_slos:
            recommendations.append(f"Investigate high burn rates for: {', '.join(high_burn_slos)}")
            recommendations.append("Consider reducing deployment frequency or improving change quality")

        # Data quality recommendations
        for slo_id, historical in dashboard.historical_data.items():
            if len(historical) < 10:  # Less than 10 data points
                slo = self.slo_definitions.get(slo_id)
                if slo:
                    recommendations.append(f"Increase monitoring data collection for SLO '{slo.name}'")

        return recommendations

    async def get_slo_trends(self, slo_id: str, days: int = 30) -> Dict[str, Any]:
        """Get SLO trends over time"""
        if slo_id not in self.measurements:
            return {"error": f"No measurements found for SLO {slo_id}"}

        measurements = self.measurements[slo_id]
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_measurements = [m for m in measurements if m.timestamp >= cutoff_date]

        if not recent_measurements:
            return {"error": "No recent measurements found"}

        # Calculate trends
        compliance_values = [m.compliance_percentage for m in recent_measurements]
        budget_values = [m.error_budget_remaining for m in recent_measurements]

        compliance_trend = self._calculate_trend(compliance_values)
        budget_trend = self._calculate_trend(budget_values)

        return {
            "slo_id": slo_id,
            "time_period_days": days,
            "compliance_trend": compliance_trend,
            "budget_trend": budget_trend,
            "data_points": len(recent_measurements),
            "latest_compliance": compliance_values[-1] if compliance_values else 0,
            "latest_budget": budget_values[-1] if budget_values else 0
        }

    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend information for a series of values"""
        if len(values) < 2:
            return {"direction": "stable", "slope": 0, "confidence": 0}

        # Simple linear regression
        n = len(values)
        x = list(range(n))
        y = values

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)

        # Determine direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "improving"
        else:
            direction = "degrading"

        return {
            "direction": direction,
            "slope": slope,
            "confidence": min(1.0, len(values) / 10)  # Simple confidence based on sample size
        }

    def export_dashboard(self, dashboard: SLODashboard, format: str = "json") -> str:
        """Export SLO dashboard in specified format"""
        data = {
            "service_name": dashboard.service_name,
            "time_period": dashboard.time_period,
            "generated_at": dashboard.generated_at.isoformat(),
            "overall_health": dashboard.overall_health.value,
            "slo_status": {slo_id: status.value for slo_id, status in dashboard.slo_status.items()},
            "error_budgets": dashboard.error_budgets,
            "alerts": dashboard.alerts,
            "recommendations": dashboard.recommendations
        }

        if format == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    # Data access methods (would be implemented with actual database queries)

    async def get_service_slos(self, service_name: str) -> List[SLODefinition]:
        """Get all SLOs for a service"""
        return [
            slo for slo in self.slo_definitions.values()
            if slo.service_name == service_name
        ]

    async def get_slo_measurements(self, slo_id: str, days: int = 7) -> List[SLOMeasurement]:
        """Get SLO measurements for a time period"""
        if slo_id not in self.measurements:
            return []

        cutoff_date = datetime.now() - timedelta(days=days)
        return [m for m in self.measurements[slo_id] if m.timestamp >= cutoff_date]

    def get_all_slos(self) -> List[SLODefinition]:
        """Get all SLO definitions"""
        return list(self.slo_definitions.values())

    async def update_slo_target(self, slo_id: str, new_target: float) -> bool:
        """Update SLO target value"""
        if slo_id not in self.slo_definitions:
            return False

        self.slo_definitions[slo_id].target_value = new_target
        self.slo_definitions[slo_id].updated_at = datetime.now()

        logger.info(f"Updated SLO {slo_id} target to {new_target}")
        return True

    async def delete_slo(self, slo_id: str) -> bool:
        """Delete an SLO"""
        if slo_id not in self.slo_definitions:
            return False

        del self.slo_definitions[slo_id]
        if slo_id in self.measurements:
            del self.measurements[slo_id]
        if slo_id in self.burn_rate_cache:
            del self.burn_rate_cache[slo_id]

        logger.info(f"Deleted SLO {slo_id}")
        return True
