"""
Backpressure manager for system stability and graceful degradation
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import psutil
import time

from ..database.config import get_postgres_session, get_clickhouse_session


logger = logging.getLogger(__name__)


class BackpressureLevel(Enum):
    """Backpressure levels"""
    NORMAL = "normal"       # No backpressure
    WARNING = "warning"     # Light backpressure
    CRITICAL = "critical"   # Heavy backpressure
    THROTTLE = "throttle"   # Severe backpressure - throttle heavily


class BackpressureStrategy(Enum):
    """Backpressure handling strategies"""
    DROP_REQUESTS = "drop_requests"           # Drop excess requests
    QUEUE_REQUESTS = "queue_requests"         # Queue requests for later processing
    DEFER_PROCESSING = "defer_processing"     # Defer non-critical processing
    SCALE_RESOURCES = "scale_resources"       # Scale up resources if possible
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Reduce functionality


@dataclass
class SystemMetrics:
    """Current system metrics"""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_connections: int
    queue_depth: int
    response_time_ms: float
    error_rate: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BackpressureRule:
    """Rule for triggering backpressure"""
    id: str
    name: str
    metric: str  # Which metric to monitor
    threshold: float
    level: BackpressureLevel
    strategy: BackpressureStrategy
    cooldown_seconds: int = 60  # Minimum time between activations
    enabled: bool = True
    last_triggered: Optional[datetime] = None


@dataclass
class BackpressureState:
    """Current backpressure state"""
    level: BackpressureLevel
    active_rules: List[str]
    strategies: List[BackpressureStrategy]
    start_time: datetime
    metrics: SystemMetrics


class BackpressureManager:
    """Manager for system backpressure and load shedding"""

    def __init__(self):
        self.rules: Dict[str, BackpressureRule] = {}
        self.current_state: Optional[BackpressureState] = None
        self.metrics_history: List[SystemMetrics] = []
        self.strategy_handlers: Dict[BackpressureStrategy, Callable] = {}
        self.alert_handlers: List[Callable] = []

        # Setup default rules and handlers
        self._setup_default_rules()
        self._setup_default_handlers()

    def _setup_default_rules(self):
        """Setup default backpressure rules"""
        # CPU usage rules
        self.add_rule(BackpressureRule(
            id="cpu_warning",
            name="High CPU Usage Warning",
            metric="cpu_percent",
            threshold=70.0,
            level=BackpressureLevel.WARNING,
            strategy=BackpressureStrategy.DEFER_PROCESSING,
            cooldown_seconds=120
        ))

        self.add_rule(BackpressureRule(
            id="cpu_critical",
            name="Critical CPU Usage",
            metric="cpu_percent",
            threshold=85.0,
            level=BackpressureLevel.CRITICAL,
            strategy=BackpressureStrategy.DROP_REQUESTS,
            cooldown_seconds=60
        ))

        self.add_rule(BackpressureRule(
            id="cpu_throttle",
            name="Extreme CPU Usage",
            metric="cpu_percent",
            threshold=95.0,
            level=BackpressureLevel.THROTTLE,
            strategy=BackpressureStrategy.DROP_REQUESTS,
            cooldown_seconds=30
        ))

        # Memory usage rules
        self.add_rule(BackpressureRule(
            id="memory_warning",
            name="High Memory Usage Warning",
            metric="memory_percent",
            threshold=75.0,
            level=BackpressureLevel.WARNING,
            strategy=BackpressureStrategy.DEFER_PROCESSING,
            cooldown_seconds=120
        ))

        self.add_rule(BackpressureRule(
            id="memory_critical",
            name="Critical Memory Usage",
            metric="memory_percent",
            threshold=90.0,
            level=BackpressureLevel.CRITICAL,
            strategy=BackpressureStrategy.DROP_REQUESTS,
            cooldown_seconds=60
        ))

        # Queue depth rules
        self.add_rule(BackpressureRule(
            id="queue_warning",
            name="High Queue Depth Warning",
            metric="queue_depth",
            threshold=1000,
            level=BackpressureLevel.WARNING,
            strategy=BackpressureStrategy.QUEUE_REQUESTS,
            cooldown_seconds=180
        ))

        self.add_rule(BackpressureRule(
            id="queue_critical",
            name="Critical Queue Depth",
            metric="queue_depth",
            threshold=5000,
            level=BackpressureLevel.CRITICAL,
            strategy=BackpressureStrategy.DROP_REQUESTS,
            cooldown_seconds=60
        ))

        # Error rate rules
        self.add_rule(BackpressureRule(
            id="error_rate_warning",
            name="High Error Rate Warning",
            metric="error_rate",
            threshold=0.10,  # 10%
            level=BackpressureLevel.WARNING,
            strategy=BackpressureStrategy.DEFER_PROCESSING,
            cooldown_seconds=300
        ))

        self.add_rule(BackpressureRule(
            id="error_rate_critical",
            name="Critical Error Rate",
            metric="error_rate",
            threshold=0.25,  # 25%
            level=BackpressureLevel.CRITICAL,
            strategy=BackpressureStrategy.DROP_REQUESTS,
            cooldown_seconds=120
        ))

    def _setup_default_handlers(self):
        """Setup default strategy handlers"""
        self.register_strategy_handler(
            BackpressureStrategy.DROP_REQUESTS,
            self._handle_drop_requests
        )

        self.register_strategy_handler(
            BackpressureStrategy.QUEUE_REQUESTS,
            self._handle_queue_requests
        )

        self.register_strategy_handler(
            BackpressureStrategy.DEFER_PROCESSING,
            self._handle_defer_processing
        )

        self.register_strategy_handler(
            BackpressureStrategy.SCALE_RESOURCES,
            self._handle_scale_resources
        )

        self.register_strategy_handler(
            BackpressureStrategy.GRACEFUL_DEGRADATION,
            self._handle_graceful_degradation
        )

    def add_rule(self, rule: BackpressureRule) -> None:
        """Add a backpressure rule"""
        self.rules[rule.id] = rule
        logger.info(f"Added backpressure rule: {rule.name} ({rule.id})")

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a backpressure rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed backpressure rule: {rule_id}")
            return True
        return False

    def register_strategy_handler(self, strategy: BackpressureStrategy, handler: Callable) -> None:
        """Register a handler for a backpressure strategy"""
        self.strategy_handlers[strategy] = handler
        logger.info(f"Registered handler for strategy: {strategy.value}")

    def register_alert_handler(self, handler: Callable) -> None:
        """Register an alert handler"""
        self.alert_handlers.append(handler)
        logger.info("Registered backpressure alert handler")

    async def check_backpressure(self) -> BackpressureState:
        """
        Check current system state and determine backpressure level

        Returns:
            Current backpressure state
        """
        # Collect current system metrics
        metrics = await self._collect_system_metrics()

        # Store metrics history
        self.metrics_history.append(metrics)

        # Keep only recent history (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.metrics_history = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]

        # Evaluate rules
        triggered_rules = []
        max_level = BackpressureLevel.NORMAL

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            # Check cooldown
            if rule.last_triggered:
                cooldown_end = rule.last_triggered + timedelta(seconds=rule.cooldown_seconds)
                if datetime.now() < cooldown_end:
                    continue

            # Evaluate rule
            metric_value = getattr(metrics, rule.metric, 0)

            if self._evaluate_rule_condition(rule, metric_value):
                triggered_rules.append(rule.id)
                rule.last_triggered = datetime.now()

                if rule.level.value > max_level.value:
                    max_level = rule.level

        # Determine active strategies
        active_strategies = []
        if triggered_rules:
            active_strategies = list(set(
                self.rules[rule_id].strategy
                for rule_id in triggered_rules
            ))

        # Create or update backpressure state
        if triggered_rules:
            if not self.current_state or self.current_state.level != max_level:
                self.current_state = BackpressureState(
                    level=max_level,
                    active_rules=triggered_rules,
                    strategies=active_strategies,
                    start_time=datetime.now(),
                    metrics=metrics
                )

                # Send alerts
                await self._send_alerts(self.current_state)

                logger.warning(f"Backpressure activated: {max_level.value} level with rules: {triggered_rules}")
        else:
            # Clear backpressure state
            if self.current_state:
                logger.info(f"Backpressure cleared (was: {self.current_state.level.value})")
                self.current_state = None

        return self.current_state or BackpressureState(
            level=BackpressureLevel.NORMAL,
            active_rules=[],
            strategies=[],
            start_time=datetime.now(),
            metrics=metrics
        )

    def _evaluate_rule_condition(self, rule: BackpressureRule, metric_value: float) -> bool:
        """Evaluate if a rule condition is met"""
        if rule.metric in ["cpu_percent", "memory_percent", "disk_usage_percent", "error_rate"]:
            return metric_value >= rule.threshold
        elif rule.metric in ["queue_depth", "network_connections"]:
            return metric_value >= rule.threshold
        elif rule.metric == "response_time_ms":
            return metric_value >= rule.threshold
        else:
            return False

    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent

            # Network connections
            network_connections = len(psutil.net_connections())

            # Queue depth (mock - would integrate with actual queue monitoring)
            queue_depth = await self._get_queue_depth()

            # Response time (mock - would integrate with actual monitoring)
            response_time_ms = await self._get_response_time()

            # Error rate (mock - would integrate with actual monitoring)
            error_rate = await self._get_error_rate()

            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage_percent=disk_usage_percent,
                network_connections=network_connections,
                queue_depth=queue_depth,
                response_time_ms=response_time_ms,
                error_rate=error_rate
            )

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # Return default metrics
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_usage_percent=0.0,
                network_connections=0,
                queue_depth=0,
                response_time_ms=0.0,
                error_rate=0.0
            )

    async def _get_queue_depth(self) -> int:
        """Get current queue depth"""
        # This would integrate with actual queue monitoring
        # For now, return a mock value
        return 50

    async def _get_response_time(self) -> float:
        """Get current response time"""
        # This would integrate with actual monitoring
        # For now, return a mock value
        return 150.0

    async def _get_error_rate(self) -> float:
        """Get current error rate"""
        # This would integrate with actual monitoring
        # For now, return a mock value
        return 0.02

    async def _send_alerts(self, state: BackpressureState) -> None:
        """Send backpressure alerts"""
        alert_data = {
            "level": state.level.value,
            "active_rules": state.active_rules,
            "strategies": [s.value for s in state.strategies],
            "metrics": {
                "cpu_percent": state.metrics.cpu_percent,
                "memory_percent": state.metrics.memory_percent,
                "queue_depth": state.metrics.queue_depth,
                "error_rate": state.metrics.error_rate
            },
            "timestamp": state.start_time.isoformat()
        }

        for handler in self.alert_handlers:
            try:
                await handler(alert_data)
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")

    async def apply_backpressure(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply backpressure to a request based on current state

        Args:
            request_context: Context of the request being processed

        Returns:
            Response indicating how to handle the request
        """
        if not self.current_state:
            return {"action": "allow", "reason": "no_backpressure"}

        # Apply strategies
        for strategy in self.current_state.strategies:
            handler = self.strategy_handlers.get(strategy)
            if handler:
                try:
                    result = await handler(request_context, self.current_state)
                    if result["action"] != "allow":
                        return result
                except Exception as e:
                    logger.error(f"Failed to apply strategy {strategy.value}: {e}")

        return {"action": "allow", "reason": "strategies_allowed"}

    async def _handle_drop_requests(self, request_context: Dict[str, Any],
                                  state: BackpressureState) -> Dict[str, Any]:
        """Handle drop requests strategy"""
        # Drop requests based on priority or randomly
        request_type = request_context.get("type", "unknown")
        priority = request_context.get("priority", "normal")

        if priority == "low" or (state.level == BackpressureLevel.THROTTLE and priority != "critical"):
            return {
                "action": "drop",
                "reason": f"backpressure_drop_{state.level.value}",
                "retry_after": 30
            }

        return {"action": "allow"}

    async def _handle_queue_requests(self, request_context: Dict[str, Any],
                                    state: BackpressureState) -> Dict[str, Any]:
        """Handle queue requests strategy"""
        # Queue non-critical requests
        priority = request_context.get("priority", "normal")

        if priority in ["low", "normal"]:
            return {
                "action": "queue",
                "reason": f"backpressure_queue_{state.level.value}",
                "estimated_delay": 60
            }

        return {"action": "allow"}

    async def _handle_defer_processing(self, request_context: Dict[str, Any],
                                      state: BackpressureState) -> Dict[str, Any]:
        """Handle defer processing strategy"""
        # Defer non-essential processing
        request_type = request_context.get("type", "unknown")

        if request_type in ["analytics", "cleanup", "background_sync"]:
            return {
                "action": "defer",
                "reason": f"backpressure_defer_{state.level.value}",
                "defer_until": (datetime.now() + timedelta(minutes=5)).isoformat()
            }

        return {"action": "allow"}

    async def _handle_scale_resources(self, request_context: Dict[str, Any],
                                     state: BackpressureState) -> Dict[str, Any]:
        """Handle scale resources strategy"""
        # This would integrate with auto-scaling systems
        logger.info(f"Scale resources triggered for backpressure level: {state.level.value}")

        # For now, just log and allow
        return {"action": "allow", "scaling_triggered": True}

    async def _handle_graceful_degradation(self, request_context: Dict[str, Any],
                                          state: BackpressureState) -> Dict[str, Any]:
        """Handle graceful degradation strategy"""
        # Reduce functionality for non-critical requests
        request_type = request_context.get("type", "unknown")

        if request_type == "api_call":
            return {
                "action": "degrade",
                "reason": f"backpressure_degrade_{state.level.value}",
                "reduced_functionality": True
            }

        return {"action": "allow"}

    def get_backpressure_status(self) -> Dict[str, Any]:
        """Get current backpressure status"""
        if not self.current_state:
            return {
                "level": "normal",
                "active": False,
                "active_rules": [],
                "strategies": []
            }

        return {
            "level": self.current_state.level.value,
            "active": True,
            "active_rules": self.current_state.active_rules,
            "strategies": [s.value for s in self.current_state.strategies],
            "start_time": self.current_state.start_time.isoformat(),
            "metrics": {
                "cpu_percent": self.current_state.metrics.cpu_percent,
                "memory_percent": self.current_state.metrics.memory_percent,
                "queue_depth": self.current_state.metrics.queue_depth,
                "error_rate": self.current_state.metrics.error_rate
            }
        }

    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            {
                "timestamp": m.timestamp.isoformat(),
                "cpu_percent": m.cpu_percent,
                "memory_percent": m.memory_percent,
                "disk_usage_percent": m.disk_usage_percent,
                "network_connections": m.network_connections,
                "queue_depth": m.queue_depth,
                "response_time_ms": m.response_time_ms,
                "error_rate": m.error_rate
            }
            for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]

    def get_rules_status(self) -> Dict[str, Any]:
        """Get status of all backpressure rules"""
        return {
            rule_id: {
                "id": rule.id,
                "name": rule.name,
                "enabled": rule.enabled,
                "metric": rule.metric,
                "threshold": rule.threshold,
                "level": rule.level.value,
                "strategy": rule.strategy.value,
                "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None
            }
            for rule_id, rule in self.rules.items()
        }

    async def reset_backpressure(self) -> bool:
        """Manually reset backpressure state"""
        if self.current_state:
            logger.info(f"Manually resetting backpressure (was: {self.current_state.level.value})")
            self.current_state = None
            return True
        return False

    def export_configuration(self) -> Dict[str, Any]:
        """Export backpressure configuration"""
        return {
            "rules": {
                rule_id: {
                    "id": rule.id,
                    "name": rule.name,
                    "metric": rule.metric,
                    "threshold": rule.threshold,
                    "level": rule.level.value,
                    "strategy": rule.strategy.value,
                    "cooldown_seconds": rule.cooldown_seconds,
                    "enabled": rule.enabled
                }
                for rule_id, rule in self.rules.items()
            },
            "exported_at": datetime.now().isoformat(),
            "version": "1.0"
        }

    def import_configuration(self, config: Dict[str, Any]) -> bool:
        """Import backpressure configuration"""
        try:
            for rule_data in config.get("rules", {}).values():
                rule = BackpressureRule(
                    id=rule_data["id"],
                    name=rule_data["name"],
                    metric=rule_data["metric"],
                    threshold=rule_data["threshold"],
                    level=BackpressureLevel(rule_data["level"]),
                    strategy=BackpressureStrategy(rule_data["strategy"]),
                    cooldown_seconds=rule_data.get("cooldown_seconds", 60),
                    enabled=rule_data.get("enabled", True)
                )
                self.add_rule(rule)

            logger.info("Imported backpressure configuration")
            return True

        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return False


# Global backpressure manager instance
backpressure_manager = BackpressureManager()
