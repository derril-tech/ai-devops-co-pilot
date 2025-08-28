"""
Rate limiter for API protection and system stability
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import hashlib

from ..database.config import get_postgres_session, get_clickhouse_session


logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"  # Fixed time window
    SLIDING_WINDOW = "sliding_window"  # Sliding time window
    TOKEN_BUCKET = "token_bucket"  # Token bucket algorithm
    LEAKY_BUCKET = "leaky_bucket"  # Leaky bucket algorithm


class RateLimitScope(Enum):
    """Scope of rate limiting"""
    GLOBAL = "global"  # Global across all requests
    USER = "user"  # Per user
    IP = "ip"  # Per IP address
    ENDPOINT = "endpoint"  # Per API endpoint
    SERVICE = "service"  # Per service


@dataclass
class RateLimitRule:
    """Rate limiting rule definition"""
    id: str
    name: str
    scope: RateLimitScope
    strategy: RateLimitStrategy
    requests_per_window: int
    window_seconds: int
    burst_limit: Optional[int] = None
    refill_rate: Optional[float] = None  # Tokens per second for token bucket
    enabled: bool = True
    exempt_users: List[str] = field(default_factory=list)
    exempt_ips: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class RateLimitState:
    """Current state of rate limiting for an entity"""
    entity_id: str
    rule_id: str
    request_count: int = 0
    window_start: datetime = field(default_factory=datetime.now)
    tokens: float = 0.0  # For token bucket
    last_refill: datetime = field(default_factory=datetime.now)
    blocked_until: Optional[datetime] = None
    violations: int = 0


@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    allowed: bool
    remaining_requests: int
    reset_time: datetime
    retry_after: Optional[int] = None
    rule_id: str = ""
    entity_id: str = ""
    blocked: bool = False


class RateLimiter:
    """Advanced rate limiter with multiple strategies and scopes"""

    def __init__(self):
        self.rules: Dict[str, RateLimitRule] = {}
        self.states: Dict[str, Dict[str, RateLimitState]] = {}  # entity_id -> rule_id -> state
        self.global_states: Dict[str, RateLimitState] = {}
        self.violation_counts: Dict[str, int] = {}  # Track violations per entity

        # Default rules
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default rate limiting rules"""
        # Global API rate limit
        self.add_rule(RateLimitRule(
            id="global_api",
            name="Global API Rate Limit",
            scope=RateLimitScope.GLOBAL,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            requests_per_window=1000,
            window_seconds=60,  # 1000 requests per minute
            burst_limit=1500
        ))

        # Per-user rate limit
        self.add_rule(RateLimitRule(
            id="per_user",
            name="Per-User Rate Limit",
            scope=RateLimitScope.USER,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            requests_per_window=100,
            window_seconds=60,
            refill_rate=1.67  # ~100 requests per minute
        ))

        # Per-IP rate limit
        self.add_rule(RateLimitRule(
            id="per_ip",
            name="Per-IP Rate Limit",
            scope=RateLimitScope.IP,
            strategy=RateLimitStrategy.FIXED_WINDOW,
            requests_per_window=50,
            window_seconds=60  # 50 requests per minute per IP
        ))

        # Intensive endpoints
        self.add_rule(RateLimitRule(
            id="intensive_endpoints",
            name="Intensive Endpoint Protection",
            scope=RateLimitScope.ENDPOINT,
            strategy=RateLimitStrategy.LEAKY_BUCKET,
            requests_per_window=10,
            window_seconds=60,
            refill_rate=0.167  # ~10 requests per minute
        ))

    def add_rule(self, rule: RateLimitRule) -> None:
        """Add a rate limiting rule"""
        self.rules[rule.id] = rule
        logger.info(f"Added rate limit rule: {rule.name} ({rule.id})")

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rate limiting rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            # Clean up associated states
            for entity_states in self.states.values():
                if rule_id in entity_states:
                    del entity_states[rule_id]
            if rule_id in self.global_states:
                del self.global_states[rule_id]
            logger.info(f"Removed rate limit rule: {rule_id}")
            return True
        return False

    async def check_rate_limit(self, entity_id: str, rule_id: str,
                              endpoint: Optional[str] = None,
                              user_id: Optional[str] = None,
                              ip_address: Optional[str] = None) -> RateLimitResult:
        """
        Check if request should be rate limited

        Args:
            entity_id: Entity identifier (user_id, ip, etc.)
            rule_id: Rate limit rule ID
            endpoint: API endpoint path
            user_id: User identifier for exemption checks
            ip_address: IP address for exemption checks

        Returns:
            RateLimitResult indicating if request is allowed
        """
        if rule_id not in self.rules:
            # No rule found, allow request
            return RateLimitResult(
                allowed=True,
                remaining_requests=-1,
                reset_time=datetime.now() + timedelta(hours=1)
            )

        rule = self.rules[rule_id]

        # Check exemptions
        if self._is_exempt(rule, user_id, ip_address):
            return RateLimitResult(
                allowed=True,
                remaining_requests=-1,
                reset_time=datetime.now() + timedelta(hours=1),
                rule_id=rule_id,
                entity_id=entity_id
            )

        # Check if currently blocked
        state = self._get_or_create_state(entity_id, rule_id)
        if state.blocked_until and datetime.now() < state.blocked_until:
            retry_after = int((state.blocked_until - datetime.now()).total_seconds())
            return RateLimitResult(
                allowed=False,
                remaining_requests=0,
                reset_time=state.blocked_until,
                retry_after=retry_after,
                rule_id=rule_id,
                entity_id=entity_id,
                blocked=True
            )

        # Apply rate limiting strategy
        if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            return self._check_fixed_window(state, rule, entity_id)
        elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._check_sliding_window(state, rule, entity_id)
        elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._check_token_bucket(state, rule, entity_id)
        elif rule.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return self._check_leaky_bucket(state, rule, entity_id)
        else:
            # Unknown strategy, allow
            return RateLimitResult(
                allowed=True,
                remaining_requests=-1,
                reset_time=datetime.now() + timedelta(hours=1),
                rule_id=rule_id,
                entity_id=entity_id
            )

    def _get_or_create_state(self, entity_id: str, rule_id: str) -> RateLimitState:
        """Get or create rate limit state for entity and rule"""
        if entity_id not in self.states:
            self.states[entity_id] = {}

        if rule_id not in self.states[entity_id]:
            rule = self.rules[rule_id]
            self.states[entity_id][rule_id] = RateLimitState(
                entity_id=entity_id,
                rule_id=rule_id,
                tokens=rule.requests_per_window if rule.strategy == RateLimitStrategy.TOKEN_BUCKET else 0.0
            )

        return self.states[entity_id][rule_id]

    def _is_exempt(self, rule: RateLimitRule, user_id: Optional[str],
                  ip_address: Optional[str]) -> bool:
        """Check if request is exempt from rate limiting"""
        if user_id and user_id in rule.exempt_users:
            return True
        if ip_address and ip_address in rule.exempt_ips:
            return True
        return False

    def _check_fixed_window(self, state: RateLimitState, rule: RateLimitRule,
                           entity_id: str) -> RateLimitResult:
        """Check rate limit using fixed window strategy"""
        now = datetime.now()
        window_end = state.window_start + timedelta(seconds=rule.window_seconds)

        # Reset window if expired
        if now >= window_end:
            state.window_start = now
            state.request_count = 0

        # Check if limit exceeded
        if state.request_count >= rule.requests_per_window:
            # Calculate reset time
            reset_time = state.window_start + timedelta(seconds=rule.window_seconds)
            retry_after = int((reset_time - now).total_seconds())

            # Increment violations
            state.violations += 1
            self._handle_violation(entity_id, rule, state.violations)

            return RateLimitResult(
                allowed=False,
                remaining_requests=0,
                reset_time=reset_time,
                retry_after=retry_after,
                rule_id=rule.id,
                entity_id=entity_id
            )

        # Allow request and increment counter
        state.request_count += 1
        remaining = rule.requests_per_window - state.request_count
        reset_time = state.window_start + timedelta(seconds=rule.window_seconds)

        return RateLimitResult(
            allowed=True,
            remaining_requests=remaining,
            reset_time=reset_time,
            rule_id=rule.id,
            entity_id=entity_id
        )

    def _check_sliding_window(self, state: RateLimitState, rule: RateLimitRule,
                             entity_id: str) -> RateLimitResult:
        """Check rate limit using sliding window strategy"""
        # For simplicity, implement as fixed window with cleanup
        # In production, you'd want a more sophisticated sliding window implementation
        return self._check_fixed_window(state, rule, entity_id)

    def _check_token_bucket(self, state: RateLimitState, rule: RateLimitRule,
                           entity_id: str) -> RateLimitResult:
        """Check rate limit using token bucket strategy"""
        now = datetime.now()

        # Refill tokens based on time elapsed
        if rule.refill_rate:
            time_elapsed = (now - state.last_refill).total_seconds()
            tokens_to_add = time_elapsed * rule.refill_rate
            state.tokens = min(rule.requests_per_window, state.tokens + tokens_to_add)
            state.last_refill = now

        # Check if we have tokens
        if state.tokens < 1:
            reset_time = now + timedelta(seconds=1 / rule.refill_rate if rule.refill_rate else 60)
            retry_after = int((reset_time - now).total_seconds())

            # Increment violations
            state.violations += 1
            self._handle_violation(entity_id, rule, state.violations)

            return RateLimitResult(
                allowed=False,
                remaining_requests=int(state.tokens),
                reset_time=reset_time,
                retry_after=retry_after,
                rule_id=rule.id,
                entity_id=entity_id
            )

        # Consume token
        state.tokens -= 1
        remaining = int(state.tokens)
        reset_time = now + timedelta(seconds=1 / rule.refill_rate if rule.refill_rate else 60)

        return RateLimitResult(
            allowed=True,
            remaining_requests=remaining,
            reset_time=reset_time,
            rule_id=rule.id,
            entity_id=entity_id
        )

    def _check_leaky_bucket(self, state: RateLimitState, rule: RateLimitRule,
                           entity_id: str) -> RateLimitResult:
        """Check rate limit using leaky bucket strategy"""
        now = datetime.now()

        # Similar to token bucket but with different semantics
        # For this implementation, we'll use a simplified version
        if state.request_count >= rule.requests_per_window:
            reset_time = state.window_start + timedelta(seconds=rule.window_seconds)
            retry_after = int((reset_time - now).total_seconds())

            state.violations += 1
            self._handle_violation(entity_id, rule, state.violations)

            return RateLimitResult(
                allowed=False,
                remaining_requests=0,
                reset_time=reset_time,
                retry_after=retry_after,
                rule_id=rule.id,
                entity_id=entity_id
            )

        state.request_count += 1
        remaining = rule.requests_per_window - state.request_count
        reset_time = state.window_start + timedelta(seconds=rule.window_seconds)

        return RateLimitResult(
            allowed=True,
            remaining_requests=remaining,
            reset_time=reset_time,
            rule_id=rule.id,
            entity_id=entity_id
        )

    def _handle_violation(self, entity_id: str, rule: RateLimitRule, violation_count: int) -> None:
        """Handle rate limit violations"""
        # Progressive penalty system
        if violation_count >= 10:
            # Temporary block for repeated violations
            block_duration = timedelta(minutes=5 * (violation_count // 10))
            state = self._get_or_create_state(entity_id, rule.id)
            state.blocked_until = datetime.now() + block_duration
            logger.warning(f"Entity {entity_id} temporarily blocked for {block_duration}")

        self.violation_counts[entity_id] = violation_count

    def get_violation_stats(self, entity_id: str) -> Dict[str, Any]:
        """Get violation statistics for an entity"""
        return {
            "entity_id": entity_id,
            "violation_count": self.violation_counts.get(entity_id, 0),
            "active_rules": list(self.states.get(entity_id, {}).keys()),
            "last_violation": None  # Would need to track timestamps
        }

    def reset_entity(self, entity_id: str) -> bool:
        """Reset rate limiting state for an entity"""
        if entity_id in self.states:
            del self.states[entity_id]
            if entity_id in self.violation_counts:
                del self.violation_counts[entity_id]
            logger.info(f"Reset rate limiting state for entity: {entity_id}")
            return True
        return False

    def get_active_rules(self) -> List[RateLimitRule]:
        """Get all active rate limiting rules"""
        return [rule for rule in self.rules.values() if rule.enabled]

    def get_rule_stats(self, rule_id: str) -> Dict[str, Any]:
        """Get statistics for a specific rule"""
        if rule_id not in self.rules:
            return {"error": f"Rule {rule_id} not found"}

        rule = self.rules[rule_id]
        entities_affected = sum(1 for entity_states in self.states.values()
                              if rule_id in entity_states)

        return {
            "rule_id": rule_id,
            "rule_name": rule.name,
            "strategy": rule.strategy.value,
            "scope": rule.scope.value,
            "requests_per_window": rule.requests_per_window,
            "window_seconds": rule.window_seconds,
            "entities_tracked": entities_affected,
            "enabled": rule.enabled
        }

    def export_configuration(self) -> Dict[str, Any]:
        """Export rate limiter configuration"""
        return {
            "rules": {
                rule_id: {
                    "id": rule.id,
                    "name": rule.name,
                    "scope": rule.scope.value,
                    "strategy": rule.strategy.value,
                    "requests_per_window": rule.requests_per_window,
                    "window_seconds": rule.window_seconds,
                    "enabled": rule.enabled,
                    "exempt_users": rule.exempt_users,
                    "exempt_ips": rule.exempt_ips
                }
                for rule_id, rule in self.rules.items()
            },
            "exported_at": datetime.now().isoformat(),
            "version": "1.0"
        }

    def import_configuration(self, config: Dict[str, Any]) -> bool:
        """Import rate limiter configuration"""
        try:
            for rule_data in config.get("rules", {}).values():
                rule = RateLimitRule(
                    id=rule_data["id"],
                    name=rule_data["name"],
                    scope=RateLimitScope(rule_data["scope"]),
                    strategy=RateLimitStrategy(rule_data["strategy"]),
                    requests_per_window=rule_data["requests_per_window"],
                    window_seconds=rule_data["window_seconds"],
                    enabled=rule_data.get("enabled", True),
                    exempt_users=rule_data.get("exempt_users", []),
                    exempt_ips=rule_data.get("exempt_ips", [])
                )
                self.add_rule(rule)

            logger.info("Imported rate limiter configuration")
            return True

        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return False

    def cleanup_expired_states(self) -> int:
        """Clean up expired rate limiting states"""
        now = datetime.now()
        cleanup_count = 0

        # Clean up per-entity states
        for entity_id in list(self.states.keys()):
            for rule_id in list(self.states[entity_id].keys()):
                state = self.states[entity_id][rule_id]
                rule = self.rules.get(rule_id)

                if rule and (now - state.window_start).total_seconds() > rule.window_seconds * 2:
                    del self.states[entity_id][rule_id]
                    cleanup_count += 1

            # Remove entity if no rules left
            if not self.states[entity_id]:
                del self.states[entity_id]

        # Clean up violation counts (remove old entries)
        old_violations = [
            entity_id for entity_id, count in self.violation_counts.items()
            if count == 0  # Remove entities with no violations
        ]

        for entity_id in old_violations:
            del self.violation_counts[entity_id]
            cleanup_count += 1

        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} expired rate limiting states")

        return cleanup_count


# Global rate limiter instance
rate_limiter = RateLimiter()
