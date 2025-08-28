"""
Fix catalog with common remediation patterns and templates
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import re

from ..rca.hypothesis_builder import HypothesisType


logger = logging.getLogger(__name__)


class FixType(Enum):
    """Types of fixes available in the catalog"""
    KUBERNETES_PATCH = "kubernetes_patch"
    HELM_UPGRADE = "helm_upgrade"
    TERRAFORM_PLAN = "terraform_plan"
    ANSIBLE_PLAYBOOK = "ansible_playbook"
    SQL_MIGRATION = "sql_migration"
    SHELL_SCRIPT = "shell_script"
    CONFIG_UPDATE = "config_update"
    SERVICE_RESTART = "service_restart"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    LOAD_BALANCER_UPDATE = "load_balancer_update"
    NETWORK_POLICY_UPDATE = "network_policy_update"
    LOG_ROTATION = "log_rotation"
    CACHE_INVALIDATION = "cache_invalidation"


class RiskLevel(Enum):
    """Risk levels for fixes"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FixTemplate:
    """Template for a remediation fix"""
    id: str
    name: str
    description: str
    fix_type: FixType
    risk_level: RiskLevel
    estimated_duration: timedelta
    success_rate: float  # Historical success rate (0-1)
    rollback_available: bool
    rollback_template: Optional[str] = None

    # Template parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Preconditions that must be met
    preconditions: List[str] = field(default_factory=list)

    # Postconditions to verify
    postconditions: List[str] = field(default_factory=list)

    # Tags for categorization
    tags: List[str] = field(default_factory=list)

    # Applicable hypothesis types
    applicable_hypotheses: List[HypothesisType] = field(default_factory=list)


@dataclass
class GeneratedFix:
    """Generated fix from a template"""
    template_id: str
    fix_type: FixType
    risk_level: RiskLevel
    scripts: List[Dict[str, Any]]  # Generated scripts/commands
    estimated_duration: timedelta
    rollback_scripts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)


class FixCatalog:
    """Catalog of remediation fixes and templates"""

    def __init__(self):
        self.templates: Dict[str, FixTemplate] = {}
        self._load_builtin_templates()

    def _load_builtin_templates(self):
        """Load built-in fix templates"""

        # Kubernetes Pod Restart
        self.templates["k8s_pod_restart"] = FixTemplate(
            id="k8s_pod_restart",
            name="Restart Kubernetes Pods",
            description="Restart pods matching specified criteria",
            fix_type=FixType.KUBERNETES_PATCH,
            risk_level=RiskLevel.MEDIUM,
            estimated_duration=timedelta(minutes=5),
            success_rate=0.95,
            rollback_available=False,
            parameters={
                "namespace": {"type": "string", "required": True},
                "selector": {"type": "string", "required": True},
                "strategy": {"type": "enum", "values": ["rolling", "immediate"], "default": "rolling"}
            },
            preconditions=[
                "Kubernetes API is accessible",
                "Pods exist and are running",
                "RBAC permissions allow pod operations"
            ],
            postconditions=[
                "Pods are in Ready state",
                "Application health checks pass",
                "No increase in error rates"
            ],
            tags=["kubernetes", "pods", "restart", "availability"],
            applicable_hypotheses=[
                HypothesisType.RESOURCE_CONSTRAINT,
                HypothesisType.CODE_BUG,
                HypothesisType.EXTERNAL_DEPENDENCY
            ]
        )

        # Database Connection Pool Reset
        self.templates["db_connection_reset"] = FixTemplate(
            id="db_connection_reset",
            name="Reset Database Connection Pool",
            description="Reset database connection pool to clear stale connections",
            fix_type=FixType.SHELL_SCRIPT,
            risk_level=RiskLevel.LOW,
            estimated_duration=timedelta(minutes=2),
            success_rate=0.90,
            rollback_available=False,
            parameters={
                "service_name": {"type": "string", "required": True},
                "pool_name": {"type": "string", "required": False}
            },
            preconditions=[
                "Database service is accessible",
                "Connection pool monitoring is enabled",
                "Application can handle brief connection interruption"
            ],
            postconditions=[
                "Connection pool is healthy",
                "New connections can be established",
                "Application recovers normal operation"
            ],
            tags=["database", "connections", "pool", "reset"],
            applicable_hypotheses=[
                HypothesisType.EXTERNAL_DEPENDENCY,
                HypothesisType.RESOURCE_CONSTRAINT
            ]
        )

        # Cache Invalidation
        self.templates["cache_invalidation"] = FixTemplate(
            id="cache_invalidation",
            name="Invalidate Application Cache",
            description="Clear application cache to resolve stale data issues",
            fix_type=FixType.SHELL_SCRIPT,
            risk_level=RiskLevel.MEDIUM,
            estimated_duration=timedelta(minutes=3),
            success_rate=0.85,
            rollback_available=False,
            parameters={
                "cache_type": {"type": "enum", "values": ["redis", "memcached", "local"], "required": True},
                "cache_keys": {"type": "string", "required": False},
                "service_name": {"type": "string", "required": True}
            },
            preconditions=[
                "Cache service is accessible",
                "Application can handle cache miss penalty",
                "Cache invalidation won't cause data loss"
            ],
            postconditions=[
                "Cache is cleared successfully",
                "Application performance returns to normal",
                "No data consistency issues"
            ],
            tags=["cache", "invalidation", "performance", "data"],
            applicable_hypotheses=[
                HypothesisType.DATA_CORRUPTION,
                HypothesisType.PERFORMANCE_DEGRADATION
            ]
        )

        # Service Scaling
        self.templates["service_scale_up"] = FixTemplate(
            id="service_scale_up",
            name="Scale Up Service",
            description="Increase service replica count to handle load",
            fix_type=FixType.KUBERNETES_PATCH,
            risk_level=RiskLevel.LOW,
            estimated_duration=timedelta(minutes=10),
            success_rate=0.92,
            rollback_available=True,
            parameters={
                "namespace": {"type": "string", "required": True},
                "service_name": {"type": "string", "required": True},
                "current_replicas": {"type": "integer", "required": True},
                "target_replicas": {"type": "integer", "required": True},
                "max_replicas": {"type": "integer", "required": False}
            },
            preconditions=[
                "Kubernetes cluster has sufficient resources",
                "Service supports horizontal scaling",
                "Load balancer can handle increased traffic"
            ],
            postconditions=[
                "Target replica count is achieved",
                "Pods are in Ready state",
                "Service performance improves",
                "Resource usage is within limits"
            ],
            tags=["kubernetes", "scaling", "horizontal", "load"],
            applicable_hypotheses=[
                HypothesisType.LOAD_SPIKE,
                HypothesisType.RESOURCE_CONSTRAINT
            ]
        )

        # Configuration Update
        self.templates["config_update"] = FixTemplate(
            id="config_update",
            name="Update Service Configuration",
            description="Update service configuration parameters",
            fix_type=FixType.CONFIG_UPDATE,
            risk_level=RiskLevel.MEDIUM,
            estimated_duration=timedelta(minutes=5),
            success_rate=0.88,
            rollback_available=True,
            parameters={
                "service_name": {"type": "string", "required": True},
                "config_path": {"type": "string", "required": True},
                "config_changes": {"type": "dict", "required": True},
                "config_format": {"type": "enum", "values": ["yaml", "json", "env"], "default": "yaml"}
            },
            preconditions=[
                "Configuration management system is accessible",
                "Configuration changes are validated",
                "Service supports configuration reload"
            ],
            postconditions=[
                "Configuration is updated successfully",
                "Service reloads configuration without restart",
                "Application behavior reflects new configuration"
            ],
            tags=["configuration", "config", "update", "reload"],
            applicable_hypotheses=[
                HypothesisType.CONFIGURATION_CHANGE,
                HypothesisType.PERFORMANCE_DEGRADATION
            ]
        )

        # Log Rotation
        self.templates["log_rotation"] = FixTemplate(
            id="log_rotation",
            name="Rotate Application Logs",
            description="Rotate and compress application log files",
            fix_type=FixType.SHELL_SCRIPT,
            risk_level=RiskLevel.LOW,
            estimated_duration=timedelta(minutes=2),
            success_rate=0.98,
            rollback_available=False,
            parameters={
                "service_name": {"type": "string", "required": True},
                "log_paths": {"type": "list", "required": True},
                "max_size_mb": {"type": "integer", "default": 100},
                "retention_days": {"type": "integer", "default": 30}
            },
            preconditions=[
                "Log files exist and are accessible",
                "Sufficient disk space for rotation",
                "Log rotation won't interrupt service"
            ],
            postconditions=[
                "Log files are rotated successfully",
                "Old logs are compressed or archived",
                "Disk space is reclaimed",
                "Logging continues normally"
            ],
            tags=["logs", "rotation", "disk", "maintenance"],
            applicable_hypotheses=[
                HypothesisType.RESOURCE_CONSTRAINT
            ]
        )

    def find_applicable_fixes(self, hypothesis_type: HypothesisType,
                                context: Dict[str, Any]) -> List[FixTemplate]:
        """Find fixes applicable to a hypothesis type and context"""
        applicable_fixes = []

        for template in self.templates.values():
            if hypothesis_type in template.applicable_hypotheses:
                # Check if context matches template requirements
                if self._template_matches_context(template, context):
                    applicable_fixes.append(template)

        # Sort by success rate and risk level
        applicable_fixes.sort(key=lambda x: (x.success_rate, -x.risk_level.value), reverse=True)

        return applicable_fixes

    def _template_matches_context(self, template: FixTemplate, context: Dict[str, Any]) -> bool:
        """Check if template matches the given context"""
        # Check service type compatibility
        service_type = context.get("service_type", "").lower()
        template_tags = [tag.lower() for tag in template.tags]

        # Basic service type matching
        if "kubernetes" in service_type and "kubernetes" not in template_tags:
            return False
        if "database" in service_type and "database" not in template_tags:
            return False

        # Check resource constraints
        if context.get("resource_issue") and "resource" not in template_tags:
            return False

        # Check for required context parameters
        for param_name, param_config in template.parameters.items():
            if param_config.get("required", False):
                context_key = f"param_{param_name}"
                if context_key not in context and param_name not in context:
                    return False

        return True

    def generate_fix(self, template_id: str, parameters: Dict[str, Any]) -> GeneratedFix:
        """Generate a fix from a template with specific parameters"""
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")

        template = self.templates[template_id]

        # Validate parameters
        validated_params = self._validate_parameters(template, parameters)

        # Generate scripts
        scripts = self._generate_scripts(template, validated_params)

        # Generate rollback scripts if available
        rollback_scripts = []
        if template.rollback_available and template.rollback_template:
            rollback_scripts = self._generate_rollback_scripts(template, validated_params)

        # Create fix metadata
        metadata = {
            "template_id": template_id,
            "parameters": validated_params,
            "generated_at": datetime.now().isoformat(),
            "risk_assessment": self._assess_risk(template, validated_params),
            "estimated_impact": self._estimate_impact(template, validated_params)
        }

        return GeneratedFix(
            template_id=template_id,
            fix_type=template.fix_type,
            risk_level=template.risk_level,
            scripts=scripts,
            estimated_duration=template.estimated_duration,
            rollback_scripts=rollback_scripts,
            metadata=metadata
        )

    def _validate_parameters(self, template: FixTemplate, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fill default parameters"""
        validated = {}

        for param_name, param_config in template.parameters.items():
            param_type = param_config["type"]

            # Check if parameter is provided
            if param_name in parameters:
                value = parameters[param_name]
            elif param_config.get("required", False):
                raise ValueError(f"Required parameter {param_name} is missing")
            elif "default" in param_config:
                value = param_config["default"]
            else:
                continue

            # Type validation
            if param_type == "string" and not isinstance(value, str):
                raise ValueError(f"Parameter {param_name} must be string")
            elif param_type == "integer" and not isinstance(value, int):
                raise ValueError(f"Parameter {param_name} must be integer")
            elif param_type == "enum" and value not in param_config.get("values", []):
                raise ValueError(f"Parameter {param_name} must be one of {param_config.get('values', [])}")
            elif param_type == "dict" and not isinstance(value, dict):
                raise ValueError(f"Parameter {param_name} must be dictionary")
            elif param_type == "list" and not isinstance(value, list):
                raise ValueError(f"Parameter {param_name} must be list")

            validated[param_name] = value

        return validated

    def _generate_scripts(self, template: FixTemplate, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate scripts for the fix"""
        scripts = []

        if template.fix_type == FixType.KUBERNETES_PATCH:
            scripts.extend(self._generate_kubernetes_scripts(template, parameters))
        elif template.fix_type == FixType.SHELL_SCRIPT:
            scripts.extend(self._generate_shell_scripts(template, parameters))
        elif template.fix_type == FixType.CONFIG_UPDATE:
            scripts.extend(self._generate_config_scripts(template, parameters))
        elif template.fix_type == FixType.SERVICE_RESTART:
            scripts.extend(self._generate_restart_scripts(template, parameters))
        elif template.fix_type == FixType.SCALE_UP:
            scripts.extend(self._generate_scale_scripts(template, parameters))

        return scripts

    def _generate_kubernetes_scripts(self, template: FixTemplate, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate Kubernetes-specific scripts"""
        scripts = []

        if template.id == "k8s_pod_restart":
            script = {
                "type": "kubectl",
                "command": "rollout",
                "args": ["restart", f"deployment/{parameters['service_name']}", "-n", parameters['namespace']],
                "description": f"Restart pods for {parameters['service_name']}",
                "timeout": 300,
                "validation": {
                    "command": "get",
                    "args": ["pods", "-l", f"app={parameters['service_name']}", "-n", parameters['namespace']],
                    "expected_status": "Running"
                }
            }
            scripts.append(script)

        elif template.id == "service_scale_up":
            script = {
                "type": "kubectl",
                "command": "scale",
                "args": ["deployment", parameters['service_name'],
                        f"--replicas={parameters['target_replicas']}",
                        "-n", parameters['namespace']],
                "description": f"Scale {parameters['service_name']} to {parameters['target_replicas']} replicas",
                "timeout": 600,
                "validation": {
                    "command": "get",
                    "args": ["deployment", parameters['service_name'], "-n", parameters['namespace']],
                    "expected_replicas": parameters['target_replicas']
                }
            }
            scripts.append(script)

        return scripts

    def _generate_shell_scripts(self, template: FixTemplate, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate shell scripts"""
        scripts = []

        if template.id == "db_connection_reset":
            script = {
                "type": "shell",
                "command": "systemctl" if parameters.get('service_name') else "docker",
                "args": ["restart", parameters.get('service_name', 'postgresql')],
                "description": f"Reset database connection pool for {parameters.get('service_name', 'database')}",
                "timeout": 120,
                "validation": {
                    "command": "pg_isready" if parameters.get('service_name') else "docker ps",
                    "expected_exit_code": 0
                }
            }
            scripts.append(script)

        elif template.id == "cache_invalidation":
            cache_type = parameters['cache_type']
            if cache_type == "redis":
                script = {
                    "type": "redis-cli",
                    "command": "FLUSHALL" if not parameters.get('cache_keys') else f"DEL {parameters['cache_keys']}",
                    "description": f"Invalidate Redis cache for {parameters['service_name']}",
                    "timeout": 30
                }
            else:
                script = {
                    "type": "shell",
                    "command": "echo",
                    "args": [f"Cache invalidation for {cache_type} not implemented"],
                    "description": f"Cache invalidation placeholder for {cache_type}"
                }
            scripts.append(script)

        return scripts

    def _generate_config_scripts(self, template: FixTemplate, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate configuration update scripts"""
        scripts = []

        if template.id == "config_update":
            script = {
                "type": "config_update",
                "config_path": parameters['config_path'],
                "changes": parameters['config_changes'],
                "format": parameters.get('config_format', 'yaml'),
                "description": f"Update configuration at {parameters['config_path']}",
                "timeout": 60,
                "validation": {
                    "type": "config_check",
                    "path": parameters['config_path'],
                    "expected_changes": parameters['config_changes']
                }
            }
            scripts.append(script)

        return scripts

    def _generate_restart_scripts(self, template: FixTemplate, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate service restart scripts"""
        return [{
            "type": "systemctl",
            "command": "restart",
            "args": [parameters['service_name']],
            "description": f"Restart service {parameters['service_name']}",
            "timeout": 120,
            "validation": {
                "command": "status",
                "args": [parameters['service_name']],
                "expected_status": "active"
            }
        }]

    def _generate_scale_scripts(self, template: FixTemplate, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate scaling scripts"""
        return [{
            "type": "kubectl",
            "command": "scale",
            "args": ["deployment", parameters['service_name'],
                    f"--replicas={parameters['target_replicas']}",
                    "-n", parameters['namespace']],
            "description": f"Scale {parameters['service_name']} to {parameters['target_replicas']} replicas",
            "timeout": 300
        }]

    def _generate_rollback_scripts(self, template: FixTemplate, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate rollback scripts"""
        rollback_scripts = []

        if template.fix_type == FixType.KUBERNETES_PATCH and template.id == "service_scale_up":
            # Rollback scaling
            rollback_scripts.append({
                "type": "kubectl",
                "command": "scale",
                "args": ["deployment", parameters['service_name'],
                        f"--replicas={parameters['current_replicas']}",
                        "-n", parameters['namespace']],
                "description": f"Rollback scaling of {parameters['service_name']} to {parameters['current_replicas']} replicas",
                "timeout": 300
            })

        elif template.fix_type == FixType.CONFIG_UPDATE:
            # Rollback config changes
            rollback_scripts.append({
                "type": "config_rollback",
                "config_path": parameters['config_path'],
                "original_config": parameters.get('original_config', {}),
                "description": f"Rollback configuration changes at {parameters['config_path']}",
                "timeout": 60
            })

        return rollback_scripts

    def _assess_risk(self, template: FixTemplate, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk of applying the fix"""
        risk_factors = []

        # Base risk from template
        if template.risk_level == RiskLevel.HIGH:
            risk_factors.append("High-risk operation")
        elif template.risk_level == RiskLevel.CRITICAL:
            risk_factors.append("Critical-risk operation")

        # Parameter-specific risks
        if template.id == "service_scale_up":
            target_replicas = parameters.get('target_replicas', 1)
            current_replicas = parameters.get('current_replicas', 1)

            if target_replicas > current_replicas * 3:
                risk_factors.append("Large scale increase may cause resource issues")

        elif template.id == "cache_invalidation":
            if not parameters.get('cache_keys'):
                risk_factors.append("Full cache invalidation may cause performance impact")

        return {
            "level": template.risk_level.value,
            "factors": risk_factors,
            "requires_approval": template.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL],
            "estimated_impact": len(risk_factors)
        }

    def _estimate_impact(self, template: FixTemplate, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate impact of the fix"""
        impact = {
            "scope": "service",  # Default to service-level impact
            "duration": template.estimated_duration.total_seconds(),
            "downtime_expected": False,
            "performance_impact": "low",
            "rollback_complexity": "low"
        }

        if template.fix_type == FixType.SERVICE_RESTART:
            impact.update({
                "downtime_expected": True,
                "performance_impact": "medium",
                "scope": "service"
            })

        elif template.fix_type == FixType.KUBERNETES_PATCH:
            impact.update({
                "downtime_expected": False,
                "performance_impact": "low",
                "scope": "deployment"
            })

        elif template.id == "cache_invalidation":
            impact.update({
                "performance_impact": "high",
                "rollback_complexity": "medium"
            })

        return impact

    def get_template_stats(self) -> Dict[str, Any]:
        """Get statistics about the fix catalog"""
        total_templates = len(self.templates)

        # Count by type
        type_counts = {}
        for template in self.templates.values():
            type_counts[template.fix_type.value] = type_counts.get(template.fix_type.value, 0) + 1

        # Count by risk level
        risk_counts = {}
        for template in self.templates.values():
            risk_counts[template.risk_level.value] = risk_counts.get(template.risk_level.value, 0) + 1

        # Success rate stats
        success_rates = [t.success_rate for t in self.templates.values()]

        return {
            "total_templates": total_templates,
            "by_type": type_counts,
            "by_risk": risk_counts,
            "avg_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0,
            "rollback_available": sum(1 for t in self.templates.values() if t.rollback_available)
        }
