"""
Automated rollback system for failed remediation attempts
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

from ..remediation.script_generator import ScriptGenerator, ToolType
from ..remediation.fix_catalog import GeneratedFix
from .canary_judge import CanaryDecision


logger = logging.getLogger(__name__)


class RollbackStatus(Enum):
    """Status of a rollback operation"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RollbackStrategy(Enum):
    """Strategy for rollback execution"""
    IMMEDIATE = "immediate"  # Rollback immediately
    GRADUAL = "gradual"     # Gradual rollback (e.g., reduce traffic)
    PHASED = "phased"       # Multi-phase rollback
    BLUE_GREEN = "blue_green"  # Switch back to previous version


@dataclass
class RollbackPlan:
    """Plan for rolling back a remediation fix"""
    id: str
    original_fix_id: str
    incident_id: str
    strategy: RollbackStrategy
    rollback_scripts: List[Dict[str, Any]]
    preconditions: List[str]
    postconditions: List[str]
    estimated_duration: timedelta
    risk_level: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RollbackExecution:
    """Execution record of a rollback operation"""
    id: str
    plan_id: str
    status: RollbackStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    script_results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    rollback_duration: Optional[timedelta] = None


class AutomatedRollback:
    """Intelligent automated rollback system"""

    def __init__(self):
        self.script_generator = ScriptGenerator()
        self.active_rollbacks: Dict[str, RollbackExecution] = {}
        self.rollback_history: Dict[str, RollbackExecution] = {}

        # Success rate tracking
        self.rollback_success_rate = 0.95  # Target: ≥ 95%

    async def create_rollback_plan(self, original_fix: GeneratedFix,
                                  incident_id: str) -> RollbackPlan:
        """
        Create a rollback plan for a given fix

        Args:
            original_fix: The original fix that may need rollback
            incident_id: ID of the incident being addressed

        Returns:
            RollbackPlan with rollback strategy and scripts
        """
        plan_id = f"rollback_{original_fix.template_id}_{int(datetime.now().timestamp())}"

        # Determine rollback strategy based on fix type and risk
        strategy = self._determine_rollback_strategy(original_fix)

        # Generate rollback scripts
        rollback_scripts = await self._generate_rollback_scripts(original_fix, strategy)

        # Define preconditions and postconditions
        preconditions = self._get_rollback_preconditions(original_fix, strategy)
        postconditions = self._get_rollback_postconditions(original_fix, strategy)

        # Estimate duration
        estimated_duration = self._estimate_rollback_duration(original_fix, strategy)

        # Assess risk
        risk_level = self._assess_rollback_risk(original_fix, strategy)

        plan = RollbackPlan(
            id=plan_id,
            original_fix_id=original_fix.template_id,
            incident_id=incident_id,
            strategy=strategy,
            rollback_scripts=rollback_scripts,
            preconditions=preconditions,
            postconditions=postconditions,
            estimated_duration=estimated_duration,
            risk_level=risk_level
        )

        logger.info(f"Created rollback plan {plan_id} for fix {original_fix.template_id}")

        return plan

    def _determine_rollback_strategy(self, fix: GeneratedFix) -> RollbackStrategy:
        """Determine the appropriate rollback strategy"""
        # Strategy based on fix type and risk
        if fix.risk_level.value in ["high", "critical"]:
            return RollbackStrategy.PHASED
        elif fix.fix_type.value in ["kubernetes_patch", "service_restart"]:
            return RollbackStrategy.BLUE_GREEN
        elif "scale" in fix.template_id:
            return RollbackStrategy.GRADUAL
        else:
            return RollbackStrategy.IMMEDIATE

    async def _generate_rollback_scripts(self, fix: GeneratedFix,
                                       strategy: RollbackStrategy) -> List[Dict[str, Any]]:
        """Generate rollback scripts for the given fix"""
        rollback_scripts = []

        # Generate rollback scripts using the script generator
        for script_data in fix.rollback_scripts:
            try:
                # Create rollback script from original script data
                rollback_script = await self._create_rollback_script(script_data, strategy)
                rollback_scripts.append(rollback_script)
            except Exception as e:
                logger.error(f"Failed to generate rollback script: {e}")
                continue

        # If no rollback scripts available, generate emergency rollback
        if not rollback_scripts:
            emergency_script = await self._generate_emergency_rollback(fix)
            if emergency_script:
                rollback_scripts.append(emergency_script)

        return rollback_scripts

    async def _create_rollback_script(self, script_data: Dict[str, Any],
                                    strategy: RollbackStrategy) -> Dict[str, Any]:
        """Create a rollback script from original script data"""
        tool_type = ToolType(script_data.get("type", "shell"))

        if tool_type == ToolType.KUBECTL:
            return await self._create_kubectl_rollback_script(script_data, strategy)
        elif tool_type == ToolType.HELM:
            return await self._create_helm_rollback_script(script_data, strategy)
        elif tool_type == ToolType.TERRAFORM:
            return await self._create_terraform_rollback_script(script_data, strategy)
        elif tool_type == ToolType.ANSIBLE:
            return await self._create_ansible_rollback_script(script_data, strategy)
        else:
            return await self._create_shell_rollback_script(script_data, strategy)

    async def _create_kubectl_rollback_script(self, script_data: Dict[str, Any],
                                            strategy: RollbackStrategy) -> Dict[str, Any]:
        """Create kubectl rollback script"""
        if strategy == RollbackStrategy.BLUE_GREEN:
            # Blue-green rollback: switch service to previous version
            script = {
                "type": "kubectl",
                "command": "set",
                "args": ["image", "deployment/my-app", "my-app=previous-version"],
                "description": "Rollback deployment to previous image version",
                "timeout": 300,
                "strategy": strategy.value
            }
        elif strategy == RollbackStrategy.PHASED:
            # Phased rollback: gradual traffic shift
            script = {
                "type": "kubectl",
                "command": "patch",
                "args": ["deployment/my-app", "-p", '{"spec":{"replicas":0}}'],
                "description": "Scale down deployment to zero replicas",
                "timeout": 600,
                "strategy": strategy.value
            }
        else:
            # Immediate rollback
            script = {
                "type": "kubectl",
                "command": "rollout",
                "args": ["undo", "deployment/my-app"],
                "description": "Undo last deployment rollout",
                "timeout": 300,
                "strategy": strategy.value
            }

        return script

    async def _create_helm_rollback_script(self, script_data: Dict[str, Any],
                                         strategy: RollbackStrategy) -> Dict[str, Any]:
        """Create Helm rollback script"""
        script = {
            "type": "helm",
            "command": "rollback",
            "args": ["my-release", "0"],  # Rollback to previous revision
            "description": "Rollback Helm release to previous version",
            "timeout": 600,
            "strategy": strategy.value
        }

        return script

    async def _create_terraform_rollback_script(self, script_data: Dict[str, Any],
                                              strategy: RollbackStrategy) -> Dict[str, Any]:
        """Create Terraform rollback script"""
        script = {
            "type": "terraform",
            "command": "destroy",
            "args": ["-auto-approve"],
            "working_dir": script_data.get("working_dir", "."),
            "description": "Destroy Terraform resources to rollback changes",
            "timeout": 900,
            "strategy": strategy.value
        }

        return script

    async def _create_ansible_rollback_script(self, script_data: Dict[str, Any],
                                            strategy: RollbackStrategy) -> Dict[str, Any]:
        """Create Ansible rollback script"""
        script = {
            "type": "ansible-playbook",
            "playbook": "rollback.yml",
            "inventory": script_data.get("inventory", "localhost,"),
            "description": "Execute rollback Ansible playbook",
            "timeout": 600,
            "strategy": strategy.value
        }

        return script

    async def _create_shell_rollback_script(self, script_data: Dict[str, Any],
                                          strategy: RollbackStrategy) -> Dict[str, Any]:
        """Create shell rollback script"""
        script = {
            "type": "shell",
            "command": "bash",
            "args": ["-c", "echo 'Rollback command here'"],
            "description": "Execute shell rollback commands",
            "timeout": 300,
            "strategy": strategy.value
        }

        return script

    async def _generate_emergency_rollback(self, fix: GeneratedFix) -> Optional[Dict[str, Any]]:
        """Generate emergency rollback script when no specific rollback is available"""
        # Generic emergency rollback - restart services
        if "kubernetes" in fix.template_id or "k8s" in fix.template_id:
            return {
                "type": "kubectl",
                "command": "rollout",
                "args": ["restart", "deployment"],
                "description": "Emergency: Restart all deployments",
                "timeout": 300,
                "emergency": True
            }
        elif "database" in fix.template_id:
            return {
                "type": "shell",
                "command": "systemctl",
                "args": ["restart", "postgresql"],
                "description": "Emergency: Restart database service",
                "timeout": 120,
                "emergency": True
            }

        return None

    def _get_rollback_preconditions(self, fix: GeneratedFix,
                                   strategy: RollbackStrategy) -> List[str]:
        """Get preconditions for rollback execution"""
        preconditions = [
            "System monitoring is operational",
            "Backup systems are available",
            "Rollback scripts are validated",
            "Required permissions are available"
        ]

        if strategy == RollbackStrategy.BLUE_GREEN:
            preconditions.extend([
                "Previous version is still available",
                "Traffic switching mechanism is working",
                "Load balancer configuration is accessible"
            ])
        elif strategy == RollbackStrategy.PHASED:
            preconditions.extend([
                "Gradual rollout capability is available",
                "Traffic distribution controls are working"
            ])

        return preconditions

    def _get_rollback_postconditions(self, fix: GeneratedFix,
                                    strategy: RollbackStrategy) -> List[str]:
        """Get postconditions for rollback verification"""
        postconditions = [
            "System health metrics are within normal ranges",
            "Application is responding to requests",
            "No new errors are being generated",
            "Previous issue symptoms are resolved"
        ]

        if "kubernetes" in fix.template_id:
            postconditions.extend([
                "All pods are in Ready state",
                "Services are accessible",
                "Network policies are correctly applied"
            ])

        return postconditions

    def _estimate_rollback_duration(self, fix: GeneratedFix,
                                  strategy: RollbackStrategy) -> timedelta:
        """Estimate rollback execution duration"""
        base_duration = timedelta(minutes=5)

        # Adjust based on strategy
        if strategy == RollbackStrategy.IMMEDIATE:
            multiplier = 1.0
        elif strategy == RollbackStrategy.GRADUAL:
            multiplier = 2.0
        elif strategy == RollbackStrategy.PHASED:
            multiplier = 3.0
        elif strategy == RollbackStrategy.BLUE_GREEN:
            multiplier = 1.5
        else:
            multiplier = 1.0

        # Adjust based on fix type
        if "kubernetes" in fix.template_id:
            type_multiplier = 1.2
        elif "database" in fix.template_id:
            type_multiplier = 1.5
        elif "infrastructure" in fix.template_id:
            type_multiplier = 2.0
        else:
            type_multiplier = 1.0

        return base_duration * multiplier * type_multiplier

    def _assess_rollback_risk(self, fix: GeneratedFix,
                             strategy: RollbackStrategy) -> str:
        """Assess risk level of rollback operation"""
        risk_score = 0

        # Base risk from original fix
        if fix.risk_level.value == "critical":
            risk_score += 2
        elif fix.risk_level.value == "high":
            risk_score += 1

        # Risk from strategy
        if strategy == RollbackStrategy.IMMEDIATE:
            risk_score += 1
        elif strategy == RollbackStrategy.BLUE_GREEN:
            risk_score += 0  # Lowest risk

        # Risk from fix type
        if "database" in fix.template_id:
            risk_score += 1
        elif "network" in fix.template_id:
            risk_score += 1

        if risk_score >= 3:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"

    async def execute_rollback(self, plan: RollbackPlan,
                              canary_decision: CanaryDecision = None) -> RollbackExecution:
        """
        Execute a rollback plan

        Args:
            plan: The rollback plan to execute
            canary_decision: Optional canary decision that triggered rollback

        Returns:
            RollbackExecution record
        """
        execution_id = f"exec_{plan.id}_{int(datetime.now().timestamp())}"

        execution = RollbackExecution(
            id=execution_id,
            plan_id=plan.id,
            status=RollbackStatus.RUNNING,
            start_time=datetime.now(),
            success_criteria=plan.postconditions
        )

        self.active_rollbacks[execution_id] = execution

        try:
            # Capture pre-rollback metrics
            execution.metrics_before = await self._capture_system_metrics()

            # Validate preconditions
            await self._validate_preconditions(plan)

            # Execute rollback scripts
            script_results = []
            for script in plan.rollback_scripts:
                result = await self._execute_rollback_script(script)
                script_results.append(result)

                # Check if script failed
                if not result.get("success", False):
                    execution.errors.append(f"Script failed: {result.get('error', 'Unknown error')}")
                    if not self._should_continue_on_failure(plan.strategy):
                        break

            execution.script_results = script_results

            # Wait for stabilization
            await self._wait_for_stabilization(plan)

            # Capture post-rollback metrics
            execution.metrics_after = await self._capture_system_metrics()

            # Verify success criteria
            success = await self._verify_success_criteria(plan, execution)

            execution.end_time = datetime.now()
            execution.rollback_duration = execution.end_time - execution.start_time

            if success and not execution.errors:
                execution.status = RollbackStatus.COMPLETED
                logger.info(f"Rollback {execution_id} completed successfully")
            else:
                execution.status = RollbackStatus.FAILED
                logger.error(f"Rollback {execution_id} failed")

        except Exception as e:
            execution.status = RollbackStatus.FAILED
            execution.errors.append(str(e))
            execution.end_time = datetime.now()
            if execution.start_time:
                execution.rollback_duration = execution.end_time - execution.start_time

            logger.error(f"Rollback {execution_id} failed with exception: {e}")

        # Move to history
        self.rollback_history[execution_id] = execution
        if execution_id in self.active_rollbacks:
            del self.active_rollbacks[execution_id]

        # Update success rate tracking
        self._update_success_rate()

        return execution

    async def _validate_preconditions(self, plan: RollbackPlan) -> None:
        """Validate rollback preconditions"""
        for precondition in plan.preconditions:
            # Implement precondition checks
            logger.debug(f"Validating precondition: {precondition}")

        # Add a small delay to simulate validation
        await asyncio.sleep(1)

    async def _execute_rollback_script(self, script: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a rollback script"""
        try:
            script_type = script.get("type", "shell")

            if script_type == "kubectl":
                result = await self._execute_kubectl_script(script)
            elif script_type == "helm":
                result = await self._execute_helm_script(script)
            elif script_type == "terraform":
                result = await self._execute_terraform_script(script)
            else:
                result = await self._execute_shell_script(script)

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "script_type": script.get("type"),
                "execution_time": datetime.now().isoformat()
            }

    def _should_continue_on_failure(self, strategy: RollbackStrategy) -> bool:
        """Determine if rollback should continue after script failure"""
        # For phased rollbacks, continue to maintain partial functionality
        return strategy in [RollbackStrategy.PHASED, RollbackStrategy.GRADUAL]

    async def _wait_for_stabilization(self, plan: RollbackPlan) -> None:
        """Wait for system to stabilize after rollback"""
        # Wait based on estimated duration
        wait_time = min(plan.estimated_duration.total_seconds(), 300)  # Max 5 minutes
        await asyncio.sleep(wait_time)

    async def _verify_success_criteria(self, plan: RollbackPlan,
                                     execution: RollbackExecution) -> bool:
        """Verify rollback success criteria"""
        success = True

        for criterion in plan.postconditions:
            # Implement criterion verification
            logger.debug(f"Verifying criterion: {criterion}")

            # For now, assume criteria are met if no errors
            if execution.errors:
                success = False
                break

        return success

    async def _capture_system_metrics(self) -> Dict[str, float]:
        """Capture current system metrics"""
        # This would integrate with monitoring systems
        # For now, return mock data
        import random
        return {
            "cpu_usage": random.uniform(0.1, 0.8),
            "memory_usage": random.uniform(0.2, 0.9),
            "error_rate": random.uniform(0.0, 0.1),
            "response_time": random.uniform(0.1, 2.0)
        }

    async def _execute_kubectl_script(self, script: Dict[str, Any]) -> Dict[str, Any]:
        """Execute kubectl rollback script"""
        # Mock execution for now
        await asyncio.sleep(2)  # Simulate execution time
        return {
            "success": True,
            "output": "kubectl command executed successfully",
            "execution_time": datetime.now().isoformat()
        }

    async def _execute_helm_script(self, script: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Helm rollback script"""
        # Mock execution for now
        await asyncio.sleep(3)
        return {
            "success": True,
            "output": "Helm rollback completed",
            "execution_time": datetime.now().isoformat()
        }

    async def _execute_terraform_script(self, script: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Terraform rollback script"""
        # Mock execution for now
        await asyncio.sleep(5)
        return {
            "success": True,
            "output": "Terraform destroy completed",
            "execution_time": datetime.now().isoformat()
        }

    async def _execute_shell_script(self, script: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shell rollback script"""
        # Mock execution for now
        await asyncio.sleep(1)
        return {
            "success": True,
            "output": "Shell script executed",
            "execution_time": datetime.now().isoformat()
        }

    def _update_success_rate(self) -> None:
        """Update rollback success rate tracking"""
        recent_rollbacks = [
            exec for exec in list(self.rollback_history.values())[-20:]  # Last 20 rollbacks
        ]

        if recent_rollbacks:
            successful = sum(1 for r in recent_rollbacks if r.status == RollbackStatus.COMPLETED)
            self.rollback_success_rate = successful / len(recent_rollbacks)

    def get_rollback_success_rate(self) -> float:
        """Get current rollback success rate"""
        return self.rollback_success_rate

    def get_active_rollbacks(self) -> List[RollbackExecution]:
        """Get currently active rollback executions"""
        return list(self.active_rollbacks.values())

    def get_rollback_history(self, limit: int = 50) -> List[RollbackExecution]:
        """Get rollback execution history"""
        return list(self.rollback_history.values())[-limit:]

    def get_rollback_stats(self) -> Dict[str, Any]:
        """Get rollback statistics"""
        all_executions = list(self.rollback_history.values())

        if not all_executions:
            return {
                "total_rollbacks": 0,
                "success_rate": 0.0,
                "avg_duration_seconds": 0,
                "by_strategy": {},
                "by_status": {}
            }

        successful = [r for r in all_executions if r.status == RollbackStatus.COMPLETED]
        failed = [r for r in all_executions if r.status == RollbackStatus.FAILED]

        # Duration stats
        completed_rollbacks = [r for r in all_executions if r.rollback_duration]
        avg_duration = (
            sum(r.rollback_duration.total_seconds() for r in completed_rollbacks) /
            len(completed_rollbacks)
        ) if completed_rollbacks else 0

        # Group by strategy
        strategy_counts = {}
        for rollback in all_executions:
            # This would need to be implemented to track strategy
            strategy = "unknown"  # Placeholder
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        # Group by status
        status_counts = {}
        for rollback in all_executions:
            status_counts[rollback.status.value] = status_counts.get(rollback.status.value, 0) + 1

        return {
            "total_rollbacks": len(all_executions),
            "success_rate": len(successful) / len(all_executions),
            "avg_duration_seconds": avg_duration,
            "by_strategy": strategy_counts,
            "by_status": status_counts,
            "recent_success_rate": self.rollback_success_rate
        }
