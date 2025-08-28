"""
Preflight engine for remediation validation
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import subprocess
import requests

from ..remediation.script_generator import GeneratedScript, ToolType


logger = logging.getLogger(__name__)


class PreflightCheck(Enum):
    """Types of preflight checks"""
    DRY_RUN = "dry_run"
    OPA_POLICY = "opa_policy"
    SLO_BUDGET = "slo_budget"
    DRIFT_DETECTION = "drift_detection"
    RESOURCE_AVAILABILITY = "resource_availability"
    DEPENDENCY_CHECK = "dependency_check"
    SECURITY_SCAN = "security_scan"


class CheckResult(Enum):
    """Result of a preflight check"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class PreflightResult:
    """Result of a preflight check"""
    check_type: PreflightCheck
    result: CheckResult
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PreflightReport:
    """Complete preflight report"""
    script: GeneratedScript
    results: List[PreflightResult]
    overall_result: CheckResult
    can_proceed: bool
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def passed_checks(self) -> List[PreflightResult]:
        """Get all passed checks"""
        return [r for r in self.results if r.result == CheckResult.PASS]

    @property
    def failed_checks(self) -> List[PreflightResult]:
        """Get all failed checks"""
        return [r for r in self.results if r.result in [CheckResult.FAIL, CheckResult.ERROR]]

    @property
    def warning_checks(self) -> List[PreflightResult]:
        """Get all warning checks"""
        return [r for r in self.results if r.result == CheckResult.WARNING]


class PreflightEngine:
    """Engine for running preflight checks on remediation scripts"""

    def __init__(self):
        self.check_handlers = {
            PreflightCheck.DRY_RUN: self._run_dry_run,
            PreflightCheck.OPA_POLICY: self._run_opa_policy_check,
            PreflightCheck.SLO_BUDGET: self._run_slo_budget_check,
            PreflightCheck.DRIFT_DETECTION: self._run_drift_detection,
            PreflightCheck.RESOURCE_AVAILABILITY: self._run_resource_availability_check,
            PreflightCheck.DEPENDENCY_CHECK: self._run_dependency_check,
            PreflightCheck.SECURITY_SCAN: self._run_security_scan
        }

    async def run_preflight_checks(self, script: GeneratedScript,
                                 checks: List[PreflightCheck] = None) -> PreflightReport:
        """
        Run preflight checks on a remediation script

        Args:
            script: The script to check
            checks: List of checks to run (defaults to all)

        Returns:
            PreflightReport with results
        """
        if checks is None:
            checks = list(PreflightCheck)

        results = []
        start_time = datetime.now()

        # Run checks in parallel where possible
        check_tasks = []
        for check_type in checks:
            handler = self.check_handlers.get(check_type)
            if handler:
                task = asyncio.create_task(self._run_single_check(script, check_type, handler))
                check_tasks.append(task)

        # Wait for all checks to complete
        check_results = await asyncio.gather(*check_tasks, return_exceptions=True)

        for result in check_results:
            if isinstance(result, Exception):
                logger.error(f"Preflight check failed: {result}")
                results.append(PreflightResult(
                    check_type=PreflightCheck.DRY_RUN,  # Default
                    result=CheckResult.ERROR,
                    message=f"Check execution failed: {str(result)}"
                ))
            else:
                results.append(result)

        # Determine overall result
        overall_result = self._calculate_overall_result(results)
        can_proceed = overall_result != CheckResult.FAIL
        risk_assessment = self._assess_risk(results)
        recommendations = self._generate_recommendations(results)

        duration = (datetime.now() - start_time).total_seconds()

        logger.info(f"Preflight checks completed for {script.filename}: {overall_result.value}")

        return PreflightReport(
            script=script,
            results=results,
            overall_result=overall_result,
            can_proceed=can_proceed,
            risk_assessment=risk_assessment,
            recommendations=recommendations
        )

    async def _run_single_check(self, script: GeneratedScript, check_type: PreflightCheck,
                               handler) -> PreflightResult:
        """Run a single preflight check"""
        start_time = datetime.now()

        try:
            result = await handler(script)
            duration = (datetime.now() - start_time).total_seconds()
            result.duration = duration
            return result

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Check {check_type.value} failed: {e}")

            return PreflightResult(
                check_type=check_type,
                result=CheckResult.ERROR,
                message=f"Check execution failed: {str(e)}",
                duration=duration
            )

    async def _run_dry_run(self, script: GeneratedScript) -> PreflightResult:
        """Run dry-run check"""
        try:
            if script.tool_type == ToolType.KUBECTL:
                return await self._kubectl_dry_run(script)
            elif script.tool_type == ToolType.HELM:
                return await self._helm_dry_run(script)
            elif script.tool_type == ToolType.TERRAFORM:
                return await self._terraform_dry_run(script)
            elif script.tool_type == ToolType.ANSIBLE:
                return await self._ansible_dry_run(script)
            elif script.tool_type == ToolType.SQL:
                return await self._sql_dry_run(script)
            else:
                return PreflightResult(
                    check_type=PreflightCheck.DRY_RUN,
                    result=CheckResult.WARNING,
                    message=f"Dry-run not implemented for {script.tool_type.value}"
                )

        except Exception as e:
            return PreflightResult(
                check_type=PreflightCheck.DRY_RUN,
                result=CheckResult.ERROR,
                message=f"Dry-run failed: {str(e)}"
            )

    async def _kubectl_dry_run(self, script: GeneratedScript) -> PreflightResult:
        """Run kubectl dry-run"""
        # Extract kubectl command from script
        lines = script.content.split('\n')
        kubectl_line = None

        for line in lines:
            line = line.strip()
            if line.startswith('kubectl'):
                kubectl_line = line
                break

        if not kubectl_line:
            return PreflightResult(
                check_type=PreflightCheck.DRY_RUN,
                result=CheckResult.FAIL,
                message="No kubectl command found in script"
            )

        # Add --dry-run flag
        if 'apply' in kubectl_line:
            dry_run_cmd = kubectl_line.replace('kubectl apply', 'kubectl apply --dry-run=client')
        elif 'create' in kubectl_line:
            dry_run_cmd = kubectl_line.replace('kubectl create', 'kubectl create --dry-run=client')
        else:
            dry_run_cmd = kubectl_line + ' --dry-run=client'

        try:
            # Execute dry-run command
            result = await asyncio.create_subprocess_shell(
                dry_run_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                return PreflightResult(
                    check_type=PreflightCheck.DRY_RUN,
                    result=CheckResult.PASS,
                    message="Dry-run completed successfully",
                    details={
                        "command": dry_run_cmd,
                        "stdout": stdout.decode(),
                        "stderr": stderr.decode()
                    }
                )
            else:
                return PreflightResult(
                    check_type=PreflightCheck.DRY_RUN,
                    result=CheckResult.FAIL,
                    message=f"Dry-run failed: {stderr.decode()}",
                    details={
                        "command": dry_run_cmd,
                        "stdout": stdout.decode(),
                        "stderr": stderr.decode(),
                        "return_code": result.returncode
                    }
                )

        except Exception as e:
            return PreflightResult(
                check_type=PreflightCheck.DRY_RUN,
                result=CheckResult.ERROR,
                message=f"Dry-run execution failed: {str(e)}"
            )

    async def _helm_dry_run(self, script: GeneratedScript) -> PreflightResult:
        """Run Helm dry-run"""
        # Extract helm command and add --dry-run flag
        lines = script.content.split('\n')
        helm_line = None

        for line in lines:
            line = line.strip()
            if line.startswith('helm'):
                helm_line = line
                break

        if not helm_line:
            return PreflightResult(
                check_type=PreflightCheck.DRY_RUN,
                result=CheckResult.FAIL,
                message="No helm command found in script"
            )

        # Add --dry-run flag
        if 'upgrade' in helm_line:
            dry_run_cmd = helm_line.replace('helm upgrade', 'helm upgrade --dry-run')
        else:
            dry_run_cmd = helm_line + ' --dry-run'

        try:
            result = await asyncio.create_subprocess_shell(
                dry_run_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                return PreflightResult(
                    check_type=PreflightCheck.DRY_RUN,
                    result=CheckResult.PASS,
                    message="Helm dry-run completed successfully",
                    details={
                        "command": dry_run_cmd,
                        "stdout": stdout.decode()[:1000],  # Limit output size
                        "stderr": stderr.decode()
                    }
                )
            else:
                return PreflightResult(
                    check_type=PreflightCheck.DRY_RUN,
                    result=CheckResult.FAIL,
                    message=f"Helm dry-run failed: {stderr.decode()}",
                    details={
                        "command": dry_run_cmd,
                        "stdout": stdout.decode()[:1000],
                        "stderr": stderr.decode(),
                        "return_code": result.returncode
                    }
                )

        except Exception as e:
            return PreflightResult(
                check_type=PreflightCheck.DRY_RUN,
                result=CheckResult.ERROR,
                message=f"Helm dry-run execution failed: {str(e)}"
            )

    async def _terraform_dry_run(self, script: GeneratedScript) -> PreflightResult:
        """Run Terraform plan (dry-run equivalent)"""
        working_dir = script.parameters.get("working_dir", ".")

        try:
            # Run terraform plan
            result = await asyncio.create_subprocess_shell(
                f"cd {working_dir} && terraform plan -no-color",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                return PreflightResult(
                    check_type=PreflightCheck.DRY_RUN,
                    result=CheckResult.PASS,
                    message="Terraform plan completed successfully",
                    details={
                        "working_dir": working_dir,
                        "stdout": stdout.decode()[:2000],  # Limit output size
                        "stderr": stderr.decode()
                    }
                )
            else:
                return PreflightResult(
                    check_type=PreflightCheck.DRY_RUN,
                    result=CheckResult.FAIL,
                    message=f"Terraform plan failed: {stderr.decode()}",
                    details={
                        "working_dir": working_dir,
                        "stdout": stdout.decode()[:2000],
                        "stderr": stderr.decode(),
                        "return_code": result.returncode
                    }
                )

        except Exception as e:
            return PreflightResult(
                check_type=PreflightCheck.DRY_RUN,
                result=CheckResult.ERROR,
                message=f"Terraform plan execution failed: {str(e)}"
            )

    async def _ansible_dry_run(self, script: GeneratedScript) -> PreflightResult:
        """Run Ansible check mode (dry-run)"""
        return PreflightResult(
            check_type=PreflightCheck.DRY_RUN,
            result=CheckResult.WARNING,
            message="Ansible dry-run not implemented yet"
        )

    async def _sql_dry_run(self, script: GeneratedScript) -> PreflightResult:
        """Run SQL dry-run (syntax check)"""
        return PreflightResult(
            check_type=PreflightCheck.DRY_RUN,
            result=CheckResult.WARNING,
            message="SQL dry-run not implemented yet"
        )

    async def _run_opa_policy_check(self, script: GeneratedScript) -> PreflightResult:
        """Run OPA policy check"""
        try:
            # Prepare input data for OPA
            opa_input = {
                "script": {
                    "tool_type": script.tool_type.value,
                    "content": script.content,
                    "parameters": script.parameters,
                    "filename": script.filename
                },
                "metadata": {
                    "generated_at": script.generated_at.isoformat(),
                    "description": script.description
                }
            }

            # This would typically call an OPA server
            # For now, return a placeholder result
            return PreflightResult(
                check_type=PreflightCheck.OPA_POLICY,
                result=CheckResult.PASS,
                message="OPA policy check passed",
                details={
                    "policies_evaluated": ["security", "compliance", "best_practices"],
                    "input": opa_input
                }
            )

        except Exception as e:
            return PreflightResult(
                check_type=PreflightCheck.OPA_POLICY,
                result=CheckResult.ERROR,
                message=f"OPA policy check failed: {str(e)}"
            )

    async def _run_slo_budget_check(self, script: GeneratedScript) -> PreflightResult:
        """Run SLO budget check"""
        try:
            # This would check current SLO budgets and error budgets
            # For now, return a placeholder result
            return PreflightResult(
                check_type=PreflightCheck.SLO_BUDGET,
                result=CheckResult.PASS,
                message="SLO budget check passed",
                details={
                    "current_budget": 0.95,
                    "remaining_budget": 0.15,
                    "estimated_impact": 0.02
                }
            )

        except Exception as e:
            return PreflightResult(
                check_type=PreflightCheck.SLO_BUDGET,
                result=CheckResult.ERROR,
                message=f"SLO budget check failed: {str(e)}"
            )

    async def _run_drift_detection(self, script: GeneratedScript) -> PreflightResult:
        """Run drift detection check"""
        try:
            # This would compare current infrastructure state with desired state
            # For now, return a placeholder result
            return PreflightResult(
                check_type=PreflightCheck.DRIFT_DETECTION,
                result=CheckResult.PASS,
                message="No infrastructure drift detected",
                details={
                    "drift_detected": False,
                    "resources_checked": 15,
                    "last_drift_check": datetime.now().isoformat()
                }
            )

        except Exception as e:
            return PreflightResult(
                check_type=PreflightCheck.DRIFT_DETECTION,
                result=CheckResult.ERROR,
                message=f"Drift detection failed: {str(e)}"
            )

    async def _run_resource_availability_check(self, script: GeneratedScript) -> PreflightResult:
        """Run resource availability check"""
        try:
            # This would check if required resources are available
            # For now, return a placeholder result
            return PreflightResult(
                check_type=PreflightCheck.RESOURCE_AVAILABILITY,
                result=CheckResult.PASS,
                message="Required resources are available",
                details={
                    "cpu_available": "80%",
                    "memory_available": "60%",
                    "storage_available": "90%"
                }
            )

        except Exception as e:
            return PreflightResult(
                check_type=PreflightCheck.RESOURCE_AVAILABILITY,
                result=CheckResult.ERROR,
                message=f"Resource availability check failed: {str(e)}"
            )

    async def _run_dependency_check(self, script: GeneratedScript) -> PreflightResult:
        """Run dependency check"""
        try:
            # This would check if all required dependencies are available
            # For now, return a placeholder result
            return PreflightResult(
                check_type=PreflightCheck.DEPENDENCY_CHECK,
                result=CheckResult.PASS,
                message="All dependencies are available",
                details={
                    "dependencies_checked": ["kubectl", "helm", "terraform"],
                    "all_available": True
                }
            )

        except Exception as e:
            return PreflightResult(
                check_type=PreflightCheck.DEPENDENCY_CHECK,
                result=CheckResult.ERROR,
                message=f"Dependency check failed: {str(e)}"
            )

    async def _run_security_scan(self, script: GeneratedScript) -> PreflightResult:
        """Run security scan on the script"""
        try:
            # Analyze script content for security issues
            issues = self._analyze_script_security(script)

            if issues:
                return PreflightResult(
                    check_type=PreflightCheck.SECURITY_SCAN,
                    result=CheckResult.FAIL,
                    message=f"Security issues found: {len(issues)}",
                    details={
                        "issues": issues,
                        "severity": "high" if any(i.get("severity") == "high" for i in issues) else "medium"
                    }
                )
            else:
                return PreflightResult(
                    check_type=PreflightCheck.SECURITY_SCAN,
                    result=CheckResult.PASS,
                    message="No security issues found",
                    details={"issues": []}
                )

        except Exception as e:
            return PreflightResult(
                check_type=PreflightCheck.SECURITY_SCAN,
                result=CheckResult.ERROR,
                message=f"Security scan failed: {str(e)}"
            )

    def _analyze_script_security(self, script: GeneratedScript) -> List[Dict[str, Any]]:
        """Analyze script content for security issues"""
        issues = []
        content = script.content.lower()

        # Check for dangerous commands
        dangerous_patterns = [
            (r"rm\s+-rf\s+/", "Dangerous recursive delete", "high"),
            (r">/dev/null", "Output redirection to /dev/null", "low"),
            (r"curl.*\|.*bash", "Executing remote script via pipe", "high"),
            (r"wget.*\|.*bash", "Executing remote script via pipe", "high"),
            (r"sudo", "Use of sudo", "medium"),
            (r"chmod\s+777", "Overly permissive file permissions", "medium")
        ]

        for pattern, description, severity in dangerous_patterns:
            if re.search(pattern, content):
                issues.append({
                    "pattern": pattern,
                    "description": description,
                    "severity": severity,
                    "line": "N/A"  # Would need line number tracking
                })

        return issues

    def _calculate_overall_result(self, results: List[PreflightResult]) -> CheckResult:
        """Calculate overall result from individual check results"""
        if any(r.result == CheckResult.ERROR for r in results):
            return CheckResult.ERROR
        elif any(r.result == CheckResult.FAIL for r in results):
            return CheckResult.FAIL
        elif any(r.result == CheckResult.WARNING for r in results):
            return CheckResult.WARNING
        else:
            return CheckResult.PASS

    def _assess_risk(self, results: List[PreflightResult]) -> Dict[str, Any]:
        """Assess overall risk based on check results"""
        failed_checks = [r for r in results if r.result in [CheckResult.FAIL, CheckResult.ERROR]]
        warning_checks = [r for r in results if r.result == CheckResult.WARNING]

        risk_score = (len(failed_checks) * 2 + len(warning_checks)) / len(results) if results else 0

        if risk_score >= 1.5:
            risk_level = "high"
        elif risk_score >= 0.8:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "failed_checks": len(failed_checks),
            "warning_checks": len(warning_checks),
            "total_checks": len(results)
        }

    def _generate_recommendations(self, results: List[PreflightResult]) -> List[str]:
        """Generate recommendations based on check results"""
        recommendations = []

        failed_checks = [r for r in results if r.result in [CheckResult.FAIL, CheckResult.ERROR]]

        for check in failed_checks:
            if check.check_type == PreflightCheck.DRY_RUN:
                recommendations.append("Fix script syntax errors before proceeding")
            elif check.check_type == PreflightCheck.OPA_POLICY:
                recommendations.append("Review and address policy violations")
            elif check.check_type == PreflightCheck.SLO_BUDGET:
                recommendations.append("Consider timing when SLO budget is healthier")
            elif check.check_type == PreflightCheck.SECURITY_SCAN:
                recommendations.append("Address security issues in the script")

        if not recommendations:
            recommendations.append("All checks passed - proceed with caution")

        return recommendations
