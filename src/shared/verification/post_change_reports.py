"""
Post-change verification reports for remediation analysis
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

from .canary_judge import CanaryResult
from .automated_rollback import RollbackExecution
from ..remediation.fix_catalog import GeneratedFix
from ..metrics.dora_metrics import DORAMetrics


logger = logging.getLogger(__name__)


class ReportSection(Enum):
    """Sections in post-change report"""
    EXECUTIVE_SUMMARY = "executive_summary"
    INCIDENT_OVERVIEW = "incident_overview"
    REMEDIATION_DETAILS = "remediation_details"
    CANARY_RESULTS = "canary_results"
    METRICS_ANALYSIS = "metrics_analysis"
    IMPACT_ASSESSMENT = "impact_assessment"
    DORA_METRICS = "dora_metrics"
    ROLLBACK_ANALYSIS = "rollback_analysis"
    RECOMMENDATIONS = "recommendations"
    LESSONS_LEARNED = "lessons_learned"


class ImpactLevel(Enum):
    """Impact level of the change"""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    CRITICAL = "critical"


@dataclass
class PostChangeReport:
    """Comprehensive post-change verification report"""
    id: str
    incident_id: str
    fix_id: str
    generated_at: datetime

    # Core components
    canary_result: Optional[CanaryResult] = None
    rollback_execution: Optional[RollbackExecution] = None
    original_fix: Optional[GeneratedFix] = None

    # Analysis data
    baseline_metrics: Dict[str, List[float]] = field(default_factory=dict)
    post_change_metrics: Dict[str, float] = field(default_factory=dict)
    impact_analysis: Dict[str, Any] = field(default_factory=dict)

    # Report sections
    sections: Dict[ReportSection, Dict[str, Any]] = field(default_factory=dict)

    # Metadata
    report_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def overall_success(self) -> bool:
        """Determine overall success of the change"""
        if self.canary_result:
            return self.canary_result.decision.value == "success"
        return False

    @property
    def impact_level(self) -> ImpactLevel:
        """Calculate overall impact level"""
        if not self.impact_analysis:
            return ImpactLevel.NONE

        # Calculate based on metrics changes and impact analysis
        max_change = max(
            abs(change) for change in self.impact_analysis.get("metric_changes", {}).values()
            if isinstance(change, (int, float))
        ) if self.impact_analysis.get("metric_changes") else 0

        if max_change > 0.5:  # 50% change
            return ImpactLevel.CRITICAL
        elif max_change > 0.25:  # 25% change
            return ImpactLevel.SIGNIFICANT
        elif max_change > 0.10:  # 10% change
            return ImpactLevel.MODERATE
        elif max_change > 0.05:  # 5% change
            return ImpactLevel.MINOR
        else:
            return ImpactLevel.NONE


class PostChangeReportGenerator:
    """Generator for comprehensive post-change verification reports"""

    def __init__(self):
        self.dora_metrics = DORAMetrics()
        self.reports: Dict[str, PostChangeReport] = {}

    async def generate_report(self, incident_id: str, fix_id: str,
                             canary_result: Optional[CanaryResult] = None,
                             rollback_execution: Optional[RollbackExecution] = None,
                             original_fix: Optional[GeneratedFix] = None) -> PostChangeReport:
        """
        Generate a comprehensive post-change verification report

        Args:
            incident_id: ID of the incident
            fix_id: ID of the applied fix
            canary_result: Results from canary analysis
            rollback_execution: Rollback execution details if applicable
            original_fix: Original fix that was applied

        Returns:
            Complete PostChangeReport
        """
        report_id = f"report_{incident_id}_{fix_id}_{int(datetime.now().timestamp())}"

        report = PostChangeReport(
            id=report_id,
            incident_id=incident_id,
            fix_id=fix_id,
            generated_at=datetime.now(),
            canary_result=canary_result,
            rollback_execution=rollback_execution,
            original_fix=original_fix
        )

        # Gather analysis data
        await self._gather_analysis_data(report)

        # Generate report sections
        await self._generate_all_sections(report)

        # Store report
        self.reports[report_id] = report

        logger.info(f"Generated post-change report {report_id}")

        return report

    async def _gather_analysis_data(self, report: PostChangeReport) -> None:
        """Gather data for report analysis"""
        # Collect baseline and post-change metrics
        report.baseline_metrics = await self._collect_baseline_metrics(report.incident_id)
        report.post_change_metrics = await self._collect_post_change_metrics()

        # Perform impact analysis
        report.impact_analysis = await self._perform_impact_analysis(
            report.baseline_metrics, report.post_change_metrics, report.canary_result
        )

        # Add metadata
        report.report_metadata = {
            "generated_by": "ai-devops-copilot",
            "version": "1.0.0",
            "data_sources": ["prometheus", "clickhouse", "kubernetes_api"],
            "analysis_window_hours": 24,
            "confidence_level": "high"
        }

    async def _generate_all_sections(self, report: PostChangeReport) -> None:
        """Generate all report sections"""
        section_generators = {
            ReportSection.EXECUTIVE_SUMMARY: self._generate_executive_summary,
            ReportSection.INCIDENT_OVERVIEW: self._generate_incident_overview,
            ReportSection.REMEDIATION_DETAILS: self._generate_remediation_details,
            ReportSection.CANARY_RESULTS: self._generate_canary_results,
            ReportSection.METRICS_ANALYSIS: self._generate_metrics_analysis,
            ReportSection.IMPACT_ASSESSMENT: self._generate_impact_assessment,
            ReportSection.DORA_METRICS: self._generate_dora_metrics,
            ReportSection.ROLLBACK_ANALYSIS: self._generate_rollback_analysis,
            ReportSection.RECOMMENDATIONS: self._generate_recommendations,
            ReportSection.LESSONS_LEARNED: self._generate_lessons_learned
        }

        for section_type, generator in section_generators.items():
            try:
                section_content = await generator(report)
                report.sections[section_type] = section_content
            except Exception as e:
                logger.error(f"Failed to generate {section_type.value} section: {e}")
                report.sections[section_type] = {"error": str(e)}

    async def _generate_executive_summary(self, report: PostChangeReport) -> Dict[str, Any]:
        """Generate executive summary section"""
        success_status = "✅ SUCCESS" if report.overall_success else "❌ FAILURE"
        impact_desc = self._get_impact_description(report.impact_level)

        summary = {
            "status": success_status,
            "impact_level": report.impact_level.value,
            "impact_description": impact_desc,
            "key_findings": self._extract_key_findings(report),
            "business_impact": self._assess_business_impact(report),
            "next_steps": self._recommend_next_steps(report),
            "confidence_level": self._calculate_confidence_level(report)
        }

        return summary

    async def _generate_incident_overview(self, report: PostChangeReport) -> Dict[str, Any]:
        """Generate incident overview section"""
        # This would query incident details from database
        incident_details = await self._get_incident_details(report.incident_id)

        return {
            "incident_id": report.incident_id,
            "title": incident_details.get("title", "Unknown Incident"),
            "severity": incident_details.get("severity", "unknown"),
            "start_time": incident_details.get("start_time"),
            "end_time": incident_details.get("end_time"),
            "duration": incident_details.get("duration_minutes", 0),
            "affected_services": incident_details.get("affected_services", []),
            "root_cause": incident_details.get("root_cause", "unknown"),
            "customer_impact": incident_details.get("customer_impact", "unknown")
        }

    async def _generate_remediation_details(self, report: PostChangeReport) -> Dict[str, Any]:
        """Generate remediation details section"""
        if not report.original_fix:
            return {"error": "No remediation details available"}

        return {
            "fix_id": report.fix_id,
            "fix_name": getattr(report.original_fix, 'name', 'Unknown Fix'),
            "fix_type": getattr(report.original_fix, 'fix_type', {}).get('value', 'unknown'),
            "risk_level": getattr(report.original_fix, 'risk_level', {}).get('value', 'unknown'),
            "applied_at": getattr(report.original_fix, 'generated_at', datetime.now()).isoformat(),
            "scripts_executed": len(getattr(report.original_fix, 'scripts', [])),
            "estimated_duration": getattr(report.original_fix, 'estimated_duration', timedelta()).total_seconds(),
            "rollback_available": getattr(report.original_fix, 'rollback_available', False),
            "success_criteria": getattr(report.original_fix, 'postconditions', [])
        }

    async def _generate_canary_results(self, report: PostChangeReport) -> Dict[str, Any]:
        """Generate canary results section"""
        if not report.canary_result:
            return {"status": "not_run", "message": "Canary analysis was not performed"}

        return {
            "decision": report.canary_result.decision.value,
            "confidence": report.canary_result.confidence,
            "metrics_improved": report.canary_result.metrics_improved,
            "metrics_degraded": report.canary_result.metrics_degraded,
            "metrics_stable": report.canary_result.metrics_stable,
            "analysis": report.canary_result.analysis,
            "evaluation_period_minutes": 30,  # Default canary window
            "recommendation": self._canary_recommendation(report.canary_result)
        }

    async def _generate_metrics_analysis(self, report: PostChangeReport) -> Dict[str, Any]:
        """Generate metrics analysis section"""
        analysis = {
            "baseline_period": "24 hours before incident",
            "post_change_period": "Current metrics",
            "key_metrics": {},
            "trends": {},
            "anomalies": []
        }

        # Analyze key metrics
        key_metrics = ["error_rate", "latency_p95", "success_rate", "throughput"]
        for metric in key_metrics:
            baseline_values = report.baseline_metrics.get(metric, [])
            current_value = report.post_change_metrics.get(metric, 0)

            if baseline_values:
                baseline_avg = statistics.mean(baseline_values)
                baseline_std = statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0

                change_percent = ((current_value - baseline_avg) / baseline_avg) if baseline_avg != 0 else 0
                z_score = abs(current_value - baseline_avg) / baseline_std if baseline_std > 0 else 0

                analysis["key_metrics"][metric] = {
                    "baseline_avg": baseline_avg,
                    "current_value": current_value,
                    "change_percent": change_percent,
                    "z_score": z_score,
                    "significant": z_score > 2.0
                }

                # Check for anomalies
                if z_score > 3.0:
                    analysis["anomalies"].append({
                        "metric": metric,
                        "severity": "high" if z_score > 5.0 else "medium",
                        "deviation": z_score,
                        "description": f"Unusual {metric} value: {change_percent:.1%} from baseline"
                    })

        return analysis

    async def _generate_impact_assessment(self, report: PostChangeReport) -> Dict[str, Any]:
        """Generate impact assessment section"""
        assessment = {
            "overall_impact": report.impact_level.value,
            "service_impact": {},
            "customer_impact": {},
            "business_impact": {},
            "recovery_time": None,
            "blast_radius": "unknown"
        }

        # Assess different types of impact
        if report.impact_analysis:
            assessment["service_impact"] = self._assess_service_impact(report.impact_analysis)
            assessment["customer_impact"] = self._assess_customer_impact(report.impact_analysis)
            assessment["business_impact"] = self._assess_business_impact(report.impact_analysis)

        # Calculate recovery time
        if report.rollback_execution and report.rollback_execution.rollback_duration:
            assessment["recovery_time"] = report.rollback_execution.rollback_duration.total_seconds()

        return assessment

    async def _generate_dora_metrics(self, report: PostChangeReport) -> Dict[str, Any]:
        """Generate DORA metrics section"""
        # Get DORA metrics for the relevant time period
        dora_data = await self.dora_metrics.get_metrics_for_incident(
            report.incident_id, days_before=7, days_after=1
        )

        return {
            "deployment_frequency": dora_data.get("deployment_frequency", 0),
            "lead_time_for_changes": dora_data.get("lead_time_for_changes", 0),
            "change_failure_rate": dora_data.get("change_failure_rate", 0),
            "time_to_restore_service": dora_data.get("time_to_restore_service", 0),
            "comparison_with_baseline": dora_data.get("comparison", {}),
            "dora_rating": self._calculate_dora_rating(dora_data)
        }

    async def _generate_rollback_analysis(self, report: PostChangeReport) -> Dict[str, Any]:
        """Generate rollback analysis section"""
        if not report.rollback_execution:
            return {"status": "not_required", "message": "No rollback was performed"}

        execution = report.rollback_execution

        analysis = {
            "rollback_performed": True,
            "success": execution.status.value == "completed",
            "duration_seconds": execution.rollback_duration.total_seconds() if execution.rollback_duration else 0,
            "scripts_executed": len(execution.script_results),
            "errors": execution.errors,
            "pre_rollback_metrics": execution.metrics_before,
            "post_rollback_metrics": execution.metrics_after,
            "recovery_effectiveness": self._assess_recovery_effectiveness(execution)
        }

        return analysis

    async def _generate_recommendations(self, report: PostChangeReport) -> Dict[str, Any]:
        """Generate recommendations section"""
        recommendations = {
            "immediate_actions": [],
            "monitoring_improvements": [],
            "process_improvements": [],
            "preventive_measures": []
        }

        # Generate recommendations based on report findings
        if not report.overall_success:
            recommendations["immediate_actions"].append(
                "Review and strengthen testing procedures for similar changes"
            )

        if report.canary_result and report.canary_result.decision.value == "failure":
            recommendations["monitoring_improvements"].append(
                "Implement more comprehensive canary monitoring"
            )

        if report.impact_level.value in ["significant", "critical"]:
            recommendations["process_improvements"].append(
                "Consider implementing change approval workflows for high-impact changes"
            )

        # Add general recommendations
        recommendations["preventive_measures"].extend([
            "Implement automated testing for rollback procedures",
            "Enhance monitoring coverage for critical metrics",
            "Establish regular disaster recovery drills"
        ])

        return recommendations

    async def _generate_lessons_learned(self, report: PostChangeReport) -> Dict[str, Any]:
        """Generate lessons learned section"""
        lessons = {
            "what_went_well": [],
            "what_could_be_improved": [],
            "preventive_actions": [],
            "follow_up_items": []
        }

        # Analyze what went well
        if report.overall_success:
            lessons["what_went_well"].append("Automated remediation successfully resolved the incident")

        if report.canary_result and report.canary_result.confidence > 0.8:
            lessons["what_went_well"].append("Canary analysis provided clear decision-making data")

        # Areas for improvement
        if report.rollback_execution and report.rollback_execution.errors:
            lessons["what_could_be_improved"].append("Rollback procedures need refinement")

        if report.impact_level.value == "critical":
            lessons["what_could_be_improved"].append("Impact assessment should trigger earlier intervention")

        # Follow-up items
        lessons["follow_up_items"].extend([
            "Schedule post-mortem meeting within 24 hours",
            "Update incident response runbooks with lessons learned",
            "Review monitoring alerts for similar incidents"
        ])

        return lessons

    def _extract_key_findings(self, report: PostChangeReport) -> List[str]:
        """Extract key findings from the report"""
        findings = []

        if report.overall_success:
            findings.append("✅ Remediation successfully resolved the incident")
        else:
            findings.append("❌ Remediation did not achieve desired outcome")

        if report.canary_result:
            confidence = report.canary_result.confidence
            findings.append(f"Canary analysis confidence: {confidence:.1%}")

        if report.impact_level != ImpactLevel.NONE:
            findings.append(f"Change impact level: {report.impact_level.value}")

        return findings

    def _assess_business_impact(self, report: PostChangeReport) -> Dict[str, Any]:
        """Assess business impact of the change"""
        impact = {
            "severity": "low",
            "affected_users": "unknown",
            "revenue_impact": "unknown",
            "brand_impact": "minimal",
            "description": ""
        }

        # Determine severity based on impact level
        if report.impact_level == ImpactLevel.CRITICAL:
            impact.update({
                "severity": "high",
                "brand_impact": "significant",
                "description": "Critical impact requiring immediate attention"
            })
        elif report.impact_level == ImpactLevel.SIGNIFICANT:
            impact.update({
                "severity": "medium",
                "brand_impact": "moderate",
                "description": "Significant impact affecting user experience"
            })

        return impact

    def _recommend_next_steps(self, report: PostChangeReport) -> List[str]:
        """Recommend next steps based on report findings"""
        next_steps = []

        if not report.overall_success:
            next_steps.append("Conduct thorough investigation of why remediation failed")
            next_steps.append("Implement alternative remediation strategy")

        if report.rollback_execution:
            next_steps.append("Verify system stability after rollback completion")

        next_steps.extend([
            "Schedule team debrief within 24 hours",
            "Update incident response documentation",
            "Monitor system for similar issues"
        ])

        return next_steps

    def _calculate_confidence_level(self, report: PostChangeReport) -> str:
        """Calculate overall confidence level in the report findings"""
        confidence_factors = []

        if report.canary_result:
            confidence_factors.append(report.canary_result.confidence)

        # Other confidence factors would be added here

        avg_confidence = statistics.mean(confidence_factors) if confidence_factors else 0.5

        if avg_confidence > 0.8:
            return "high"
        elif avg_confidence > 0.6:
            return "medium"
        else:
            return "low"

    def _get_impact_description(self, impact_level: ImpactLevel) -> str:
        """Get human-readable description of impact level"""
        descriptions = {
            ImpactLevel.NONE: "No measurable impact detected",
            ImpactLevel.MINOR: "Minor impact on system performance",
            ImpactLevel.MODERATE: "Moderate impact requiring monitoring",
            ImpactLevel.SIGNIFICANT: "Significant impact affecting user experience",
            ImpactLevel.CRITICAL: "Critical impact requiring immediate action"
        }
        return descriptions.get(impact_level, "Unknown impact level")

    def _canary_recommendation(self, canary_result: CanaryResult) -> str:
        """Generate recommendation based on canary results"""
        if canary_result.decision.value == "success":
            return "Proceed with full deployment"
        elif canary_result.decision.value == "failure":
            return "Rollback immediately and investigate"
        elif canary_result.decision.value == "rollback":
            return "Emergency rollback required"
        else:
            return "Continue monitoring and re-evaluate"

    def _assess_service_impact(self, impact_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess impact on services"""
        return {
            "affected_services": impact_analysis.get("affected_services", []),
            "service_degradation": impact_analysis.get("service_degradation", "unknown"),
            "recovery_status": impact_analysis.get("recovery_status", "unknown")
        }

    def _assess_customer_impact(self, impact_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess impact on customers"""
        return {
            "user_experience": impact_analysis.get("user_experience", "unknown"),
            "error_rate_increase": impact_analysis.get("error_rate_increase", 0),
            "latency_increase": impact_analysis.get("latency_increase", 0)
        }

    def _calculate_dora_rating(self, dora_data: Dict[str, Any]) -> str:
        """Calculate DORA performance rating"""
        # Simplified DORA rating calculation
        deploy_freq = dora_data.get("deployment_frequency", 0)
        lead_time = dora_data.get("lead_time_for_changes", 0)
        failure_rate = dora_data.get("change_failure_rate", 0)
        restore_time = dora_data.get("time_to_restore_service", 0)

        # Elite criteria (simplified)
        if (deploy_freq >= 1/7 and  # Multiple deploys per day
            lead_time <= 60 and     # Less than 1 hour
            failure_rate <= 0.15 and # Less than 15%
            restore_time <= 3600):  # Less than 1 hour
            return "Elite"
        elif (deploy_freq >= 1/30 and
              lead_time <= 1440 and
              failure_rate <= 0.30 and
              restore_time <= 14400):
            return "High"
        elif (deploy_freq >= 1/90 and
              lead_time <= 10080 and
              failure_rate <= 0.45 and
              restore_time <= 86400):
            return "Medium"
        else:
            return "Low"

    def _assess_recovery_effectiveness(self, rollback_execution: RollbackExecution) -> Dict[str, Any]:
        """Assess effectiveness of recovery/rollback"""
        effectiveness = {
            "recovery_successful": rollback_execution.status.value == "completed",
            "recovery_time_minutes": (
                rollback_execution.rollback_duration.total_seconds() / 60
                if rollback_execution.rollback_duration else 0
            ),
            "error_count": len(rollback_execution.errors),
            "metrics_improved": False
        }

        # Check if metrics improved after rollback
        if rollback_execution.metrics_before and rollback_execution.metrics_after:
            # Simple check: if error rate decreased
            before_error = rollback_execution.metrics_before.get("error_rate", 0)
            after_error = rollback_execution.metrics_after.get("error_rate", 0)
            effectiveness["metrics_improved"] = after_error < before_error

        return effectiveness

    async def _collect_baseline_metrics(self, incident_id: str) -> Dict[str, List[float]]:
        """Collect baseline metrics before incident"""
        # This would query historical metrics
        # For now, return mock data
        import random
        return {
            "error_rate": [random.uniform(0.01, 0.05) for _ in range(20)],
            "latency_p95": [random.uniform(0.1, 0.5) for _ in range(20)],
            "success_rate": [random.uniform(0.95, 0.99) for _ in range(20)],
            "throughput": [random.uniform(100, 200) for _ in range(20)]
        }

    async def _collect_post_change_metrics(self) -> Dict[str, float]:
        """Collect current post-change metrics"""
        # This would query current metrics
        # For now, return mock data
        import random
        return {
            "error_rate": random.uniform(0.01, 0.1),
            "latency_p95": random.uniform(0.1, 1.0),
            "success_rate": random.uniform(0.9, 0.99),
            "throughput": random.uniform(80, 220)
        }

    async def _perform_impact_analysis(self, baseline: Dict[str, List[float]],
                                     current: Dict[str, float],
                                     canary_result: Optional[CanaryResult]) -> Dict[str, Any]:
        """Perform comprehensive impact analysis"""
        analysis = {
            "metric_changes": {},
            "significant_changes": [],
            "overall_impact_score": 0,
            "affected_services": [],
            "service_degradation": "unknown"
        }

        for metric_name, baseline_values in baseline.items():
            current_value = current.get(metric_name, 0)

            if baseline_values:
                baseline_avg = statistics.mean(baseline_values)
                change_percent = ((current_value - baseline_avg) / baseline_avg) if baseline_avg != 0 else 0
                analysis["metric_changes"][metric_name] = change_percent

                # Check for significant changes
                if abs(change_percent) > 0.1:  # 10% change
                    analysis["significant_changes"].append({
                        "metric": metric_name,
                        "change_percent": change_percent,
                        "severity": "high" if abs(change_percent) > 0.25 else "medium"
                    })

        # Calculate overall impact score
        if analysis["metric_changes"]:
            max_change = max(abs(change) for change in analysis["metric_changes"].values())
            analysis["overall_impact_score"] = max_change

        return analysis

    async def _get_incident_details(self, incident_id: str) -> Dict[str, Any]:
        """Get incident details from database"""
        # This would query the incident database
        # For now, return mock data
        return {
            "title": f"Incident {incident_id}",
            "severity": "high",
            "start_time": (datetime.now() - timedelta(hours=2)).isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_minutes": 120,
            "affected_services": ["api-gateway", "user-service"],
            "root_cause": "Database connection pool exhaustion",
            "customer_impact": "Degraded response times for 15% of users"
        }

    def export_report(self, report_id: str, format: str = "json") -> str:
        """Export report in specified format"""
        if report_id not in self.reports:
            raise ValueError(f"Report {report_id} not found")

        report = self.reports[report_id]

        if format == "json":
            return json.dumps(self._report_to_dict(report), indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _report_to_dict(self, report: PostChangeReport) -> Dict[str, Any]:
        """Convert report to dictionary for export"""
        return {
            "id": report.id,
            "incident_id": report.incident_id,
            "fix_id": report.fix_id,
            "generated_at": report.generated_at.isoformat(),
            "overall_success": report.overall_success,
            "impact_level": report.impact_level.value,
            "sections": {section.value: content for section, content in report.sections.items()},
            "metadata": report.report_metadata
        }

    def get_report_summary(self, report_id: str) -> Dict[str, Any]:
        """Get a summary of the report"""
        if report_id not in self.reports:
            raise ValueError(f"Report {report_id} not found")

        report = self.reports[report_id]

        return {
            "report_id": report.id,
            "incident_id": report.incident_id,
            "success": report.overall_success,
            "impact_level": report.impact_level.value,
            "generated_at": report.generated_at.isoformat(),
            "sections_count": len(report.sections),
            "has_canary": report.canary_result is not None,
            "has_rollback": report.rollback_execution is not None
        }

    def get_reports_for_incident(self, incident_id: str) -> List[Dict[str, Any]]:
        """Get all reports for a specific incident"""
        incident_reports = [
            self.get_report_summary(report.id)
            for report in self.reports.values()
            if report.incident_id == incident_id
        ]

        return sorted(incident_reports, key=lambda x: x["generated_at"], reverse=True)
