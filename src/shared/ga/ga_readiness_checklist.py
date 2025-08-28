"""
GA (General Availability) readiness checklist
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

from ..database.config import get_postgres_session, get_clickhouse_session


logger = logging.getLogger(__name__)


class ChecklistCategory(Enum):
    """Categories for GA readiness checklist"""
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    MONITORING = "monitoring"
    DOCUMENTATION = "documentation"
    OPERATIONS = "operations"
    COMPLIANCE = "compliance"


class ChecklistItemStatus(Enum):
    """Status of a checklist item"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    DEFERRED = "deferred"


class ChecklistPriority(Enum):
    """Priority levels for checklist items"""
    CRITICAL = "critical"     # Must be completed for GA
    HIGH = "high"            # Should be completed for GA
    MEDIUM = "medium"        # Nice to have for GA
    LOW = "low"              # Can be deferred post-GA


@dataclass
class ChecklistItem:
    """Individual checklist item"""
    id: str
    title: str
    description: str
    category: ChecklistCategory
    priority: ChecklistPriority
    status: ChecklistItemStatus = ChecklistItemStatus.PENDING
    assigned_to: Optional[str] = None
    due_date: Optional[datetime] = None
    prerequisites: List[str] = field(default_factory=list)  # IDs of prerequisite items
    verification_steps: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    verified_by: Optional[str] = None


@dataclass
class GAReadinessAssessment:
    """Overall GA readiness assessment"""
    id: str
    assessment_date: datetime
    overall_score: float  # 0-100
    readiness_level: str  # "not_ready", "needs_work", "almost_ready", "ga_ready"
    critical_items_remaining: int
    high_priority_items_remaining: int
    completion_percentage: float
    estimated_ga_date: Optional[datetime] = None
    recommendations: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)


class GAReadinessChecklist:
    """Manager for GA readiness checklist and assessment"""

    def __init__(self):
        self.items: Dict[str, ChecklistItem] = {}
        self.assessments: List[GAReadinessAssessment] = []
        self.item_dependencies: Dict[str, Set[str]] = {}  # item_id -> dependent_item_ids

        # Setup default checklist items
        self._setup_default_checklist()

    def _setup_default_checklist(self):
        """Setup comprehensive default GA readiness checklist"""
        # Infrastructure items
        self._add_item(ChecklistItem(
            id="infra_production_environment",
            title="Production Environment Setup",
            description="Complete production environment configuration with proper networking, security groups, and access controls",
            category=ChecklistCategory.INFRASTRUCTURE,
            priority=ChecklistPriority.CRITICAL,
            verification_steps=[
                "Verify production Kubernetes cluster is configured",
                "Check network security groups and firewall rules",
                "Validate load balancer configuration",
                "Confirm DNS records are properly configured"
            ]
        ))

        self._add_item(ChecklistItem(
            id="infra_database_production",
            title="Production Database Setup",
            description="Production database cluster with backups, monitoring, and high availability",
            category=ChecklistCategory.INFRASTRUCTURE,
            priority=ChecklistPriority.CRITICAL,
            verification_steps=[
                "Verify database cluster is running in production",
                "Check backup schedules and retention policies",
                "Validate replication and failover mechanisms",
                "Confirm monitoring and alerting are active"
            ]
        ))

        self._add_item(ChecklistItem(
            id="infra_cdn_caching",
            title="CDN and Caching Setup",
            description="Content delivery network and caching layers configured for global performance",
            category=ChecklistCategory.INFRASTRUCTURE,
            priority=ChecklistPriority.HIGH,
            verification_steps=[
                "Verify CDN configuration and edge locations",
                "Check caching policies and invalidation strategies",
                "Validate SSL/TLS certificate configuration",
                "Confirm performance improvements with CDN"
            ]
        ))

        # Security items
        self._add_item(ChecklistItem(
            id="security_penetration_testing",
            title="Penetration Testing",
            description="Complete penetration testing and vulnerability assessment",
            category=ChecklistCategory.SECURITY,
            priority=ChecklistPriority.CRITICAL,
            verification_steps=[
                "Conduct comprehensive penetration testing",
                "Address all critical and high-severity findings",
                "Obtain security team sign-off",
                "Document security testing results"
            ]
        ))

        self._add_item(ChecklistItem(
            id="security_secrets_management",
            title="Secrets Management",
            description="Secure secrets management system with proper access controls and rotation",
            category=ChecklistCategory.SECURITY,
            priority=ChecklistPriority.CRITICAL,
            verification_steps=[
                "Verify secrets vault is configured and accessible",
                "Check secret rotation policies and schedules",
                "Validate access controls and audit logging",
                "Confirm secrets are not exposed in code or logs"
            ]
        ))

        self._add_item(ChecklistItem(
            id="security_compliance_audit",
            title="Compliance Audit",
            description="Complete security compliance audit (SOC 2, GDPR, HIPAA as applicable)",
            category=ChecklistCategory.SECURITY,
            priority=ChecklistPriority.HIGH,
            verification_steps=[
                "Conduct compliance audit with third-party firm",
                "Address all compliance gaps",
                "Obtain compliance certification",
                "Document compliance controls and procedures"
            ]
        ))

        # Performance items
        self._add_item(ChecklistItem(
            id="perf_load_testing",
            title="Load Testing",
            description="Complete load testing to validate performance under production load",
            category=ChecklistCategory.PERFORMANCE,
            priority=ChecklistPriority.CRITICAL,
            verification_steps=[
                "Execute comprehensive load testing scenarios",
                "Verify performance meets SLAs under peak load",
                "Check auto-scaling behavior under load",
                "Validate performance degradation is graceful"
            ]
        ))

        self._add_item(ChecklistItem(
            id="perf_optimization",
            title="Performance Optimization",
            description="Performance optimizations and query optimizations completed",
            category=ChecklistCategory.PERFORMANCE,
            priority=ChecklistPriority.HIGH,
            verification_steps=[
                "Optimize database queries and indexes",
                "Implement caching strategies",
                "Optimize API response times",
                "Reduce memory usage and improve garbage collection"
            ]
        ))

        # Reliability items
        self._add_item(ChecklistItem(
            id="reliability_disaster_recovery",
            title="Disaster Recovery Plan",
            description="Comprehensive disaster recovery plan with tested procedures",
            category=ChecklistCategory.RELIABILITY,
            priority=ChecklistPriority.CRITICAL,
            verification_steps=[
                "Document complete disaster recovery procedures",
                "Test disaster recovery procedures",
                "Verify RTO/RPO targets are achievable",
                "Confirm backup and restore processes work"
            ]
        ))

        self._add_item(ChecklistItem(
            id="reliability_rollback_procedures",
            title="Rollback Procedures",
            description="Tested rollback procedures for all deployment types",
            category=ChecklistCategory.RELIABILITY,
            priority=ChecklistPriority.CRITICAL,
            prerequisites=["perf_load_testing"],  # Depends on load testing
            verification_steps=[
                "Document rollback procedures for all services",
                "Test rollback procedures in staging",
                "Verify rollback completes within acceptable time",
                "Confirm data integrity after rollback"
            ]
        ))

        self._add_item(ChecklistItem(
            id="reliability_high_availability",
            title="High Availability Setup",
            description="High availability configuration across all critical components",
            category=ChecklistCategory.RELIABILITY,
            priority=ChecklistPriority.HIGH,
            verification_steps=[
                "Verify multi-zone/multi-region deployment",
                "Check load balancer health checks",
                "Validate failover mechanisms",
                "Confirm redundancy at all levels"
            ]
        ))

        # Monitoring items
        self._add_item(ChecklistItem(
            id="monitoring_comprehensive",
            title="Comprehensive Monitoring",
            description="Complete monitoring setup covering all services and dependencies",
            category=ChecklistCategory.MONITORING,
            priority=ChecklistPriority.CRITICAL,
            verification_steps=[
                "Verify all services have monitoring",
                "Check alerting rules are comprehensive",
                "Validate dashboards are created and working",
                "Confirm monitoring data retention policies"
            ]
        ))

        self._add_item(ChecklistItem(
            id="monitoring_slos_defined",
            title="SLOs and SLIs Defined",
            description="Service Level Objectives and Indicators defined and monitored",
            category=ChecklistCategory.MONITORING,
            priority=ChecklistPriority.HIGH,
            verification_steps=[
                "Define SLOs for all critical services",
                "Implement SLI monitoring",
                "Set up error budget tracking",
                "Create SLO dashboards and alerts"
            ]
        ))

        # Documentation items
        self._add_item(ChecklistItem(
            id="docs_api_documentation",
            title="API Documentation",
            description="Complete API documentation with examples and guides",
            category=ChecklistCategory.DOCUMENTATION,
            priority=ChecklistPriority.HIGH,
            verification_steps=[
                "Generate OpenAPI/Swagger documentation",
                "Create API usage examples",
                "Document authentication and authorization",
                "Publish documentation in accessible format"
            ]
        ))

        self._add_item(ChecklistItem(
            id="docs_runbooks",
            title="Operations Runbooks",
            description="Complete runbooks for common operations and troubleshooting",
            category=ChecklistCategory.DOCUMENTATION,
            priority=ChecklistPriority.CRITICAL,
            verification_steps=[
                "Document deployment procedures",
                "Create incident response runbooks",
                "Document troubleshooting procedures",
                "Create maintenance and upgrade guides"
            ]
        ))

        self._add_item(ChecklistItem(
            id="docs_user_guides",
            title="User Documentation",
            description="User guides, tutorials, and getting started documentation",
            category=ChecklistCategory.DOCUMENTATION,
            priority=ChecklistPriority.MEDIUM,
            verification_steps=[
                "Create getting started guide",
                "Write user tutorials and examples",
                "Document best practices",
                "Create FAQ and troubleshooting sections"
            ]
        ))

        # Operations items
        self._add_item(ChecklistItem(
            id="ops_deployment_pipeline",
            title="Production Deployment Pipeline",
            description="Robust CI/CD pipeline for production deployments",
            category=ChecklistCategory.OPERATIONS,
            priority=ChecklistPriority.CRITICAL,
            verification_steps=[
                "Verify CI/CD pipeline is working",
                "Check deployment automation",
                "Validate testing in deployment pipeline",
                "Confirm rollback capabilities in pipeline"
            ]
        ))

        self._add_item(ChecklistItem(
            id="ops_oncall_rotation",
            title="On-Call Rotation",
            description="Established on-call rotation and escalation procedures",
            category=ChecklistCategory.OPERATIONS,
            priority=ChecklistPriority.CRITICAL,
            verification_steps=[
                "Define on-call schedules and rotations",
                "Set up escalation procedures",
                "Test alert routing and notifications",
                "Document incident response procedures"
            ]
        ))

        # Compliance items
        self._add_item(ChecklistItem(
            id="compliance_data_privacy",
            title="Data Privacy Compliance",
            description="Compliance with data privacy regulations (GDPR, CCPA, etc.)",
            category=ChecklistCategory.COMPLIANCE,
            priority=ChecklistPriority.HIGH,
            verification_steps=[
                "Conduct data privacy audit",
                "Implement data subject access procedures",
                "Set up data retention policies",
                "Document compliance procedures"
            ]
        ))

        self._add_item(ChecklistItem(
            id="compliance_access_control",
            title="Access Control and Audit",
            description="Proper access control and audit logging in place",
            category=ChecklistCategory.COMPLIANCE,
            priority=ChecklistPriority.CRITICAL,
            verification_steps=[
                "Implement role-based access control",
                "Set up comprehensive audit logging",
                "Conduct access control review",
                "Verify audit logs are tamper-proof"
            ]
        ))

    def _add_item(self, item: ChecklistItem) -> None:
        """Add a checklist item and track dependencies"""
        self.items[item.id] = item

        # Track dependencies
        for prereq_id in item.prerequisites:
            if prereq_id not in self.item_dependencies:
                self.item_dependencies[prereq_id] = set()
            self.item_dependencies[prereq_id].add(item.id)

    def update_item_status(self, item_id: str, status: ChecklistItemStatus,
                          notes: str = "", updated_by: str = "system") -> bool:
        """
        Update the status of a checklist item

        Args:
            item_id: ID of the item to update
            status: New status
            notes: Optional notes about the update
            updated_by: User making the update

        Returns:
            True if update was successful
        """
        if item_id not in self.items:
            return False

        item = self.items[item_id]

        # Check prerequisites for completion
        if status == ChecklistItemStatus.COMPLETED:
            if not self._check_prerequisites_completed(item):
                return False
            item.completed_at = datetime.now()
            item.verified_by = updated_by

        item.status = status
        item.notes = notes
        item.updated_at = datetime.now()

        logger.info(f"Updated checklist item {item_id} to {status.value} by {updated_by}")

        return True

    def _check_prerequisites_completed(self, item: ChecklistItem) -> bool:
        """Check if all prerequisites are completed"""
        for prereq_id in item.prerequisites:
            prereq_item = self.items.get(prereq_id)
            if not prereq_item or prereq_item.status != ChecklistItemStatus.COMPLETED:
                return False
        return True

    def assign_item(self, item_id: str, assigned_to: str, due_date: Optional[datetime] = None) -> bool:
        """Assign a checklist item to a user"""
        if item_id not in self.items:
            return False

        item = self.items[item_id]
        item.assigned_to = assigned_to
        item.due_date = due_date
        item.updated_at = datetime.now()

        logger.info(f"Assigned checklist item {item_id} to {assigned_to}")
        return True

    def get_blocked_items(self) -> List[ChecklistItem]:
        """Get items that are blocked by incomplete prerequisites"""
        blocked_items = []

        for item in self.items.values():
            if item.status == ChecklistItemStatus.PENDING and item.prerequisites:
                if not self._check_prerequisites_completed(item):
                    blocked_items.append(item)

        return blocked_items

    def get_overdue_items(self) -> List[ChecklistItem]:
        """Get items that are overdue"""
        now = datetime.now()
        overdue_items = []

        for item in self.items.values():
            if (item.due_date and item.due_date < now and
                item.status not in [ChecklistItemStatus.COMPLETED]):
                overdue_items.append(item)

        return overdue_items

    def generate_assessment(self) -> GAReadinessAssessment:
        """
        Generate a comprehensive GA readiness assessment

        Returns:
            GA readiness assessment
        """
        assessment_id = f"assessment_{int(datetime.now().timestamp())}"

        # Calculate completion statistics
        total_items = len(self.items)
        completed_items = len([item for item in self.items.values()
                              if item.status == ChecklistItemStatus.COMPLETED])

        critical_items = [item for item in self.items.values()
                         if item.priority == ChecklistPriority.CRITICAL]
        critical_completed = len([item for item in critical_items
                                 if item.status == ChecklistItemStatus.COMPLETED])

        high_items = [item for item in self.items.values()
                     if item.priority == ChecklistPriority.HIGH]
        high_completed = len([item for item in high_items
                             if item.status == ChecklistItemStatus.COMPLETED])

        # Calculate overall score
        completion_percentage = (completed_items / total_items) * 100 if total_items > 0 else 0

        # Weight critical items more heavily
        critical_weight = 0.5
        high_weight = 0.3
        other_weight = 0.2

        critical_score = (critical_completed / len(critical_items)) * 100 if critical_items else 100
        high_score = (high_completed / len(high_items)) * 100 if high_items else 100
        other_items = [item for item in self.items.values()
                      if item.priority not in [ChecklistPriority.CRITICAL, ChecklistPriority.HIGH]]
        other_completed = len([item for item in other_items
                              if item.status == ChecklistItemStatus.COMPLETED])
        other_score = (other_completed / len(other_items)) * 100 if other_items else 100

        overall_score = (
            critical_score * critical_weight +
            high_score * high_weight +
            other_score * other_weight
        )

        # Determine readiness level
        if overall_score >= 95 and critical_score == 100:
            readiness_level = "ga_ready"
        elif overall_score >= 80 and critical_score >= 90:
            readiness_level = "almost_ready"
        elif overall_score >= 60:
            readiness_level = "needs_work"
        else:
            readiness_level = "not_ready"

        # Calculate estimated GA date
        estimated_ga_date = None
        if readiness_level != "not_ready":
            # Estimate based on remaining work
            remaining_critical = len(critical_items) - critical_completed
            remaining_high = len(high_items) - high_completed

            # Assume 2 days per critical item, 1 day per high item
            days_needed = (remaining_critical * 2) + remaining_high
            estimated_ga_date = datetime.now() + timedelta(days=days_needed)

        # Generate recommendations and risks
        recommendations, risks, blockers = self._generate_assessment_insights(
            critical_score, high_score, readiness_level
        )

        assessment = GAReadinessAssessment(
            id=assessment_id,
            assessment_date=datetime.now(),
            overall_score=overall_score,
            readiness_level=readiness_level,
            critical_items_remaining=len(critical_items) - critical_completed,
            high_priority_items_remaining=len(high_items) - high_completed,
            completion_percentage=completion_percentage,
            estimated_ga_date=estimated_ga_date,
            recommendations=recommendations,
            risks=risks,
            blockers=blockers
        )

        self.assessments.append(assessment)

        logger.info(f"Generated GA readiness assessment: {readiness_level} ({overall_score:.1f}%)")

        return assessment

    def _generate_assessment_insights(self, critical_score: float, high_score: float,
                                    readiness_level: str) -> tuple:
        """Generate assessment insights, recommendations, and risks"""
        recommendations = []
        risks = []
        blockers = []

        if critical_score < 100:
            blockers.append("Critical items must be completed before GA")
            recommendations.append("Prioritize completion of all critical checklist items")

        if high_score < 90:
            risks.append("High-priority items may impact production stability")
            recommendations.append("Complete remaining high-priority items")

        if readiness_level == "not_ready":
            risks.append("System is not ready for production deployment")
            recommendations.append("Focus on infrastructure and security items first")
        elif readiness_level == "needs_work":
            risks.append("Several areas need attention before GA")
            recommendations.append("Address monitoring and reliability gaps")
        elif readiness_level == "almost_ready":
            recommendations.append("Complete final testing and documentation")
        else:  # ga_ready
            recommendations.append("Schedule GA deployment and monitor closely")

        # Add specific recommendations based on common issues
        if not any(item.status == ChecklistItemStatus.COMPLETED and
                  item.category == ChecklistCategory.SECURITY
                  for item in self.items.values()):
            risks.append("Security readiness not confirmed")
            recommendations.append("Complete security review and penetration testing")

        if not any(item.status == ChecklistItemStatus.COMPLETED and
                  item.category == ChecklistCategory.RELIABILITY
                  for item in self.items.values()):
            risks.append("Reliability testing incomplete")
            recommendations.append("Execute disaster recovery drills and load testing")

        return recommendations, risks, blockers

    def get_items_by_category(self, category: ChecklistCategory) -> List[ChecklistItem]:
        """Get checklist items by category"""
        return [item for item in self.items.values() if item.category == category]

    def get_items_by_priority(self, priority: ChecklistPriority) -> List[ChecklistItem]:
        """Get checklist items by priority"""
        return [item for item in self.items.values() if item.priority == priority]

    def get_items_by_assignee(self, assignee: str) -> List[ChecklistItem]:
        """Get checklist items assigned to a specific user"""
        return [item for item in self.items.values() if item.assigned_to == assignee]

    def get_completion_summary(self) -> Dict[str, Any]:
        """Get completion summary by category and priority"""
        summary = {
            "by_category": {},
            "by_priority": {},
            "by_status": {},
            "total_items": len(self.items),
            "completed_items": 0,
            "in_progress_items": 0,
            "pending_items": 0,
            "blocked_items": 0
        }

        for item in self.items.values():
            # By category
            category = item.category.value
            if category not in summary["by_category"]:
                summary["by_category"][category] = {"total": 0, "completed": 0}
            summary["by_category"][category]["total"] += 1
            if item.status == ChecklistItemStatus.COMPLETED:
                summary["by_category"][category]["completed"] += 1

            # By priority
            priority = item.priority.value
            if priority not in summary["by_priority"]:
                summary["by_priority"][priority] = {"total": 0, "completed": 0}
            summary["by_priority"][priority]["total"] += 1
            if item.status == ChecklistItemStatus.COMPLETED:
                summary["by_priority"][priority]["completed"] += 1

            # By status
            status = item.status.value
            if status not in summary["by_status"]:
                summary["by_status"][status] = 0
            summary["by_status"][status] += 1

            # Overall counters
            if item.status == ChecklistItemStatus.COMPLETED:
                summary["completed_items"] += 1
            elif item.status == ChecklistItemStatus.IN_PROGRESS:
                summary["in_progress_items"] += 1
            elif item.status == ChecklistItemStatus.PENDING:
                summary["pending_items"] += 1
            elif item.status == ChecklistItemStatus.BLOCKED:
                summary["blocked_items"] += 1

        return summary

    def export_checklist(self, format: str = "json") -> str:
        """Export the complete checklist"""
        data = {
            "items": {
                item_id: {
                    "id": item.id,
                    "title": item.title,
                    "description": item.description,
                    "category": item.category.value,
                    "priority": item.priority.value,
                    "status": item.status.value,
                    "assigned_to": item.assigned_to,
                    "due_date": item.due_date.isoformat() if item.due_date else None,
                    "prerequisites": item.prerequisites,
                    "verification_steps": item.verification_steps,
                    "notes": item.notes,
                    "created_at": item.created_at.isoformat(),
                    "updated_at": item.updated_at.isoformat(),
                    "completed_at": item.completed_at.isoformat() if item.completed_at else None,
                    "verified_by": item.verified_by
                }
                for item_id, item in self.items.items()
            },
            "assessments": [
                {
                    "id": assessment.id,
                    "assessment_date": assessment.assessment_date.isoformat(),
                    "overall_score": assessment.overall_score,
                    "readiness_level": assessment.readiness_level,
                    "critical_items_remaining": assessment.critical_items_remaining,
                    "high_priority_items_remaining": assessment.high_priority_items_remaining,
                    "completion_percentage": assessment.completion_percentage,
                    "estimated_ga_date": assessment.estimated_ga_date.isoformat() if assessment.estimated_ga_date else None,
                    "recommendations": assessment.recommendations,
                    "risks": assessment.risks,
                    "blockers": assessment.blockers
                }
                for assessment in self.assessments
            ],
            "exported_at": datetime.now().isoformat(),
            "version": "1.0"
        }

        if format == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_upcoming_deadlines(self, days_ahead: int = 7) -> List[ChecklistItem]:
        """Get items with upcoming deadlines"""
        cutoff_date = datetime.now() + timedelta(days=days_ahead)

        upcoming_items = []
        for item in self.items.values():
            if (item.due_date and
                item.due_date <= cutoff_date and
                item.status != ChecklistItemStatus.COMPLETED):
                upcoming_items.append(item)

        return sorted(upcoming_items, key=lambda x: x.due_date)

    def get_recent_activity(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """Get recent activity on checklist items"""
        cutoff_date = datetime.now() - timedelta(days=days_back)

        recent_activity = []
        for item in self.items.values():
            if item.updated_at >= cutoff_date:
                recent_activity.append({
                    "item_id": item.id,
                    "title": item.title,
                    "status": item.status.value,
                    "updated_at": item.updated_at.isoformat(),
                    "updated_by": "system"  # Would track actual user in real implementation
                })

        return sorted(recent_activity, key=lambda x: x["updated_at"], reverse=True)

    def calculate_burndown(self) -> Dict[str, Any]:
        """Calculate checklist burndown metrics"""
        total_items = len(self.items)
        completed_items = len([item for item in self.items.values()
                              if item.status == ChecklistItemStatus.COMPLETED])

        # Group by priority
        critical_total = len([item for item in self.items.values()
                             if item.priority == ChecklistPriority.CRITICAL])
        critical_completed = len([item for item in self.items.values()
                                 if item.priority == ChecklistPriority.CRITICAL
                                 and item.status == ChecklistItemStatus.COMPLETED])

        high_total = len([item for item in self.items.values()
                         if item.priority == ChecklistPriority.HIGH])
        high_completed = len([item for item in self.items.values()
                             if item.priority == ChecklistPriority.HIGH
                             and item.status == ChecklistItemStatus.COMPLETED])

        return {
            "total_items": total_items,
            "completed_items": completed_items,
            "remaining_items": total_items - completed_items,
            "completion_percentage": (completed_items / total_items) * 100 if total_items > 0 else 0,
            "by_priority": {
                "critical": {
                    "total": critical_total,
                    "completed": critical_completed,
                    "remaining": critical_total - critical_completed
                },
                "high": {
                    "total": high_total,
                    "completed": high_completed,
                    "remaining": high_total - high_completed
                }
            },
            "calculated_at": datetime.now().isoformat()
        }


# Global GA readiness checklist instance
ga_readiness_checklist = GAReadinessChecklist()
