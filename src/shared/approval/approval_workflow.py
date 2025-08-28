"""
Approval workflow system for remediation scripts
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

from ..preflight.preflight_engine import PreflightReport


logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of an approval request"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalType(Enum):
    """Type of approval required"""
    AUTOMATIC = "automatic"  # No approval needed
    PEER_REVIEW = "peer_review"  # Any team member can approve
    MANAGER_APPROVAL = "manager_approval"  # Manager approval required
    SECURITY_REVIEW = "security_review"  # Security team approval
    EXECUTIVE_APPROVAL = "executive_approval"  # Executive approval for critical changes


@dataclass
class ApprovalRequest:
    """Approval request for a remediation script"""
    id: str
    script_id: str
    script_name: str
    requester_id: str
    requester_name: str
    approval_type: ApprovalType
    status: ApprovalStatus
    risk_level: str
    description: str
    justification: str
    preflight_report: PreflightReport
    required_approvers: List[str] = field(default_factory=list)
    approved_by: List[Dict[str, Any]] = field(default_factory=list)
    rejected_by: List[Dict[str, Any]] = field(default_factory=list)
    comments: List[Dict[str, Any]] = field(default_factory=list)
    expires_at: datetime
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def is_expired(self) -> bool:
        """Check if the approval request has expired"""
        return datetime.now() > self.expires_at

    @property
    def approval_progress(self) -> Dict[str, Any]:
        """Get approval progress information"""
        total_required = len(self.required_approvers) if self.required_approvers else 1
        approved_count = len(self.approved_by)
        rejected_count = len(self.rejected_by)

        return {
            "total_required": total_required,
            "approved_count": approved_count,
            "rejected_count": rejected_count,
            "progress_percentage": (approved_count / total_required) * 100 if total_required > 0 else 100,
            "is_complete": approved_count >= total_required,
            "has_rejections": rejected_count > 0
        }

    def can_be_approved_by(self, user_id: str, user_role: str) -> bool:
        """Check if a user can approve this request"""
        if self.status != ApprovalStatus.PENDING:
            return False

        if self.is_expired:
            return False

        # Check if user has already acted on this request
        has_approved = any(approval["user_id"] == user_id for approval in self.approved_by)
        has_rejected = any(rejection["user_id"] == user_id for rejection in self.rejected_by)

        if has_approved or has_rejected:
            return False

        # Check approval type requirements
        if self.approval_type == ApprovalType.AUTOMATIC:
            return True
        elif self.approval_type == ApprovalType.PEER_REVIEW:
            return True  # Any authenticated user can approve
        elif self.approval_type == ApprovalType.MANAGER_APPROVAL:
            return user_role in ["manager", "admin"]
        elif self.approval_type == ApprovalType.SECURITY_REVIEW:
            return user_role in ["security", "admin"]
        elif self.approval_type == ApprovalType.EXECUTIVE_APPROVAL:
            return user_role in ["executive", "admin"]

        return False


@dataclass
class ApprovalAction:
    """An approval or rejection action"""
    request_id: str
    user_id: str
    user_name: str
    action: str  # 'approve' or 'reject'
    comments: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ApprovalWorkflowEngine:
    """Engine for managing approval workflows"""

    def __init__(self):
        self.pending_requests: Dict[str, ApprovalRequest] = {}
        self.completed_requests: Dict[str, ApprovalRequest] = {}

        # Default approval policies by risk level
        self.approval_policies = {
            "low": ApprovalType.AUTOMATIC,
            "medium": ApprovalType.PEER_REVIEW,
            "high": ApprovalType.MANAGER_APPROVAL,
            "critical": ApprovalType.EXECUTIVE_APPROVAL
        }

        # Default expiration times
        self.expiration_times = {
            "low": timedelta(hours=24),
            "medium": timedelta(hours=12),
            "high": timedelta(hours=4),
            "critical": timedelta(hours=1)
        }

    def create_approval_request(self, script_id: str, script_name: str, requester_id: str,
                               requester_name: str, risk_level: str, description: str,
                               justification: str, preflight_report: PreflightReport,
                               custom_approval_type: Optional[ApprovalType] = None) -> ApprovalRequest:
        """
        Create a new approval request

        Args:
            script_id: ID of the script to approve
            script_name: Name of the script
            requester_id: ID of the user requesting approval
            requester_name: Name of the user requesting approval
            risk_level: Risk level of the script
            description: Description of what the script does
            justification: Why this script needs to be executed
            preflight_report: Preflight check results
            custom_approval_type: Override default approval type

        Returns:
            ApprovalRequest object
        """
        # Determine approval type
        approval_type = custom_approval_type or self.approval_policies.get(risk_level, ApprovalType.PEER_REVIEW)

        # Determine required approvers based on type
        required_approvers = self._get_required_approvers(approval_type, risk_level)

        # Calculate expiration time
        expiration_delta = self.expiration_times.get(risk_level, timedelta(hours=24))
        expires_at = datetime.now() + expiration_delta

        # Create request
        request = ApprovalRequest(
            id=str(uuid.uuid4()),
            script_id=script_id,
            script_name=script_name,
            requester_id=requester_id,
            requester_name=requester_name,
            approval_type=approval_type,
            status=ApprovalStatus.PENDING,
            risk_level=risk_level,
            description=description,
            justification=justification,
            preflight_report=preflight_report,
            required_approvers=required_approvers,
            expires_at=expires_at
        )

        # Store request
        self.pending_requests[request.id] = request

        logger.info(f"Created approval request {request.id} for script {script_name}")

        return request

    def _get_required_approvers(self, approval_type: ApprovalType, risk_level: str) -> List[str]:
        """Get list of required approvers for a request"""
        # This would typically query user roles from a database
        # For now, return placeholder approver IDs

        if approval_type == ApprovalType.AUTOMATIC:
            return []  # No approvers needed
        elif approval_type == ApprovalType.PEER_REVIEW:
            return ["peer_approver_1"]  # Any peer can approve
        elif approval_type == ApprovalType.MANAGER_APPROVAL:
            return ["manager_1", "manager_2"]  # Multiple managers for redundancy
        elif approval_type == ApprovalType.SECURITY_REVIEW:
            return ["security_1", "security_2"]
        elif approval_type == ApprovalType.EXECUTIVE_APPROVAL:
            return ["executive_1"]

        return []

    def approve_request(self, request_id: str, user_id: str, user_name: str,
                       user_role: str, comments: str = "") -> Tuple[bool, str]:
        """
        Approve an approval request

        Args:
            request_id: ID of the request to approve
            user_id: ID of the approving user
            user_name: Name of the approving user
            user_role: Role of the approving user
            comments: Optional approval comments

        Returns:
            Tuple of (success, message)
        """
        request = self.pending_requests.get(request_id)
        if not request:
            return False, "Approval request not found"

        if not request.can_be_approved_by(user_id, user_role):
            return False, "User is not authorized to approve this request"

        # Add approval
        approval_action = {
            "user_id": user_id,
            "user_name": user_name,
            "user_role": user_role,
            "comments": comments,
            "timestamp": datetime.now().isoformat()
        }

        request.approved_by.append(approval_action)
        request.updated_at = datetime.now()

        # Check if request is now complete
        progress = request.approval_progress
        if progress["is_complete"]:
            request.status = ApprovalStatus.APPROVED
            self._move_to_completed(request)
            logger.info(f"Approval request {request_id} fully approved")

            return True, f"Request approved successfully. {progress['approved_count']}/{progress['total_required']} approvals received."
        else:
            logger.info(f"Partial approval for request {request_id}: {progress['approved_count']}/{progress['total_required']}")

            return True, f"Approval recorded. {progress['approved_count']}/{progress['total_required']} approvals received."

    def reject_request(self, request_id: str, user_id: str, user_name: str,
                      user_role: str, comments: str = "") -> Tuple[bool, str]:
        """
        Reject an approval request

        Args:
            request_id: ID of the request to reject
            user_id: ID of the rejecting user
            user_name: Name of the rejecting user
            user_role: Role of the rejecting user
            comments: Rejection comments

        Returns:
            Tuple of (success, message)
        """
        request = self.pending_requests.get(request_id)
        if not request:
            return False, "Approval request not found"

        if not request.can_be_approved_by(user_id, user_role):
            return False, "User is not authorized to reject this request"

        # Add rejection
        rejection_action = {
            "user_id": user_id,
            "user_name": user_name,
            "user_role": user_role,
            "comments": comments,
            "timestamp": datetime.now().isoformat()
        }

        request.rejected_by.append(rejection_action)
        request.status = ApprovalStatus.REJECTED
        request.updated_at = datetime.now()

        self._move_to_completed(request)

        logger.info(f"Approval request {request_id} rejected by {user_name}")

        return True, "Request rejected successfully"

    def cancel_request(self, request_id: str, user_id: str) -> Tuple[bool, str]:
        """Cancel an approval request"""
        request = self.pending_requests.get(request_id)
        if not request:
            return False, "Approval request not found"

        # Only requester can cancel
        if request.requester_id != user_id:
            return False, "Only the requester can cancel this request"

        if request.status != ApprovalStatus.PENDING:
            return False, "Cannot cancel a request that is not pending"

        request.status = ApprovalStatus.CANCELLED
        request.updated_at = datetime.now()

        self._move_to_completed(request)

        logger.info(f"Approval request {request_id} cancelled")

        return True, "Request cancelled successfully"

    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get an approval request by ID"""
        return self.pending_requests.get(request_id) or self.completed_requests.get(request_id)

    def get_pending_requests(self, user_id: Optional[str] = None) -> List[ApprovalRequest]:
        """Get all pending approval requests"""
        requests = list(self.pending_requests.values())

        if user_id:
            # Filter to requests the user can act on
            # This would need user role information
            pass

        return requests

    def get_requests_for_user(self, user_id: str) -> List[ApprovalRequest]:
        """Get all approval requests for a specific user"""
        user_requests = []

        # Check pending requests
        for request in self.pending_requests.values():
            if request.requester_id == user_id:
                user_requests.append(request)

        # Check completed requests
        for request in self.completed_requests.values():
            if request.requester_id == user_id:
                user_requests.append(request)

        return sorted(user_requests, key=lambda x: x.created_at, reverse=True)

    def check_expired_requests(self) -> List[str]:
        """Check for expired approval requests and update their status"""
        expired_ids = []

        for request_id, request in list(self.pending_requests.items()):
            if request.is_expired:
                request.status = ApprovalStatus.EXPIRED
                expired_ids.append(request_id)
                self._move_to_completed(request)

                logger.info(f"Approval request {request_id} expired")

        return expired_ids

    def _move_to_completed(self, request: ApprovalRequest):
        """Move a request from pending to completed"""
        if request.id in self.pending_requests:
            del self.pending_requests[request.id]
            self.completed_requests[request.id] = request

    def get_approval_stats(self) -> Dict[str, Any]:
        """Get approval workflow statistics"""
        total_pending = len(self.pending_requests)
        total_completed = len(self.completed_requests)

        if total_completed > 0:
            completed_requests = list(self.completed_requests.values())

            approved_count = sum(1 for r in completed_requests if r.status == ApprovalStatus.APPROVED)
            rejected_count = sum(1 for r in completed_requests if r.status == ApprovalStatus.REJECTED)
            expired_count = sum(1 for r in completed_requests if r.status == ApprovalStatus.EXPIRED)
            cancelled_count = sum(1 for r in completed_requests if r.status == ApprovalStatus.CANCELLED)

            approval_rate = approved_count / total_completed
            avg_approval_time = sum(
                (r.updated_at - r.created_at).total_seconds()
                for r in completed_requests
                if r.status == ApprovalStatus.APPROVED
            ) / approved_count if approved_count > 0 else 0
        else:
            approved_count = rejected_count = expired_count = cancelled_count = 0
            approval_rate = 0
            avg_approval_time = 0

        return {
            "total_pending": total_pending,
            "total_completed": total_completed,
            "approved_count": approved_count,
            "rejected_count": rejected_count,
            "expired_count": expired_count,
            "cancelled_count": cancelled_count,
            "approval_rate": approval_rate,
            "avg_approval_time_seconds": avg_approval_time
        }

    def get_requests_by_risk_level(self) -> Dict[str, List[ApprovalRequest]]:
        """Get approval requests grouped by risk level"""
        risk_groups = {
            "low": [],
            "medium": [],
            "high": [],
            "critical": []
        }

        for request in list(self.pending_requests.values()) + list(self.completed_requests.values()):
            if request.risk_level in risk_groups:
                risk_groups[request.risk_level].append(request)

        return risk_groups

    def add_comment(self, request_id: str, user_id: str, user_name: str,
                   comment: str) -> Tuple[bool, str]:
        """Add a comment to an approval request"""
        request = self.get_request(request_id)
        if not request:
            return False, "Approval request not found"

        comment_data = {
            "user_id": user_id,
            "user_name": user_name,
            "comment": comment,
            "timestamp": datetime.now().isoformat()
        }

        request.comments.append(comment_data)
        request.updated_at = datetime.now()

        logger.info(f"Comment added to approval request {request_id} by {user_name}")

        return True, "Comment added successfully"

    def get_request_history(self, request_id: str) -> List[Dict[str, Any]]:
        """Get the complete history of an approval request"""
        request = self.get_request(request_id)
        if not request:
            return []

        history = []

        # Add creation event
        history.append({
            "event": "created",
            "timestamp": request.created_at.isoformat(),
            "user_id": request.requester_id,
            "user_name": request.requester_name,
            "details": {
                "description": request.description,
                "justification": request.justification,
                "risk_level": request.risk_level
            }
        })

        # Add approval events
        for approval in request.approved_by:
            history.append({
                "event": "approved",
                "timestamp": approval.get("timestamp"),
                "user_id": approval.get("user_id"),
                "user_name": approval.get("user_name"),
                "details": {
                    "comments": approval.get("comments", ""),
                    "user_role": approval.get("user_role", "")
                }
            })

        # Add rejection events
        for rejection in request.rejected_by:
            history.append({
                "event": "rejected",
                "timestamp": rejection.get("timestamp"),
                "user_id": rejection.get("user_id"),
                "user_name": rejection.get("user_name"),
                "details": {
                    "comments": rejection.get("comments", ""),
                    "user_role": rejection.get("user_role", "")
                }
            })

        # Add comment events
        for comment in request.comments:
            history.append({
                "event": "commented",
                "timestamp": comment.get("timestamp"),
                "user_id": comment.get("user_id"),
                "user_name": comment.get("user_name"),
                "details": {
                    "comment": comment.get("comment")
                }
            })

        # Sort by timestamp
        history.sort(key=lambda x: x["timestamp"])

        return history


# Global approval workflow engine instance
approval_engine = ApprovalWorkflowEngine()
