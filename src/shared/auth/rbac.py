"""
Role-Based Access Control (RBAC) utilities
"""
from typing import Optional, Dict, Any
from enum import Enum
from uuid import UUID


class UserRole(Enum):
    """User role enumeration"""
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class Permission(Enum):
    """Permission enumeration"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


class ResourceType(Enum):
    """Resource type enumeration"""
    ORGANIZATION = "organization"
    USER = "user"
    CONNECTOR = "connector"
    SIGNAL = "signal"
    INCIDENT = "incident"
    FIX_PLAN = "fix_plan"
    DOCUMENT = "document"


# Role permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: {
        ResourceType.ORGANIZATION: [Permission.READ, Permission.WRITE, Permission.ADMIN],
        ResourceType.USER: [Permission.READ, Permission.WRITE, Permission.ADMIN],
        ResourceType.CONNECTOR: [Permission.READ, Permission.WRITE, Permission.EXECUTE],
        ResourceType.SIGNAL: [Permission.READ, Permission.WRITE],
        ResourceType.INCIDENT: [Permission.READ, Permission.WRITE, Permission.EXECUTE],
        ResourceType.FIX_PLAN: [Permission.READ, Permission.WRITE, Permission.EXECUTE],
        ResourceType.DOCUMENT: [Permission.READ, Permission.WRITE],
    },
    UserRole.MEMBER: {
        ResourceType.ORGANIZATION: [Permission.READ],
        ResourceType.USER: [Permission.READ],
        ResourceType.CONNECTOR: [Permission.READ, Permission.WRITE],
        ResourceType.SIGNAL: [Permission.READ, Permission.WRITE],
        ResourceType.INCIDENT: [Permission.READ, Permission.WRITE, Permission.EXECUTE],
        ResourceType.FIX_PLAN: [Permission.READ, Permission.WRITE, Permission.EXECUTE],
        ResourceType.DOCUMENT: [Permission.READ, Permission.WRITE],
    },
    UserRole.VIEWER: {
        ResourceType.ORGANIZATION: [Permission.READ],
        ResourceType.USER: [Permission.READ],
        ResourceType.CONNECTOR: [Permission.READ],
        ResourceType.SIGNAL: [Permission.READ],
        ResourceType.INCIDENT: [Permission.READ],
        ResourceType.FIX_PLAN: [Permission.READ],
        ResourceType.DOCUMENT: [Permission.READ],
    },
}


class RBACManager:
    """Role-Based Access Control manager"""

    @staticmethod
    def has_permission(
        user_role: str,
        resource_type: ResourceType,
        permission: Permission,
        resource_org_id: Optional[UUID] = None,
        user_org_id: Optional[UUID] = None
    ) -> bool:
        """
        Check if user has permission for a specific resource and action

        Args:
            user_role: User's role
            resource_type: Type of resource being accessed
            permission: Permission being requested
            resource_org_id: Organization ID of the resource
            user_org_id: Organization ID of the user

        Returns:
            True if user has permission, False otherwise
        """
        try:
            role_enum = UserRole(user_role)
        except ValueError:
            return False

        # Check if role has the required permission for resource type
        role_permissions = ROLE_PERMISSIONS.get(role_enum, {})
        resource_permissions = role_permissions.get(resource_type, [])

        if permission not in resource_permissions:
            return False

        # Check organization context (cross-org access not allowed)
        if resource_org_id and user_org_id and resource_org_id != user_org_id:
            return False

        return True

    @staticmethod
    def get_user_permissions(user_role: str) -> Dict[str, Any]:
        """
        Get all permissions for a user role

        Args:
            user_role: User's role

        Returns:
            Dictionary of resource permissions
        """
        try:
            role_enum = UserRole(user_role)
        except ValueError:
            return {}

        role_permissions = ROLE_PERMISSIONS.get(role_enum, {})

        # Convert enums to strings for JSON serialization
        return {
            resource_type.value: [perm.value for perm in permissions]
            for resource_type, permissions in role_permissions.items()
        }

    @staticmethod
    def can_create_user(creator_role: str, target_role: str) -> bool:
        """
        Check if user can create another user with specific role

        Args:
            creator_role: Role of the user creating the account
            target_role: Role being assigned to new user

        Returns:
            True if creation is allowed, False otherwise
        """
        try:
            creator = UserRole(creator_role)
            target = UserRole(target_role)
        except ValueError:
            return False

        # Admins can create any role
        if creator == UserRole.ADMIN:
            return True

        # Members can create viewers and other members
        if creator == UserRole.MEMBER:
            return target in [UserRole.VIEWER, UserRole.MEMBER]

        # Viewers cannot create users
        return False

    @staticmethod
    def can_modify_user(modifier_role: str, target_role: str) -> bool:
        """
        Check if user can modify another user's role

        Args:
            modifier_role: Role of the user making changes
            target_role: Current role of target user

        Returns:
            True if modification is allowed, False otherwise
        """
        try:
            modifier = UserRole(modifier_role)
            target = UserRole(target_role)
        except ValueError:
            return False

        # Admins can modify any role
        if modifier == UserRole.ADMIN:
            return True

        # Members can only modify viewers
        if modifier == UserRole.MEMBER:
            return target == UserRole.VIEWER

        # Viewers cannot modify anyone
        return False

    @staticmethod
    def can_execute_fix_plan(user_role: str) -> bool:
        """
        Check if user can execute fix plans

        Args:
            user_role: User's role

        Returns:
            True if execution is allowed, False otherwise
        """
        try:
            role = UserRole(user_role)
        except ValueError:
            return False

        return role in [UserRole.ADMIN, UserRole.MEMBER]

    @staticmethod
    def can_manage_connectors(user_role: str) -> bool:
        """
        Check if user can manage connectors

        Args:
            user_role: User's role

        Returns:
            True if connector management is allowed, False otherwise
        """
        try:
            role = UserRole(user_role)
        except ValueError:
            return False

        return role in [UserRole.ADMIN, UserRole.MEMBER]


def require_permission(resource_type: ResourceType, permission: Permission):
    """
    Decorator to require specific permission for endpoint

    Args:
        resource_type: Type of resource being accessed
        permission: Permission required

    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This would be implemented with actual user context
            # For now, just return the function
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_role(required_role: UserRole):
    """
    Decorator to require specific role for endpoint

    Args:
        required_role: Role required

    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This would be implemented with actual user context
            # For now, just return the function
            return func(*args, **kwargs)
        return wrapper
    return decorator
