"""
Idempotency middleware for handling duplicate requests
"""
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from uuid import uuid4

from ..database.config import get_postgres_session
from ..models.base import IdempotencyKey


logger = logging.getLogger(__name__)


class IdempotencyError(Exception):
    """Base exception for idempotency errors"""
    pass


class DuplicateRequestError(IdempotencyError):
    """Exception raised when a duplicate request is detected"""
    pass


class IdempotencyManager:
    """Manager for handling idempotent requests"""

    def __init__(self, ttl_seconds: int = 86400):  # 24 hours default
        self.ttl_seconds = ttl_seconds

    async def process_request(
        self,
        idempotency_key: str,
        request_data: Dict[str, Any],
        org_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process an idempotent request

        Args:
            idempotency_key: Unique key for the request
            request_data: Request payload data
            org_id: Organization ID
            user_id: Optional user ID

        Returns:
            Stored response if duplicate, None if new request

        Raises:
            DuplicateRequestError: If duplicate request with different data
        """
        try:
            async with get_postgres_session() as session:
                # Check if key already exists
                existing_key = await session.execute(
                    """
                    SELECT response_data, request_hash, created_at
                    FROM idempotency_keys
                    WHERE key = :key AND org_id = :org_id
                    AND created_at > :cutoff_time
                    """,
                    {
                        "key": idempotency_key,
                        "org_id": org_id,
                        "cutoff_time": datetime.utcnow() - timedelta(seconds=self.ttl_seconds)
                    }
                )

                result = existing_key.fetchone()

                if result:
                    stored_response, stored_hash, created_at = result

                    # Calculate current request hash
                    current_hash = self._calculate_request_hash(request_data)

                    if stored_hash != current_hash:
                        logger.warning(f"Duplicate idempotency key with different request data: {idempotency_key}")
                        raise DuplicateRequestError("Idempotency key used with different request data")

                    logger.info(f"Returning cached response for idempotency key: {idempotency_key}")
                    return stored_response

                # Store new idempotency key
                request_hash = self._calculate_request_hash(request_data)

                new_key = IdempotencyKey(
                    key=idempotency_key,
                    org_id=org_id,
                    user_id=user_id,
                    request_hash=request_hash,
                    request_data=request_data,
                    response_data=None,  # Will be set after processing
                    created_at=datetime.utcnow()
                )

                session.add(new_key)
                await session.commit()

                logger.info(f"Created new idempotency key: {idempotency_key}")
                return None

        except Exception as e:
            logger.error(f"Error processing idempotent request: {e}")
            # Don't fail the request, just proceed without idempotency
            return None

    async def store_response(self, idempotency_key: str, response_data: Dict[str, Any], org_id: str) -> None:
        """
        Store the response for an idempotency key

        Args:
            idempotency_key: The idempotency key
            response_data: Response data to store
            org_id: Organization ID
        """
        try:
            async with get_postgres_session() as session:
                await session.execute(
                    """
                    UPDATE idempotency_keys
                    SET response_data = :response_data,
                        updated_at = :updated_at
                    WHERE key = :key AND org_id = :org_id
                    """,
                    {
                        "key": idempotency_key,
                        "org_id": org_id,
                        "response_data": response_data,
                        "updated_at": datetime.utcnow()
                    }
                )
                await session.commit()

                logger.debug(f"Stored response for idempotency key: {idempotency_key}")

        except Exception as e:
            logger.error(f"Error storing response for idempotency key: {e}")

    def _calculate_request_hash(self, request_data: Dict[str, Any]) -> str:
        """Calculate hash of request data for comparison"""
        # Convert to JSON string with sorted keys for consistent hashing
        json_str = json.dumps(request_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    async def cleanup_expired_keys(self) -> int:
        """Clean up expired idempotency keys

        Returns:
            Number of keys deleted
        """
        try:
            async with get_postgres_session() as session:
                result = await session.execute(
                    """
                    DELETE FROM idempotency_keys
                    WHERE created_at < :cutoff_time
                    """,
                    {"cutoff_time": datetime.utcnow() - timedelta(seconds=self.ttl_seconds)}
                )

                deleted_count = result.rowcount
                await session.commit()

                logger.info(f"Cleaned up {deleted_count} expired idempotency keys")
                return deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up expired keys: {e}")
            return 0

    def generate_key(self, prefix: str = "") -> str:
        """Generate a unique idempotency key"""
        key = str(uuid4())
        if prefix:
            key = f"{prefix}:{key}"
        return key


class IdempotencyMiddleware:
    """Middleware for handling idempotent requests"""

    def __init__(self, manager: IdempotencyManager):
        self.manager = manager

    async def __call__(self, request_data: Dict[str, Any], org_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Process request through idempotency middleware"""
        # Extract idempotency key from request
        idempotency_key = request_data.get("idempotency_key")
        if not idempotency_key:
            # No idempotency key, process normally
            return request_data

        # Remove idempotency key from request data for processing
        clean_request_data = {k: v for k, v in request_data.items() if k != "idempotency_key"}

        # Check for duplicate request
        cached_response = await self.manager.process_request(
            idempotency_key=idempotency_key,
            request_data=clean_request_data,
            org_id=org_id,
            user_id=user_id
        )

        if cached_response is not None:
            # Return cached response
            return {
                "cached": True,
                "data": cached_response
            }

        # Add processing metadata
        clean_request_data["_idempotency_key"] = idempotency_key
        clean_request_data["_org_id"] = org_id
        clean_request_data["_user_id"] = user_id

        return clean_request_data

    async def store_response(self, response_data: Dict[str, Any], request_data: Dict[str, Any]) -> None:
        """Store response for idempotency key if present"""
        idempotency_key = request_data.get("_idempotency_key")
        org_id = request_data.get("_org_id")

        if idempotency_key and org_id:
            await self.manager.store_response(idempotency_key, response_data, org_id)


# Global idempotency manager instance
idempotency_manager = IdempotencyManager()

# Global middleware instance
idempotency_middleware = IdempotencyMiddleware(idempotency_manager)
