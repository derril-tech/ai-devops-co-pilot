"""
Request-ID middleware for tracing requests across the system
"""
import logging
import threading
from contextvars import ContextVar
from typing import Optional, Dict, Any
from uuid import uuid4

# Context variable to store request ID across async calls
request_id_context: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class RequestIDManager:
    """Manager for handling request IDs"""

    def __init__(self, header_name: str = "X-Request-ID"):
        self.header_name = header_name
        self.logger = logging.getLogger(__name__)

    def get_request_id(self) -> Optional[str]:
        """Get current request ID from context"""
        return request_id_context.get()

    def set_request_id(self, request_id: Optional[str]) -> None:
        """Set request ID in context"""
        request_id_context.set(request_id)

    def generate_request_id(self) -> str:
        """Generate a new unique request ID"""
        return str(uuid4())

    def extract_from_headers(self, headers: Dict[str, Any]) -> Optional[str]:
        """Extract request ID from HTTP headers"""
        for key, value in headers.items():
            if key.lower() == self.header_name.lower():
                return str(value)
        return None

    def inject_into_headers(self, headers: Dict[str, Any]) -> Dict[str, Any]:
        """Inject current request ID into headers"""
        if headers is None:
            headers = {}

        current_id = self.get_request_id()
        if current_id:
            headers[self.header_name] = current_id

        return headers

    def inject_into_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Inject current request ID into metadata"""
        if metadata is None:
            metadata = {}

        current_id = self.get_request_id()
        if current_id:
            metadata["request_id"] = current_id

        return metadata


class RequestIDMiddleware:
    """Middleware for handling request IDs"""

    def __init__(self, manager: RequestIDManager):
        self.manager = manager

    async def process_request(self, headers: Dict[str, Any]) -> str:
        """Process incoming request and set up request ID"""
        # Extract existing request ID from headers
        request_id = self.manager.extract_from_headers(headers)

        # Generate new ID if none provided
        if not request_id:
            request_id = self.manager.generate_request_id()
            self.manager.logger.debug(f"Generated new request ID: {request_id}")
        else:
            self.manager.logger.debug(f"Using existing request ID: {request_id}")

        # Set in context
        self.manager.set_request_id(request_id)

        return request_id

    async def process_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process outgoing response and inject request ID"""
        if response_data is None:
            response_data = {}

        # Inject request ID into response metadata
        response_data = self.manager.inject_into_metadata(response_data)

        return response_data

    def get_current_request_id(self) -> Optional[str]:
        """Get current request ID"""
        return self.manager.get_request_id()


class LoggingAdapter(logging.LoggerAdapter):
    """Logging adapter that includes request ID in log messages"""

    def __init__(self, logger: logging.Logger, manager: RequestIDManager):
        self.manager = manager
        super().__init__(logger, {})

    def process(self, msg: str, kwargs: Any) -> tuple:
        """Add request ID to log message"""
        request_id = self.manager.get_request_id()
        if request_id:
            msg = f"[request_id={request_id}] {msg}"
        return msg, kwargs


# Global instances
request_id_manager = RequestIDManager()
request_id_middleware = RequestIDMiddleware(request_id_manager)


def get_logger(name: str) -> LoggingAdapter:
    """Get logger with request ID support"""
    logger = logging.getLogger(name)
    return LoggingAdapter(logger, request_id_manager)


def with_request_id():
    """Decorator to ensure function has request ID context"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Ensure we have a request ID
            current_id = request_id_manager.get_request_id()
            if not current_id:
                request_id = request_id_manager.generate_request_id()
                request_id_manager.set_request_id(request_id)

            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Utility functions for easy access
def current_request_id() -> Optional[str]:
    """Get current request ID"""
    return request_id_manager.get_request_id()


def set_current_request_id(request_id: str) -> None:
    """Set current request ID"""
    request_id_manager.set_request_id(request_id)


def generate_request_id() -> str:
    """Generate a new request ID"""
    return request_id_manager.generate_request_id()
