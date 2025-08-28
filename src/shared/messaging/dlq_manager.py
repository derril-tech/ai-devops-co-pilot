"""
Dead Letter Queue (DLQ) manager for handling failed messages
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


class DLQMessageStatus(Enum):
    """Status of messages in DLQ"""
    PENDING = "pending"      # Waiting for processing
    RETRYING = "retrying"   # Currently being retried
    PROCESSED = "processed" # Successfully processed
    FAILED = "failed"       # Permanently failed
    EXPIRED = "expired"     # Expired and removed


class DLQStrategy(Enum):
    """DLQ processing strategies"""
    IMMEDIATE_RETRY = "immediate_retry"  # Retry immediately
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Exponential backoff
    FIXED_DELAY = "fixed_delay"  # Fixed delay between retries
    MANUAL_REVIEW = "manual_review"  # Require manual review
    DISCARD = "discard"  # Discard failed messages


@dataclass
class DLQMessage:
    """Message in the dead letter queue"""
    id: str
    original_topic: str
    original_message: Dict[str, Any]
    error_message: str
    error_details: Dict[str, Any]
    retry_count: int = 0
    max_retries: int = 3
    status: DLQMessageStatus = DLQMessageStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    last_retry_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DLQConfiguration:
    """Configuration for a DLQ"""
    name: str
    topics: List[str]  # Topics this DLQ handles
    strategy: DLQStrategy
    max_retries: int = 3
    retry_delay_seconds: int = 60
    exponential_backoff_multiplier: float = 2.0
    max_retry_delay_seconds: int = 3600  # 1 hour
    message_ttl_seconds: int = 604800  # 7 days
    batch_size: int = 10
    enabled: bool = True
    alert_threshold: int = 100  # Alert when queue size exceeds this


class DLQManager:
    """Manager for Dead Letter Queues"""

    def __init__(self):
        self.queues: Dict[str, DLQConfiguration] = {}
        self.messages: Dict[str, Dict[str, DLQMessage]] = {}  # queue_name -> message_id -> message
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.retry_handlers: Dict[str, Callable] = {}
        self.alert_handlers: Dict[str, Callable] = {}

        # Setup default queues
        self._setup_default_queues()

    def _setup_default_queues(self):
        """Setup default DLQ configurations"""
        # Remediation queue DLQ
        self.create_queue(DLQConfiguration(
            name="remediation_dlq",
            topics=["remediation.*"],
            strategy=DLQStrategy.EXPONENTIAL_BACKOFF,
            max_retries=5,
            retry_delay_seconds=30,
            message_ttl_seconds=259200  # 3 days
        ))

        # Verification queue DLQ
        self.create_queue(DLQConfiguration(
            name="verification_dlq",
            topics=["verify.*", "canary.*", "rollback.*"],
            strategy=DLQStrategy.EXPONENTIAL_BACKOFF,
            max_retries=3,
            retry_delay_seconds=60,
            message_ttl_seconds=86400  # 1 day
        ))

        # Metrics queue DLQ
        self.create_queue(DLQConfiguration(
            name="metrics_dlq",
            topics=["metrics.*", "slo.*", "dora.*"],
            strategy=DLQStrategy.FIXED_DELAY,
            max_retries=10,
            retry_delay_seconds=300,  # 5 minutes
            message_ttl_seconds=604800  # 7 days
        ))

        # Generic fallback DLQ
        self.create_queue(DLQConfiguration(
            name="generic_dlq",
            topics=["*"],
            strategy=DLQStrategy.MANUAL_REVIEW,
            max_retries=0,
            message_ttl_seconds=2592000  # 30 days
        ))

    def create_queue(self, config: DLQConfiguration) -> None:
        """Create a new DLQ"""
        self.queues[config.name] = config
        self.messages[config.name] = {}
        logger.info(f"Created DLQ: {config.name} for topics: {config.topics}")

        # Start processing task for this queue
        self._start_queue_processor(config.name)

    def delete_queue(self, queue_name: str) -> bool:
        """Delete a DLQ"""
        if queue_name in self.queues:
            # Stop processing task
            if queue_name in self.processing_tasks:
                self.processing_tasks[queue_name].cancel()
                del self.processing_tasks[queue_name]

            # Clean up messages and configuration
            del self.queues[queue_name]
            del self.messages[queue_name]

            logger.info(f"Deleted DLQ: {queue_name}")
            return True
        return False

    def register_retry_handler(self, queue_name: str, handler: Callable) -> None:
        """Register a retry handler for a DLQ"""
        self.retry_handlers[queue_name] = handler
        logger.info(f"Registered retry handler for DLQ: {queue_name}")

    def register_alert_handler(self, queue_name: str, handler: Callable) -> None:
        """Register an alert handler for a DLQ"""
        self.alert_handlers[queue_name] = handler
        logger.info(f"Registered alert handler for DLQ: {queue_name}")

    async def enqueue_message(self, topic: str, message: Dict[str, Any],
                             error_message: str, error_details: Dict[str, Any] = None) -> str:
        """
        Enqueue a failed message to the appropriate DLQ

        Args:
            topic: Original message topic
            message: Original message content
            error_message: Error description
            error_details: Additional error details

        Returns:
            Message ID in DLQ
        """
        queue_name = self._find_queue_for_topic(topic)
        if not queue_name:
            logger.error(f"No DLQ found for topic: {topic}")
            return ""

        config = self.queues[queue_name]

        message_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(seconds=config.message_ttl_seconds)

        dlq_message = DLQMessage(
            id=message_id,
            original_topic=topic,
            original_message=message,
            error_message=error_message,
            error_details=error_details or {},
            max_retries=config.max_retries,
            expires_at=expires_at,
            next_retry_at=self._calculate_next_retry(config, 0)
        )

        self.messages[queue_name][message_id] = dlq_message

        logger.info(f"Enqueued message {message_id} to DLQ {queue_name}")

        # Check if we need to send alerts
        await self._check_alert_threshold(queue_name)

        return message_id

    def _find_queue_for_topic(self, topic: str) -> Optional[str]:
        """Find the appropriate DLQ for a topic"""
        # Check specific topic matches first
        for queue_name, config in self.queues.items():
            if topic in config.topics:
                return queue_name

        # Check wildcard matches
        for queue_name, config in self.queues.items():
            for topic_pattern in config.topics:
                if topic_pattern.endswith("*"):
                    prefix = topic_pattern[:-1]
                    if topic.startswith(prefix):
                        return queue_name

        # Return generic DLQ if available
        return "generic_dlq" if "generic_dlq" in self.queues else None

    def _calculate_next_retry(self, config: DLQConfiguration, retry_count: int) -> Optional[datetime]:
        """Calculate next retry time based on strategy"""
        if retry_count >= config.max_retries:
            return None

        now = datetime.now()

        if config.strategy == DLQStrategy.IMMEDIATE_RETRY:
            return now
        elif config.strategy == DLQStrategy.FIXED_DELAY:
            return now + timedelta(seconds=config.retry_delay_seconds)
        elif config.strategy == DLQStrategy.EXPONENTIAL_BACKOFF:
            delay = config.retry_delay_seconds * (config.exponential_backoff_multiplier ** retry_count)
            delay = min(delay, config.max_retry_delay_seconds)
            return now + timedelta(seconds=delay)
        elif config.strategy == DLQStrategy.MANUAL_REVIEW:
            return None  # No automatic retry
        else:
            return None

    def _start_queue_processor(self, queue_name: str) -> None:
        """Start the processing task for a DLQ"""
        if queue_name in self.processing_tasks:
            self.processing_tasks[queue_name].cancel()

        task = asyncio.create_task(self._process_queue(queue_name))
        self.processing_tasks[queue_name] = task

        logger.info(f"Started processor for DLQ: {queue_name}")

    async def _process_queue(self, queue_name: str) -> None:
        """Process messages in a DLQ"""
        while True:
            try:
                config = self.queues.get(queue_name)
                if not config or not config.enabled:
                    await asyncio.sleep(60)
                    continue

                # Get pending messages
                pending_messages = [
                    msg for msg in self.messages[queue_name].values()
                    if msg.status == DLQMessageStatus.PENDING
                    and (not msg.next_retry_at or datetime.now() >= msg.next_retry_at)
                    and (not msg.expires_at or datetime.now() < msg.expires_at)
                ]

                if not pending_messages:
                    await asyncio.sleep(10)
                    continue

                # Process messages in batches
                batch = pending_messages[:config.batch_size]

                for message in batch:
                    await self._process_message(queue_name, message)

                # Clean up expired messages
                await self._cleanup_expired_messages(queue_name)

                await asyncio.sleep(5)  # Small delay between batches

            except Exception as e:
                logger.error(f"Error processing DLQ {queue_name}: {e}")
                await asyncio.sleep(60)

    async def _process_message(self, queue_name: str, message: DLQMessage) -> None:
        """Process a single DLQ message"""
        config = self.queues[queue_name]

        # Check if message has expired
        if message.expires_at and datetime.now() >= message.expires_at:
            message.status = DLQMessageStatus.EXPIRED
            logger.info(f"Message {message.id} in DLQ {queue_name} has expired")
            return

        # Check if max retries exceeded
        if message.retry_count >= message.max_retries:
            message.status = DLQMessageStatus.FAILED
            logger.warning(f"Message {message.id} in DLQ {queue_name} exceeded max retries")
            return

        # Check if it's time to retry
        if message.next_retry_at and datetime.now() < message.next_retry_at:
            return

        # Attempt to retry the message
        message.status = DLQMessageStatus.RETRYING
        message.last_retry_at = datetime.now()
        message.retry_count += 1

        success = await self._retry_message(queue_name, message)

        if success:
            message.status = DLQMessageStatus.PROCESSED
            message.processed_at = datetime.now()
            logger.info(f"Successfully processed message {message.id} from DLQ {queue_name}")
        else:
            # Calculate next retry time
            message.next_retry_at = self._calculate_next_retry(config, message.retry_count)
            message.status = DLQMessageStatus.PENDING

            if message.next_retry_at is None:
                message.status = DLQMessageStatus.FAILED
                logger.error(f"Failed to process message {message.id} from DLQ {queue_name}")

    async def _retry_message(self, queue_name: str, message: DLQMessage) -> bool:
        """Retry processing a message"""
        handler = self.retry_handlers.get(queue_name)
        if not handler:
            logger.error(f"No retry handler registered for DLQ {queue_name}")
            return False

        try:
            # Call the retry handler
            success = await handler(message.original_topic, message.original_message)

            if success:
                logger.info(f"Successfully retried message {message.id}")
            else:
                logger.warning(f"Retry failed for message {message.id}")

            return success

        except Exception as e:
            logger.error(f"Error retrying message {message.id}: {e}")
            return False

    async def _check_alert_threshold(self, queue_name: str) -> None:
        """Check if queue size exceeds alert threshold"""
        config = self.queues[queue_name]
        queue_size = len(self.messages[queue_name])

        if queue_size >= config.alert_threshold:
            alert_handler = self.alert_handlers.get(queue_name)
            if alert_handler:
                await alert_handler(queue_name, queue_size, config.alert_threshold)
            else:
                logger.warning(f"DLQ {queue_name} size ({queue_size}) exceeds threshold ({config.alert_threshold})")

    async def _cleanup_expired_messages(self, queue_name: str) -> int:
        """Clean up expired messages from DLQ"""
        now = datetime.now()
        expired_messages = [
            msg_id for msg_id, msg in self.messages[queue_name].items()
            if msg.expires_at and now >= msg.expires_at
        ]

        for msg_id in expired_messages:
            del self.messages[queue_name][msg_id]

        if expired_messages:
            logger.info(f"Cleaned up {len(expired_messages)} expired messages from DLQ {queue_name}")

        return len(expired_messages)

    def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """Get statistics for a DLQ"""
        if queue_name not in self.queues:
            return {"error": f"Queue {queue_name} not found"}

        messages = self.messages[queue_name]
        config = self.queues[queue_name]

        stats = {
            "queue_name": queue_name,
            "total_messages": len(messages),
            "pending_messages": len([m for m in messages.values() if m.status == DLQMessageStatus.PENDING]),
            "retrying_messages": len([m for m in messages.values() if m.status == DLQMessageStatus.RETRYING]),
            "processed_messages": len([m for m in messages.values() if m.status == DLQMessageStatus.PROCESSED]),
            "failed_messages": len([m for m in messages.values() if m.status == DLQMessageStatus.FAILED]),
            "expired_messages": len([m for m in messages.values() if m.status == DLQMessageStatus.EXPIRED]),
            "configuration": {
                "strategy": config.strategy.value,
                "max_retries": config.max_retries,
                "message_ttl_seconds": config.message_ttl_seconds,
                "enabled": config.enabled
            }
        }

        return stats

    def get_all_queue_stats(self) -> Dict[str, Any]:
        """Get statistics for all DLQs"""
        all_stats = {}
        for queue_name in self.queues.keys():
            all_stats[queue_name] = self.get_queue_stats(queue_name)

        return {
            "queues": all_stats,
            "summary": {
                "total_queues": len(self.queues),
                "total_messages": sum(stats["total_messages"] for stats in all_stats.values()),
                "generated_at": datetime.now().isoformat()
            }
        }

    def get_message_details(self, queue_name: str, message_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific message"""
        if queue_name not in self.messages:
            return None

        message = self.messages[queue_name].get(message_id)
        if not message:
            return None

        return {
            "id": message.id,
            "original_topic": message.original_topic,
            "error_message": message.error_message,
            "error_details": message.error_details,
            "retry_count": message.retry_count,
            "max_retries": message.max_retries,
            "status": message.status.value,
            "created_at": message.created_at.isoformat(),
            "last_retry_at": message.last_retry_at.isoformat() if message.last_retry_at else None,
            "next_retry_at": message.next_retry_at.isoformat() if message.next_retry_at else None,
            "processed_at": message.processed_at.isoformat() if message.processed_at else None,
            "expires_at": message.expires_at.isoformat() if message.expires_at else None,
            "original_message": message.original_message
        }

    def manually_retry_message(self, queue_name: str, message_id: str) -> bool:
        """Manually retry a message in the DLQ"""
        if queue_name not in self.messages:
            return False

        message = self.messages[queue_name].get(message_id)
        if not message:
            return False

        # Reset retry state
        message.status = DLQMessageStatus.PENDING
        message.next_retry_at = datetime.now()

        logger.info(f"Manually queued message {message_id} for retry in DLQ {queue_name}")
        return True

    def delete_message(self, queue_name: str, message_id: str) -> bool:
        """Delete a message from the DLQ"""
        if queue_name not in self.messages:
            return False

        if message_id in self.messages[queue_name]:
            del self.messages[queue_name][message_id]
            logger.info(f"Deleted message {message_id} from DLQ {queue_name}")
            return True

        return False

    def export_queue_data(self, queue_name: str) -> Dict[str, Any]:
        """Export all data from a DLQ for backup/analysis"""
        if queue_name not in self.queues:
            return {"error": f"Queue {queue_name} not found"}

        config = self.queues[queue_name]
        messages = self.messages[queue_name]

        return {
            "queue_name": queue_name,
            "configuration": {
                "topics": config.topics,
                "strategy": config.strategy.value,
                "max_retries": config.max_retries,
                "retry_delay_seconds": config.retry_delay_seconds,
                "message_ttl_seconds": config.message_ttl_seconds,
                "enabled": config.enabled
            },
            "messages": {
                msg_id: {
                    "id": msg.id,
                    "original_topic": msg.original_topic,
                    "original_message": msg.original_message,
                    "error_message": msg.error_message,
                    "error_details": msg.error_details,
                    "retry_count": msg.retry_count,
                    "status": msg.status.value,
                    "created_at": msg.created_at.isoformat(),
                    "last_retry_at": msg.last_retry_at.isoformat() if msg.last_retry_at else None,
                    "expires_at": msg.expires_at.isoformat() if msg.expires_at else None
                }
                for msg_id, msg in messages.items()
            },
            "exported_at": datetime.now().isoformat(),
            "total_messages": len(messages)
        }

    async def shutdown(self) -> None:
        """Shutdown the DLQ manager"""
        logger.info("Shutting down DLQ manager...")

        # Cancel all processing tasks
        for task in self.processing_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks.values(), return_exceptions=True)

        logger.info("DLQ manager shutdown complete")


# Global DLQ manager instance
dlq_manager = DLQManager()
