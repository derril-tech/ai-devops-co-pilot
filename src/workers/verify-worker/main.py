"""
Verify worker for automated remediation verification
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

import nats
from pydantic import BaseModel

from ...shared.database.config import get_postgres_session, get_clickhouse_session
from ...shared.verification.canary_judge import CanaryJudge, CanaryDecision
from ...shared.verification.automated_rollback import AutomatedRollback
from ...shared.verification.post_change_reports import PostChangeReportGenerator
from ...shared.metrics.dora_metrics import DORAMetrics
from ...shared.metrics.slo_metrics import SLOMetrics


logger = logging.getLogger(__name__)


class VerifyConfig(BaseModel):
    """Verify worker configuration"""
    canary_enabled: bool = True
    auto_rollback_enabled: bool = True
    report_generation_enabled: bool = True
    slo_monitoring_enabled: bool = True
    dora_metrics_enabled: bool = True
    max_concurrent_verifications: int = 10
    canary_window_hours: int = 2
    baseline_window_days: int = 7
    verification_timeout: int = 3600  # 1 hour
    rollback_timeout: int = 1800  # 30 minutes


class VerificationTask:
    """Verification task state"""
    def __init__(self, task_id: str, incident_id: str, fix_id: str):
        self.task_id = task_id
        self.incident_id = incident_id
        self.fix_id = fix_id
        self.status = "running"
        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None
        self.canary_result: Optional[Any] = None
        self.rollback_result: Optional[Any] = None
        self.report: Optional[Any] = None
        self.errors: List[str] = []


class VerifyWorker:
    """Worker for automated remediation verification"""

    def __init__(self, config: VerifyConfig):
        self.config = config
        self.nc: Optional[nats.NATS] = None
        self.running = False
        self.tasks: Dict[str, VerificationTask] = {}

        # Initialize verification components
        self.canary_judge = CanaryJudge()
        self.rollback_system = AutomatedRollback()
        self.report_generator = PostChangeReportGenerator()
        self.dora_metrics = DORAMetrics()
        self.slo_metrics = SLOMetrics()

        # Active tasks tracking
        self.active_tasks: Dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        """Start the verify worker"""
        logger.info("Starting verify worker")

        # Connect to NATS
        self.nc = await nats.connect("nats://localhost:4222")

        # Subscribe to verification topics
        await self._subscribe_topics()

        # Start maintenance tasks
        self.running = True
        maintenance_task = asyncio.create_task(self._maintenance_loop())

        self.active_tasks["maintenance"] = maintenance_task

        logger.info("Verify worker started successfully")

    async def stop(self) -> None:
        """Stop the verify worker"""
        logger.info("Stopping verify worker")
        self.running = False

        # Cancel all active tasks
        for task_id, task in list(self.active_tasks.items()):
            if not task.done():
                task.cancel()

        # Cancel any running verifications
        for task_id, verification_task in list(self.tasks.items()):
            if verification_task.status == "running":
                await self._cancel_verification(task_id)

        # Close NATS connection
        if self.nc:
            await self.nc.close()

        logger.info("Verify worker stopped")

    async def _subscribe_topics(self) -> None:
        """Subscribe to NATS topics"""
        # Subscribe to verification requests
        await self.nc.subscribe("verify.remediation", cb=self._handle_verification_request)

        # Subscribe to canary results
        await self.nc.subscribe("canary.result", cb=self._handle_canary_result)

        # Subscribe to rollback results
        await self.nc.subscribe("rollback.result", cb=self._handle_rollback_result)

        # Subscribe to status queries
        await self.nc.subscribe("verify.status", cb=self._handle_status_query)

    async def _maintenance_loop(self) -> None:
        """Periodic maintenance loop"""
        while self.running:
            try:
                await self._perform_maintenance()
                await asyncio.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(60)

    async def _handle_verification_request(self, msg) -> None:
        """Handle verification requests"""
        try:
            data = json.loads(msg.data.decode())
            incident_id = data.get("incident_id")
            fix_id = data.get("fix_id")
            canary_duration_hours = data.get("canary_duration_hours", self.config.canary_window_hours)
            baseline_window_days = data.get("baseline_window_days", self.config.baseline_window_days)

            # Start verification process
            task_id = await self.start_verification(
                incident_id, fix_id, canary_duration_hours, baseline_window_days
            )

            # Respond with task ID
            await self.nc.publish(msg.reply, json.dumps({"task_id": task_id}).encode())

        except Exception as e:
            logger.error(f"Failed to handle verification request: {e}")

    async def _handle_canary_result(self, msg) -> None:
        """Handle canary analysis results"""
        try:
            data = json.loads(msg.data.decode())
            task_id = data.get("task_id")
            canary_result = data.get("result")

            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.canary_result = canary_result

                # Check if rollback is needed
                if canary_result.get("decision") in ["failure", "rollback"]:
                    await self._initiate_rollback(task)

        except Exception as e:
            logger.error(f"Failed to handle canary result: {e}")

    async def _handle_rollback_result(self, msg) -> None:
        """Handle rollback completion results"""
        try:
            data = json.loads(msg.data.decode())
            task_id = data.get("task_id")
            rollback_result = data.get("result")

            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.rollback_result = rollback_result

                # Complete verification process
                await self._complete_verification(task_id)

        except Exception as e:
            logger.error(f"Failed to handle rollback result: {e}")

    async def _handle_status_query(self, msg) -> None:
        """Handle verification status queries"""
        try:
            data = json.loads(msg.data.decode())
            task_id = data.get("task_id")

            if task_id in self.tasks:
                task = self.tasks[task_id]
                status = {
                    "task_id": task.task_id,
                    "status": task.status,
                    "created_at": task.created_at.isoformat(),
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "canary_result": task.canary_result,
                    "rollback_result": task.rollback_result,
                    "errors": task.errors
                }
            else:
                status = {"error": f"Task {task_id} not found"}

            await self.nc.publish(msg.reply, json.dumps(status).encode())

        except Exception as e:
            logger.error(f"Failed to handle status query: {e}")

    async def start_verification(self, incident_id: str, fix_id: str,
                               canary_duration_hours: int = 2,
                               baseline_window_days: int = 7) -> str:
        """
        Start a verification process for a remediation

        Args:
            incident_id: ID of the incident
            fix_id: ID of the applied fix
            canary_duration_hours: Hours to run canary analysis
            baseline_window_days: Days of baseline data to use

        Returns:
            Task ID for tracking
        """
        task_id = f"verify_{incident_id}_{fix_id}_{int(datetime.now().timestamp())}"

        task = VerificationTask(task_id, incident_id, fix_id)
        self.tasks[task_id] = task

        # Start verification task
        verification_task = asyncio.create_task(
            self._run_verification(task, canary_duration_hours, baseline_window_days)
        )

        self.active_tasks[task_id] = verification_task

        logger.info(f"Started verification task {task_id}")

        return task_id

    async def _run_verification(self, task: VerificationTask,
                              canary_duration_hours: int,
                              baseline_window_days: int) -> None:
        """Run the verification process"""
        try:
            # Step 1: Canary analysis
            if self.config.canary_enabled:
                await self._run_canary_analysis(task, canary_duration_hours, baseline_window_days)

            # Step 2: SLO/SLO monitoring
            if self.config.slo_monitoring_enabled:
                await self._check_slo_compliance(task)

            # Step 3: DORA metrics analysis
            if self.config.dora_metrics_enabled:
                await self._analyze_dora_metrics(task)

            # Step 4: Generate post-change report
            if self.config.report_generation_enabled:
                await self._generate_post_change_report(task)

            # If no issues found, complete successfully
            if not task.errors and task.canary_result:
                canary_decision = task.canary_result.get("decision")
                if canary_decision == "success":
                    await self._complete_verification(task.task_id)

        except Exception as e:
            logger.error(f"Verification task {task.task_id} failed: {e}")
            task.errors.append(str(e))
            task.status = "failed"
            task.completed_at = datetime.now()

    async def _run_canary_analysis(self, task: VerificationTask,
                                 duration_hours: int, baseline_days: int) -> None:
        """Run canary analysis"""
        try:
            logger.info(f"Running canary analysis for task {task.task_id}")

            # Start canary experiment
            experiment_id = await self.canary_judge.start_canary_experiment(
                incident_id=task.incident_id,
                fix_id=task.fix_id,
                baseline_window_days=baseline_days
            )

            # Wait for canary duration
            await asyncio.sleep(duration_hours * 3600)

            # Evaluate canary results
            canary_result = await self.canary_judge.evaluate_canary(experiment_id)

            # Store result
            task.canary_result = {
                "experiment_id": experiment_id,
                "decision": canary_result.decision.value,
                "confidence": canary_result.confidence,
                "improved_metrics": [m for m in canary_result.metrics_improved],
                "degraded_metrics": [m for m in canary_result.metrics_degraded],
                "stable_metrics": [m for m in canary_result.metrics_stable]
            }

            # Publish canary result
            await self.nc.publish("canary.result", json.dumps({
                "task_id": task.task_id,
                "result": task.canary_result
            }).encode())

        except Exception as e:
            logger.error(f"Canary analysis failed for task {task.task_id}: {e}")
            task.errors.append(f"Canary analysis failed: {str(e)}")

    async def _check_slo_compliance(self, task: VerificationTask) -> None:
        """Check SLO compliance"""
        try:
            # Get service name from incident/fix data
            service_name = await self._get_service_name(task.incident_id)

            if service_name:
                # Generate SLO dashboard
                dashboard = await self.slo_metrics.generate_slo_dashboard(service_name)

                # Check for SLO breaches or warnings
                alerts = dashboard.alerts
                if alerts:
                    task.errors.extend([f"SLO Alert: {alert['message']}" for alert in alerts])

        except Exception as e:
            logger.error(f"SLO check failed for task {task.task_id}: {e}")
            task.errors.append(f"SLO check failed: {str(e)}")

    async def _analyze_dora_metrics(self, task: VerificationTask) -> None:
        """Analyze DORA metrics"""
        try:
            # Get DORA metrics for the incident period
            dora_data = await self.dora_metrics.get_metrics_for_incident(
                task.incident_id, days_before=7, days_after=1
            )

            # Check for significant changes
            if "comparison" in dora_data:
                comparison = dora_data["comparison"]
                for metric, data in comparison.items():
                    change_percent = data.get("change_percent", 0)
                    if abs(change_percent) > 25:  # 25% change threshold
                        task.errors.append(
                            f"DORA metric {metric} changed by {change_percent:.1f}%"
                        )

        except Exception as e:
            logger.error(f"DORA analysis failed for task {task.task_id}: {e}")
            task.errors.append(f"DORA analysis failed: {str(e)}")

    async def _generate_post_change_report(self, task: VerificationTask) -> None:
        """Generate post-change verification report"""
        try:
            # Generate comprehensive report
            report = await self.report_generator.generate_report(
                incident_id=task.incident_id,
                fix_id=task.fix_id,
                canary_result=task.canary_result,
                rollback_execution=task.rollback_result
            )

            task.report = {
                "report_id": report.id,
                "overall_success": report.overall_success,
                "impact_level": report.impact_level.value,
                "generated_at": report.generated_at.isoformat()
            }

            logger.info(f"Generated post-change report for task {task.task_id}")

        except Exception as e:
            logger.error(f"Report generation failed for task {task.task_id}: {e}")
            task.errors.append(f"Report generation failed: {str(e)}")

    async def _initiate_rollback(self, task: VerificationTask) -> None:
        """Initiate rollback process"""
        try:
            logger.info(f"Initiating rollback for task {task.task_id}")

            if not self.config.auto_rollback_enabled:
                logger.info("Auto-rollback disabled, manual intervention required")
                return

            # Create rollback plan
            # This would need actual fix details
            rollback_plan = await self.rollback_system.create_rollback_plan(
                original_fix=None,  # Would need to get actual fix
                incident_id=task.incident_id
            )

            # Execute rollback
            rollback_execution = await self.rollback_system.execute_rollback(
                plan=rollback_plan,
                canary_decision=CanaryDecision.FAILURE
            )

            task.rollback_result = {
                "execution_id": rollback_execution.id,
                "success": rollback_execution.status.value == "completed",
                "duration_seconds": rollback_execution.rollback_duration.total_seconds() if rollback_execution.rollback_duration else 0,
                "errors": rollback_execution.errors
            }

            # Publish rollback result
            await self.nc.publish("rollback.result", json.dumps({
                "task_id": task.task_id,
                "result": task.rollback_result
            }).encode())

        except Exception as e:
            logger.error(f"Rollback initiation failed for task {task.task_id}: {e}")
            task.errors.append(f"Rollback failed: {str(e)}")

    async def _complete_verification(self, task_id: str) -> None:
        """Complete verification process"""
        if task_id not in self.tasks:
            return

        task = self.tasks[task_id]
        task.status = "completed"
        task.completed_at = datetime.now()

        # Clean up active task
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]

        logger.info(f"Verification task {task_id} completed")

        # Publish completion event
        await self.nc.publish("verify.completed", json.dumps({
            "task_id": task_id,
            "status": "completed",
            "canary_result": task.canary_result,
            "rollback_result": task.rollback_result,
            "report": task.report,
            "errors": task.errors
        }).encode())

    async def _cancel_verification(self, task_id: str) -> None:
        """Cancel a verification task"""
        if task_id not in self.tasks:
            return

        task = self.tasks[task_id]
        task.status = "cancelled"
        task.completed_at = datetime.now()

        # Cancel active task
        if task_id in self.active_tasks:
            self.active_tasks[task_id].cancel()
            del self.active_tasks[task_id]

        logger.info(f"Verification task {task_id} cancelled")

    async def _perform_maintenance(self) -> None:
        """Perform maintenance tasks"""
        try:
            # Check for timed-out tasks
            timeout_cutoff = datetime.now() - timedelta(seconds=self.config.verification_timeout)

            timed_out_tasks = [
                task_id for task_id, task in self.tasks.items()
                if task.status == "running" and task.created_at < timeout_cutoff
            ]

            for task_id in timed_out_tasks:
                logger.warning(f"Verification task {task_id} timed out")
                task = self.tasks[task_id]
                task.errors.append("Verification timed out")
                await self._cancel_verification(task_id)

            # Clean up old completed tasks (keep last 7 days)
            cleanup_cutoff = datetime.now() - timedelta(days=7)

            old_tasks = [
                task_id for task_id, task in list(self.tasks.items())
                if task.status in ["completed", "failed", "cancelled"]
                and task.completed_at and task.completed_at < cleanup_cutoff
            ]

            for task_id in old_tasks:
                del self.tasks[task_id]

            # Log status
            active_count = len([t for t in self.tasks.values() if t.status == "running"])
            logger.info(f"Active verification tasks: {active_count}")

        except Exception as e:
            logger.error(f"Failed to perform maintenance: {e}")

    async def _get_service_name(self, incident_id: str) -> Optional[str]:
        """Get service name from incident data"""
        # This would query the incident database
        # For now, return a mock service name
        return "api-service"

    def get_verification_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a verification task"""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]
        return {
            "task_id": task.task_id,
            "status": task.status,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "canary_result": task.canary_result,
            "rollback_result": task.rollback_result,
            "report": task.report,
            "errors": task.errors
        }

    def get_active_verifications(self) -> List[Dict[str, Any]]:
        """Get all active verification tasks"""
        active = [task for task in self.tasks.values() if task.status == "running"]
        return [self.get_verification_status(task.task_id) for task in active]

    def get_verification_stats(self) -> Dict[str, Any]:
        """Get verification statistics"""
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks.values() if t.status == "completed"])
        failed_tasks = len([t for t in self.tasks.values() if t.status == "failed"])
        cancelled_tasks = len([t for t in self.tasks.values() if t.status == "cancelled"])
        running_tasks = len([t for t in self.tasks.values() if t.status == "running"])

        success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "cancelled_tasks": cancelled_tasks,
            "running_tasks": running_tasks,
            "success_rate": success_rate
        }


async def main():
    """Main entry point"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    config = VerifyConfig()

    # Create worker
    worker = VerifyWorker(config)

    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info("Received signal, shutting down...")
        asyncio.create_task(worker.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start worker
        await worker.start()

        # Keep running until stopped
        while worker.running:
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Worker failed: {e}")
        sys.exit(1)
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
