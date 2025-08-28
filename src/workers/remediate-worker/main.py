"""
Remediate worker for automated incident remediation
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
from ...shared.remediation.fix_catalog import FixCatalog
from ...shared.remediation.script_generator import ScriptGenerator
from ...shared.preflight.preflight_engine import PreflightEngine, PreflightCheck
from ...shared.approval.approval_workflow import ApprovalWorkflowEngine, ApprovalType
from ...shared.rca.hypothesis_builder import HypothesisType


logger = logging.getLogger(__name__)


class RemediateConfig(BaseModel):
    """Remediate worker configuration"""
    auto_approve_low_risk: bool = True
    require_preflight: bool = True
    max_concurrent_fixes: int = 5
    execution_timeout: int = 3600  # 1 hour
    rollback_on_failure: bool = True


class RemediateWorker:
    """Worker for automated incident remediation"""

    def __init__(self, config: RemediateConfig):
        self.config = config
        self.nc: Optional[nats.NATS] = None
        self.running = False
        self.tasks: List[asyncio.Task] = []

        # Initialize remediation components
        self.fix_catalog = FixCatalog()
        self.script_generator = ScriptGenerator()
        self.preflight_engine = PreflightEngine()
        self.approval_engine = ApprovalWorkflowEngine()

        # Active fixes tracking
        self.active_fixes: Dict[str, Dict[str, Any]] = {}

    async def start(self) -> None:
        """Start the remediate worker"""
        logger.info("Starting remediate worker")

        # Connect to NATS
        self.nc = await nats.connect("nats://localhost:4222")

        # Subscribe to remediation topics
        await self._subscribe_topics()

        # Start maintenance tasks
        self.running = True
        maintenance_task = asyncio.create_task(self._maintenance_loop())

        self.tasks.append(maintenance_task)

        logger.info("Remediate worker started successfully")

    async def stop(self) -> None:
        """Stop the remediate worker"""
        logger.info("Stopping remediate worker")
        self.running = False

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Cancel any active fixes
        for fix_id, fix_data in self.active_fixes.items():
            await self._cancel_fix(fix_id)

        # Close NATS connection
        if self.nc:
            await self.nc.close()

        logger.info("Remediate worker stopped")

    async def _subscribe_topics(self) -> None:
        """Subscribe to NATS topics"""
        # Subscribe to incident remediation requests
        await self.nc.subscribe("remediation.generate", cb=self._handle_remediation_request)

        # Subscribe to fix execution requests
        await self.nc.subscribe("remediation.execute", cb=self._handle_fix_execution)

        # Subscribe to approval responses
        await self.nc.subscribe("approval.response", cb=self._handle_approval_response)

        # Subscribe to fix status updates
        await self.nc.subscribe("fix.status", cb=self._handle_fix_status_update)

    async def _maintenance_loop(self) -> None:
        """Periodic maintenance loop"""
        while self.running:
            try:
                await self._perform_maintenance()
                await asyncio.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(60)

    async def _handle_remediation_request(self, msg) -> None:
        """Handle remediation generation requests"""
        try:
            data = json.loads(msg.data.decode())
            incident_id = data.get("incident_id")
            hypothesis_type = data.get("hypothesis_type")
            context = data.get("context", {})

            # Generate remediation plan
            remediation_plan = await self._generate_remediation_plan(
                incident_id, hypothesis_type, context
            )

            # Respond with remediation plan
            await self.nc.publish(msg.reply, json.dumps(remediation_plan).encode())

        except Exception as e:
            logger.error(f"Failed to handle remediation request: {e}")

    async def _handle_fix_execution(self, msg) -> None:
        """Handle fix execution requests"""
        try:
            data = json.loads(msg.data.decode())
            fix_id = data.get("fix_id")
            auto_approve = data.get("auto_approve", False)

            # Execute fix
            execution_result = await self._execute_fix(fix_id, auto_approve)

            # Respond with execution result
            await self.nc.publish(msg.reply, json.dumps(execution_result).encode())

        except Exception as e:
            logger.error(f"Failed to handle fix execution: {e}")

    async def _handle_approval_response(self, msg) -> None:
        """Handle approval responses"""
        try:
            data = json.loads(msg.data.decode())
            request_id = data.get("request_id")
            approved = data.get("approved")
            approver_id = data.get("approver_id")
            comments = data.get("comments", "")

            # Process approval response
            await self._process_approval_response(request_id, approved, approver_id, comments)

        except Exception as e:
            logger.error(f"Failed to handle approval response: {e}")

    async def _handle_fix_status_update(self, msg) -> None:
        """Handle fix status updates"""
        try:
            data = json.loads(msg.data.decode())
            fix_id = data.get("fix_id")
            status = data.get("status")
            details = data.get("details", {})

            # Update fix status
            await self._update_fix_status(fix_id, status, details)

        except Exception as e:
            logger.error(f"Failed to handle fix status update: {e}")

    async def _generate_remediation_plan(self, incident_id: str, hypothesis_type: str,
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive remediation plan"""
        try:
            # Parse hypothesis type
            hypothesis_enum = HypothesisType(hypothesis_type) if hypothesis_type else None

            # Find applicable fixes
            applicable_fixes = []
            if hypothesis_enum:
                applicable_fixes = self.fix_catalog.find_applicable_fixes(hypothesis_enum, context)

            # If no specific fixes found, get general fixes
            if not applicable_fixes:
                applicable_fixes = list(self.fix_catalog.templates.values())[:5]  # Top 5 general fixes

            # Generate scripts for applicable fixes
            fix_options = []
            for fix_template in applicable_fixes:
                try:
                    # Extract parameters from context
                    parameters = self._extract_parameters_from_context(fix_template, context)

                    # Generate fix
                    generated_fix = self.fix_catalog.generate_fix(fix_template.id, parameters)

                    # Generate scripts
                    scripts = self.script_generator.generate_scripts(generated_fix)

                    fix_option = {
                        "fix_id": generated_fix.template_id,
                        "name": fix_template.name,
                        "description": fix_template.description,
                        "risk_level": fix_template.risk_level.value,
                        "estimated_duration": fix_template.estimated_duration.total_seconds(),
                        "success_rate": fix_template.success_rate,
                        "scripts": [
                            {
                                "filename": script.filename,
                                "description": script.description,
                                "tool_type": script.tool_type.value
                            }
                            for script in scripts
                        ]
                    }

                    fix_options.append(fix_option)

                except Exception as e:
                    logger.error(f"Failed to generate fix for template {fix_template.id}: {e}")
                    continue

            return {
                "incident_id": incident_id,
                "hypothesis_type": hypothesis_type,
                "fix_options": fix_options,
                "total_options": len(fix_options),
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to generate remediation plan: {e}")
            return {"error": str(e)}

    async def _execute_fix(self, fix_id: str, auto_approve: bool = False) -> Dict[str, Any]:
        """Execute a remediation fix"""
        try:
            # Get fix details (this would come from a database or cache)
            fix_details = await self._get_fix_details(fix_id)
            if not fix_details:
                return {"error": f"Fix {fix_id} not found"}

            # Check if approval is needed
            if not auto_approve and self._requires_approval(fix_details):
                # Create approval request
                approval_request = await self._create_approval_request(fix_details)
                return {
                    "status": "pending_approval",
                    "approval_request_id": approval_request.id,
                    "message": "Fix requires approval before execution"
                }

            # Run preflight checks
            if self.config.require_preflight:
                preflight_result = await self._run_preflight_checks(fix_details)
                if not preflight_result.can_proceed:
                    return {
                        "status": "preflight_failed",
                        "preflight_report": {
                            "overall_result": preflight_result.overall_result.value,
                            "issues": [r.message for r in preflight_result.results if r.result.name in ["FAIL", "ERROR"]]
                        }
                    }

            # Execute the fix
            execution_result = await self._execute_fix_scripts(fix_details)

            # Track active fix
            self.active_fixes[fix_id] = {
                "started_at": datetime.now(),
                "status": "running",
                "fix_details": fix_details
            }

            return {
                "status": "executing",
                "execution_id": execution_result.get("execution_id"),
                "message": "Fix execution started"
            }

        except Exception as e:
            logger.error(f"Failed to execute fix {fix_id}: {e}")
            return {"error": str(e)}

    async def _run_preflight_checks(self, fix_details: Dict[str, Any]) -> Any:
        """Run preflight checks on a fix"""
        # This would create proper PreflightReport objects
        # For now, return a mock result
        from ...shared.preflight.preflight_engine import PreflightReport, CheckResult

        mock_report = PreflightReport(
            script=None,  # Would need actual script object
            results=[],
            overall_result=CheckResult.PASS,
            can_proceed=True,
            risk_assessment={"risk_level": "low"},
            recommendations=[]
        )

        return mock_report

    async def _create_approval_request(self, fix_details: Dict[str, Any]) -> Any:
        """Create an approval request for a fix"""
        # This would create a proper ApprovalRequest object
        # For now, return a mock object
        class MockApprovalRequest:
            def __init__(self):
                self.id = f"approval_{datetime.now().timestamp()}"

        return MockApprovalRequest()

    async def _execute_fix_scripts(self, fix_details: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fix scripts"""
        # This would actually execute the scripts
        # For now, return a mock result
        execution_id = f"exec_{datetime.now().timestamp()}"

        return {
            "execution_id": execution_id,
            "status": "started",
            "scripts_executed": len(fix_details.get("scripts", []))
        }

    async def _process_approval_response(self, request_id: str, approved: bool,
                                       approver_id: str, comments: str) -> None:
        """Process approval response"""
        try:
            if approved:
                # Find the approved fix and execute it
                # This would need to map request_id back to fix_id
                logger.info(f"Fix approved by {approver_id}: {comments}")
            else:
                logger.info(f"Fix rejected by {approver_id}: {comments}")

        except Exception as e:
            logger.error(f"Failed to process approval response: {e}")

    async def _update_fix_status(self, fix_id: str, status: str, details: Dict[str, Any]) -> None:
        """Update fix execution status"""
        try:
            if fix_id in self.active_fixes:
                self.active_fixes[fix_id]["status"] = status
                self.active_fixes[fix_id]["last_updated"] = datetime.now()

                if status in ["completed", "failed", "cancelled"]:
                    # Clean up completed fixes
                    del self.active_fixes[fix_id]

                logger.info(f"Fix {fix_id} status updated to {status}")

        except Exception as e:
            logger.error(f"Failed to update fix status: {e}")

    def _extract_parameters_from_context(self, fix_template: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters from context for fix template"""
        parameters = {}

        # Extract common parameters
        if "namespace" in context:
            parameters["namespace"] = context["namespace"]
        if "service_name" in context:
            parameters["service_name"] = context["service_name"]
        if "current_replicas" in context:
            parameters["current_replicas"] = context["current_replicas"]
        if "target_replicas" in context:
            parameters["target_replicas"] = context["target_replicas"]

        # Add defaults for required parameters
        for param_name, param_config in fix_template.parameters.items():
            if param_name not in parameters and "default" in param_config:
                parameters[param_name] = param_config["default"]

        return parameters

    def _requires_approval(self, fix_details: Dict[str, Any]) -> bool:
        """Check if a fix requires approval"""
        risk_level = fix_details.get("risk_level", "medium")

        if risk_level in ["high", "critical"]:
            return True

        if risk_level == "medium" and not self.config.auto_approve_low_risk:
            return True

        return False

    async def _get_fix_details(self, fix_id: str) -> Optional[Dict[str, Any]]:
        """Get fix details by ID"""
        # This would query a database or cache
        # For now, return mock data
        return {
            "fix_id": fix_id,
            "name": "Sample Fix",
            "risk_level": "medium",
            "scripts": []
        }

    async def _cancel_fix(self, fix_id: str) -> None:
        """Cancel an active fix"""
        try:
            if fix_id in self.active_fixes:
                logger.info(f"Cancelling active fix {fix_id}")
                # This would send cancellation signals to running scripts
                del self.active_fixes[fix_id]

        except Exception as e:
            logger.error(f"Failed to cancel fix {fix_id}: {e}")

    async def _perform_maintenance(self) -> None:
        """Perform maintenance tasks"""
        try:
            # Check for expired approvals
            expired_requests = self.approval_engine.check_expired_requests()
            if expired_requests:
                logger.info(f"Found {len(expired_requests)} expired approval requests")

            # Check for stuck executions
            stuck_fixes = []
            cutoff_time = datetime.now() - timedelta(hours=2)

            for fix_id, fix_data in self.active_fixes.items():
                started_at = fix_data.get("started_at")
                if started_at and started_at < cutoff_time:
                    stuck_fixes.append(fix_id)

            for fix_id in stuck_fixes:
                logger.warning(f"Fix {fix_id} appears to be stuck, cancelling")
                await self._cancel_fix(fix_id)

            # Log active fixes count
            logger.info(f"Active fixes: {len(self.active_fixes)}")

        except Exception as e:
            logger.error(f"Failed to perform maintenance: {e}")


async def main():
    """Main entry point"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    config = RemediateConfig()

    # Create worker
    worker = RemediateWorker(config)

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
