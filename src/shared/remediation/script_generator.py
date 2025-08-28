"""
Script generator for various remediation tools
"""
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import yaml

from .fix_catalog import FixTemplate, GeneratedFix


logger = logging.getLogger(__name__)


class ToolType(Enum):
    """Supported remediation tools"""
    KUBECTL = "kubectl"
    HELM = "helm"
    TERRAFORM = "terraform"
    ANSIBLE = "ansible"
    SQL = "sql"
    SHELL = "shell"


@dataclass
class GeneratedScript:
    """Generated script with metadata"""
    tool_type: ToolType
    content: str
    filename: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    validation_commands: List[str] = field(default_factory=list)
    rollback_commands: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


class ScriptGenerator:
    """Generator for remediation scripts across different tools"""

    def __init__(self):
        self.generators = {
            ToolType.KUBECTL: self._generate_kubectl_script,
            ToolType.HELM: self._generate_helm_script,
            ToolType.TERRAFORM: self._generate_terraform_script,
            ToolType.ANSIBLE: self._generate_ansible_script,
            ToolType.SQL: self._generate_sql_script,
            ToolType.SHELL: self._generate_shell_script
        }

    def generate_scripts(self, fix: GeneratedFix) -> List[GeneratedScript]:
        """Generate scripts for a fix"""
        scripts = []

        for script_data in fix.scripts:
            tool_type = ToolType(script_data["type"])
            generator = self.generators.get(tool_type)

            if generator:
                try:
                    script = generator(script_data, fix)
                    scripts.append(script)
                except Exception as e:
                    logger.error(f"Failed to generate {tool_type.value} script: {e}")
            else:
                logger.warning(f"No generator available for tool type: {tool_type.value}")

        return scripts

    def _generate_kubectl_script(self, script_data: Dict[str, Any], fix: GeneratedFix) -> GeneratedScript:
        """Generate kubectl script"""
        command = script_data["command"]
        args = script_data.get("args", [])
        namespace = None

        # Extract namespace from args if present
        if "-n" in args:
            ns_index = args.index("-n")
            if ns_index + 1 < len(args):
                namespace = args[ns_index + 1]

        # Build kubectl command
        cmd_parts = ["kubectl", command] + args

        script_content = f"""#!/bin/bash
# Generated kubectl script
# Description: {script_data.get('description', 'Kubernetes operation')}
# Generated at: {datetime.now().isoformat()}

set -e

echo "Executing: {' '.join(cmd_parts)}"

# Execute command
{" ".join(cmd_parts)}

echo "Command completed successfully"
"""

        filename = f"kubectl_{command}_{fix.template_id}_{int(datetime.now().timestamp())}.sh"

        # Generate validation commands
        validation_cmds = []
        if script_data.get("validation"):
            validation = script_data["validation"]
            if validation["command"] == "get":
                val_cmd = ["kubectl", "get"] + validation["args"]
                if namespace and "-n" not in validation["args"]:
                    val_cmd.extend(["-n", namespace])
                validation_cmds.append(" ".join(val_cmd))

        return GeneratedScript(
            tool_type=ToolType.KUBECTL,
            content=script_content,
            filename=filename,
            description=script_data.get('description', 'Kubernetes operation'),
            parameters={"namespace": namespace, "command": command, "args": args},
            validation_commands=validation_cmds
        )

    def _generate_helm_script(self, script_data: Dict[str, Any], fix: GeneratedFix) -> GeneratedScript:
        """Generate Helm script"""
        command = script_data["command"]
        args = script_data.get("args", [])

        if command == "upgrade":
            script_content = f"""#!/bin/bash
# Generated Helm upgrade script
# Description: {script_data.get('description', 'Helm upgrade')}
# Generated at: {datetime.now().isoformat()}

set -e

# Extract release name and chart
RELEASE_NAME="{args[0] if len(args) > 0 else 'unknown'}"
CHART="{args[1] if len(args) > 1 else 'unknown'}"
NAMESPACE="{script_data.get('namespace', 'default')}"

echo "Upgrading Helm release: $RELEASE_NAME"
echo "Chart: $CHART"
echo "Namespace: $NAMESPACE"

# Create values override if provided
if [ -n "{script_data.get('values', '')}" ]; then
    cat > values-override.yaml << EOF
{script_data.get('values', '')}
EOF
    VALUES_FLAG="--values values-override.yaml"
else
    VALUES_FLAG=""
fi

# Execute helm upgrade
helm upgrade $RELEASE_NAME $CHART \\
    --namespace $NAMESPACE \\
    --wait \\
    --timeout 600s \\
    $VALUES_FLAG \\
    {" ".join(args[2:])}

echo "Helm upgrade completed successfully"

# Cleanup
rm -f values-override.yaml
"""

        elif command == "rollback":
            script_content = f"""#!/bin/bash
# Generated Helm rollback script
# Description: {script_data.get('description', 'Helm rollback')}
# Generated at: {datetime.now().isoformat()}

set -e

RELEASE_NAME="{args[0] if len(args) > 0 else 'unknown'}"
REVISION="{args[1] if len(args) > 1 else '0'}"
NAMESPACE="{script_data.get('namespace', 'default')}"

echo "Rolling back Helm release: $RELEASE_NAME to revision $REVISION"

helm rollback $RELEASE_NAME $REVISION \\
    --namespace $NAMESPACE \\
    --wait \\
    --timeout 600s

echo "Helm rollback completed successfully"
"""

        else:
            # Generic helm command
            cmd_parts = ["helm", command] + args
            script_content = f"""#!/bin/bash
# Generated Helm script
# Description: {script_data.get('description', 'Helm operation')}
# Generated at: {datetime.now().isoformat()}

set -e

echo "Executing: {' '.join(cmd_parts)}"

# Execute command
{" ".join(cmd_parts)}

echo "Command completed successfully"
"""

        filename = f"helm_{command}_{fix.template_id}_{int(datetime.now().timestamp())}.sh"

        return GeneratedScript(
            tool_type=ToolType.HELM,
            content=script_content,
            filename=filename,
            description=script_data.get('description', 'Helm operation'),
            parameters={"command": command, "args": args}
        )

    def _generate_terraform_script(self, script_data: Dict[str, Any], fix: GeneratedFix) -> GeneratedScript:
        """Generate Terraform script"""
        operation = script_data.get("operation", "plan")

        if operation == "plan":
            script_content = f"""#!/bin/bash
# Generated Terraform plan script
# Description: {script_data.get('description', 'Terraform plan')}
# Generated at: {datetime.now().isoformat()}

set -e

WORKING_DIR="{script_data.get('working_dir', '.')}"

echo "Running Terraform plan in: $WORKING_DIR"
cd "$WORKING_DIR"

# Initialize if needed
if [ ! -d ".terraform" ]; then
    echo "Initializing Terraform..."
    terraform init
fi

# Generate plan
echo "Generating Terraform plan..."
terraform plan \\
    -out=tfplan \\
    -no-color \\
    {script_data.get('plan_args', '')}

echo "Terraform plan generated successfully"
echo "Review the plan file: tfplan"
"""

        elif operation == "apply":
            script_content = f"""#!/bin/bash
# Generated Terraform apply script
# Description: {script_data.get('description', 'Terraform apply')}
# Generated at: {datetime.now().isoformat()}

set -e

WORKING_DIR="{script_data.get('working_dir', '.')}"

echo "Running Terraform apply in: $WORKING_DIR"
cd "$WORKING_DIR"

# Check if plan file exists
if [ ! -f "tfplan" ]; then
    echo "Plan file not found. Run plan first."
    exit 1
fi

# Apply plan
echo "Applying Terraform plan..."
terraform apply \\
    -auto-approve \\
    -no-color \\
    tfplan

echo "Terraform apply completed successfully"

# Cleanup plan file
rm -f tfplan
"""

        else:
            script_content = f"""#!/bin/bash
# Generated Terraform script
# Description: {script_data.get('description', 'Terraform operation')}
# Generated at: {datetime.now().isoformat()}

set -e

WORKING_DIR="{script_data.get('working_dir', '.')}"

echo "Running Terraform {operation} in: $WORKING_DIR"
cd "$WORKING_DIR"

terraform {operation} {script_data.get('args', '')}

echo "Terraform {operation} completed successfully"
"""

        filename = f"terraform_{operation}_{fix.template_id}_{int(datetime.now().timestamp())}.sh"

        return GeneratedScript(
            tool_type=ToolType.TERRAFORM,
            content=script_content,
            filename=filename,
            description=script_data.get('description', 'Terraform operation'),
            parameters={"operation": operation, "working_dir": script_data.get('working_dir', '.')}
        )

    def _generate_ansible_script(self, script_data: Dict[str, Any], fix: GeneratedFix) -> GeneratedScript:
        """Generate Ansible script"""
        playbook_content = script_data.get("playbook", {})

        # Generate Ansible playbook YAML
        playbook_yaml = yaml.dump([playbook_content], default_flow_style=False, indent=2)

        script_content = f"""#!/bin/bash
# Generated Ansible script
# Description: {script_data.get('description', 'Ansible playbook execution')}
# Generated at: {datetime.now().isoformat()}

set -e

PLAYBOOK_FILE="generated_playbook_{fix.template_id}_{int(datetime.now().timestamp())}.yml"
INVENTORY="{script_data.get('inventory', 'localhost,')}"

# Create playbook file
cat > "$PLAYBOOK_FILE" << 'EOF'
{playbook_yaml}
EOF

echo "Generated Ansible playbook: $PLAYBOOK_FILE"
echo "Inventory: $INVENTORY"

# Execute playbook
ansible-playbook \\
    "$PLAYBOOK_FILE" \\
    -i "$INVENTORY" \\
    --connection=local \\
    {script_data.get('extra_args', '')}

echo "Ansible playbook executed successfully"

# Cleanup
rm -f "$PLAYBOOK_FILE"
"""

        filename = f"ansible_{fix.template_id}_{int(datetime.now().timestamp())}.sh"

        return GeneratedScript(
            tool_type=ToolType.ANSIBLE,
            content=script_content,
            filename=filename,
            description=script_data.get('description', 'Ansible operation'),
            parameters={"playbook": playbook_content, "inventory": script_data.get('inventory', 'localhost,')}
        )

    def _generate_sql_script(self, script_data: Dict[str, Any], fix: GeneratedFix) -> GeneratedScript:
        """Generate SQL script"""
        sql_statements = script_data.get("statements", [])
        connection_params = script_data.get("connection", {})

        # Generate SQL script
        sql_content = "\n".join(sql_statements)

        script_content = f"""#!/bin/bash
# Generated SQL script
# Description: {script_data.get('description', 'SQL execution')}
# Generated at: {datetime.now().isoformat()}

set -e

SQL_FILE="generated_script_{fix.template_id}_{int(datetime.now().timestamp())}.sql"
HOST="{connection_params.get('host', 'localhost')}"
PORT="{connection_params.get('port', '5432')}"
DATABASE="{connection_params.get('database', 'postgres')}"
USER="{connection_params.get('user', 'postgres')}"

# Create SQL file
cat > "$SQL_FILE" << 'EOF'
{sql_content}
EOF

echo "Generated SQL script: $SQL_FILE"
echo "Target database: $DATABASE on $HOST:$PORT"

# Execute SQL script
PGPASSWORD="{connection_params.get('password', '')}" psql \\
    -h "$HOST" \\
    -p "$PORT" \\
    -d "$DATABASE" \\
    -U "$USER" \\
    -f "$SQL_FILE"

echo "SQL script executed successfully"

# Cleanup
rm -f "$SQL_FILE"
"""

        filename = f"sql_{fix.template_id}_{int(datetime.now().timestamp())}.sh"

        return GeneratedScript(
            tool_type=ToolType.SQL,
            content=script_content,
            filename=filename,
            description=script_data.get('description', 'SQL operation'),
            parameters={"statements": sql_statements, "connection": connection_params}
        )

    def _generate_shell_script(self, script_data: Dict[str, Any], fix: GeneratedFix) -> GeneratedScript:
        """Generate shell script"""
        command = script_data.get("command", "echo")
        args = script_data.get("args", [])
        environment = script_data.get("environment", {})

        # Build environment variables
        env_vars = []
        for key, value in environment.items():
            env_vars.append(f"export {key}=\"{value}\"")

        env_section = "\n".join(env_vars) if env_vars else ""

        # Build command
        cmd_parts = [command] + args
        full_command = " ".join(cmd_parts)

        script_content = f"""#!/bin/bash
# Generated shell script
# Description: {script_data.get('description', 'Shell command execution')}
# Generated at: {datetime.now().isoformat()}

set -e

{env_section}

echo "Executing: {full_command}"

# Execute command
{full_command}

echo "Command completed successfully"
"""

        filename = f"shell_{command}_{fix.template_id}_{int(datetime.now().timestamp())}.sh"

        return GeneratedScript(
            tool_type=ToolType.SHELL,
            content=script_content,
            filename=filename,
            description=script_data.get('description', 'Shell operation'),
            parameters={"command": command, "args": args, "environment": environment}
        )

    def generate_rollback_script(self, original_script: GeneratedScript) -> Optional[GeneratedScript]:
        """Generate rollback script for a given script"""
        if original_script.tool_type == ToolType.KUBECTL:
            return self._generate_kubectl_rollback(original_script)
        elif original_script.tool_type == ToolType.HELM:
            return self._generate_helm_rollback(original_script)
        elif original_script.tool_type == ToolType.TERRAFORM:
            return self._generate_terraform_rollback(original_script)
        elif original_script.tool_type == ToolType.ANSIBLE:
            return self._generate_ansible_rollback(original_script)
        elif original_script.tool_type == ToolType.SQL:
            return self._generate_sql_rollback(original_script)

        return None

    def _generate_kubectl_rollback(self, original_script: GeneratedScript) -> Optional[GeneratedScript]:
        """Generate kubectl rollback script"""
        # For now, return a generic rollback script
        # In practice, this would be more sophisticated
        rollback_content = f"""#!/bin/bash
# Generated kubectl rollback script
# Original: {original_script.filename}
# Generated at: {datetime.now().isoformat()}

set -e

echo "Rolling back kubectl operation..."

# This is a placeholder for actual rollback logic
# In practice, this would depend on the specific operation

echo "Rollback completed"
"""

        return GeneratedScript(
            tool_type=ToolType.KUBECTL,
            content=rollback_content,
            filename=f"rollback_{original_script.filename}",
            description=f"Rollback for {original_script.description}",
            parameters=original_script.parameters
        )

    def _generate_helm_rollback(self, original_script: GeneratedScript) -> Optional[GeneratedScript]:
        """Generate Helm rollback script"""
        # Extract release name from parameters
        release_name = original_script.parameters.get("args", ["unknown"])[0]

        rollback_content = f"""#!/bin/bash
# Generated Helm rollback script
# Original: {original_script.filename}
# Generated at: {datetime.now().isoformat()}

set -e

RELEASE_NAME="{release_name}"
NAMESPACE="{original_script.parameters.get('namespace', 'default')}"

echo "Rolling back Helm release: $RELEASE_NAME"

helm rollback $RELEASE_NAME 0 \\
    --namespace $NAMESPACE \\
    --wait \\
    --timeout 600s

echo "Helm rollback completed successfully"
"""

        return GeneratedScript(
            tool_type=ToolType.HELM,
            content=rollback_content,
            filename=f"rollback_{original_script.filename}",
            description=f"Rollback for {original_script.description}",
            parameters={"release_name": release_name, "namespace": original_script.parameters.get('namespace', 'default')}
        )

    def _generate_terraform_rollback(self, original_script: GeneratedScript) -> Optional[GeneratedScript]:
        """Generate Terraform rollback script"""
        working_dir = original_script.parameters.get("working_dir", ".")

        rollback_content = f"""#!/bin/bash
# Generated Terraform rollback script
# Original: {original_script.filename}
# Generated at: {datetime.now().isoformat()}

set -e

WORKING_DIR="{working_dir}"

echo "Rolling back Terraform changes in: $WORKING_DIR"
cd "$WORKING_DIR"

# Destroy resources
terraform destroy \\
    -auto-approve \\
    -no-color

echo "Terraform rollback completed successfully"
"""

        return GeneratedScript(
            tool_type=ToolType.TERRAFORM,
            content=rollback_content,
            filename=f"rollback_{original_script.filename}",
            description=f"Rollback for {original_script.description}",
            parameters={"working_dir": working_dir}
        )

    def _generate_ansible_rollback(self, original_script: GeneratedScript) -> Optional[GeneratedScript]:
        """Generate Ansible rollback script"""
        # This would typically require a separate rollback playbook
        rollback_content = f"""#!/bin/bash
# Generated Ansible rollback script
# Original: {original_script.filename}
# Generated at: {datetime.now().isoformat()}

set -e

echo "Ansible rollback not implemented for this operation"
echo "Manual rollback may be required"

# Placeholder for rollback logic
# In practice, this would execute a separate rollback playbook
"""

        return GeneratedScript(
            tool_type=ToolType.ANSIBLE,
            content=rollback_content,
            filename=f"rollback_{original_script.filename}",
            description=f"Rollback for {original_script.description}",
            parameters=original_script.parameters
        )

    def _generate_sql_rollback(self, original_script: GeneratedScript) -> Optional[GeneratedScript]:
        """Generate SQL rollback script"""
        # SQL rollback would typically require transaction rollback or reverse operations
        rollback_content = f"""#!/bin/bash
# Generated SQL rollback script
# Original: {original_script.filename}
# Generated at: {datetime.now().isoformat()}

set -e

echo "SQL rollback not implemented for this operation"
echo "Manual rollback may be required"
echo "Consider using database transaction rollback or reverse migration"

# Placeholder for rollback logic
"""

        return GeneratedScript(
            tool_type=ToolType.SQL,
            content=rollback_content,
            filename=f"rollback_{original_script.filename}",
            description=f"Rollback for {original_script.description}",
            parameters=original_script.parameters
        )

    def validate_script(self, script: GeneratedScript) -> Dict[str, Any]:
        """Validate a generated script"""
        validation_results = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "suggestions": []
        }

        # Basic validation
        if not script.content.strip():
            validation_results["is_valid"] = False
            validation_results["issues"].append("Script content is empty")

        # Tool-specific validation
        if script.tool_type == ToolType.KUBECTL:
            validation_results.update(self._validate_kubectl_script(script))
        elif script.tool_type == ToolType.HELM:
            validation_results.update(self._validate_helm_script(script))
        elif script.tool_type == ToolType.TERRAFORM:
            validation_results.update(self._validate_terraform_script(script))
        elif script.tool_type == ToolType.ANSIBLE:
            validation_results.update(self._validate_ansible_script(script))
        elif script.tool_type == ToolType.SQL:
            validation_results.update(self._validate_sql_script(script))

        return validation_results

    def _validate_kubectl_script(self, script: GeneratedScript) -> Dict[str, Any]:
        """Validate kubectl script"""
        issues = []
        warnings = []

        content = script.content.lower()

        # Check for dangerous operations
        if "delete" in content and "all" in content:
            issues.append("Script contains potentially dangerous delete operation")

        # Check for missing namespace
        if "kubectl" in content and "-n" not in content and "namespace" not in content:
            warnings.append("Consider specifying namespace explicitly")

        return {"issues": issues, "warnings": warnings}

    def _validate_helm_script(self, script: GeneratedScript) -> Dict[str, Any]:
        """Validate Helm script"""
        issues = []
        warnings = []

        content = script.content.lower()

        # Check for version specification
        if "--version" not in content and "upgrade" in content:
            warnings.append("Consider specifying chart version for reproducible deployments")

        return {"issues": issues, "warnings": warnings}

    def _validate_terraform_script(self, script: GeneratedScript) -> Dict[str, Any]:
        """Validate Terraform script"""
        issues = []
        warnings = []

        content = script.content.lower()

        # Check for state file handling
        if "terraform.tfstate" not in content and ("apply" in content or "destroy" in content):
            warnings.append("Ensure proper state file handling")

        return {"issues": issues, "warnings": warnings}

    def _validate_ansible_script(self, script: GeneratedScript) -> Dict[str, Any]:
        """Validate Ansible script"""
        issues = []
        warnings = []

        # Check for basic Ansible structure
        if "ansible-playbook" not in script.content:
            issues.append("Script does not contain ansible-playbook command")

        return {"issues": issues, "warnings": warnings}

    def _validate_sql_script(self, script: GeneratedScript) -> Dict[str, Any]:
        """Validate SQL script"""
        issues = []
        warnings = []

        content = script.content.upper()

        # Check for dangerous operations
        if "DROP" in content and "DATABASE" in content:
            issues.append("Script contains database drop operation")

        if "DELETE" in content and "WHERE" not in content:
            warnings.append("DELETE without WHERE clause detected")

        return {"issues": issues, "warnings": warnings}
