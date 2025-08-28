"""
Secrets vault for secure storage of connector credentials
"""
import os
import json
import base64
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..database.config import get_postgres_session
from ..models.base import Secret


logger = logging.getLogger(__name__)


class SecretError(Exception):
    """Base exception for secret operations"""
    pass


class SecretNotFoundError(SecretError):
    """Exception raised when secret is not found"""
    pass


class SecretAccessError(SecretError):
    """Exception raised when access to secret is denied"""
    pass


class SecretsVault:
    """Secure vault for storing and retrieving secrets"""

    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or os.getenv("VAULT_MASTER_KEY")
        if not self.master_key:
            raise SecretError("Master key not provided")

        self.fernet = self._create_cipher()

    def _create_cipher(self) -> Fernet:
        """Create Fernet cipher for encryption/decryption"""
        # Derive encryption key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'ai-devops-copilot',  # Fixed salt for consistent encryption
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        return Fernet(key)

    def encrypt(self, data: Dict[str, Any]) -> str:
        """Encrypt data dictionary"""
        json_str = json.dumps(data, sort_keys=True)
        encrypted = self.fernet.encrypt(json_str.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt data to dictionary"""
        try:
            encrypted = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted)
            return json.loads(decrypted.decode())
        except Exception as e:
            raise SecretError(f"Failed to decrypt data: {e}")

    async def store_secret(
        self,
        org_id: str,
        secret_name: str,
        secret_data: Dict[str, Any],
        user_id: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """Store a secret in the vault"""
        try:
            async with get_postgres_session() as session:
                # Encrypt the secret data
                encrypted_data = self.encrypt(secret_data)

                # Create secret record
                secret = Secret(
                    org_id=org_id,
                    name=secret_name,
                    encrypted_data=encrypted_data,
                    description=description,
                    created_by=user_id,
                    updated_by=user_id
                )

                session.add(secret)
                await session.commit()
                await session.refresh(secret)

                logger.info(f"Stored secret: {secret_name} for org: {org_id}")
                return str(secret.id)

        except Exception as e:
            logger.error(f"Failed to store secret {secret_name}: {e}")
            raise SecretError(f"Failed to store secret: {e}")

    async def retrieve_secret(
        self,
        org_id: str,
        secret_name: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieve a secret from the vault"""
        try:
            async with get_postgres_session() as session:
                # Find the secret
                result = await session.execute(
                    """
                    SELECT id, encrypted_data, created_at, updated_at
                    FROM secrets
                    WHERE org_id = :org_id AND name = :name
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    {"org_id": org_id, "name": secret_name}
                )

                row = result.fetchone()
                if not row:
                    raise SecretNotFoundError(f"Secret not found: {secret_name}")

                secret_id, encrypted_data, created_at, updated_at = row

                # Decrypt the data
                secret_data = self.decrypt(encrypted_data)

                # Log access
                await self._log_secret_access(secret_id, user_id, "retrieve")

                logger.info(f"Retrieved secret: {secret_name} for org: {org_id}")
                return secret_data

        except SecretNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            raise SecretError(f"Failed to retrieve secret: {e}")

    async def update_secret(
        self,
        org_id: str,
        secret_name: str,
        secret_data: Dict[str, Any],
        user_id: Optional[str] = None,
        description: Optional[str] = None
    ) -> None:
        """Update an existing secret"""
        try:
            async with get_postgres_session() as session:
                # Encrypt the new data
                encrypted_data = self.encrypt(secret_data)

                # Update the secret
                result = await session.execute(
                    """
                    UPDATE secrets
                    SET encrypted_data = :encrypted_data,
                        description = :description,
                        updated_by = :user_id,
                        updated_at = :updated_at
                    WHERE org_id = :org_id AND name = :name
                    """,
                    {
                        "org_id": org_id,
                        "name": secret_name,
                        "encrypted_data": encrypted_data,
                        "description": description,
                        "user_id": user_id,
                        "updated_at": datetime.utcnow()
                    }
                )

                if result.rowcount == 0:
                    raise SecretNotFoundError(f"Secret not found: {secret_name}")

                await session.commit()

                logger.info(f"Updated secret: {secret_name} for org: {org_id}")

        except SecretNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update secret {secret_name}: {e}")
            raise SecretError(f"Failed to update secret: {e}")

    async def delete_secret(
        self,
        org_id: str,
        secret_name: str,
        user_id: Optional[str] = None
    ) -> None:
        """Delete a secret from the vault"""
        try:
            async with get_postgres_session() as session:
                # Find the secret first for logging
                result = await session.execute(
                    """
                    SELECT id FROM secrets
                    WHERE org_id = :org_id AND name = :name
                    """,
                    {"org_id": org_id, "name": secret_name}
                )

                row = result.fetchone()
                if not row:
                    raise SecretNotFoundError(f"Secret not found: {secret_name}")

                secret_id = row[0]

                # Delete the secret
                await session.execute(
                    """
                    DELETE FROM secrets
                    WHERE org_id = :org_id AND name = :name
                    """,
                    {"org_id": org_id, "name": secret_name}
                )

                await session.commit()

                # Log deletion
                await self._log_secret_access(secret_id, user_id, "delete")

                logger.info(f"Deleted secret: {secret_name} for org: {org_id}")

        except SecretNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete secret {secret_name}: {e}")
            raise SecretError(f"Failed to delete secret: {e}")

    async def list_secrets(self, org_id: str) -> list:
        """List all secrets for an organization"""
        try:
            async with get_postgres_session() as session:
                result = await session.execute(
                    """
                    SELECT name, description, created_at, updated_at
                    FROM secrets
                    WHERE org_id = :org_id
                    ORDER BY name
                    """,
                    {"org_id": org_id}
                )

                secrets = []
                for row in result.fetchall():
                    secrets.append({
                        "name": row[0],
                        "description": row[1],
                        "created_at": row[2],
                        "updated_at": row[3]
                    })

                return secrets

        except Exception as e:
            logger.error(f"Failed to list secrets for org {org_id}: {e}")
            raise SecretError(f"Failed to list secrets: {e}")

    async def _log_secret_access(
        self,
        secret_id: str,
        user_id: Optional[str],
        action: str
    ) -> None:
        """Log secret access for audit purposes"""
        try:
            async with get_postgres_session() as session:
                await session.execute(
                    """
                    INSERT INTO audit_log (org_id, user_id, action, target, target_id, meta)
                    SELECT s.org_id, :user_id, :action, 'secret', :secret_id, '{}'
                    FROM secrets s WHERE s.id = :secret_id
                    """,
                    {
                        "secret_id": secret_id,
                        "user_id": user_id,
                        "action": action
                    }
                )
                await session.commit()

        except Exception as e:
            logger.error(f"Failed to log secret access: {e}")
            # Don't fail the main operation if logging fails

    async def rotate_master_key(self, new_master_key: str) -> None:
        """Rotate the master encryption key"""
        try:
            # This would need to decrypt all secrets with old key and re-encrypt with new key
            # Implementation would depend on specific requirements
            logger.warning("Master key rotation not fully implemented")
            raise NotImplementedError("Master key rotation not implemented")

        except Exception as e:
            logger.error(f"Failed to rotate master key: {e}")
            raise SecretError(f"Failed to rotate master key: {e}")


class ConnectorSecretsManager:
    """Manager for connector-specific secrets"""

    def __init__(self, vault: SecretsVault):
        self.vault = vault

    async def store_connector_secret(
        self,
        org_id: str,
        connector_id: str,
        connector_type: str,
        secret_data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> str:
        """Store secrets for a specific connector"""
        secret_name = f"connector:{connector_type}:{connector_id}"
        description = f"Secrets for {connector_type} connector {connector_id}"

        return await self.vault.store_secret(
            org_id=org_id,
            secret_name=secret_name,
            secret_data=secret_data,
            user_id=user_id,
            description=description
        )

    async def retrieve_connector_secret(
        self,
        org_id: str,
        connector_id: str,
        connector_type: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieve secrets for a specific connector"""
        secret_name = f"connector:{connector_type}:{connector_id}"

        return await self.vault.retrieve_secret(
            org_id=org_id,
            secret_name=secret_name,
            user_id=user_id
        )

    async def update_connector_secret(
        self,
        org_id: str,
        connector_id: str,
        connector_type: str,
        secret_data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> None:
        """Update secrets for a specific connector"""
        secret_name = f"connector:{connector_type}:{connector_id}"
        description = f"Secrets for {connector_type} connector {connector_id}"

        await self.vault.update_secret(
            org_id=org_id,
            secret_name=secret_name,
            secret_data=secret_data,
            user_id=user_id,
            description=description
        )

    async def delete_connector_secret(
        self,
        org_id: str,
        connector_id: str,
        connector_type: str,
        user_id: Optional[str] = None
    ) -> None:
        """Delete secrets for a specific connector"""
        secret_name = f"connector:{connector_type}:{connector_id}"

        await self.vault.delete_secret(
            org_id=org_id,
            secret_name=secret_name,
            user_id=user_id
        )


# Global instances
secrets_vault = SecretsVault()
connector_secrets_manager = ConnectorSecretsManager(secrets_vault)
