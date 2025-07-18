"""
Google Secret Manager Integration
"""
import os
import logging
from typing import Optional
from google.cloud import secretmanager
from google.api_core import exceptions

logger = logging.getLogger(__name__)


class SecretManagerClient:
    """Client for accessing secrets from Google Secret Manager."""
    
    def __init__(self, project_id: Optional[str] = None):
        """Initialize Secret Manager client."""
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID must be set")
            
        self.client = secretmanager.SecretManagerServiceClient()
        self._cache = {}  # Cache secrets to avoid repeated API calls
        
    def get_secret(self, secret_id: str, version: str = "latest") -> str:
        """
        Retrieve a secret from Secret Manager.
        
        Args:
            secret_id: The ID of the secret
            version: The version of the secret (default: "latest")
            
        Returns:
            The secret value as a string
        """
        cache_key = f"{secret_id}:{version}"
        
        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        try:
            # Build the resource name
            name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
            
            # Access the secret
            response = self.client.access_secret_version(request={"name": name})
            secret_value = response.payload.data.decode("UTF-8")
            
            # Cache the result
            self._cache[cache_key] = secret_value
            
            logger.info(f"Successfully retrieved secret: {secret_id}")
            return secret_value
            
        except exceptions.NotFound:
            logger.error(f"Secret not found: {secret_id}")
            raise ValueError(f"Secret '{secret_id}' not found in project {self.project_id}")
        except Exception as e:
            logger.error(f"Error accessing secret {secret_id}: {e}")
            raise
            
    def get_secret_or_env(self, secret_id: str, env_var: str) -> str:
        """
        Try to get secret from Secret Manager, fall back to environment variable.
        
        Args:
            secret_id: The ID of the secret in Secret Manager
            env_var: The environment variable name to fall back to
            
        Returns:
            The secret value
        """
        # First try environment variable (for local development)
        env_value = os.getenv(env_var)
        if env_value:
            logger.debug(f"Using {env_var} from environment")
            return env_value
            
        # Then try Secret Manager (for production)
        try:
            return self.get_secret(secret_id)
        except Exception as e:
            logger.error(f"Failed to get secret {secret_id} and no env var {env_var}: {e}")
            raise ValueError(f"Could not retrieve {secret_id} from Secret Manager or {env_var} from environment")
            
    def create_secret(self, secret_id: str, secret_value: str) -> None:
        """
        Create a new secret in Secret Manager.
        
        Args:
            secret_id: The ID for the new secret
            secret_value: The secret value to store
        """
        try:
            # Create the secret
            parent = f"projects/{self.project_id}"
            secret = self.client.create_secret(
                request={
                    "parent": parent,
                    "secret_id": secret_id,
                    "secret": {"replication": {"automatic": {}}},
                }
            )
            
            # Add secret version
            self.client.add_secret_version(
                request={
                    "parent": secret.name,
                    "payload": {"data": secret_value.encode("UTF-8")},
                }
            )
            
            logger.info(f"Created secret: {secret_id}")
            
        except exceptions.AlreadyExists:
            logger.warning(f"Secret already exists: {secret_id}")
        except Exception as e:
            logger.error(f"Error creating secret {secret_id}: {e}")
            raise
            
    def update_secret(self, secret_id: str, secret_value: str) -> None:
        """
        Update an existing secret by adding a new version.
        
        Args:
            secret_id: The ID of the secret to update
            secret_value: The new secret value
        """
        try:
            parent = f"projects/{self.project_id}/secrets/{secret_id}"
            
            # Add new version
            self.client.add_secret_version(
                request={
                    "parent": parent,
                    "payload": {"data": secret_value.encode("UTF-8")},
                }
            )
            
            # Clear cache
            self._cache = {k: v for k, v in self._cache.items() if not k.startswith(f"{secret_id}:")}
            
            logger.info(f"Updated secret: {secret_id}")
            
        except Exception as e:
            logger.error(f"Error updating secret {secret_id}: {e}")
            raise
            
    def create_or_update_secret(self, secret_id: str, secret_value: str) -> None:
        """
        Create a secret if it doesn't exist, or update it if it does.
        
        Args:
            secret_id: The ID of the secret
            secret_value: The secret value to store
        """
        try:
            # Try to update first (secret exists)
            self.update_secret(secret_id, secret_value)
        except exceptions.NotFound:
            # Secret doesn't exist, create it
            self.create_secret(secret_id, secret_value)


# Global instance
_secret_manager: Optional[SecretManagerClient] = None


def get_secret_manager() -> SecretManagerClient:
    """Get or create the global Secret Manager client."""
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = SecretManagerClient()
    return _secret_manager


def get_secret(secret_id: str, env_var: Optional[str] = None) -> str:
    """
    Convenience function to get a secret.
    
    Args:
        secret_id: The secret ID in Secret Manager
        env_var: Optional environment variable to check first
        
    Returns:
        The secret value
    """
    manager = get_secret_manager()
    
    if env_var:
        return manager.get_secret_or_env(secret_id, env_var)
    else:
        return manager.get_secret(secret_id)


def create_or_update_secret(secret_id: str, secret_value: str) -> None:
    """
    Convenience function to create or update a secret.
    
    Args:
        secret_id: The secret ID in Secret Manager
        secret_value: The secret value to store
    """
    manager = get_secret_manager()
    manager.create_or_update_secret(secret_id, secret_value)