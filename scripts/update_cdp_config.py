#!/usr/bin/env python3
"""
Update CDP credentials in Google Secret Manager
"""
import os
import json
import sys
from google.cloud import secretmanager

def update_cdp_secrets(project_id: str, cdp_json_path: str):
    """Update CDP credentials in Secret Manager."""
    
    # Initialize Secret Manager client
    client = secretmanager.SecretManagerServiceClient()
    project_path = f"projects/{project_id}"
    
    # Load CDP credentials from JSON file
    print(f"Loading CDP credentials from: {cdp_json_path}")
    with open(cdp_json_path, 'r') as f:
        cdp_data = json.load(f)
    
    # Extract credentials
    api_key_id = cdp_data.get('id')
    api_key_secret = cdp_data.get('privateKey')
    
    if not api_key_id or not api_key_secret:
        print("❌ Error: Invalid CDP JSON file format")
        return False
    
    print(f"CDP API Key ID: {api_key_id}")
    print(f"CDP API Key Secret: {api_key_secret[:20]}...")
    
    # Update secrets
    secrets_to_update = {
        'cdp-api-key': api_key_id,
        'cdp-api-secret': api_key_secret
    }
    
    for secret_id, secret_value in secrets_to_update.items():
        try:
            # Check if secret exists
            secret_name = f"{project_path}/secrets/{secret_id}"
            try:
                secret = client.get_secret(request={"name": secret_name})
                print(f"✅ Secret '{secret_id}' exists")
                
                # Delete old versions (keeping only the latest)
                print(f"  Cleaning up old versions...")
                versions = client.list_secret_versions(
                    request={"parent": secret_name}
                )
                version_count = 0
                for version in versions:
                    version_count += 1
                    if version.state == secretmanager.SecretVersion.State.ENABLED and version_count > 1:
                        # Destroy old versions
                        client.destroy_secret_version(
                            request={"name": version.name}
                        )
                        print(f"  Destroyed old version: {version.name.split('/')[-1]}")
                
            except Exception:
                # Secret doesn't exist, create it
                print(f"📝 Creating new secret '{secret_id}'")
                secret = client.create_secret(
                    request={
                        "parent": project_path,
                        "secret_id": secret_id,
                        "secret": {"replication": {"automatic": {}}},
                    }
                )
            
            # Add new secret version
            print(f"  Adding new version...")
            version = client.add_secret_version(
                request={
                    "parent": secret_name,
                    "payload": {"data": secret_value.encode("UTF-8")},
                }
            )
            print(f"✅ Updated '{secret_id}' with new version: {version.name.split('/')[-1]}")
            
        except Exception as e:
            print(f"❌ Error updating secret '{secret_id}': {e}")
            return False
    
    # Check for wallet secret
    wallet_secret = os.environ.get('CDP_WALLET_SECRET')
    if wallet_secret:
        print("\n📝 Saving CDP wallet secret...")
        try:
            secret_id = 'cdp-wallet-secret'
            secret_name = f"{project_path}/secrets/{secret_id}"
            
            # Try to create the secret
            try:
                secret = client.create_secret(
                    request={
                        "parent": project_path,
                        "secret_id": secret_id,
                        "secret": {"replication": {"automatic": {}}},
                    }
                )
                print(f"✅ Created new secret '{secret_id}'")
            except:
                print(f"✅ Secret '{secret_id}' already exists")
            
            # Add secret version
            version = client.add_secret_version(
                request={
                    "parent": secret_name,
                    "payload": {"data": wallet_secret.encode("UTF-8")},
                }
            )
            print(f"✅ Saved wallet secret")
        except Exception as e:
            print(f"⚠️  Warning: Could not save wallet secret: {e}")
    
    print("\n✅ CDP secrets updated successfully!")
    print("\nNext steps:")
    print("1. Enable appropriate permissions in CDP dashboard (Trade, Transfer)")
    print("2. Download new API key after updating permissions")
    print("3. Run this script again with the new JSON file")
    
    return True


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python update_cdp_secrets.py <cdp_api_key.json>")
        print("\nExample:")
        print("  python update_cdp_secrets.py '/Users/you/Downloads/cdp_api_key.json'")
        sys.exit(1)
    
    cdp_json_path = sys.argv[1]
    if not os.path.exists(cdp_json_path):
        print(f"❌ Error: File not found: {cdp_json_path}")
        sys.exit(1)
    
    # Get project ID from environment or settings
    project_id = os.environ.get('GCP_PROJECT_ID', 'athena-defi-agent-1752635199')
    
    print("🔐 CDP Secret Manager Update")
    print("=" * 50)
    print(f"Project ID: {project_id}")
    print(f"CDP JSON: {cdp_json_path}")
    print()
    
    success = update_cdp_secrets(project_id, cdp_json_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()