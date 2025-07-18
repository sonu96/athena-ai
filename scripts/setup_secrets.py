#!/usr/bin/env python3
"""
Script to set up secrets in Google Secret Manager
"""
import os
import sys
from getpass import getpass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gcp.secret_manager import SecretManagerClient


def main():
    """Set up secrets in Google Secret Manager."""
    print("🔐 Athena AI - Secret Setup")
    print("=" * 40)
    
    # Check for project ID
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        project_id = input("Enter your GCP Project ID: ")
        os.environ["GCP_PROJECT_ID"] = project_id
    
    print(f"\nUsing project: {project_id}")
    
    # Initialize Secret Manager
    try:
        secret_manager = SecretManagerClient(project_id)
    except Exception as e:
        print(f"❌ Failed to initialize Secret Manager: {e}")
        print("Make sure you have authenticated with gcloud:")
        print("  gcloud auth application-default login")
        return
    
    # Secrets to create
    secrets = [
        {
            "id": "cdp-api-key",
            "name": "CDP API Key",
            "required": True,
            "description": "Your CDP API key from Coinbase Developer Platform"
        },
        {
            "id": "cdp-api-secret", 
            "name": "CDP API Secret",
            "required": True,
            "description": "Your CDP API secret from Coinbase Developer Platform"
        },
        {
            "id": "cdp-client-api-key",
            "name": "CDP Client API Key",
            "required": False,
            "description": "CDP Client API Key for authenticated RPC access (real-time data)"
        },
        {
            "id": "langsmith-api-key",
            "name": "LangSmith API Key",
            "required": False,
            "description": "Optional: For observability (from smith.langchain.com)"
        },
        {
            "id": "mem0-api-key",
            "name": "Mem0 API Key",
            "required": False,
            "description": "Optional: For cloud memory storage"
        },
        {
            "id": "base-rpc-url",
            "name": "Base RPC URL",
            "required": False,
            "description": "Optional: Custom Base RPC URL (default: https://mainnet.base.org)"
        }
    ]
    
    print("\n📝 Let's set up your secrets...")
    print("(Press Enter to skip optional secrets)\n")
    
    created_count = 0
    
    for secret in secrets:
        print(f"\n{secret['name']} {'(Required)' if secret['required'] else '(Optional)'}")
        print(f"  {secret['description']}")
        
        # Check if secret already exists
        try:
            existing = secret_manager.get_secret(secret['id'])
            overwrite = input(f"  Secret already exists. Overwrite? (y/N): ").lower() == 'y'
            if not overwrite:
                continue
        except:
            pass  # Secret doesn't exist
        
        # Get secret value
        if secret['id'].endswith('-secret') or secret['id'].endswith('-key'):
            value = getpass(f"  Enter value: ")
        else:
            value = input(f"  Enter value: ")
        
        # Skip if empty and optional
        if not value and not secret['required']:
            print("  Skipped (optional)")
            continue
        
        # Validate required
        if not value and secret['required']:
            print(f"  ❌ {secret['name']} is required!")
            continue
        
        # Create or update secret
        try:
            secret_manager.create_secret(secret['id'], value)
            print(f"  ✅ Created secret: {secret['id']}")
            created_count += 1
        except Exception as e:
            if "already exists" in str(e):
                # Update existing secret
                try:
                    secret_manager.update_secret(secret['id'], value)
                    print(f"  ✅ Updated secret: {secret['id']}")
                    created_count += 1
                except Exception as update_error:
                    print(f"  ❌ Failed to update: {update_error}")
            else:
                print(f"  ❌ Failed to create: {e}")
    
    print(f"\n✅ Setup complete! Created/updated {created_count} secrets.")
    
    # Show next steps
    print("\n📋 Next steps:")
    print("1. Make sure you have enabled required Google Cloud APIs:")
    print("   gcloud services enable secretmanager.googleapis.com")
    print("   gcloud services enable aiplatform.googleapis.com")
    print("   gcloud services enable firestore.googleapis.com")
    print("\n2. Grant your service account access to secrets:")
    print("   gcloud secrets add-iam-policy-binding <secret-id> \\")
    print("     --member='serviceAccount:YOUR-SA@PROJECT.iam.gserviceaccount.com' \\")
    print("     --role='roles/secretmanager.secretAccessor'")
    print("\n3. Run Athena:")
    print("   python run.py")


if __name__ == "__main__":
    main()