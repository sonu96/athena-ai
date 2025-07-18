#!/usr/bin/env python3
"""
Test CDP Setup and Configuration
"""
import asyncio
import os
import sys

# Set environment
os.environ['GCP_PROJECT_ID'] = 'athena-defi-agent-1752635199'

async def test_cdp_setup():
    """Test the CDP setup."""
    print("🧪 Testing CDP Setup")
    print("=" * 50)
    
    # Test 1: Version check
    print("\n1️⃣ Testing CDP SDK version...")
    try:
        import cdp
        print(f"✅ CDP SDK version: {cdp.__version__}")
    except Exception as e:
        print(f"❌ Version check failed: {e}")
        return False
    
    # Test 2: Settings loading
    print("\n2️⃣ Testing settings...")
    try:
        from config.settings import settings
        print(f"✅ GCP Project: {settings.gcp_project_id}")
        print(f"✅ CDP API Key: {settings.cdp_api_key}")
        print(f"✅ CDP API Secret: {'*' * 20}...")
        print(f"✅ Wallet Secret: {'Loaded' if settings.cdp_wallet_secret else 'Not set (will be generated)'}")
    except Exception as e:
        print(f"❌ Settings failed: {e}")
        return False
    
    # Test 3: CDP Client initialization
    print("\n3️⃣ Testing CDP client...")
    try:
        from cdp import CdpClient
        
        # Create client
        client = CdpClient(
            api_key_id=settings.cdp_api_key,
            api_key_secret=settings.cdp_api_secret
        )
        print("✅ CDP client created")
        
        # Test API access
        accounts = await client.evm.list_accounts()
        print(f"✅ API access works! Found {len(accounts.data) if hasattr(accounts, 'data') else 0} accounts")
        
        await client.close()
    except Exception as e:
        print(f"❌ CDP client failed: {e}")
        return False
    
    # Test 4: Base client initialization (without wallet creation)
    print("\n4️⃣ Testing BaseClient...")
    try:
        from src.cdp.base_client import BaseClient
        base_client = BaseClient()
        print("✅ BaseClient created")
        
        # Don't initialize (would try to create wallet)
        print("⚠️  Skipping wallet creation test (requires proper permissions)")
    except Exception as e:
        print(f"❌ BaseClient failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ All tests passed!")
    print("\n📋 Next steps:")
    print("1. Enable 'Trade' permission in CDP dashboard")
    print("2. Download new API key with proper permissions")
    print("3. Run: python scripts/update_cdp_secrets.py <new_json_file>")
    print("4. Run the agent: python run.py")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_cdp_setup())
    sys.exit(0 if success else 1)