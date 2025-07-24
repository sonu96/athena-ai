#!/usr/bin/env python3
"""
Diagnose pool profile creation issues.
"""
import os
from google.cloud import firestore
from datetime import datetime, timedelta
from collections import defaultdict

# Set up Google Cloud project
project_id = os.getenv('GCP_PROJECT_ID', 'athena-defi-agent-1752635199')
os.environ['GOOGLE_CLOUD_PROJECT'] = project_id

# Initialize Firestore
db = firestore.Client(project=project_id)

def check_pool_memories():
    """Check what pools are being tracked in memories."""
    print("Checking pool-related memories...\n")
    
    # Check observed_patterns for pool mentions
    patterns_ref = db.collection('observed_patterns').order_by('discovered_at', direction=firestore.Query.DESCENDING).limit(100)
    patterns = list(patterns_ref.stream())
    
    pool_mentions = defaultdict(int)
    unique_addresses = set()
    
    for doc in patterns:
        data = doc.to_dict()
        description = data.get('description', '')
        
        # Look for pool addresses in patterns
        if '0x' in description:
            # Extract potential addresses
            words = description.split()
            for word in words:
                if word.startswith('0x') and len(word) == 42:
                    unique_addresses.add(word.lower())
        
        # Count pool pair mentions
        for pool in ['WETH/USDC', 'AERO/USDC', 'AERO/WETH', 'WETH/DAI', 'USDC/DAI', 'USDC/USDbC']:
            if pool in description:
                pool_mentions[pool] += 1
    
    print(f"Pool pairs mentioned in patterns:")
    for pool, count in sorted(pool_mentions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pool}: {count} times")
    
    print(f"\nUnique pool addresses found: {len(unique_addresses)}")
    for addr in sorted(unique_addresses)[:5]:  # Show first 5
        print(f"  {addr}")

def check_recent_cycles():
    """Check what pools the agent has observed."""
    print("\n\nChecking recent agent cycles...\n")
    
    cycles_ref = db.collection('cycles').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(5)
    cycles = list(cycles_ref.stream())
    
    pools_observed = defaultdict(int)
    pool_addresses = set()
    
    for doc in cycles:
        data = doc.to_dict()
        state = data.get('state', {})
        observations = state.get('observations', [])
        
        print(f"Cycle {doc.id} at {data.get('timestamp', 'unknown')}:")
        
        for obs in observations:
            if obs.get('type') == 'pools':
                pool_data = obs.get('data', {})
                if isinstance(pool_data, dict):
                    address = pool_data.get('address')
                    if address:
                        pool_addresses.add(address)
                        print(f"  - Pool observed: {address}")
                    
                    # Check for pool errors
                    if 'error' in pool_data:
                        print(f"  - Pool error: {pool_data['error']}")
    
    print(f"\nUnique pool addresses observed by agent: {len(pool_addresses)}")
    for addr in pool_addresses:
        print(f"  {addr}")

def check_pool_scanner_data():
    """Check if pool scanner is storing data."""
    print("\n\nChecking pool scanner activity...\n")
    
    # Look for recent patterns that indicate pool scanning
    now = datetime.utcnow()
    recent_cutoff = now - timedelta(hours=1)
    
    patterns_ref = db.collection('observed_patterns').where('discovered_at', '>=', recent_cutoff).stream()
    patterns = list(patterns_ref)
    
    scanner_patterns = []
    for doc in patterns:
        data = doc.to_dict()
        if 'pool' in data.get('type', '').lower() or 'apr' in data.get('description', '').lower():
            scanner_patterns.append(data)
    
    print(f"Recent pool-related patterns (last hour): {len(scanner_patterns)}")
    
    # Show sample patterns
    for pattern in scanner_patterns[:5]:
        print(f"  - {pattern.get('type')}: {pattern.get('description', '')[:80]}...")

def check_firestore_errors():
    """Check for any Firestore write errors."""
    print("\n\nChecking for potential issues...\n")
    
    # Check pool_profiles collection permissions
    try:
        test_ref = db.collection('pool_profiles').document('test_write')
        test_ref.set({'test': True, 'timestamp': datetime.utcnow()})
        test_ref.delete()
        print("✅ Firestore write permissions: OK")
    except Exception as e:
        print(f"❌ Firestore write error: {e}")
    
    # Check if pool profiles have required fields
    profiles_ref = db.collection('pool_profiles').stream()
    profiles = list(profiles_ref)
    
    for doc in profiles:
        data = doc.to_dict()
        required_fields = ['pool_address', 'observations_count', 'confidence_score']
        missing = [f for f in required_fields if f not in data]
        if missing:
            print(f"⚠️  Profile {doc.id} missing fields: {missing}")

if __name__ == "__main__":
    print("=== Pool Profile Diagnostic Report ===\n")
    
    check_pool_memories()
    check_recent_cycles()
    check_pool_scanner_data()
    check_firestore_errors()
    
    print("\n=== Summary ===")
    print("The issue appears to be that:")
    print("1. The agent only observes one pool (WETH/USDC) directly")
    print("2. The pool scanner may not be getting pool addresses from the CDP client")
    print("3. Without addresses, pool profiles cannot be created")