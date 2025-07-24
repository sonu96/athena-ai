#!/usr/bin/env python3
"""
Analyze what pools are being tracked in memory and patterns.
"""
import os
from google.cloud import firestore
from datetime import datetime
from collections import defaultdict

# Set up Google Cloud project
project_id = os.getenv('GCP_PROJECT_ID', 'athena-ai-435923')
os.environ['GOOGLE_CLOUD_PROJECT'] = project_id

# Initialize Firestore
db = firestore.Client(project=project_id)

def analyze_patterns():
    """Analyze observed patterns for pool references."""
    print("Analyzing observed patterns for pool data...\n")
    
    patterns_ref = db.collection('observed_patterns')
    patterns = list(patterns_ref.stream())
    
    pool_patterns = defaultdict(list)
    pool_mentions = defaultdict(int)
    
    for doc in patterns:
        data = doc.to_dict()
        description = data.get('description', '')
        pattern_type = data.get('type', '')
        
        # Look for pool references
        pools = []
        if 'AERO/USDC' in description:
            pools.append('AERO/USDC')
        if 'AERO/WETH' in description:
            pools.append('AERO/WETH')
        if 'WETH/USDC' in description:
            pools.append('WETH/USDC')
        if 'USDC/USDbC' in description:
            pools.append('USDC/USDbC')
        if 'cbETH/WETH' in description:
            pools.append('cbETH/WETH')
        
        for pool in pools:
            pool_mentions[pool] += 1
            pool_patterns[pool].append({
                'type': pattern_type,
                'description': description[:100],
                'confidence': data.get('confidence', 0)
            })
    
    print(f"Total patterns analyzed: {len(patterns)}")
    print(f"\nPools mentioned in patterns:")
    for pool, count in sorted(pool_mentions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pool}: {count} mentions")
    
    # Show sample patterns for each pool
    print("\nSample patterns by pool:")
    for pool, patterns in pool_patterns.items():
        print(f"\n{pool}:")
        for p in patterns[:3]:  # Show first 3
            print(f"  - {p['type']}: {p['description']}... (conf: {p['confidence']:.2f})")

def check_pool_profiles():
    """Check existing pool profiles."""
    print("\n\nChecking pool profiles...\n")
    
    profiles_ref = db.collection('pool_profiles')
    profiles = list(profiles_ref.stream())
    
    print(f"Total pool profiles: {len(profiles)}")
    
    for doc in profiles:
        data = doc.to_dict()
        print(f"\nProfile: {doc.id}")
        print(f"  Pair: {data.get('pair', 'N/A')}")
        print(f"  Observations: {data.get('observations_count', 0)}")
        print(f"  Confidence: {data.get('confidence_score', 0):.2%}")
        print(f"  APR Range: {data.get('apr_range', [0, 0])}")
        print(f"  TVL Range: ${data.get('tvl_range', [0, 0])[0]:,.0f} - ${data.get('tvl_range', [0, 0])[1]:,.0f}")
        print(f"  Volume Range: ${data.get('volume_range', [0, 0])[0]:,.0f} - ${data.get('volume_range', [0, 0])[1]:,.0f}")

def check_cycles_for_pools():
    """Check reasoning cycles for pool observations."""
    print("\n\nChecking recent cycles for pool data...\n")
    
    cycles_ref = db.collection('cycles').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(10)
    cycles = list(cycles_ref.stream())
    
    pools_observed = defaultdict(int)
    
    for doc in cycles:
        data = doc.to_dict()
        state = data.get('state', {})
        observations = state.get('observations', [])
        
        for obs in observations:
            if obs.get('type') == 'pools':
                pool_data = obs.get('data', {})
                # Try to identify which pool
                if isinstance(pool_data, dict):
                    pair = pool_data.get('pair', 'unknown')
                    pools_observed[pair] += 1
    
    print(f"Pools observed in last 10 cycles:")
    for pool, count in pools_observed.items():
        print(f"  {pool}: {count} times")

if __name__ == "__main__":
    analyze_patterns()
    check_pool_profiles()
    check_cycles_for_pools()