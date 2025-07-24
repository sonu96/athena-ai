#!/usr/bin/env python3
"""
Script to inspect raw pool profile data in Firestore.
Shows all fields and nested data structures.
"""

import os
import json
from datetime import datetime
from google.cloud import firestore
from pprint import pprint


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def main():
    """Inspect pool profiles with raw data."""
    # Get project ID from environment
    project_id = os.environ.get('GCP_PROJECT_ID', 'athena-defi-agent-1752635199')
    
    print(f"Connecting to Firestore project: {project_id}")
    print("=" * 80)
    
    try:
        # Initialize Firestore client
        db = firestore.Client(project=project_id)
        
        # Get all documents from pool_profiles collection
        pool_profiles_ref = db.collection('pool_profiles')
        docs = list(pool_profiles_ref.stream())
        
        print(f"Found {len(docs)} pool profiles\n")
        
        if not docs:
            print("No pool profiles found in the database.")
            return
        
        # Inspect each profile
        for i, doc in enumerate(docs, 1):
            print(f"\n{'='*80}")
            print(f"POOL PROFILE #{i}")
            print(f"Document ID (Address): {doc.id}")
            print(f"{'='*80}\n")
            
            # Get the raw data
            data = doc.to_dict()
            
            # Pretty print the entire data structure
            print("RAW DATA:")
            print("-" * 80)
            pprint(data, width=100, depth=4)
            
            # Show specific nested structures if they exist
            print("\n" + "-" * 80)
            print("DETAILED BREAKDOWN:")
            print("-" * 80)
            
            # Token information
            if 'token0' in data or 'token1' in data:
                print("\nTOKEN INFORMATION:")
                if 'token0' in data:
                    print(f"  Token0: {data.get('token0')}")
                if 'token1' in data:
                    print(f"  Token1: {data.get('token1')}")
            
            # Observations
            if 'observations' in data:
                print(f"\nOBSERVATIONS ({len(data['observations'])} total):")
                # Show last 3 observations
                recent_obs = list(data['observations'].values())[-3:]
                for obs in recent_obs:
                    print(f"  - {obs}")
            
            # Behavior patterns
            if 'behavior_patterns' in data:
                print("\nBEHAVIOR PATTERNS:")
                pprint(data['behavior_patterns'], indent=2)
            
            # Historical metrics
            if 'historical_metrics' in data:
                print("\nHISTORICAL METRICS:")
                pprint(data['historical_metrics'], indent=2)
            
            # Save to JSON file for further analysis
            filename = f"pool_profile_{doc.id[:8]}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=json_serial)
            print(f"\nSaved raw data to: {filename}")
        
        # Also check other collections
        print("\n" + "=" * 80)
        print("OTHER COLLECTIONS:")
        print("-" * 80)
        
        collections = ['pool_metrics', 'observed_patterns', 'pattern_confidence', 
                      'pattern_correlations', 'agent_state', 'cycles']
        
        for collection_name in collections:
            try:
                collection = db.collection(collection_name)
                count = len(list(collection.limit(1).stream()))
                if count > 0:
                    # Get actual count (up to 100)
                    docs = list(collection.limit(100).stream())
                    print(f"{collection_name}: {len(docs)} documents")
                else:
                    print(f"{collection_name}: 0 documents")
            except Exception as e:
                print(f"{collection_name}: Error - {e}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()