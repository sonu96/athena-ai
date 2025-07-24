#!/usr/bin/env python3
"""
Complete Firestore summary for Athena AI project.
Shows all collections and their document counts with key insights.
"""

import os
from datetime import datetime, timedelta
from google.cloud import firestore
from collections import defaultdict


def main():
    """Generate complete Firestore summary."""
    # Get project ID from environment
    project_id = os.environ.get('GCP_PROJECT_ID', 'athena-defi-agent-1752635199')
    
    print("=" * 80)
    print("ATHENA AI - FIRESTORE DATABASE SUMMARY")
    print("=" * 80)
    print(f"Project: {project_id}")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print("=" * 80)
    
    try:
        # Initialize Firestore client
        db = firestore.Client(project=project_id)
        
        # Define collections to check
        collections = {
            'pool_profiles': 'Pool behavior profiles',
            'pool_metrics': 'Time-series pool metrics',
            'observed_patterns': 'Discovered trading patterns',
            'pattern_confidence': 'Pattern success tracking',
            'pattern_correlations': 'Cross-pool correlations',
            'agent_state': 'Current agent state',
            'cycles': 'Reasoning cycle history',
            'positions': 'Trading positions',
            'performance': 'Performance metrics'
        }
        
        summary = {}
        
        print("\nCOLLECTION SUMMARY:")
        print("-" * 80)
        
        for collection_name, description in collections.items():
            try:
                collection = db.collection(collection_name)
                # Get up to 1000 documents for count
                docs = list(collection.limit(1000).stream())
                count = len(docs)
                
                summary[collection_name] = {
                    'count': count,
                    'docs': docs[:5]  # Keep first 5 for analysis
                }
                
                print(f"{collection_name:<25} {count:>6} docs   - {description}")
            except Exception as e:
                print(f"{collection_name:<25}  Error - {e}")
        
        # Detailed analysis
        print("\n" + "=" * 80)
        print("DETAILED ANALYSIS:")
        print("=" * 80)
        
        # Pool Profiles Analysis
        if 'pool_profiles' in summary and summary['pool_profiles']['count'] > 0:
            print("\n📊 POOL PROFILES:")
            print("-" * 40)
            print(f"Total pools monitored: {summary['pool_profiles']['count']}")
            
            for doc in summary['pool_profiles']['docs']:
                data = doc.to_dict()
                pair = data.get('pair', 'Unknown')
                obs = data.get('observations_count', 0)
                conf = data.get('confidence_score', 0)
                tvl = data.get('tvl_range', [0, 0])
                
                print(f"\nPool: {doc.id[:10]}...")
                print(f"  Pair: {pair if pair else 'Not specified'}")
                print(f"  Observations: {obs}")
                print(f"  Confidence: {conf:.3f}")
                print(f"  TVL Range: ${tvl[0]:,.2f} - ${tvl[1]:,.2f}")
        
        # Observed Patterns Analysis
        if 'observed_patterns' in summary and summary['observed_patterns']['count'] > 0:
            print("\n🔍 OBSERVED PATTERNS:")
            print("-" * 40)
            print(f"Total patterns discovered: {summary['observed_patterns']['count']}")
            
            pattern_types = defaultdict(int)
            for doc in summary['observed_patterns']['docs']:
                data = doc.to_dict()
                pattern_type = data.get('pattern_type', 'unknown')
                pattern_types[pattern_type] += 1
            
            if pattern_types:
                print("\nPattern types found:")
                for ptype, count in pattern_types.items():
                    print(f"  - {ptype}: {count}")
        
        # Agent State
        if 'agent_state' in summary and summary['agent_state']['count'] > 0:
            print("\n🤖 AGENT STATE:")
            print("-" * 40)
            
            for doc in summary['agent_state']['docs']:
                data = doc.to_dict()
                state = data.get('current_state', 'unknown')
                emotion = data.get('emotional_state', 'unknown')
                last_update = data.get('last_update')
                
                print(f"Current state: {state}")
                print(f"Emotional state: {emotion}")
                if last_update:
                    print(f"Last update: {last_update}")
        
        # Reasoning Cycles
        if 'cycles' in summary and summary['cycles']['count'] > 0:
            print("\n🔄 REASONING CYCLES:")
            print("-" * 40)
            print(f"Total cycles completed: {summary['cycles']['count']}")
            
            # Get most recent cycle
            recent_cycle = max(summary['cycles']['docs'], 
                             key=lambda d: d.to_dict().get('cycle_number', 0))
            if recent_cycle:
                data = recent_cycle.to_dict()
                print(f"Latest cycle: #{data.get('cycle_number', 'N/A')}")
                print(f"Timestamp: {data.get('timestamp', 'N/A')}")
        
        # Trading Performance
        if 'positions' in summary and summary['positions']['count'] > 0:
            print("\n💰 TRADING POSITIONS:")
            print("-" * 40)
            print(f"Total positions: {summary['positions']['count']}")
            
            active_positions = 0
            for doc in summary['positions']['docs']:
                data = doc.to_dict()
                if data.get('status') == 'active':
                    active_positions += 1
            
            print(f"Active positions: {active_positions}")
        
        print("\n" + "=" * 80)
        print("SUMMARY INSIGHTS:")
        print("=" * 80)
        
        # Calculate totals
        total_docs = sum(s['count'] for s in summary.values())
        print(f"\n📈 Total documents across all collections: {total_docs}")
        
        # Key findings
        print("\n🔑 KEY FINDINGS:")
        print("-" * 40)
        
        if summary.get('pool_profiles', {}).get('count', 0) == 1:
            print("• Only 1 pool profile exists - system is in early observation phase")
        
        if summary.get('observed_patterns', {}).get('count', 0) > 50:
            print(f"• {summary['observed_patterns']['count']} patterns discovered - active learning ongoing")
        
        if summary.get('pool_metrics', {}).get('count', 0) == 0:
            print("• No time-series metrics stored yet")
        
        if summary.get('positions', {}).get('count', 0) == 0:
            print("• No trading positions created - likely still in observation mode")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()