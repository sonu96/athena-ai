#!/usr/bin/env python3
"""
Monitor pool profile creation in real-time.
"""
import os
import time
from google.cloud import firestore
from datetime import datetime

# Set up Google Cloud project
project_id = os.getenv('GCP_PROJECT_ID', 'athena-defi-agent-1752635199')
os.environ['GOOGLE_CLOUD_PROJECT'] = project_id

# Initialize Firestore
db = firestore.Client(project=project_id)

def monitor_profiles():
    """Monitor pool profiles."""
    print("Monitoring pool profiles... (Press Ctrl+C to stop)\n")
    
    last_count = 0
    expected_pools = [
        "WETH/USDC", "AERO/USDC", "AERO/WETH", 
        "WETH/DAI", "USDC/DAI", "USDC/USDbC"
    ]
    
    while True:
        try:
            # Get all pool profiles
            profiles_ref = db.collection('pool_profiles').stream()
            profiles = list(profiles_ref)
            
            current_count = len(profiles)
            
            # If count changed, show update
            if current_count != last_count:
                print(f"\n{'='*60}")
                print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Pool profiles: {current_count}")
                print(f"{'='*60}")
                
                # Show each profile
                pools_found = []
                for doc in profiles:
                    data = doc.to_dict()
                    pair = data.get('pair', 'Unknown')
                    pools_found.append(pair)
                    
                    print(f"\n{pair}:")
                    print(f"  Address: {doc.id}")
                    print(f"  Observations: {data.get('observations_count', 0)}")
                    print(f"  Confidence: {data.get('confidence_score', 0):.2%}")
                    print(f"  TVL: ${data.get('tvl_range', [0, 0])[1]:,.0f}")
                    print(f"  APR: {data.get('apr_range', [0, 0])[1]:.1f}%")
                    print(f"  Last update: {data.get('last_updated', 'Never')}")
                
                # Check which pools are missing
                missing_pools = [p for p in expected_pools if p not in pools_found]
                if missing_pools:
                    print(f"\n⚠️  Missing pools: {', '.join(missing_pools)}")
                else:
                    print(f"\n✅ All expected pools have profiles!")
                
                last_count = current_count
            
            # Wait before next check
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_profiles()