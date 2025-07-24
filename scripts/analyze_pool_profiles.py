#!/usr/bin/env python3
"""
Script to analyze pool profiles in Firestore.
Connects to Firestore and provides detailed statistics about pool profiles.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict
from statistics import mean, median

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gcp.firestore_client import FirestoreClient
from config.settings import settings


def format_number(num: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    elif num >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"


def analyze_pool_profiles():
    """Analyze pool profiles in Firestore."""
    print("=" * 80)
    print("POOL PROFILES ANALYSIS")
    print("=" * 80)
    print(f"Project ID: {settings.gcp_project_id}")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print("=" * 80)
    
    # Initialize Firestore client
    try:
        client = FirestoreClient(settings.gcp_project_id)
        print("✓ Connected to Firestore")
    except Exception as e:
        print(f"✗ Failed to connect to Firestore: {e}")
        return
    
    # Get all pool profiles
    profiles = client.get_all_pool_profiles()
    
    if not profiles:
        print("\nNo pool profiles found in Firestore.")
        return
    
    print(f"\nTotal Pool Profiles: {len(profiles)}")
    print("-" * 80)
    
    # Collect statistics
    observations_counts = []
    confidence_scores = []
    volumes = []
    aprs = []
    profiles_by_pair = defaultdict(list)
    
    # Display individual pool profiles
    print("\nINDIVIDUAL POOL PROFILES:")
    print("-" * 80)
    print(f"{'Pair':<20} {'Address':<15} {'Obs':<6} {'Conf':<8} {'Volume':<12} {'APR':<8} {'Updated'}")
    print("-" * 80)
    
    for address, profile in profiles.items():
        # Extract data with safe defaults
        pair = profile.get('pair', 'Unknown')
        observations_count = profile.get('observations_count', 0)
        confidence_score = profile.get('confidence_score', 0.0)
        avg_volume = profile.get('avg_volume_24h', 0.0)
        avg_apr = profile.get('avg_apr', 0.0)
        last_updated = profile.get('updated_at', profile.get('last_observation'))
        
        # Collect statistics
        observations_counts.append(observations_count)
        confidence_scores.append(confidence_score)
        volumes.append(avg_volume)
        aprs.append(avg_apr)
        profiles_by_pair[pair].append(profile)
        
        # Format display
        address_short = f"{address[:6]}...{address[-4:]}" if len(address) > 15 else address
        
        # Format timestamp
        if isinstance(last_updated, datetime):
            try:
                # Handle timezone-aware and naive datetimes
                now = datetime.utcnow()
                if last_updated.tzinfo is not None:
                    # Convert to naive UTC datetime
                    last_updated = last_updated.replace(tzinfo=None)
                
                time_ago = now - last_updated
                if time_ago.days > 0:
                    time_str = f"{time_ago.days}d ago"
                elif time_ago.seconds > 3600:
                    time_str = f"{time_ago.seconds // 3600}h ago"
                else:
                    time_str = f"{time_ago.seconds // 60}m ago"
            except Exception as e:
                time_str = "Unknown"
        else:
            time_str = "Unknown"
        
        print(f"{pair:<20} {address_short:<15} {observations_count:<6} {confidence_score:<8.2f} {format_number(avg_volume):<12} {avg_apr:<8.1f}% {time_str}")
    
    # Calculate and display statistics
    print("\n" + "=" * 80)
    print("STATISTICS:")
    print("-" * 80)
    
    if observations_counts:
        print(f"\nObservations per Pool:")
        print(f"  Average: {mean(observations_counts):.1f}")
        print(f"  Median:  {median(observations_counts):.1f}")
        print(f"  Max:     {max(observations_counts)}")
        print(f"  Min:     {min(observations_counts)}")
    
    if confidence_scores:
        print(f"\nConfidence Scores:")
        print(f"  Average: {mean(confidence_scores):.3f}")
        print(f"  Median:  {median(confidence_scores):.3f}")
        print(f"  Max:     {max(confidence_scores):.3f}")
        print(f"  Min:     {min(confidence_scores):.3f}")
    
    if volumes:
        print(f"\n24h Volume Statistics:")
        print(f"  Average: {format_number(mean(volumes))}")
        print(f"  Median:  {format_number(median(volumes))}")
        print(f"  Total:   {format_number(sum(volumes))}")
    
    if aprs:
        print(f"\nAPR Statistics:")
        print(f"  Average: {mean(aprs):.1f}%")
        print(f"  Median:  {median(aprs):.1f}%")
        print(f"  Max:     {max(aprs):.1f}%")
        print(f"  Min:     {min(aprs):.1f}%")
    
    # Top pools by different metrics
    print("\n" + "=" * 80)
    print("TOP POOLS:")
    print("-" * 80)
    
    # Sort by confidence
    sorted_by_confidence = sorted(profiles.items(), key=lambda x: x[1].get('confidence_score', 0), reverse=True)[:5]
    print("\nTop 5 by Confidence Score:")
    for address, profile in sorted_by_confidence:
        pair = profile.get('pair', 'Unknown')
        conf = profile.get('confidence_score', 0.0)
        print(f"  {pair:<20} ({address[:6]}...{address[-4:]}): {conf:.3f}")
    
    # Sort by volume
    sorted_by_volume = sorted(profiles.items(), key=lambda x: x[1].get('avg_volume_24h', 0), reverse=True)[:5]
    print("\nTop 5 by 24h Volume:")
    for address, profile in sorted_by_volume:
        pair = profile.get('pair', 'Unknown')
        vol = profile.get('avg_volume_24h', 0.0)
        print(f"  {pair:<20} ({address[:6]}...{address[-4:]}): {format_number(vol)}")
    
    # Sort by APR
    sorted_by_apr = sorted(profiles.items(), key=lambda x: x[1].get('avg_apr', 0), reverse=True)[:5]
    print("\nTop 5 by APR:")
    for address, profile in sorted_by_apr:
        pair = profile.get('pair', 'Unknown')
        apr = profile.get('avg_apr', 0.0)
        print(f"  {pair:<20} ({address[:6]}...{address[-4:]}): {apr:.1f}%")
    
    # Pair distribution
    print("\n" + "=" * 80)
    print("PAIR DISTRIBUTION:")
    print("-" * 80)
    
    for pair, pair_profiles in sorted(profiles_by_pair.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"{pair}: {len(pair_profiles)} pools")
    
    # Activity analysis
    print("\n" + "=" * 80)
    print("ACTIVITY ANALYSIS:")
    print("-" * 80)
    
    # Recent updates
    recent_cutoff = datetime.utcnow() - timedelta(hours=1)
    recent_updates = 0
    stale_profiles = 0
    stale_cutoff = datetime.utcnow() - timedelta(days=1)
    
    for profile in profiles.values():
        last_updated = profile.get('updated_at', profile.get('last_observation'))
        if isinstance(last_updated, datetime):
            # Handle timezone-aware and naive datetimes
            if last_updated.tzinfo is not None:
                last_updated = last_updated.replace(tzinfo=None)
            if last_updated > recent_cutoff:
                recent_updates += 1
            elif last_updated < stale_cutoff:
                stale_profiles += 1
    
    print(f"Recently updated (last hour): {recent_updates}")
    print(f"Stale profiles (>24h old): {stale_profiles}")
    print(f"Active profiles: {len(profiles) - stale_profiles}")
    
    # Pattern insights
    print("\n" + "=" * 80)
    print("PATTERN INSIGHTS:")
    print("-" * 80)
    
    # Count pools with significant patterns
    pools_with_patterns = 0
    high_volatility_pools = 0
    stable_pools = 0
    
    for profile in profiles.values():
        patterns = profile.get('behavior_patterns', {})
        if patterns:
            pools_with_patterns += 1
            
            volatility = patterns.get('volatility', 'unknown')
            if volatility == 'high':
                high_volatility_pools += 1
            elif volatility == 'stable':
                stable_pools += 1
    
    print(f"Pools with identified patterns: {pools_with_patterns}")
    print(f"High volatility pools: {high_volatility_pools}")
    print(f"Stable pools: {stable_pools}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        analyze_pool_profiles()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()