#!/usr/bin/env python3
"""
Monitor real data collection and Firestore updates.

This script monitors:
1. Pool profiles being created/updated
2. Gauge data collection
3. Event volume tracking
4. Memory storage
"""
import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
import os

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Force set GCP project ID for testing
os.environ['GCP_PROJECT_ID'] = 'athena-ai-436008'

async def monitor_firestore():
    """Monitor Firestore collections for real data."""
    try:
        from src.gcp.firestore_client import FirestoreClient
        
        logger.info("Initializing Firestore client...")
        firestore = FirestoreClient()
        
        logger.info("\n=== Monitoring Firestore Collections ===")
        
        # Check pool profiles
        logger.info("\n1. Pool Profiles:")
        pool_profiles = firestore.db.collection('pool_profiles').stream()
        profile_count = 0
        for doc in pool_profiles:
            profile_count += 1
            data = doc.to_dict()
            logger.info(f"   - {doc.id}: {data.get('pair', 'Unknown')} "
                       f"(APR range: {data.get('apr_range', [0,0])}, "
                       f"observations: {data.get('observations_count', 0)})")
        logger.info(f"   Total profiles: {profile_count}")
        
        # Check recent pool metrics
        logger.info("\n2. Recent Pool Metrics (last hour):")
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        pool_metrics = firestore.db.collection('pool_metrics')\
            .where('timestamp', '>=', one_hour_ago)\
            .stream()
        metrics_count = 0
        for doc in pool_metrics:
            metrics_count += 1
            data = doc.to_dict()
            logger.info(f"   - {data.get('pool_address', 'Unknown')}: "
                       f"APR={data.get('apr', 0):.2f}%, "
                       f"Fee APR={data.get('fee_apr', 0):.2f}%, "
                       f"Emission APR={data.get('incentive_apr', 0):.2f}%")
        logger.info(f"   Total recent metrics: {metrics_count}")
        
        # Check gauge data
        logger.info("\n3. Gauge Data:")
        gauge_data = firestore.db.collection('gauge_data').stream()
        gauge_count = 0
        for doc in gauge_data:
            gauge_count += 1
            data = doc.to_dict()
            logger.info(f"   - {doc.id}: reward_rate={data.get('reward_rate', 0):.6f} AERO/sec, "
                       f"daily={data.get('aero_per_day', 0):.2f} AERO")
        logger.info(f"   Total gauges monitored: {gauge_count}")
        
        # Check event volumes
        logger.info("\n4. Event Volumes (recent):")
        event_volumes = firestore.db.collection('event_volumes')\
            .limit(10)\
            .stream()
        volume_count = 0
        for doc in event_volumes:
            volume_count += 1
            data = doc.to_dict()
            logger.info(f"   - {data.get('pool_address', 'Unknown')} @ {data.get('hour_key', '')}: "
                       f"volume=${data.get('swap_volume', 0):,.2f}")
        logger.info(f"   Recent volume records: {volume_count}")
        
        # Check memories with new categories
        logger.info("\n5. Memories (new categories):")
        new_categories = ['gauge_emissions', 'volume_tracking', 'arbitrage_opportunity', 
                         'new_pool', 'apr_anomaly', 'fee_collection']
        
        for category in new_categories:
            memories = firestore.db.collection('memories')\
                .where('category', '==', category)\
                .limit(5)\
                .stream()
            count = 0
            for _ in memories:
                count += 1
            logger.info(f"   - {category}: {count} memories")
            
    except Exception as e:
        logger.error(f"Firestore monitoring error: {e}")
        

async def monitor_real_data_collection():
    """Monitor the real data collection process."""
    try:
        logger.info("\n=== Testing Real Data Collection Components ===")
        
        # Test gauge reader
        logger.info("\n1. Testing Gauge Reader:")
        from src.aerodrome.gauge_reader import GaugeReader
        from src.blockchain.rpc_reader import RPCReader
        from config.settings import settings
        
        async with RPCReader(settings.base_rpc_url) as rpc_reader:
            async with GaugeReader(rpc_reader) as gauge_reader:
                # Test with a known pool
                test_pool = "0xcDAc0d6c6C59727a65F871236188350531885C43"  # WETH-USDC
                gauge_address = await gauge_reader.get_gauge_address(test_pool)
                
                if gauge_address:
                    logger.info(f"   ✓ Found gauge: {gauge_address}")
                    reward_rate = await gauge_reader.get_reward_rate(gauge_address)
                    logger.info(f"   ✓ Reward rate: {reward_rate:.6f} AERO/sec")
                else:
                    logger.warning("   ✗ No gauge found for test pool")
        
        # Test pool scanner integration
        logger.info("\n2. Testing Pool Scanner with Real APR:")
        from src.cdp.base_client import BaseClient
        from src.agent.memory import AthenaMemory
        from src.collectors.pool_scanner import PoolScanner
        
        # Initialize components
        base_client = BaseClient()
        await base_client.initialize()
        
        memory = AthenaMemory()
        scanner = PoolScanner(base_client, memory)
        
        # Get a single pool's data
        pool_data = await scanner._scan_pool("WETH", "USDC", False)
        if pool_data:
            logger.info(f"   ✓ Pool scanned: {pool_data['pair']}")
            logger.info(f"   ✓ Total APR: {pool_data['apr']:.2f}%")
            logger.info(f"   ✓ Fee APR: {pool_data['fee_apr']:.2f}%")
            logger.info(f"   ✓ Emission APR: {pool_data['incentive_apr']:.2f}%")
        else:
            logger.warning("   ✗ Failed to scan pool")
            
    except Exception as e:
        logger.error(f"Real data collection test error: {e}")


async def main():
    """Main monitoring function."""
    logger.info("Starting Athena AI Real Data Monitoring...")
    
    # Run monitoring tasks
    await monitor_firestore()
    await monitor_real_data_collection()
    
    logger.info("\n=== Monitoring Complete ===")
    logger.info("Real data collection is working if you see:")
    logger.info("- Pool profiles with observation counts > 0")
    logger.info("- Recent pool metrics with real APR breakdowns")
    logger.info("- Gauge data with reward rates")
    logger.info("- Event volumes being tracked")
    logger.info("- Memories in new categories")


if __name__ == "__main__":
    asyncio.run(main())