# main.py
import asyncio
import logging
import os
from datetime import datetime, timedelta
from src.agent.core import AthenaAgent
from src.agent.memory import AthenaMemory
from src.cdp.base_client import BaseClient
from src.collectors.gas_monitor import GasMonitor
from src.collectors.pool_scanner import PoolScanner
from src.gcp.firestore_client import FirestoreClient
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    print("🚀 Initializing Athena AI...")
    print(f"🧠 I am learning 24/7 to maximize DeFi profits on Aerodrome")
    
    # Check observation mode
    if settings.observation_mode:
        print(f"🔍 Starting in OBSERVATION MODE for {settings.observation_days} days")
        print(f"📊 Will collect patterns and learn before trading")
        if settings.observation_start_time:
            start_time = datetime.fromisoformat(settings.observation_start_time)
        else:
            start_time = datetime.utcnow()
            # Save start time to settings for persistence
            os.environ["OBSERVATION_START_TIME"] = start_time.isoformat()
        print(f"⏰ Observation started: {start_time.strftime('%Y-%m-%d %H:%M UTC')}")
    else:
        print(f"💰 Trading mode ACTIVE - executing strategies")
    
    # Initialize components
    memory = AthenaMemory()
    base_client = BaseClient()
    firestore = FirestoreClient(settings.gcp_project_id)
    
    # Initialize CDP client
    await base_client.initialize()
    print(f"💳 Wallet address: {base_client.address}")
    
    # Create agent
    agent = AthenaAgent(memory, base_client, firestore)
    
    # Start collectors
    gas_monitor = GasMonitor(base_client, memory)
    pool_scanner = PoolScanner(base_client, memory)
    
    print("👀 Starting 24/7 monitoring...")
    print("📊 Tracking gas prices, pool APRs, and market opportunities")
    
    # Run collectors in background
    asyncio.create_task(gas_monitor.start_monitoring())
    asyncio.create_task(pool_scanner.start_scanning())
    
    # Run agent reasoning loop
    cycle_count = 0
    while True:
        cycle_count += 1
        logger.info(f"🔄 Starting reasoning cycle #{cycle_count}")
        
        try:
            # Run through agent graph
            state = {
                "observations": [],
                "current_analysis": "",
                "theories": [],
                "emotions": agent.emotions,
                "memories": [],
                "decisions": [],
                "next_action": "",
                "messages": []
            }
            
            # Execute workflow
            result = await agent.graph.ainvoke(state)
            
            # Save to Firestore
            firestore.save_agent_state({
                'cycle_count': cycle_count,
                'emotions': agent.emotions,
                'performance': agent.performance,
                'status': 'observing' if agent._is_observation_mode() else 'active',
                'observation_mode': agent._is_observation_mode()
            })
            
            firestore.save_cycle_result(cycle_count, {
                'observations': result.get('observations', []),
                'theories': result.get('theories', []),
                'decisions': result.get('decisions', []),
                'next_action': result.get('next_action', ''),
                'observation_mode': agent._is_observation_mode()
            })
            
            firestore.update_performance(agent.performance)
            
            # Track observation metrics
            if agent._is_observation_mode():
                firestore.save_observation_metrics({
                    'patterns_discovered': len(agent.patterns_discovered),
                    'cycles_completed': cycle_count,
                    'unique_theories': len(set(result.get('theories', []))),
                    'observations_collected': len(result.get('observations', [])),
                    'start_time': agent.observation_start.isoformat(),
                    'days_observed': (datetime.utcnow() - agent.observation_start).days
                })
            
            # Log results
            logger.info(f"✅ Cycle #{cycle_count} complete")
            logger.info(f"🎭 Emotional state: {agent.emotions}")
            
            if agent._is_observation_mode():
                logger.info(f"📊 Patterns discovered: {len(agent.patterns_discovered)}")
                # Check if transitioning soon
                observation_end = agent.observation_start + timedelta(days=settings.observation_days)
                remaining = observation_end - datetime.utcnow()
                if remaining.total_seconds() < 3600:  # Less than 1 hour
                    logger.info("⚡ Observation period ending soon - preparing for trading!")
                    # Load high confidence patterns
                    high_conf_patterns = firestore.get_high_confidence_patterns(settings.min_pattern_confidence)
                    logger.info(f"🎯 Found {len(high_conf_patterns)} high-confidence patterns")
            else:
                logger.info(f"💰 Total profit: ${agent.performance['total_profit']}")
            
        except Exception as e:
            logger.error(f"Error in cycle #{cycle_count}: {e}")
            
        # Wait before next reasoning cycle
        await asyncio.sleep(settings.agent_cycle_time)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Athena AI shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise