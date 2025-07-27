# main.py
import asyncio
import logging
import os
from datetime import datetime, timedelta
from src.agent.core_new import AthenaAgent
from src.agent.memory import AthenaMemory
from src.mcp.quicknode_mcp import QuickNodeMCP
from src.agentkit.agent_client import AthenaAgentKit
from src.collectors.gas_monitor import GasMonitor
from src.collectors.pool_scanner_new import PoolScanner
from src.gcp.firestore_client import FirestoreClient
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    print("🚀 Initializing Athena AI with MCP & AgentKit...")
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
    firestore = FirestoreClient(settings.gcp_project_id)
    
    # Initialize QuickNode MCP
    mcp = QuickNodeMCP(settings.quicknode_api_key)
    await mcp.initialize()
    print("✅ QuickNode MCP initialized")
    
    # Initialize AgentKit
    agentkit = AthenaAgentKit()
    await agentkit.initialize()
    print(f"💳 Wallet address: {agentkit.address}")
    
    # Create agent
    agent = AthenaAgent(memory, firestore)
    
    # Initialize agent's blockchain clients
    await agent.agentkit.initialize()
    
    # Start collectors with MCP
    # Note: GasMonitor will be updated to use MCP in next iteration
    pool_scanner = PoolScanner(mcp, memory)
    
    print("👀 Starting 24/7 monitoring...")
    print("📊 Tracking gas prices, pool APRs, and market opportunities")
    
    # Run collectors in background
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
                "messages": [],
                "rebalance_recommendations": [],
                "compound_recommendations": []
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
            
            logger.info(f"✅ Cycle #{cycle_count} complete")
            logger.info(f"🎭 Emotional state: {agent.emotions}")
            logger.info(f"💰 Total profit: ${agent.performance['total_profit']}")
            
        except Exception as e:
            logger.error(f"Error in cycle #{cycle_count}: {e}")
            
        # Wait before next reasoning cycle
        await asyncio.sleep(settings.agent_cycle_time)
    
    # Cleanup
    await mcp.close()


if __name__ == "__main__":
    try:
        print("=" * 60)
        print("ATHENA AI - 24/7 DeFi Agent")
        print("Powered by QuickNode MCP & Coinbase AgentKit")
        print("=" * 60)
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Athena AI shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise