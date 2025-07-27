"""
Run Athena AI with API server - Enhanced with MCP & AgentKit
"""
import asyncio
import logging
import threading
import uvicorn
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from src.agent.core_new import AthenaAgent
from src.agent.memory import AthenaMemory
from src.mcp.quicknode_mcp import QuickNodeMCP
from src.agentkit.agent_client import AthenaAgentKit
from src.gcp.firestore_client import FirestoreClient
from src.collectors.pool_scanner_new import PoolScanner
from src.agent.rebalancer_new import SmartRebalancer
from src.api.main import app, set_agent_references
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_api_server():
    """Run FastAPI server in a separate thread."""
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower()
    )


async def main():
    print("🚀 Initializing Athena AI with MCP & AgentKit...")
    print(f"🧠 AI-powered DeFi agent with natural language blockchain interaction")
    
    # Initialize components
    memory = AthenaMemory()
    firestore = FirestoreClient(project_id=settings.gcp_project_id)
    
    # Initialize QuickNode MCP
    print("🔌 Connecting to QuickNode MCP...")
    mcp = QuickNodeMCP(settings.quicknode_api_key)
    await mcp.initialize()
    print("✅ QuickNode MCP connected - natural language queries enabled")
    
    # Initialize AgentKit
    print("🤖 Initializing Coinbase AgentKit...")
    agentkit = AthenaAgentKit(
        api_key=settings.cdp_api_key,
        api_secret=settings.cdp_api_secret,
        wallet_data=settings.agent_wallet_id
    )
    await agentkit.initialize()
    print(f"💳 Wallet address: {agentkit.address}")
    
    # Create agent with simplified architecture
    agent = AthenaAgent(memory, firestore)
    
    # Create smart rebalancer with new components
    rebalancer = SmartRebalancer(memory, mcp, agentkit)
    
    # Start collectors
    pool_scanner = PoolScanner(mcp, memory)
    
    # Set references for API (update needed for new components)
    # Note: This will need updating in the API module
    # set_agent_references(agent, memory, gas_monitor, pool_scanner)
    
    # Start API server in background thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    print(f"🌐 API server starting on http://{settings.api_host}:{settings.api_port}")
    print(f"📚 API docs available at http://{settings.api_host}:{settings.api_port}/docs")
    
    # Wait for API to start
    await asyncio.sleep(2)
    
    print("👀 Starting 24/7 monitoring with enhanced capabilities...")
    print("🔍 Natural language blockchain queries via QuickNode MCP")
    print("⚡ AI-native transactions via Coinbase AgentKit")
    
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
                "rebalance_recommendations": [],
                "compound_recommendations": [],
                "emotions": agent.emotions,
                "memories": [],
                "decisions": [],
                "next_action": "",
                "messages": []
            }
            
            # Execute workflow
            result = await agent.graph.ainvoke(state)
            
            # Handle rebalancing if recommended
            if result.get("rebalance_recommendations"):
                for rec in result["rebalance_recommendations"]:
                    if rec["confidence"] > 0.7:
                        logger.info(f"Executing rebalance: {rec['reason']}")
                        await rebalancer.execute_rebalance(rec)
            
            # Log results
            logger.info(f"✅ Cycle #{cycle_count} complete")
            logger.info(f"🎭 Emotional state: {agent.emotions}")
            logger.info(f"💰 Total profit: ${agent.performance['total_profit']}")
            
        except Exception as e:
            logger.error(f"Error in cycle #{cycle_count}: {e}")
            
        # Wait before next reasoning cycle
        await asyncio.sleep(settings.agent_cycle_time)


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