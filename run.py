"""
Run Athena AI with API server - Production Version
"""
import os
import asyncio
import logging
import threading
import uvicorn
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from src.agent.core_rpc import AthenaAgent
from src.agent.memory import AthenaMemory
from src.agentkit.agent_client import AthenaAgentKit
from src.gcp.firestore_client import FirestoreClient
from src.collectors.aerodrome_api_scanner import AerodromeAPIScanner
from src.collectors.gas_monitor import GasMonitor
from src.collectors.pool_scanner import PoolScanner
from src.agent.rebalancer import SmartRebalancer
from src.api.main import app
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_api_server():
    """Run FastAPI server in a separate thread."""
    # Use PORT environment variable for Cloud Run compatibility
    port = int(os.environ.get("PORT", settings.api_port))
    uvicorn.run(
        app,
        host=settings.api_host,
        port=port,
        log_level=settings.log_level.lower()
    )


async def main():
    print("🚀 Initializing Athena AI Production System...")
    print(f"🧠 AI-powered DeFi agent with QuickNode API & AgentKit")
    
    try:
        # Initialize core components
        memory = AthenaMemory()
        firestore = FirestoreClient(project_id=settings.gcp_project_id)
        
        # Initialize data collectors
        print("📊 Initializing data collectors...")
        api_scanner = AerodromeAPIScanner(memory)
        gas_monitor = GasMonitor(memory)
        pool_scanner = PoolScanner(memory)
        
        # Initialize AgentKit
        print("🤖 Initializing Coinbase AgentKit...")
        agentkit = AthenaAgentKit()
        await agentkit.initialize()
        print(f"💳 Wallet address: {agentkit.address}")
        
        # Create agent with current architecture
        agent = AthenaAgent(memory, firestore)
        
        # Create smart rebalancer
        rebalancer = SmartRebalancer(memory, agentkit)
        
        # Start API server in background thread
        api_thread = threading.Thread(target=run_api_server, daemon=True)
        api_thread.start()
        port = int(os.environ.get("PORT", settings.api_port))
        print(f"🌐 API server starting on http://{settings.api_host}:{port}")
        print(f"📚 API docs available at http://{settings.api_host}:{port}/docs")
        
        # Wait for API to start
        await asyncio.sleep(2)
        
        print("👀 Starting 24/7 monitoring with current capabilities...")
        print("🔍 QuickNode Aerodrome API for pool data")
        print("⚡ AI-native transactions via Coinbase AgentKit")
        
        # Run collectors in background
        asyncio.create_task(api_scanner.start_monitoring())
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
                
            except Exception as e:
                logger.error(f"Error in cycle #{cycle_count}: {e}")
                
            # Wait before next reasoning cycle
            await asyncio.sleep(settings.agent_cycle_time)
            
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        # Just run the API server if agent fails
        print("⚠️ Running in API-only mode")
        api_thread = threading.Thread(target=run_api_server, daemon=True)
        api_thread.start()
        
        # Keep the process alive
        while True:
            await asyncio.sleep(60)


if __name__ == "__main__":
    try:
        print("=" * 60)
        print("ATHENA AI - 24/7 DeFi Agent")
        print("Powered by QuickNode API & Coinbase AgentKit")
        print("=" * 60)
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Athena AI shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise