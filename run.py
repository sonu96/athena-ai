"""
Run Athena AI with API server
"""
import asyncio
import logging
import threading
import uvicorn
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from src.agent.core import AthenaAgent
from src.agent.memory import AthenaMemory
from src.cdp.base_client import BaseClient
from src.gcp.firestore_client import FirestoreClient
from src.collectors.gas_monitor import GasMonitor
from src.collectors.pool_scanner import PoolScanner
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
    print("🚀 Initializing Athena AI with API...")
    print(f"🧠 I am learning 24/7 to maximize DeFi profits on Aerodrome")
    
    # Initialize components
    memory = AthenaMemory()
    base_client = BaseClient()
    firestore = FirestoreClient(project_id=settings.gcp_project_id)
    
    # Initialize CDP client
    await base_client.initialize()
    print(f"💳 Wallet address: {base_client.address}")
    
    # Create agent with firestore client
    agent = AthenaAgent(memory, base_client, firestore)
    
    # Start collectors
    gas_monitor = GasMonitor(base_client, memory)
    pool_scanner = PoolScanner(base_client, memory, firestore)
    
    # Set references for API
    set_agent_references(agent, memory, gas_monitor, pool_scanner)
    
    # Start API server in background thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    print(f"🌐 API server starting on http://{settings.api_host}:{settings.api_port}")
    print(f"📚 API docs available at http://{settings.api_host}:{settings.api_port}/docs")
    
    # Wait for API to start
    await asyncio.sleep(2)
    
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
        print("=" * 60)
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Athena AI shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise