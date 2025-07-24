"""
Run Athena AI with API server (Fixed for Cloud Run)
"""
import asyncio
import logging
from dotenv import load_dotenv
import uvicorn
from contextlib import asynccontextmanager

# Load environment variables first
load_dotenv()

from src.agent.core import AthenaAgent
from src.agent.memory import AthenaMemory
from src.cdp.base_client import BaseClient
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

# Global references
agent = None
memory = None
gas_monitor = None
pool_scanner = None


async def start_agent():
    """Initialize and start the agent components."""
    global agent, memory, gas_monitor, pool_scanner
    
    print("🚀 Initializing Athena AI...")
    print(f"🧠 I am learning 24/7 to maximize DeFi profits on Aerodrome")
    
    # Initialize components
    memory = AthenaMemory()
    base_client = BaseClient()
    
    # Initialize CDP client
    await base_client.initialize()
    print(f"💳 Wallet address: {base_client.address}")
    
    # Create agent
    agent = AthenaAgent(memory, base_client)
    
    # Start collectors
    gas_monitor = GasMonitor(base_client, memory)
    pool_scanner = PoolScanner(base_client, memory)
    
    # Set references for API
    set_agent_references(agent, memory, gas_monitor, pool_scanner)
    
    print("👀 Starting 24/7 monitoring...")
    print("📊 Tracking gas prices, pool APRs, and market opportunities")
    
    # Run collectors in background
    asyncio.create_task(gas_monitor.start_monitoring())
    asyncio.create_task(pool_scanner.start_scanning())
    
    # Run agent reasoning loop in background
    asyncio.create_task(agent_reasoning_loop())


async def agent_reasoning_loop():
    """Run the agent reasoning loop."""
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


@asynccontextmanager
async def lifespan(app):
    """Manage application lifecycle."""
    # Startup
    await start_agent()
    yield
    # Shutdown
    logger.info("Shutting down Athena AI...")


# Set the lifespan for the FastAPI app
app.router.lifespan_context = lifespan


if __name__ == "__main__":
    try:
        print("=" * 60)
        print("ATHENA AI - 24/7 DeFi Agent")
        print("=" * 60)
        print(f"🌐 Starting API server on http://{settings.api_host}:{settings.api_port}")
        print(f"📚 API docs available at http://{settings.api_host}:{settings.api_port}/docs")
        
        # Run uvicorn directly as the main process
        # Use PORT env var from Cloud Run, fallback to settings
        import os
        port = int(os.environ.get('PORT', settings.api_port))
        uvicorn.run(
            "run_fixed:app",
            host=settings.api_host,
            port=port,
            log_level=settings.log_level.lower(),
            reload=False
        )
    except KeyboardInterrupt:
        print("\n👋 Athena AI shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise