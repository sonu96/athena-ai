"""
Run Athena AI with QuickNode Aerodrome API

This version uses QuickNode Aerodrome API for pool data
and CDP AgentKit for transaction execution.
"""
import asyncio
import logging
import uvicorn
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from src.agent.core_rpc import AthenaAgent
from src.agent.memory import AthenaMemory
from src.gcp.firestore_client import FirestoreClient
from src.api.main import app
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_api_server():
    """Run FastAPI server asynchronously."""
    config = uvicorn.Config(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower()
    )
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    print("=" * 70)
    print("🚀 ATHENA AI - DeFi Agent with Real Blockchain Data")
    print("=" * 70)
    print(f"📊 Data Source: QuickNode RPC (Real-time blockchain data)")
    print(f"💼 Transactions: CDP AgentKit (Natural language execution)")
    print(f"🧠 AI Model: {settings.google_ai_model}")
    print(f"⛓️ Network: Base (Chain ID: 8453)")
    print("=" * 70)
    
    # Initialize components
    memory = AthenaMemory()
    firestore = FirestoreClient(project_id=settings.gcp_project_id)
    
    # Create agent with RPC-based data fetching
    agent = AthenaAgent(memory, firestore)
    
    # Start pool scanner in background
    scanner_task = asyncio.create_task(agent.pool_scanner.scan_pools())
    
    # Start API server in background task
    api_task = asyncio.create_task(run_api_server())
    print(f"\n🌐 API server starting on http://{settings.api_host}:{settings.api_port}")
    print(f"📚 API docs available at http://{settings.api_host}:{settings.api_port}/docs")
    
    # Wait for API and scanner to start
    await asyncio.sleep(5)
    
    # Show initial data
    print("\n📈 Initial Market Data:")
    print("   Fetching pool data from Aerodrome API...")
    
    # Check observation mode
    if agent._is_observation_mode():
        days_remaining = settings.observation_days - (datetime.utcnow() - agent.observation_start).days
        print(f"\n👁️ OBSERVATION MODE ACTIVE")
        print(f"   Days remaining: {days_remaining}")
        print(f"   Pattern confidence threshold: {settings.min_pattern_confidence}")
        print(f"   The agent will observe and learn patterns before trading")
    else:
        print(f"\n💰 TRADING MODE ACTIVE")
        print(f"   Max position size: ${settings.agent_max_position_size}")
        print(f"   Risk limit: {settings.agent_risk_limit * 100}%")
    
    print("\n🔄 Starting reasoning cycles...")
    print("=" * 70)
    
    # Run agent reasoning loop
    cycle_count = 0
    while True:
        cycle_count += 1
        
        try:
            print(f"\n⏰ Cycle #{cycle_count} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            # Run through agent workflow
            result = await agent.run_cycle()
            
            # Log key results
            if result.get("observations"):
                high_apr_count = len([o for o in result["observations"] 
                                    if o.get("type") == "high_yield_pool"])
                print(f"   📊 Found {high_apr_count} high APR pools")
                
            if result.get("decisions"):
                print(f"   🎯 Made {len(result['decisions'])} decisions:")
                for decision in result["decisions"]:
                    print(f"      - {decision['action']}: {decision['reason']}")
            else:
                print(f"   💭 No actions needed this cycle")
                
            # Show emotional state
            print(f"   🎭 Emotions: " + ", ".join(
                f"{emotion}: {value:.2f}" 
                for emotion, value in agent.emotions.items()
            ))
            
            # Show performance
            if agent.performance["current_positions"]:
                print(f"   💼 Active positions: {len(agent.performance['current_positions'])}")
                for pos in agent.performance["current_positions"]:
                    print(f"      - {pos['pool_name']}: entered at {pos['entry_apr']:.1f}% APR")
                    
            # Memory stats
            memory_count = await memory.count_all()
            print(f"   🧠 Total memories: {memory_count}")
            
        except Exception as e:
            logger.error(f"Error in cycle #{cycle_count}: {e}", exc_info=True)
            
        # Wait before next cycle
        print(f"\n💤 Waiting {settings.agent_cycle_time} seconds until next cycle...")
        await asyncio.sleep(settings.agent_cycle_time)


if __name__ == "__main__":
    try:
        # Add missing import
        from datetime import datetime
        
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Athena AI shutting down gracefully...")
        print("📊 Final statistics:")
        # Could add final stats here
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise