"""
FastAPI Backend for Athena AI
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import settings

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Athena AI API",
    description="24/7 DeFi Agent API for monitoring and control",
    version="1.0.0"
)

# Add CORS middleware
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Response models
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    agent_active: bool
    wallet_address: Optional[str]


class PerformanceResponse(BaseModel):
    period: str
    total_profit: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    current_positions: List[Dict]


class PositionResponse(BaseModel):
    positions: List[Dict]
    total_value: float
    timestamp: datetime


class StrategyResponse(BaseModel):
    active_strategies: List[str]
    strategy_details: Dict
    last_execution: Optional[datetime]


class MemoryResponse(BaseModel):
    recent_memories: List[Dict]
    total_memories: int
    patterns_discovered: int


# Global references (will be set when API starts)
agent = None
memory = None
gas_monitor = None
pool_scanner = None


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint."""
    return {
        "name": "Athena AI",
        "description": "24/7 Autonomous DeFi Agent",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if agent else "initializing",
        timestamp=datetime.utcnow(),
        agent_active=agent is not None,
        wallet_address=agent.base_client.address if agent else None,
    )


@app.get("/performance/{period}", response_model=PerformanceResponse)
async def get_performance(period: str = "24h"):
    """Get performance metrics for specified period."""
    if not agent:
        return PerformanceResponse(
            period=period,
            total_profit=0,
            win_rate=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            current_positions=[],
        )
        
    # Calculate based on period
    # TODO: Implement actual time-based filtering
    performance = agent.performance
    
    total_trades = performance["winning_trades"] + performance["losing_trades"]
    win_rate = performance["winning_trades"] / total_trades if total_trades > 0 else 0
    
    return PerformanceResponse(
        period=period,
        total_profit=float(performance["total_profit"]),
        win_rate=win_rate,
        total_trades=total_trades,
        winning_trades=performance["winning_trades"],
        losing_trades=performance["losing_trades"],
        current_positions=performance["current_positions"],
    )


@app.get("/positions", response_model=PositionResponse)
async def get_positions():
    """Get current positions."""
    if not agent:
        return PositionResponse(
            positions=[],
            total_value=0,
            timestamp=datetime.utcnow(),
        )
        
    # Get balances
    balances = await agent.base_client.get_all_balances()
    
    positions = []
    total_value = 0
    
    for token, balance in balances.items():
        if balance > 0:
            # TODO: Get actual USD values
            usd_value = float(balance) * 1.0  # Mock pricing
            positions.append({
                "token": token,
                "balance": float(balance),
                "usd_value": usd_value,
            })
            total_value += usd_value
            
    return PositionResponse(
        positions=positions,
        total_value=total_value,
        timestamp=datetime.utcnow(),
    )


@app.get("/strategies/active", response_model=StrategyResponse)
async def get_active_strategies():
    """Get active strategies."""
    from config.settings import STRATEGIES
    
    active = [name for name, config in STRATEGIES.items() if config["enabled"]]
    
    return StrategyResponse(
        active_strategies=active,
        strategy_details=STRATEGIES,
        last_execution=datetime.utcnow() if agent else None,
    )


@app.get("/memories/recent", response_model=MemoryResponse)
async def get_recent_memories():
    """Get recent memories."""
    if not memory:
        return MemoryResponse(
            recent_memories=[],
            total_memories=0,
            patterns_discovered=0,
        )
        
    # Get recent observations and patterns
    recent = await memory.recall(
        query="recent market activity",
        limit=10
    )
    
    return MemoryResponse(
        recent_memories=recent,
        total_memories=memory.stats["total_memories"],
        patterns_discovered=memory.stats["patterns_discovered"],
    )


@app.get("/gas/recommendation")
async def get_gas_recommendation():
    """Get gas price recommendation."""
    if not gas_monitor:
        return {"error": "Gas monitor not initialized"}
        
    return gas_monitor.get_gas_recommendation()


@app.get("/pools/opportunities")
async def get_pool_opportunities(category: Optional[str] = None):
    """Get pool opportunities."""
    if not pool_scanner:
        return {"error": "Pool scanner not initialized"}
        
    return {
        "opportunities": pool_scanner.get_opportunities(category),
        "summary": pool_scanner.get_summary(),
    }


@app.websocket("/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await websocket.accept()
    
    try:
        while True:
            # Send periodic updates
            if agent:
                await websocket.send_json({
                    "type": "status",
                    "timestamp": datetime.utcnow().isoformat(),
                    "emotions": agent.emotions,
                    "performance": {
                        "total_profit": float(agent.performance["total_profit"]),
                    },
                    "gas": gas_monitor.stats["current_price"] if gas_monitor else 0,
                })
                
            # Wait before next update
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        

@app.post("/strategies/override")
async def override_strategy(strategy: str, action: str):
    """Manual strategy override (for emergencies)."""
    # TODO: Implement strategy override
    return {
        "message": f"Strategy {strategy} override: {action}",
        "success": True,
    }


def set_agent_references(agent_ref, memory_ref, gas_ref, pool_ref):
    """Set global references to agent components."""
    global agent, memory, gas_monitor, pool_scanner
    agent = agent_ref
    memory = memory_ref
    gas_monitor = gas_ref
    pool_scanner = pool_ref


# Add this for running with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)