"""
Athena AI Agent Core - Using QuickNode Aerodrome API

Enhanced version using QuickNode Aerodrome API for pool data
and CDP AgentKit for transaction execution.
"""
import asyncio
import logging
from typing import Dict, List, TypedDict, Annotated, Sequence, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal

from langgraph.graph import StateGraph, END
import google.generativeai as genai

from src.agent.memory import AthenaMemory, MemoryType
from src.agent.pool_profiles import PoolProfileManager
from src.collectors.aerodrome_api_scanner import AerodromeAPIScanner
from src.agentkit.agent_client import AthenaAgentKit
from src.gcp.firestore_client import FirestoreClient
from config.settings import settings, STRATEGIES, EMOTIONAL_STATES

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State schema for Athena's thought process."""
    observations: List[Dict]
    current_analysis: str
    theories: List[str]
    rebalance_recommendations: List[Dict]
    compound_recommendations: List[Dict]
    emotions: Dict[str, float]
    memories: List[Dict]
    decisions: List[Dict]
    next_action: str
    messages: Annotated[Sequence[Dict], "The messages in the conversation"]


class AthenaAgent:
    """
    Athena's core consciousness - a learning DeFi agent.
    
    Now powered by:
    - QuickNode RPC for real blockchain data
    - CDP AgentKit for AI-native transaction execution
    """
    
    def __init__(self, memory: AthenaMemory, firestore_client: FirestoreClient):
        """Initialize Athena with QuickNode RPC data fetching."""
        self.memory = memory
        self.firestore = firestore_client
        
        # Initialize Aerodrome API scanner for pool data
        self.pool_scanner = AerodromeAPIScanner(memory)
        logger.info(f"Initialized Aerodrome API scanner")
        
        # Initialize AgentKit for transactions
        self.agentkit = AthenaAgentKit(
            api_key=settings.cdp_api_key,
            api_secret=settings.cdp_api_secret,
            wallet_data=settings.agent_wallet_id
        )
        
        # Initialize pool profile manager
        self.pool_profiles = PoolProfileManager(firestore_client)
        
        # Initialize Gemini
        if settings.google_api_key:
            genai.configure(api_key=settings.google_api_key)
            self.model = genai.GenerativeModel(
                model_name=settings.google_ai_model,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 2048,
                }
            )
        else:
            logger.warning("Google API key not found - LLM features disabled")
            self.model = None
        
        # Emotional state
        self.emotions = {
            "confidence": 0.7,
            "curiosity": 0.8,
            "caution": 0.3,
            "satisfaction": 0.5,
        }
        
        # Performance tracking
        self.performance = {
            "total_profit": Decimal("0"),
            "winning_trades": 0,
            "losing_trades": 0,
            "current_positions": [],
        }
        
        # Build the graph
        self.graph = self._build_graph()
        
        # Track observation mode
        self.observation_start = datetime.fromisoformat(settings.observation_start_time) if settings.observation_start_time else datetime.utcnow()
        self.patterns_discovered = []
        
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("observe", self.observe)
        workflow.add_node("analyze", self.analyze)
        workflow.add_node("decide", self.decide)
        workflow.add_node("execute", self.execute)
        workflow.add_node("learn", self.learn)
        
        # Add edges
        workflow.add_edge("observe", "analyze")
        workflow.add_edge("analyze", "decide")
        workflow.add_edge("decide", "execute")
        workflow.add_edge("execute", "learn")
        workflow.add_edge("learn", END)
        
        # Set entry point
        workflow.set_entry_point("observe")
        
        return workflow.compile()
        
    def _is_observation_mode(self) -> bool:
        """Check if still in observation mode."""
        if not settings.observation_mode:
            return False
            
        days_elapsed = (datetime.utcnow() - self.observation_start).days
        return days_elapsed < settings.observation_days
        
    async def observe(self, state: AgentState) -> AgentState:
        """OBSERVE: Gather real data from QuickNode Aerodrome API."""
        logger.info("👁️ OBSERVE: Scanning pools with Aerodrome API...")
        
        observations = []
        
        # Fetch pools from API
        pools = await self.pool_scanner.fetch_pools(limit=50, min_apr=5.0)
        
        # Process high APR pools
        for pool in pools:
            if pool["apr"] >= settings.min_apr_for_memory:
                observations.append({
                    "type": "high_yield_pool",
                    "pool": pool["symbol"],
                    "pool_address": pool["address"],
                    "apr": pool["apr"],
                    "tvl": pool["tvl"],
                    "volume_24h": pool["volume_24h"],
                    "stable": pool["is_stable"],
                    "data": pool
                })
                
        # Process high volume pools
        for pool in pools:
            if pool["volume_24h"] >= settings.min_volume_for_memory:
                observations.append({
                    "type": "high_volume_pool",
                    "pool": pool["symbol"],
                    "pool_address": pool["address"],
                    "volume": pool["volume_24h"],
                    "tvl": pool["tvl"],
                    "apr": pool["apr"],
                    "data": pool
                })
                
        # Get current gas price
        gas_price = await self.pool_scanner.get_gas_price()
        if gas_price:
            observations.append({
                "type": "gas_price",
                "price_gwei": gas_price,
                "status": "low" if gas_price < 1 else "normal" if gas_price < 5 else "high"
            })
        
        # Get pool predictions
        predictions = await self.pool_profiles.predict_opportunities(
            datetime.utcnow() + timedelta(hours=1)
        )
        
        for pred in predictions[:3]:
            observations.append({
                "type": "pool_prediction",
                "pool": pred["pool"],
                "predicted_apr": pred["predicted_apr"],
                "confidence": pred["confidence"]
            })
            
        state["observations"] = observations
        logger.info(f"📊 Collected {len(observations)} observations from QuickNode")
        
        # Get relevant memories
        memories = []
        
        # Get memories for high-yield pools
        for obs in observations:
            if obs["type"] == "high_yield_pool":
                pool = obs["pool"]
                
                # Get pool-specific memories
                pool_memories = await self.memory.recall_pool_memories(
                    pool_pair=pool,
                    time_window_hours=168  # Last week
                )
                memories.extend(pool_memories)
                
                # Get pattern memories
                pattern_query = f"APR patterns for {pool} pools"
                patterns = await self.memory.recall(
                    pattern_query,
                    memory_type=MemoryType.PATTERN,
                    min_confidence=0.7,
                    limit=5
                )
                memories.extend(patterns)
                
        # Get gas optimization patterns
        gas_patterns = await self.memory.recall(
            "gas price optimization windows Base network",
            category="gas_optimization_windows",
            limit=10
        )
        memories.extend(gas_patterns)
        
        state["memories"] = memories
        logger.info(f"🗂️ Retrieved {len(memories)} relevant memories")
        
        return state
        
    async def analyze(self, state: AgentState) -> AgentState:
        """ANALYZE: Process observations with LLM."""
        logger.info("🤔 ANALYZE: Processing market data...")
        
        if not self.model:
            state["current_analysis"] = "LLM analysis unavailable"
            return state
            
        # Prepare context
        observations_text = "\n".join([
            f"- {obs['type']}: {obs.get('pool', obs.get('data', obs))}"
            for obs in state["observations"][:10]
        ])
        
        memories_text = "\n".join([
            f"- {mem.get('content', mem)}"
            for mem in state["memories"][:10]
        ])
        
        prompt = f"""
        You are Athena, an AI DeFi agent analyzing Aerodrome on Base.
        
        Current Market Observations (REAL DATA from QuickNode):
        {observations_text}
        
        Relevant Historical Patterns:
        {memories_text}
        
        Current Emotional State: {self.emotions}
        
        Analyze the current market conditions and identify:
        1. Key opportunities (pools with high APR that are sustainable)
        2. Risks to watch (overheated pools, gas spikes)
        3. Patterns that match historical data
        4. Recommended focus areas
        
        Be specific about pool names, APRs, and actionable insights.
        This is REAL blockchain data, not simulated.
        """
        
        response = self.model.generate_content(prompt)
        analysis = response.text
        
        state["current_analysis"] = analysis
        logger.info("✅ Analysis complete")
        
        return state
        
    async def decide(self, state: AgentState) -> AgentState:
        """DECIDE: Make strategic decisions."""
        logger.info("🎯 DECIDE: Evaluating opportunities...")
        
        decisions = []
        
        # Check if in observation mode
        if self._is_observation_mode():
            logger.info("📝 Observation mode - recording patterns only")
            
            # Look for patterns to remember
            for obs in state["observations"]:
                if obs["type"] == "high_yield_pool" and obs["apr"] > 50:
                    # Record high APR pattern
                    await self.memory.remember(
                        content=f"High APR pool discovered: {obs['pool']} at {obs['apr']:.1f}% APR",
                        memory_type=MemoryType.PATTERN,
                        category="apr_anomaly",
                        metadata={
                            "pool": obs["pool"],
                            "apr": obs["apr"],
                            "hour": datetime.utcnow().hour,
                            "day_of_week": datetime.utcnow().weekday()
                        },
                        confidence=0.7
                    )
                    
        else:
            # Active trading mode - make real decisions
            # Evaluate rebalancing opportunities
            for position in self.performance.get("current_positions", []):
                current_pool_data = None
                
                # Find current pool in observations
                for obs in state["observations"]:
                    if obs.get("pool_address") == position.get("pool_address"):
                        current_pool_data = obs["data"]
                        break
                        
                if current_pool_data:
                    # Check if APR dropped significantly
                    original_apr = position.get("entry_apr", 0)
                    current_apr = current_pool_data["apr_total"]
                    
                    if current_apr < original_apr * 0.7:  # 30% drop
                        # Find better alternative
                        better_pools = [
                            obs for obs in state["observations"]
                            if obs["type"] == "high_yield_pool" 
                            and obs["apr"] > current_apr * 1.5
                            and obs["tvl"] > 100000
                        ]
                        
                        if better_pools:
                            best_alternative = max(better_pools, key=lambda x: x["apr"])
                            
                            decisions.append({
                                "action": "rebalance",
                                "from_pool": position["pool_address"],
                                "to_pool": best_alternative["pool_address"],
                                "to_pool_name": best_alternative["pool"],
                                "reason": f"APR dropped from {original_apr:.1f}% to {current_apr:.1f}%",
                                "expected_apr": best_alternative["apr"],
                                "confidence": 0.8
                            })
                            
            # Look for new opportunities if not fully invested
            if len(self.performance.get("current_positions", [])) < 5:  # Max 5 positions
                best_opportunities = sorted(
                    [obs for obs in state["observations"] 
                     if obs["type"] == "high_yield_pool" 
                     and obs["tvl"] > 100000
                     and obs["apr"] > settings.min_apr_for_memory],
                    key=lambda x: x["apr"],
                    reverse=True
                )[:3]
                
                for opp in best_opportunities:
                    if not any(p["pool_address"] == opp["pool_address"] 
                              for p in self.performance.get("current_positions", [])):
                        decisions.append({
                            "action": "add_liquidity",
                            "pool_address": opp["pool_address"],
                            "pool_name": opp["pool"],
                            "token_a": opp["data"]["token0"],
                            "token_b": opp["data"]["token1"],
                            "apr": opp["apr"],
                            "reason": f"High APR opportunity: {opp['apr']:.1f}%",
                            "confidence": 0.7
                        })
                        
        state["decisions"] = decisions
        logger.info(f"📋 Made {len(decisions)} decisions")
        
        return state
        
    async def execute(self, state: AgentState) -> AgentState:
        """EXECUTE: Perform actions using AgentKit."""
        logger.info("⚡ EXECUTE: Taking action...")
        
        # Initialize AgentKit if needed
        if not hasattr(self, '_agentkit_initialized') or not self._agentkit_initialized:
            await self.agentkit.initialize()
            self._agentkit_initialized = True
            logger.info(f"💳 Wallet initialized: {self.agentkit.address}")
        
        # Check if we're in observation mode
        if self._is_observation_mode():
            logger.info("👁️ Observation mode - no trades executed")
            state["next_action"] = "observe_only"
            return state
            
        # Execute decisions
        for decision in state["decisions"]:
            try:
                if decision["confidence"] < settings.min_pattern_confidence:
                    logger.info(f"Skipping low confidence decision: {decision['reason']}")
                    continue
                    
                if decision["action"] == "add_liquidity":
                    # Use natural language execution
                    result = await self.agentkit.execute_natural_language(
                        f"Add liquidity to {decision['pool_name']} pool on Aerodrome "
                        f"with balanced amounts, expecting {decision['apr']:.1f}% APR"
                    )
                    
                    logger.info(f"✅ Liquidity added to {decision['pool_name']}: {result}")
                    
                    # Track position
                    self.performance["current_positions"].append({
                        "pool_address": decision["pool_address"],
                        "pool_name": decision["pool_name"],
                        "entry_apr": decision["apr"],
                        "entry_time": datetime.utcnow(),
                        "tx_hash": result.get("tx_hash")
                    })
                    
                elif decision["action"] == "rebalance":
                    # First remove liquidity
                    remove_result = await self.agentkit.execute_natural_language(
                        f"Remove all liquidity from {decision['from_pool']} pool on Aerodrome"
                    )
                    
                    # Then add to new pool
                    add_result = await self.agentkit.execute_natural_language(
                        f"Add the removed liquidity to {decision['to_pool_name']} pool on Aerodrome "
                        f"expecting {decision['expected_apr']:.1f}% APR"
                    )
                    
                    logger.info(f"✅ Rebalanced from {decision['from_pool']} to {decision['to_pool_name']}")
                    
                    # Update positions
                    self.performance["current_positions"] = [
                        p for p in self.performance["current_positions"]
                        if p["pool_address"] != decision["from_pool"]
                    ]
                    
                    self.performance["current_positions"].append({
                        "pool_address": decision["to_pool"],
                        "pool_name": decision["to_pool_name"],
                        "entry_apr": decision["expected_apr"],
                        "entry_time": datetime.utcnow()
                    })
                    
            except Exception as e:
                logger.error(f"Failed to execute {decision['action']}: {e}")
                
        state["next_action"] = "continue_monitoring"
        return state
        
    async def learn(self, state: AgentState) -> AgentState:
        """LEARN: Update knowledge and emotional state."""
        logger.info("🧠 LEARN: Updating knowledge...")
        
        # Learn from observations
        for obs in state["observations"]:
            if obs["type"] == "high_yield_pool":
                # Update pool profile
                await self.pool_profiles.update_pool_behavior(
                    pool_address=obs["pool_address"],
                    apr=obs["apr"],
                    tvl=obs["tvl"],
                    volume_24h=obs["volume_24h"]
                )
                
        # Update emotional state based on decisions
        if state["decisions"]:
            self.emotions["confidence"] = min(0.9, self.emotions["confidence"] + 0.05)
            self.emotions["satisfaction"] = min(0.9, self.emotions["satisfaction"] + 0.1)
        else:
            self.emotions["curiosity"] = min(0.9, self.emotions["curiosity"] + 0.05)
            
        # Store learnings as memories
        if state.get("current_analysis"):
            await self.memory.remember(
                content=f"Market analysis: {state['current_analysis'][:200]}...",
                memory_type=MemoryType.LEARNING,
                category="market_analysis",
                metadata={
                    "timestamp": datetime.utcnow().isoformat(),
                    "decision_count": len(state["decisions"]),
                    "top_apr": max([obs["apr"] for obs in state["observations"] 
                                   if obs.get("apr")], default=0)
                }
            )
            
        logger.info("✅ Learning complete")
        return state
        
    async def run_cycle(self) -> Dict[str, Any]:
        """Run a complete observation-analysis-decision cycle."""
        initial_state = {
            "observations": [],
            "current_analysis": "",
            "theories": [],
            "rebalance_recommendations": [],
            "compound_recommendations": [],
            "emotions": self.emotions,
            "memories": [],
            "decisions": [],
            "next_action": "",
            "messages": []
        }
        
        result = await self.graph.ainvoke(initial_state)
        return result