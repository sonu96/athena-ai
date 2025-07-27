"""
Athena AI Agent Core - LangGraph Implementation with MCP & AgentKit

Enhanced version using QuickNode MCP for data and Coinbase AgentKit for transactions.
"""
import asyncio
import logging
from typing import Dict, List, TypedDict, Annotated, Sequence, Optional
from datetime import datetime, timedelta
from decimal import Decimal

from langgraph.graph import StateGraph, END
import google.generativeai as genai

from src.agent.memory import AthenaMemory, MemoryType
from src.agent.pool_profiles import PoolProfileManager
from src.mcp.quicknode_mcp import QuickNodeMCP
from src.agentkit.agent_client import AthenaAgentKit
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
    - QuickNode MCP for natural language blockchain queries
    - Coinbase AgentKit for AI-native transaction execution
    """
    
    def __init__(self, memory: AthenaMemory, firestore_client=None):
        """Initialize Athena's consciousness with new architecture."""
        self.memory = memory
        self.firestore = firestore_client
        
        # Initialize MCP for blockchain data
        self.mcp = QuickNodeMCP(settings.quicknode_api_key)
        
        # Initialize AgentKit for transactions
        self.agentkit = AthenaAgentKit(
            api_key=settings.cdp_api_key,
            api_secret=settings.cdp_api_secret,
            wallet_data=settings.agent_wallet_id
        )
        
        # Initialize pool profile manager
        self.pool_profiles = PoolProfileManager(firestore_client)
        
        # Initialize Gemini directly without LangChain
        if settings.google_api_key:
            genai.configure(api_key=settings.google_api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash-latest",
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
        """Build Athena's reasoning graph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each state
        workflow.add_node("observe", self.observe)
        workflow.add_node("remember", self.remember)
        workflow.add_node("analyze", self.analyze)
        workflow.add_node("theorize", self.theorize)
        workflow.add_node("strategize", self.strategize)
        workflow.add_node("decide", self.decide)
        workflow.add_node("execute", self.execute)
        workflow.add_node("learn", self.learn)
        workflow.add_node("reflect", self.reflect)
        
        # Define transitions
        workflow.add_edge("observe", "remember")
        workflow.add_edge("remember", "analyze")
        workflow.add_edge("analyze", "theorize")
        workflow.add_edge("theorize", "strategize")
        workflow.add_edge("strategize", "decide")
        workflow.add_edge("decide", "execute")
        workflow.add_edge("execute", "learn")
        workflow.add_edge("learn", "reflect")
        workflow.add_edge("reflect", END)
        
        # Set entry point
        workflow.set_entry_point("observe")
        
        return workflow.compile()
        
    async def observe(self, state: AgentState) -> AgentState:
        """OBSERVE: Gather market data using MCP natural language queries."""
        logger.info("🔍 OBSERVE: Gathering market intelligence...")
        
        observations = []
        
        # Use MCP for comprehensive market overview
        market_overview = await self.mcp.get_market_overview()
        observations.append({
            "type": "market_overview",
            "data": market_overview,
            "timestamp": datetime.utcnow()
        })
        
        # Get high-yield opportunities
        opportunities = await self.mcp.get_aerodrome_pools(
            min_apr=settings.min_apr_for_memory,
            min_tvl=settings.min_volume_for_memory
        )
        
        for pool in opportunities[:10]:  # Top 10
            observations.append({
                "type": "high_yield_pool",
                "pool": pool.get("pair"),
                "apr": float(pool.get("apr", 0)),
                "tvl": float(pool.get("tvl", 0)),
                "volume_24h": float(pool.get("volume_24h", 0)),
                "address": pool.get("address"),
                "timestamp": datetime.utcnow()
            })
            
        # Get gas optimization insights
        gas_info = await self.mcp.optimize_gas_timing()
        observations.append({
            "type": "gas_optimization",
            "data": gas_info,
            "timestamp": datetime.utcnow()
        })
        
        # Check our positions using AgentKit
        if hasattr(self, '_initialized') and self._initialized:
            balance_eth = await self.agentkit.get_balance("eth")
            balance_usdc = await self.agentkit.get_balance("usdc")
            
            observations.append({
                "type": "wallet_status",
                "balances": {
                    "ETH": float(balance_eth),
                    "USDC": float(balance_usdc)
                },
                "address": self.agentkit.address,
                "timestamp": datetime.utcnow()
            })
        
        state["observations"] = observations
        logger.info(f"📊 Collected {len(observations)} observations")
        
        return state
        
    async def remember(self, state: AgentState) -> AgentState:
        """REMEMBER: Recall relevant memories and patterns."""
        logger.info("🧠 REMEMBER: Recalling relevant patterns...")
        
        memories = []
        
        # For each observation, recall relevant memories
        for obs in state["observations"]:
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
        
        Current Market Observations:
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
        """
        
        response = self.model.generate_content(prompt)
        analysis = response.text
        
        state["current_analysis"] = analysis
        logger.info("✅ Analysis complete")
        
        return state
        
    async def execute(self, state: AgentState) -> AgentState:
        """EXECUTE: Perform actions using AgentKit."""
        logger.info("⚡ EXECUTE: Taking action...")
        
        # Initialize AgentKit if needed
        if not hasattr(self, '_initialized') or not self._initialized:
            await self.agentkit.initialize()
            self._initialized = True
            logger.info(f"💳 Wallet initialized: {self.agentkit.address}")
        
        # Check if we're in observation mode
        if self._is_observation_mode():
            logger.info("👁️ Observation mode - no trades executed")
            state["next_action"] = "observe_only"
            return state
            
        # Execute decisions
        for decision in state["decisions"]:
            if decision["action"] == "add_liquidity":
                result = await self.agentkit.add_liquidity(
                    token_a=decision["token_a"],
                    token_b=decision["token_b"],
                    amount_a=Decimal(str(decision["amount_a"])),
                    amount_b=Decimal(str(decision["amount_b"])),
                    pool_type=decision.get("pool_type", "volatile")
                )
                
                logger.info(f"✅ Liquidity added: {result}")
                
            elif decision["action"] == "rebalance":
                # Remove from old pool
                remove_result = await self.agentkit.remove_liquidity(
                    pool_address=decision["from_pool"],
                    lp_amount=Decimal(str(decision["lp_amount"]))
                )
                
                # Add to new pool  
                add_result = await self.agentkit.add_liquidity(
                    token_a=decision["to_token_a"],
                    token_b=decision["to_token_b"],
                    amount_a=Decimal(str(decision["amount_a"])),
                    amount_b=Decimal(str(decision["amount_b"]))
                )
                
                logger.info(f"✅ Rebalanced: {remove_result} -> {add_result}")
                
            elif decision["action"] == "claim_rewards":
                result = await self.agentkit.claim_rewards(
                    pool_address=decision["pool_address"]
                )
                
                logger.info(f"✅ Rewards claimed: {result}")
                
        state["next_action"] = "continue_monitoring"
        return state
        
    async def strategize(self, state: AgentState) -> AgentState:
        """STRATEGIZE: Plan optimal actions using MCP insights."""
        logger.info("📋 STRATEGIZE: Planning optimal moves...")
        
        # Get rebalancing recommendations from MCP
        rebalance_recs = []
        
        for position in self.performance.get("current_positions", []):
            analysis = await self.mcp.analyze_rebalance_opportunity(
                current_pool=position.get("pool"),
                current_apr=position.get("apr", 0),
                position_value=position.get("value_usd", 0)
            )
            
            if analysis.get("recommendation") == "rebalance":
                rebalance_recs.append(analysis)
                
        state["rebalance_recommendations"] = rebalance_recs
        
        # Get compound timing recommendations
        compound_recs = []
        
        for position in self.performance.get("current_positions", []):
            if position.get("pending_rewards", 0) > settings.compound_min_value:
                compound_recs.append({
                    "pool": position["pool"],
                    "rewards": position["pending_rewards"],
                    "recommended_action": "compound",
                    "optimal_time": "next_gas_window"
                })
                
        state["compound_recommendations"] = compound_recs
        
        logger.info(f"📊 Strategy: {len(rebalance_recs)} rebalances, {len(compound_recs)} compounds")
        
        return state
        
    def _is_observation_mode(self) -> bool:
        """Check if agent is in observation mode."""
        if not settings.observation_mode:
            return False
            
        days_elapsed = (datetime.utcnow() - self.observation_start).days
        return days_elapsed < settings.observation_days
        
    async def theorize(self, state: AgentState) -> AgentState:
        """THEORIZE: Form hypotheses about market behavior."""
        logger.info("🎯 THEORIZE: Forming market hypotheses...")
        
        theories = []
        
        # Analyze high APR pools
        high_apr_pools = [
            obs for obs in state["observations"] 
            if obs.get("type") == "high_yield_pool" and obs.get("apr", 0) > 50
        ]
        
        if high_apr_pools:
            theories.append(
                f"High APR opportunity: {len(high_apr_pools)} pools offering >50% APR. "
                f"Top pool {high_apr_pools[0]['pool']} at {high_apr_pools[0]['apr']:.1f}% APR."
            )
            
        # Gas optimization theory
        gas_obs = next((obs for obs in state["observations"] if obs["type"] == "gas_optimization"), None)
        if gas_obs and gas_obs.get("data", {}).get("recommended_window"):
            theories.append(
                f"Gas optimization window: Best time for transactions is "
                f"{gas_obs['data']['recommended_window']}"
            )
            
        state["theories"] = theories
        logger.info(f"💡 Formed {len(theories)} theories")
        
        return state
        
    async def decide(self, state: AgentState) -> AgentState:
        """DECIDE: Make actionable decisions."""
        logger.info("🎯 DECIDE: Making strategic decisions...")
        
        decisions = []
        
        # Decision logic based on analysis and theories
        if self._is_observation_mode():
            decisions.append({
                "action": "observe",
                "reason": "Still in observation period",
                "confidence": 1.0
            })
        else:
            # Check for rebalancing opportunities
            for rec in state.get("rebalance_recommendations", []):
                if rec.get("expected_profit", 0) > 50:  # $50 minimum
                    decisions.append({
                        "action": "rebalance",
                        "from_pool": rec["current_pool"],
                        "to_pool": rec["recommended_pool"],
                        "reason": rec["reason"],
                        "expected_profit": rec["expected_profit"],
                        "confidence": self.emotions["confidence"]
                    })
                    
            # Check for new high-yield positions
            if not decisions:  # No rebalancing needed
                top_pools = [
                    obs for obs in state["observations"]
                    if obs.get("type") == "high_yield_pool" and obs.get("apr", 0) > 30
                ]
                
                if top_pools and self.emotions["confidence"] > 0.7:
                    pool = top_pools[0]
                    decisions.append({
                        "action": "add_liquidity",
                        "pool": pool["pool"],
                        "token_a": pool["pool"].split("/")[0],
                        "token_b": pool["pool"].split("/")[1],
                        "amount_a": 100,  # Start small
                        "amount_b": 100,
                        "reason": f"High APR opportunity at {pool['apr']:.1f}%",
                        "confidence": self.emotions["confidence"]
                    })
                    
        state["decisions"] = decisions
        logger.info(f"⚖️ Made {len(decisions)} decisions")
        
        return state
        
    async def learn(self, state: AgentState) -> AgentState:
        """LEARN: Extract patterns and store learnings."""
        logger.info("📚 LEARN: Extracting patterns...")
        
        # Store observations as memories
        for obs in state["observations"]:
            if obs.get("type") == "high_yield_pool":
                await self.memory.remember(
                    content=f"Pool {obs['pool']} offering {obs['apr']:.1f}% APR with ${obs['tvl']:,.0f} TVL",
                    memory_type=MemoryType.OBSERVATION,
                    category="pool_analysis",
                    metadata={
                        "pool": obs["pool"],
                        "apr": obs["apr"],
                        "tvl": obs["tvl"],
                        "volume": obs.get("volume_24h", 0)
                    }
                )
                
        # Learn from decisions made
        for decision in state.get("decisions", []):
            if decision["action"] != "observe":
                await self.memory.remember(
                    content=f"Decided to {decision['action']} because {decision['reason']}",
                    memory_type=MemoryType.STRATEGY,
                    category="strategy_performance",
                    metadata={
                        "action": decision["action"],
                        "confidence": decision["confidence"],
                        "expected_outcome": decision.get("expected_profit", 0)
                    }
                )
                
        logger.info("✅ Learning complete")
        return state
        
    async def reflect(self, state: AgentState) -> AgentState:
        """REFLECT: Self-evaluate and adjust emotional state."""
        logger.info("🪞 REFLECT: Self-evaluation...")
        
        # Adjust emotions based on outcomes
        successful_decisions = len([d for d in state["decisions"] if d.get("confidence", 0) > 0.7])
        total_decisions = len(state["decisions"])
        
        if total_decisions > 0:
            success_rate = successful_decisions / total_decisions
            
            # Update confidence
            self.emotions["confidence"] = min(0.95, self.emotions["confidence"] * (1 + success_rate * 0.1))
            
            # Update satisfaction
            self.emotions["satisfaction"] = success_rate
            
        # Store emotional state
        if self.firestore:
            self.firestore.save_agent_state({
                "emotions": self.emotions,
                "performance": {
                    "total_profit": float(self.performance["total_profit"]),
                    "winning_trades": self.performance["winning_trades"],
                    "losing_trades": self.performance["losing_trades"]
                },
                "observation_mode": self._is_observation_mode()
            })
            
        logger.info(f"🎭 Emotional state: {self.emotions}")
        return state