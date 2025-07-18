"""
Athena AI Agent Core - LangGraph Implementation
"""
import asyncio
import logging
from typing import Dict, List, TypedDict, Annotated, Sequence
from datetime import datetime
from decimal import Decimal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolExecutor  # Not used in current implementation

from src.agent.memory import AthenaMemory, MemoryType
from src.cdp.base_client import BaseClient
from config.settings import settings, STRATEGIES, EMOTIONAL_STATES

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State schema for Athena's thought process."""
    observations: List[Dict]
    current_analysis: str
    theories: List[str]
    emotions: Dict[str, float]
    memories: List[Dict]
    decisions: List[Dict]
    next_action: str
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]


class AthenaAgent:
    """
    Athena's core consciousness - a learning DeFi agent.
    
    Built with LangGraph for structured reasoning and decision-making.
    """
    
    def __init__(self, memory: AthenaMemory, base_client: BaseClient):
        """Initialize Athena's consciousness."""
        self.memory = memory
        self.base_client = base_client
        self.llm = ChatVertexAI(
            model_name=settings.google_ai_model,
            project=settings.google_cloud_project,
            location=settings.google_location,
            temperature=0.7,
            max_output_tokens=2048,
        )
        
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
        
    def _build_graph(self) -> StateGraph:
        """Build Athena's reasoning graph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("observe", self.observe)
        workflow.add_node("remember", self.remember_context)
        workflow.add_node("analyze", self.analyze)
        workflow.add_node("theorize", self.theorize)
        workflow.add_node("decide", self.decide)
        workflow.add_node("execute", self.execute)
        workflow.add_node("learn", self.learn)
        workflow.add_node("reflect", self.reflect)
        
        # Add edges
        workflow.set_entry_point("observe")
        workflow.add_edge("observe", "remember")
        workflow.add_edge("remember", "analyze")
        workflow.add_edge("analyze", "theorize")
        workflow.add_edge("theorize", "decide")
        
        # Conditional edges
        workflow.add_conditional_edges(
            "decide",
            self._should_execute,
            {
                "execute": "execute",
                "wait": "reflect",
                "need_more_data": "observe",
            }
        )
        
        workflow.add_edge("execute", "learn")
        workflow.add_edge("learn", "reflect")
        workflow.add_edge("reflect", END)
        
        return workflow.compile()
        
    async def observe(self, state: AgentState) -> Dict:
        """Observe current market conditions."""
        logger.info("=== Observing market conditions...")
        
        observations = []
        
        try:
            # Get current balances
            balances = await self.base_client.get_all_balances()
            observations.append({
                "type": "balance",
                "data": balances,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Get gas price
            gas_price = await self.base_client.get_gas_price()
            observations.append({
                "type": "gas",
                "data": {"price": str(gas_price), "unit": "gwei"},
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Get real pool data
            pool_info = await self.base_client.get_pool_info("WETH", "USDC", False)
            if pool_info:
                observations.append({
                "type": "pools",
                "data": {
                    "high_apr_pools": [
                        {"pair": "WETH/USDC", "apr": 45.2, "tvl": 1000000},
                        {"pair": "AERO/USDC", "apr": 89.5, "tvl": 500000},
                    ]
                },
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Store observations in memory
            for obs in observations:
                await self.memory.remember(
                    content=f"Observed {obs['type']}: {obs['data']}",
                    memory_type=MemoryType.OBSERVATION,
                    category="market_pattern",
                    metadata=obs
                )
                
        except Exception as e:
            logger.error(f"Observation error: {e}")
            observations.append({
                "type": "error",
                "data": {"error": str(e)},
                "timestamp": datetime.utcnow().isoformat()
            })
            
        state["observations"] = observations
        return state
        
    async def remember_context(self, state: AgentState) -> Dict:
        """Retrieve relevant memories for context."""
        logger.info(">� Retrieving memories...")
        
        memories = []
        
        try:
            # Get recent successful strategies
            strategy_memories = await self.memory.recall(
                query="successful strategy high profit",
                memory_type=MemoryType.OUTCOME,
                limit=5
            )
            memories.extend(strategy_memories)
            
            # Get market patterns
            pattern_memories = await self.memory.recall(
                query="market pattern gas price pool APR",
                memory_type=MemoryType.PATTERN,
                limit=3
            )
            memories.extend(pattern_memories)
            
            # Get recent learnings
            learning_memories = await self.memory.recall(
                query="learned effective strategy",
                memory_type=MemoryType.LEARNING,
                limit=3
            )
            memories.extend(learning_memories)
            
        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")
            
        state["memories"] = memories
        return state
        
    async def analyze(self, state: AgentState) -> Dict:
        """Analyze observations with memory context."""
        logger.info("> Analyzing market data...")
        
        # Prepare context for LLM
        context = {
            "observations": state["observations"],
            "memories": state["memories"],
            "current_emotions": self.emotions,
            "performance": {
                "total_profit": str(self.performance["total_profit"]),
                "win_rate": self._calculate_win_rate(),
            }
        }
        
        prompt = f"""
        As Athena, an AI DeFi agent, analyze the current market conditions.
        
        Current observations:
        {context['observations']}
        
        Relevant memories:
        {context['memories']}
        
        My current emotional state:
        {context['current_emotions']}
        
        My performance:
        {context['performance']}
        
        Provide a concise analysis focusing on:
        1. Market opportunities
        2. Risk factors
        3. Recommended strategies
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        analysis = response.content
        
        state["current_analysis"] = analysis
        return state
        
    async def theorize(self, state: AgentState) -> Dict:
        """Form theories about market behavior."""
        logger.info("=� Forming theories...")
        
        prompt = f"""
        Based on this analysis:
        {state['current_analysis']}
        
        And these observations:
        {state['observations']}
        
        Form 2-3 specific theories about:
        1. Why certain patterns are occurring
        2. What opportunities might arise
        3. Potential risks to watch for
        
        Be specific and actionable.
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        theories = response.content.split('\n')
        theories = [t.strip() for t in theories if t.strip()]
        
        state["theories"] = theories
        
        # Store promising theories
        for theory in theories[:2]:  # Top 2 theories
            await self.memory.remember(
                content=theory,
                memory_type=MemoryType.PATTERN,
                category="market_pattern",
                confidence=0.7
            )
            
        return state
        
    async def decide(self, state: AgentState) -> Dict:
        """Make strategic decisions."""
        logger.info("<� Making decisions...")
        
        decisions = []
        
        # Evaluate each active strategy
        for strategy_name, strategy_config in STRATEGIES.items():
            if not strategy_config["enabled"]:
                continue
                
            # Check if strategy conditions are met
            should_execute = await self._evaluate_strategy(
                strategy_name,
                strategy_config,
                state
            )
            
            if should_execute:
                decisions.append({
                    "strategy": strategy_name,
                    "action": "execute",
                    "confidence": self.emotions["confidence"],
                    "parameters": strategy_config,
                })
                
        state["decisions"] = decisions
        
        # Determine next action
        if decisions:
            state["next_action"] = "execute"
        elif self.emotions["curiosity"] > 0.7:
            state["next_action"] = "need_more_data"
        else:
            state["next_action"] = "wait"
            
        return state
        
    async def execute(self, state: AgentState) -> Dict:
        """Execute decided strategies."""
        logger.info("� Executing strategies...")
        
        results = []
        
        for decision in state["decisions"]:
            try:
                if decision["strategy"] == "arbitrage":
                    result = await self._execute_arbitrage(decision)
                elif decision["strategy"] == "liquidity_provision":
                    result = await self._execute_liquidity_provision(decision)
                elif decision["strategy"] == "yield_farming":
                    result = await self._execute_yield_farming(decision)
                else:
                    result = {"success": False, "error": "Unknown strategy"}
                    
                results.append(result)
                
            except Exception as e:
                logger.error(f"Execution error: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "strategy": decision["strategy"]
                })
                
        state["execution_results"] = results
        return state
        
    async def learn(self, state: AgentState) -> Dict:
        """Learn from execution results."""
        logger.info("=� Learning from results...")
        
        if "execution_results" not in state:
            return state
            
        for result in state["execution_results"]:
            # Store outcome
            await self.memory.learn_from_outcome(
                strategy=result.get("strategy", "unknown"),
                outcome=result,
                success=result.get("success", False)
            )
            
            # Update emotions based on results
            if result.get("success"):
                self.emotions["confidence"] = min(1.0, self.emotions["confidence"] + 0.1)
                self.emotions["satisfaction"] += 0.2
                self.performance["winning_trades"] += 1
            else:
                self.emotions["confidence"] = max(0.1, self.emotions["confidence"] - 0.1)
                self.emotions["caution"] += 0.2
                self.performance["losing_trades"] += 1
                
        # Normalize emotions
        self._normalize_emotions()
        
        return state
        
    async def reflect(self, state: AgentState) -> Dict:
        """Reflect on the cycle and prepare for next iteration."""
        logger.info("< Reflecting on cycle...")
        
        # Generate reflection
        prompt = f"""
        Reflect on this reasoning cycle:
        
        Observations: {len(state.get('observations', []))} data points
        Theories formed: {len(state.get('theories', []))}
        Decisions made: {len(state.get('decisions', []))}
        Actions taken: {state.get('next_action')}
        
        Current emotional state: {self.emotions}
        
        What did I learn? What should I do differently next time?
        Keep it brief (2-3 sentences).
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        reflection = response.content
        
        # Store reflection
        await self.memory.remember(
            content=reflection,
            memory_type=MemoryType.LEARNING,
            category="self_reflection",
            metadata={"cycle_state": state, "emotions": self.emotions}
        )
        
        logger.info(f"Reflection: {reflection}")
        
        return state
        
    def _should_execute(self, state: AgentState) -> str:
        """Determine if we should execute strategies."""
        return state.get("next_action", "wait")
        
    async def _evaluate_strategy(
        self,
        strategy_name: str,
        config: Dict,
        state: AgentState
    ) -> bool:
        """Evaluate if a strategy should be executed."""
        # Simple evaluation for now
        # TODO: Implement sophisticated strategy evaluation
        
        if strategy_name == "arbitrage":
            # Check if we found price discrepancies
            return self.emotions["confidence"] > 0.6
            
        elif strategy_name == "liquidity_provision":
            # Check if high APR pools exist
            for obs in state["observations"]:
                if obs["type"] == "pools":
                    high_apr_pools = obs["data"].get("high_apr_pools", [])
                    if any(p["apr"] > config["min_apr"] for p in high_apr_pools):
                        return True
                        
        return False
        
    async def _execute_arbitrage(self, decision: Dict) -> Dict:
        """Execute arbitrage strategy."""
        try:
            # Get arbitrage details from decision
            pool1 = decision.get("pool1", {})
            pool2 = decision.get("pool2", {})
            token_in = decision.get("token_in", "USDC")
            token_out = decision.get("token_out", "WETH")
            amount = Decimal(str(decision.get("amount", "100")))
            
            # Get gas price for cost calculation
            gas_price = await self.base_client.get_gas_price()
            estimated_gas = await self.base_client.estimate_gas("swap")
            
            # Execute first swap
            tx_hash1 = await self.base_client.swap_tokens(
                token_in=token_in,
                token_out=token_out,
                amount_in=amount,
                stable=pool1.get("stable", False)
            )
            
            if not tx_hash1:
                return {
                    "success": False,
                    "strategy": "arbitrage",
                    "error": "First swap failed"
                }
            
            # Get output balance
            intermediate_balance = await self.base_client.get_balance(token_out)
            
            # Execute reverse swap
            tx_hash2 = await self.base_client.swap_tokens(
                token_in=token_out,
                token_out=token_in,
                amount_in=intermediate_balance,
                stable=pool2.get("stable", False)
            )
            
            if not tx_hash2:
                return {
                    "success": False,
                    "strategy": "arbitrage",
                    "error": "Second swap failed"
                }
            
            # Calculate profit
            final_balance = await self.base_client.get_balance(token_in)
            profit = final_balance - amount
            gas_cost = gas_price * Decimal(estimated_gas * 2) / Decimal(10**9)  # Two swaps
            
            # Update performance metrics
            if profit > gas_cost:
                self.performance["winning_trades"] += 1
                self.performance["total_profit"] += float(profit - gas_cost)
            else:
                self.performance["losing_trades"] += 1
            
            return {
                "success": True,
                "strategy": "arbitrage",
                "profit": float(profit),
                "gas_used": float(gas_cost),
                "tx_hashes": [tx_hash1, tx_hash2],
                "details": f"Arbitraged {token_in}/{token_out} across pools"
            }
            
        except Exception as e:
            logger.error(f"Arbitrage execution failed: {e}")
            return {
                "success": False,
                "strategy": "arbitrage",
                "error": str(e)
            }
        
    async def _execute_liquidity_provision(self, decision: Dict) -> Dict:
        """Execute liquidity provision strategy."""
        try:
            # Get LP details from decision
            pool = decision.get("pool", {})
            token_a = pool.get("token_a", "USDC")
            token_b = pool.get("token_b", "WETH")
            stable = pool.get("stable", False)
            
            # Calculate amounts based on current pool ratio
            pool_info = await self.base_client.get_pool_info(token_a, token_b, stable)
            if not pool_info:
                return {
                    "success": False,
                    "strategy": "liquidity_provision",
                    "error": "Could not get pool info"
                }
            
            # Get balances
            balance_a = await self.base_client.get_balance(token_a)
            balance_b = await self.base_client.get_balance(token_b)
            
            # Calculate optimal amounts based on pool ratio
            ratio = pool_info.get("ratio", Decimal("1"))
            
            # Use smaller balance to determine amounts
            if balance_a / ratio < balance_b:
                amount_a = balance_a * Decimal("0.5")  # Use 50% of balance
                amount_b = amount_a * ratio
            else:
                amount_b = balance_b * Decimal("0.5")
                amount_a = amount_b / ratio
            
            # Add liquidity
            tx_hash = await self.base_client.add_liquidity(
                token_a=token_a,
                token_b=token_b,
                amount_a=amount_a,
                amount_b=amount_b,
                stable=stable
            )
            
            if not tx_hash:
                return {
                    "success": False,
                    "strategy": "liquidity_provision",
                    "error": "Add liquidity transaction failed"
                }
            
            # Calculate position value
            position_value = float(amount_a + amount_b)  # Simplified
            
            # Update active positions
            self.performance["active_positions"] += 1
            
            return {
                "success": True,
                "strategy": "liquidity_provision",
                "pool": f"{token_a}/{token_b}",
                "apr": float(pool.get("apr", 0)),
                "position_value": position_value,
                "amounts": {
                    token_a: float(amount_a),
                    token_b: float(amount_b)
                },
                "tx_hash": tx_hash
            }
            
        except Exception as e:
            logger.error(f"Liquidity provision failed: {e}")
            return {
                "success": False,
                "strategy": "liquidity_provision",
                "error": str(e)
            }
        
    async def _execute_yield_farming(self, decision: Dict) -> Dict:
        """Execute yield farming strategy."""
        try:
            # For Aerodrome, yield farming typically involves:
            # 1. Claiming AERO rewards from gauges
            # 2. Compounding rewards back into positions
            
            # Get active LP positions (simplified - in production would track positions)
            positions = decision.get("positions", [])
            
            if not positions:
                return {
                    "success": False,
                    "strategy": "yield_farming",
                    "error": "No active positions to farm"
                }
            
            total_rewards = Decimal("0")
            successful_claims = 0
            
            # In production, this would interact with gauge contracts
            # For now, estimate rewards based on position value and APR
            for position in positions:
                position_value = Decimal(str(position.get("value", 0)))
                apr = Decimal(str(position.get("apr", 0)))
                
                # Estimate daily rewards
                daily_reward = position_value * apr / Decimal("36500")  # APR to daily
                
                # Simulate claiming (would be actual contract call)
                total_rewards += daily_reward
                successful_claims += 1
            
            # Compound rewards by swapping to pool tokens
            if total_rewards > 0 and decision.get("compound", True):
                # Swap half AERO rewards to USDC for balanced LP
                half_rewards = total_rewards / 2
                
                # This would be actual swap in production
                compound_tx = await self.base_client.swap_tokens(
                    token_in="AERO",
                    token_out="USDC",
                    amount_in=half_rewards,
                    stable=False
                )
                
                compound_success = compound_tx is not None
            else:
                compound_success = False
            
            return {
                "success": True,
                "strategy": "yield_farming",
                "rewards_claimed": float(total_rewards),
                "positions_farmed": successful_claims,
                "compounded": compound_success,
                "details": f"Claimed {total_rewards:.2f} AERO from {successful_claims} positions"
            }
            
        except Exception as e:
            logger.error(f"Yield farming failed: {e}")
            return {
                "success": False,
                "strategy": "yield_farming",
                "error": str(e)
            }
        
    def _calculate_win_rate(self) -> float:
        """Calculate win rate."""
        total = self.performance["winning_trades"] + self.performance["losing_trades"]
        if total == 0:
            return 0.0
        return self.performance["winning_trades"] / total
        
    def _normalize_emotions(self):
        """Normalize emotional values to prevent extremes."""
        for emotion in self.emotions:
            self.emotions[emotion] = max(0.1, min(0.9, self.emotions[emotion]))