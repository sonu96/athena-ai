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
            
            # Get pool data (mock for now)
            # TODO: Implement real pool scanning
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
        # Mock implementation
        # TODO: Implement real arbitrage
        return {
            "success": True,
            "strategy": "arbitrage",
            "profit": 25.50,
            "gas_used": 5.20,
            "details": "Arbitraged WETH/USDC price discrepancy"
        }
        
    async def _execute_liquidity_provision(self, decision: Dict) -> Dict:
        """Execute liquidity provision strategy."""
        # Mock implementation
        # TODO: Implement real LP
        return {
            "success": True,
            "strategy": "liquidity_provision",
            "pool": "AERO/USDC",
            "apr": 89.5,
            "position_value": 1000,
        }
        
    async def _execute_yield_farming(self, decision: Dict) -> Dict:
        """Execute yield farming strategy."""
        # Mock implementation
        # TODO: Implement real yield farming
        return {
            "success": True,
            "strategy": "yield_farming",
            "rewards_claimed": 15.75,
            "compounded": True,
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