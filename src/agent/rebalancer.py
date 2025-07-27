"""
Smart Rebalancer - Enhanced with MCP and AgentKit

Uses QuickNode MCP for analysis and Coinbase AgentKit for execution.
"""
import logging
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta

from src.agent.memory import AthenaMemory, MemoryType
from src.mcp.quicknode_mcp import QuickNodeMCP
from src.agentkit.agent_client import AthenaAgentKit
from config.settings import settings

logger = logging.getLogger(__name__)


class SmartRebalancer:
    """
    Memory-driven rebalancing system using natural language analysis.
    
    Combines MCP's analytical power with AgentKit's execution simplicity.
    """
    
    def __init__(self, 
                 memory: AthenaMemory,
                 mcp_client: QuickNodeMCP,
                 agentkit_client: AthenaAgentKit):
        """Initialize the smart rebalancer."""
        self.memory = memory
        self.mcp = mcp_client
        self.agentkit = agentkit_client
        
        # Rebalancing parameters from settings
        self.min_apr_threshold = settings.rebalance_min_apr_threshold
        self.apr_drop_trigger = settings.rebalance_apr_drop_trigger
        self.cost_multiplier = settings.rebalance_cost_multiplier
        self.compound_min_value = settings.compound_min_value
        self.compound_optimal_gas = settings.compound_optimal_gas
        
        # Track rebalancing history
        self.rebalance_history = []
        
    async def analyze_positions(self, positions: List[Dict]) -> List[Dict]:
        """Analyze all positions for rebalancing opportunities using MCP.
        
        Args:
            positions: List of current positions with pool, APR, value
            
        Returns:
            List of rebalancing recommendations
        """
        recommendations = []
        
        for position in positions:
            # Use MCP to analyze each position
            analysis = await self.mcp.analyze_rebalance_opportunity(
                current_pool=position.get("pool_pair", position.get("pool")),
                current_apr=position.get("apr", 0),
                position_value=position.get("value_usd", 0)
            )
            
            # Check if rebalancing is recommended
            if analysis.get("recommendation") == "rebalance":
                # Get memories about the target pool
                target_pool = analysis.get("recommended_pool", {})
                if target_pool:
                    pool_memories = await self.memory.recall_pool_memories(
                        pool_pair=target_pool.get("pair"),
                        time_window_hours=168  # Last week
                    )
                    
                    # Analyze patterns
                    confidence = await self._calculate_confidence(
                        target_pool, pool_memories
                    )
                    
                    recommendation = {
                        "position": position,
                        "target_pool": target_pool,
                        "expected_improvement": analysis.get("apr_improvement", 0),
                        "expected_profit": analysis.get("expected_profit_usd", 0),
                        "gas_cost": analysis.get("estimated_gas_cost", 0),
                        "confidence": confidence,
                        "reason": analysis.get("reason", "Higher APR opportunity"),
                        "action_steps": analysis.get("steps", [])
                    }
                    
                    recommendations.append(recommendation)
                    
        # Sort by expected profit
        recommendations.sort(key=lambda x: x["expected_profit"], reverse=True)
        
        logger.info(f"Found {len(recommendations)} rebalancing opportunities")
        return recommendations
        
    async def should_compound(self, position: Dict, gas_price: Decimal) -> bool:
        """Determine if rewards should be compounded using MCP analysis.
        
        Args:
            position: Position with pending rewards
            gas_price: Current gas price in gwei
            
        Returns:
            True if compounding is profitable
        """
        pending_rewards = position.get("pending_rewards_usd", 0)
        
        # Skip if rewards too small
        if pending_rewards < self.compound_min_value:
            return False
            
        # Use MCP to analyze compound profitability
        analysis = await self.mcp.query(
            f"Should I compound {pending_rewards} USD rewards from {position['pool_pair']} "
            f"pool with current gas at {gas_price} gwei? Consider compound frequency optimization."
        )
        
        return analysis.get("recommendation") == "compound"
        
    async def execute_rebalance(self, recommendation: Dict) -> Dict:
        """Execute a rebalancing operation using AgentKit.
        
        Args:
            recommendation: Rebalancing recommendation from analyze_positions
            
        Returns:
            Execution result
        """
        logger.info(f"Executing rebalance from {recommendation['position']['pool_pair']} "
                   f"to {recommendation['target_pool']['pair']}")
                   
        try:
            # Natural language execution with AgentKit
            prompt = (
                f"Rebalance my position: "
                f"1. Remove all liquidity from {recommendation['position']['pool_address']} "
                f"2. Swap tokens as needed to match {recommendation['target_pool']['pair']} ratio "
                f"3. Add liquidity to {recommendation['target_pool']['address']} "
                f"with 0.5% slippage protection"
            )
            
            result = await self.agentkit.execute_natural_language(prompt)
            
            # Store outcome in memory
            await self._store_rebalance_outcome(recommendation, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Rebalance execution failed: {e}")
            return {"error": str(e)}
            
    async def execute_compound(self, position: Dict) -> Dict:
        """Execute reward compounding using AgentKit.
        
        Args:
            position: Position with pending rewards
            
        Returns:
            Execution result
        """
        logger.info(f"Compounding rewards for {position['pool_pair']}")
        
        try:
            # Simple natural language command
            prompt = (
                f"Claim and compound all rewards from {position['pool_address']} "
                f"back into the same pool"
            )
            
            result = await self.agentkit.execute_natural_language(prompt)
            
            # Learn from outcome
            await self.memory.remember(
                content=f"Compounded {position['pending_rewards_usd']} USD rewards for {position['pool_pair']}",
                memory_type=MemoryType.OUTCOME,
                category="compound_execution",
                metadata={
                    "pool": position["pool_pair"],
                    "rewards": position["pending_rewards_usd"],
                    "success": not result.get("error"),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Compound execution failed: {e}")
            return {"error": str(e)}
            
    async def find_emergency_exits(self, positions: List[Dict]) -> List[Dict]:
        """Find positions that need emergency exit using MCP.
        
        Args:
            positions: Current positions
            
        Returns:
            List of emergency exit recommendations
        """
        emergencies = []
        
        # Use MCP to analyze all positions at once
        analysis = await self.mcp.query(
            f"Analyze these positions for emergency exit needs: {positions}. "
            f"Check for: rug pull risks, extreme IL, liquidity drainage, "
            f"or APR collapse below {self.min_apr_threshold}%"
        )
        
        for recommendation in analysis.get("emergency_exits", []):
            emergencies.append({
                "position": recommendation["position"],
                "reason": recommendation["reason"],
                "urgency": recommendation["urgency"],
                "estimated_loss": recommendation.get("estimated_loss", 0)
            })
            
        return emergencies
        
    async def optimize_gas_timing(self) -> Dict:
        """Get optimal transaction timing using MCP."""
        return await self.mcp.optimize_gas_timing()
        
    async def _calculate_confidence(self, target_pool: Dict, memories: List[Dict]) -> float:
        """Calculate confidence score for a rebalancing decision.
        
        Args:
            target_pool: Target pool information
            memories: Historical memories about the pool
            
        Returns:
            Confidence score (0-1)
        """
        if not memories:
            return 0.5  # Neutral confidence for new pools
            
        # Analyze historical performance
        positive_outcomes = 0
        total_outcomes = 0
        
        for memory in memories:
            if memory.get("memory_type") == "OUTCOME":
                total_outcomes += 1
                if memory.get("metadata", {}).get("profit", 0) > 0:
                    positive_outcomes += 1
                    
        if total_outcomes > 0:
            success_rate = positive_outcomes / total_outcomes
        else:
            success_rate = 0.5
            
        # Factor in pool age and stability
        pool_age_days = target_pool.get("age_days", 0)
        age_factor = min(pool_age_days / 30, 1.0)  # Max confidence at 30 days
        
        # Combine factors
        confidence = (success_rate * 0.6) + (age_factor * 0.4)
        
        return min(confidence, 0.95)  # Cap at 95%
        
    async def _store_rebalance_outcome(self, recommendation: Dict, result: Dict):
        """Store rebalancing outcome for learning.
        
        Args:
            recommendation: Original recommendation
            result: Execution result
        """
        success = not result.get("error")
        
        content = (
            f"Rebalanced from {recommendation['position']['pool_pair']} "
            f"to {recommendation['target_pool']['pair']} - "
            f"{'Success' if success else 'Failed'}"
        )
        
        metadata = {
            "from_pool": recommendation["position"]["pool_pair"],
            "to_pool": recommendation["target_pool"]["pair"],
            "expected_profit": recommendation["expected_profit"],
            "gas_cost": recommendation["gas_cost"],
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if success:
            metadata["tx_hash"] = result.get("transaction_hash")
            
        await self.memory.remember(
            content=content,
            memory_type=MemoryType.OUTCOME,
            category="rebalance_execution",
            metadata=metadata
        )
        
        # Update history
        self.rebalance_history.append({
            "timestamp": datetime.utcnow(),
            "recommendation": recommendation,
            "result": result
        })