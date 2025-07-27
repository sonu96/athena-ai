"""
Aerodrome-Specific Trading Strategies

Implements platform-specific strategies that leverage Aerodrome's unique mechanics.
Each strategy uses platform knowledge to make informed decisions.
"""
import logging
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta

from src.platforms.aerodrome.knowledge_base import AerodromeKnowledgeBase
from src.platforms.aerodrome.tokenomics import AerodromeTokenomics

logger = logging.getLogger(__name__)


class AerodromeStrategies:
    """
    Strategy implementations for Aerodrome platform.
    
    Leverages deep platform knowledge to execute:
    - Emission farming
    - Bribe optimization
    - Pool rotation
    - Risk management
    """
    
    def __init__(self, knowledge_base: AerodromeKnowledgeBase, tokenomics: AerodromeTokenomics):
        """Initialize with platform knowledge systems."""
        self.knowledge = knowledge_base
        self.tokenomics = tokenomics
        
    async def analyze_stable_farming_opportunity(
        self,
        pool_data: Dict[str, Any],
        capital: Decimal
    ) -> Dict[str, Any]:
        """
        Analyze stable pool farming opportunity.
        
        Uses knowledge of:
        - Optimal TVL ranges
        - Vote stability patterns
        - Risk factors
        """
        analysis = {
            "pool": pool_data.get("pair"),
            "strategy": "stable_farming",
            "viable": False,
            "score": 0,
            "reasons": [],
            "warnings": []
        }
        
        # Check against known optimal characteristics
        tvl = pool_data.get("tvl", 0)
        apr = pool_data.get("apr", 0)
        votes = pool_data.get("gauge_votes", 0)
        
        # TVL check (from knowledge base)
        if 1000000 <= tvl <= 50000000:  # $1M - $50M optimal
            analysis["score"] += 25
            analysis["reasons"].append("TVL in optimal range")
        else:
            analysis["warnings"].append(f"TVL outside optimal range: ${tvl:,.0f}")
            
        # APR threshold
        if apr >= 8:  # Minimum for stable farming
            analysis["score"] += 25
            analysis["reasons"].append(f"Good APR: {apr:.1f}%")
        else:
            analysis["warnings"].append(f"Low APR: {apr:.1f}%")
            
        # Vote stability (need historical data)
        if votes > 1000000:  # 1M+ veAERO votes
            analysis["score"] += 25
            analysis["reasons"].append("Strong vote support")
        else:
            analysis["warnings"].append("Low vote allocation")
            
        # Capital efficiency
        if capital >= 500:  # Minimum from knowledge
            analysis["score"] += 25
            analysis["reasons"].append("Sufficient capital")
        else:
            analysis["warnings"].append("Capital below minimum")
            
        # Final assessment
        analysis["viable"] = analysis["score"] >= 75
        analysis["expected_return"] = capital * Decimal(apr) / 100 / 365  # Daily return
        analysis["risk_assessment"] = "Low" if analysis["score"] >= 75 else "Medium"
        
        return analysis
        
    async def calculate_veaero_strategy(
        self,
        available_aero: Decimal,
        time_horizon_days: int,
        risk_tolerance: str = "medium"
    ) -> Dict[str, Any]:
        """
        Calculate optimal veAERO locking strategy.
        
        Considers:
        - Lock duration vs returns
        - Bribe income projections
        - Liquidity needs
        """
        # Get tokenomics analysis
        strategy = await self.tokenomics.analyze_veaero_strategy(
            available_aero,
            time_horizon_days
        )
        
        # Enhance with knowledge-based recommendations
        if risk_tolerance == "low":
            # Recommend shorter locks
            filtered_scenarios = [
                s for s in strategy["all_scenarios"]
                if s["lock_period_days"] <= 365
            ]
            strategy["recommended"] = filtered_scenarios[0] if filtered_scenarios else None
            strategy["reasoning"] = "Shorter lock preserves liquidity flexibility"
            
        elif risk_tolerance == "high":
            # Recommend maximum lock
            strategy["recommended"] = strategy["optimal_strategy"]
            strategy["reasoning"] = "Maximum lock for highest returns"
            
        else:
            # Balanced approach
            one_year = next(
                (s for s in strategy["all_scenarios"] if s["lock_period_days"] == 365),
                None
            )
            strategy["recommended"] = one_year or strategy["optimal_strategy"]
            strategy["reasoning"] = "1-year lock balances returns and flexibility"
            
        return strategy
        
    async def find_emission_arbitrage(
        self,
        current_positions: List[Dict[str, Any]],
        available_pools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find emission arbitrage opportunities.
        
        Identifies:
        - Overvalued positions (low emission/TVL)
        - Undervalued pools (high emission/TVL)
        - Profitable swaps after costs
        """
        opportunities = []
        
        for position in current_positions:
            position_efficiency = self._calculate_emission_efficiency(position)
            
            for pool in available_pools:
                pool_efficiency = self._calculate_emission_efficiency(pool)
                
                # Check if swap would be profitable
                if pool_efficiency > position_efficiency * 1.2:  # 20% better
                    
                    # Calculate costs
                    exit_slippage = position["liquidity"] * Decimal("0.005")  # 0.5%
                    entry_slippage = position["liquidity"] * Decimal("0.005")
                    gas_cost = Decimal("10")  # Estimated $10 gas
                    
                    total_cost = exit_slippage + entry_slippage + gas_cost
                    
                    # Calculate benefit (1 week)
                    current_weekly = position["liquidity"] * Decimal(position["apr"]) / 100 / 52
                    new_weekly = position["liquidity"] * Decimal(pool["apr"]) / 100 / 52
                    weekly_gain = new_weekly - current_weekly
                    
                    # Breakeven time
                    breakeven_weeks = total_cost / weekly_gain if weekly_gain > 0 else float('inf')
                    
                    if breakeven_weeks < 4:  # Profitable within a month
                        opportunities.append({
                            "from_pool": position["pool"],
                            "to_pool": pool["pair"],
                            "liquidity": position["liquidity"],
                            "current_apr": position["apr"],
                            "new_apr": pool["apr"],
                            "weekly_gain": float(weekly_gain),
                            "total_cost": float(total_cost),
                            "breakeven_weeks": float(breakeven_weeks),
                            "score": 100 / breakeven_weeks  # Higher score = faster payback
                        })
                        
        # Sort by score
        opportunities.sort(key=lambda x: x["score"], reverse=True)
        return opportunities[:5]  # Top 5
        
    async def analyze_bribe_opportunity(
        self,
        pool_address: str,
        pool_data: Dict[str, Any],
        veaero_balance: Decimal
    ) -> Dict[str, Any]:
        """
        Analyze bribe voting opportunity.
        
        Calculates:
        - Expected bribe income
        - Optimal vote allocation
        - Risk/reward ratio
        """
        analysis = {
            "pool": pool_data.get("pair"),
            "viable": False,
            "expected_bribes": Decimal("0"),
            "expected_fees": Decimal("0"),
            "total_apr": Decimal("0"),
            "recommended_allocation": Decimal("0")
        }
        
        # Get current bribes (would be fetched from bribe contract)
        current_bribes = pool_data.get("bribes_usd", 0)
        pool_votes = pool_data.get("gauge_votes", 1)
        
        # Calculate bribe efficiency
        bribe_per_vote = Decimal(current_bribes) / Decimal(pool_votes)
        
        # Calculate expected income
        if veaero_balance > 0:
            expected_bribes = bribe_per_vote * veaero_balance
            
            # Add fee income (simplified)
            pool_fees_daily = Decimal(pool_data.get("volume_24h", 0)) * Decimal("0.003") * Decimal("0.5")
            vote_share = veaero_balance / self.tokenomics.total_veaero
            expected_fees_weekly = pool_fees_daily * 7 * vote_share
            
            # Calculate APR
            veaero_value = veaero_balance * Decimal("1.0")  # Assume $1 per veAERO
            weekly_income = expected_bribes + expected_fees_weekly
            apr = (weekly_income * 52 / veaero_value) * 100
            
            analysis["expected_bribes"] = expected_bribes
            analysis["expected_fees"] = expected_fees_weekly
            analysis["total_apr"] = apr
            
            # Viability check
            if apr > 15:  # 15% minimum for bribe voting
                analysis["viable"] = True
                
                # Calculate optimal allocation (max 20% per pool for diversification)
                max_allocation = veaero_balance * Decimal("0.2")
                if bribe_per_vote > Decimal("0.001"):  # Good bribe ratio
                    analysis["recommended_allocation"] = max_allocation
                else:
                    analysis["recommended_allocation"] = veaero_balance * Decimal("0.1")
                    
        return analysis
        
    async def design_pool_rotation_schedule(
        self,
        positions: List[Dict[str, Any]],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Design optimal pool rotation schedule.
        
        Based on:
        - Emission decay cycles
        - Vote migration patterns
        - Gas optimization
        """
        schedule = {
            "immediate_actions": [],
            "weekly_actions": [],
            "monthly_actions": [],
            "rationale": {}
        }
        
        current_epoch = self._get_current_epoch()
        
        for position in positions:
            # Check position health
            health = await self._assess_position_health(position)
            
            if health["score"] < 30:
                # Immediate exit needed
                schedule["immediate_actions"].append({
                    "action": "exit",
                    "pool": position["pool"],
                    "reason": health["primary_issue"],
                    "urgency": "high"
                })
                
            elif health["score"] < 60:
                # Plan exit this week
                schedule["weekly_actions"].append({
                    "action": "reduce",
                    "pool": position["pool"],
                    "amount": "50%",
                    "reason": health["primary_issue"],
                    "target_epoch": current_epoch + 1
                })
                
            elif health["trending"] == "down":
                # Monitor for next month
                schedule["monthly_actions"].append({
                    "action": "monitor",
                    "pool": position["pool"],
                    "threshold": "APR < 15%",
                    "reason": "Declining performance"
                })
                
        # Add rationale
        schedule["rationale"] = {
            "immediate": "Positions with critical issues requiring immediate action",
            "weekly": "Positions to adjust at next epoch for gas efficiency",
            "monthly": "Positions to review based on trend analysis"
        }
        
        return schedule
        
    async def calculate_compound_schedule(
        self,
        position_value: Decimal,
        daily_rewards: Decimal,
        gas_price: Decimal
    ) -> Dict[str, Any]:
        """
        Calculate optimal compound frequency.
        
        Balances:
        - Compound benefits
        - Gas costs
        - Time value
        """
        # Calculate compound frequencies
        frequencies = [1, 7, 14, 30]  # Daily, weekly, bi-weekly, monthly
        
        results = []
        
        for days in frequencies:
            # Rewards per period
            period_rewards = daily_rewards * days
            
            # Gas cost
            compound_gas = gas_price * Decimal("2")  # Claim + reinvest
            
            # Net benefit
            net_rewards = period_rewards - compound_gas
            
            # APR impact (simplified compound interest)
            compounds_per_year = 365 / days
            compound_apr_boost = (1 + net_rewards / position_value) ** compounds_per_year - 1
            
            # Gas cost as % of rewards
            gas_percentage = (compound_gas / period_rewards * 100) if period_rewards > 0 else 100
            
            results.append({
                "frequency_days": days,
                "period_rewards": float(period_rewards),
                "gas_cost": float(compound_gas),
                "net_rewards": float(net_rewards),
                "apr_boost": float(compound_apr_boost * 100),
                "gas_percentage": float(gas_percentage),
                "viable": gas_percentage < 10  # Less than 10% to gas
            })
            
        # Find optimal
        viable_options = [r for r in results if r["viable"]]
        optimal = min(viable_options, key=lambda x: x["frequency_days"]) if viable_options else results[-1]
        
        return {
            "optimal_frequency": optimal["frequency_days"],
            "reasoning": f"Compound every {optimal['frequency_days']} days for best ROI",
            "all_options": results,
            "recommendation": self._get_compound_recommendation(optimal, position_value)
        }
        
    # Helper methods
    
    def _calculate_emission_efficiency(self, pool_data: Dict[str, Any]) -> Decimal:
        """Calculate emission efficiency (emissions per dollar of TVL)."""
        emissions = pool_data.get("weekly_emissions", 0)
        tvl = pool_data.get("tvl", 1)
        return Decimal(emissions) / Decimal(tvl) * 1000000  # Per million TVL
        
    def _get_current_epoch(self) -> int:
        """Get current epoch number."""
        # Calculate weeks since Aerodrome launch
        launch_date = datetime(2023, 8, 31)  # Approximate
        weeks_elapsed = (datetime.utcnow() - launch_date).days // 7
        return weeks_elapsed
        
    async def _assess_position_health(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Assess health score of a position."""
        score = 100
        issues = []
        
        # Check APR
        if position.get("apr", 0) < 10:
            score -= 40
            issues.append("APR below 10%")
            
        # Check IL
        if position.get("impermanent_loss", 0) > 10:
            score -= 30
            issues.append("High impermanent loss")
            
        # Check vote trend (would need historical data)
        if position.get("vote_trend", "stable") == "declining":
            score -= 20
            issues.append("Declining vote support")
            
        # Check pool balance
        if abs(position.get("pool_ratio", 1) - 1) > 0.2:
            score -= 10
            issues.append("Pool imbalanced")
            
        return {
            "score": max(0, score),
            "issues": issues,
            "primary_issue": issues[0] if issues else None,
            "trending": "down" if score < 70 else "stable"
        }
        
    def _get_compound_recommendation(self, optimal: Dict[str, Any], position_value: Decimal) -> str:
        """Get compound frequency recommendation."""
        if optimal["frequency_days"] == 1:
            return "Daily compounding optimal for large positions"
        elif optimal["frequency_days"] == 7:
            return "Weekly compounding balances gas costs and growth"
        elif optimal["frequency_days"] == 14:
            return "Bi-weekly compounding for medium positions"
        else:
            return "Monthly compounding due to high gas costs"
            
    async def execute_strategy(
        self,
        strategy_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a specific strategy with given parameters.
        
        This is the main entry point for the agent to execute strategies.
        """
        logger.info(f"Executing strategy: {strategy_name}")
        
        if strategy_name == "stable_farming":
            return await self.analyze_stable_farming_opportunity(
                parameters["pool_data"],
                parameters["capital"]
            )
            
        elif strategy_name == "veaero_maximizer":
            return await self.calculate_veaero_strategy(
                parameters["available_aero"],
                parameters["time_horizon_days"],
                parameters.get("risk_tolerance", "medium")
            )
            
        elif strategy_name == "emission_arbitrage":
            return await self.find_emission_arbitrage(
                parameters["current_positions"],
                parameters["available_pools"]
            )
            
        elif strategy_name == "bribe_optimization":
            return await self.analyze_bribe_opportunity(
                parameters["pool_address"],
                parameters["pool_data"],
                parameters["veaero_balance"]
            )
            
        else:
            return {
                "error": f"Unknown strategy: {strategy_name}",
                "available_strategies": [
                    "stable_farming",
                    "veaero_maximizer",
                    "emission_arbitrage",
                    "bribe_optimization"
                ]
            }