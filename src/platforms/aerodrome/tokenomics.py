"""
Aerodrome Tokenomics Model

Deep understanding of AERO token economics including:
- Emission schedules and decay
- veAERO mechanics and voting power
- Reward distribution formulas
- Bribe economics
"""
import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AerodromeTokenomics:
    """
    Model of Aerodrome's token economics.
    
    Provides calculations and predictions for:
    - AERO emissions over time
    - veAERO voting power
    - Optimal locking strategies
    - Bribe efficiency
    """
    
    def __init__(self):
        """Initialize tokenomics model."""
        # Core parameters
        self.initial_supply = Decimal("500000000")  # 500M AERO
        self.initial_weekly_emission = Decimal("10000000")  # 10M first week
        self.emission_decay = Decimal("0.98")  # 2% weekly decay
        self.max_lock_time = 4 * 365  # 4 years in days
        
        # Distribution splits
        self.lp_share = Decimal("0.67")  # 67% to LPs
        self.veaero_share = Decimal("0.33")  # 33% to veAERO
        
        # Boost parameters
        self.min_boost = Decimal("1.0")
        self.max_boost = Decimal("2.5")
        self.boost_factor = Decimal("1.5")
        
        # Current state (would be fetched on-chain)
        self.current_week = 52  # Weeks since launch
        self.total_veaero = Decimal("150000000")  # Total veAERO
        self.circulating_supply = Decimal("200000000")  # Circulating AERO
        
    async def get_full_tokenomics(self) -> Dict[str, Any]:
        """Get comprehensive tokenomics overview."""
        return {
            "token_info": {
                "name": "AERO",
                "initial_supply": str(self.initial_supply),
                "current_circulating": str(self.circulating_supply),
                "locked_as_veaero": str(self.total_veaero),
                "percent_locked": f"{(self.total_veaero / self.circulating_supply * 100):.1f}%"
            },
            
            "emission_model": {
                "type": "Decaying emissions",
                "initial_weekly": str(self.initial_weekly_emission),
                "decay_rate": "2% per week",
                "current_weekly": str(self.calculate_current_emission()),
                "projected_1_year": str(self.project_emissions(52)),
                "terminal_emissions": "Approaching zero over time"
            },
            
            "veaero_mechanics": {
                "lock_period": "1 week to 4 years",
                "voting_power": "1 AERO locked 4 years = 1 veAERO",
                "decay": "Linear decay to zero at unlock",
                "benefits": [
                    "Direct weekly emissions to pools",
                    "Earn trading fees from voted pools",
                    "Collect bribes for votes",
                    "Boost LP rewards up to 2.5x",
                    "Governance participation"
                ],
                "rebase": "Weekly distribution to veAERO holders"
            },
            
            "distribution_model": {
                "liquidity_providers": "67% of emissions",
                "veaero_holders": "33% of emissions (rebase)",
                "team_allocation": "Vested over 4 years",
                "no_premine": True,
                "fair_launch": True
            },
            
            "value_accrual": {
                "trading_fees": "100% to liquidity providers",
                "bribes": "100% to veAERO voters",
                "emissions": "Split 67/33 between LPs and veAERO",
                "protocol_revenue": "0% - fully decentralized"
            },
            
            "game_theory": {
                "ve_3_3": "Combination of ve-model and (3,3) cooperation",
                "flywheel": "More locks → More value → More demand → More locks",
                "nash_equilibrium": "Optimal to lock for maximum duration",
                "coordination": "Voters and LPs aligned for maximum value"
            }
        }
        
    def calculate_current_emission(self) -> Decimal:
        """Calculate current weekly emission based on decay."""
        return self.initial_weekly_emission * (self.emission_decay ** self.current_week)
        
    def project_emissions(self, weeks_ahead: int) -> Decimal:
        """Project total emissions for next N weeks."""
        current_emission = self.calculate_current_emission()
        total = Decimal("0")
        
        for week in range(weeks_ahead):
            week_emission = current_emission * (self.emission_decay ** week)
            total += week_emission
            
        return total
        
    def calculate_veaero_from_lock(
        self,
        aero_amount: Decimal,
        lock_days: int
    ) -> Decimal:
        """Calculate veAERO received from locking AERO."""
        if lock_days < 7:
            return Decimal("0")
        if lock_days > self.max_lock_time:
            lock_days = self.max_lock_time
            
        # Linear scaling: 4 years = 100%, 1 week ≈ 0.5%
        lock_ratio = Decimal(lock_days) / Decimal(self.max_lock_time)
        return aero_amount * lock_ratio
        
    def calculate_voting_power_decay(
        self,
        initial_veaero: Decimal,
        days_elapsed: int,
        total_lock_days: int
    ) -> Decimal:
        """Calculate remaining voting power after time elapsed."""
        if days_elapsed >= total_lock_days:
            return Decimal("0")
            
        remaining_ratio = Decimal(total_lock_days - days_elapsed) / Decimal(total_lock_days)
        return initial_veaero * remaining_ratio
        
    def calculate_boost_multiplier(
        self,
        user_veaero: Decimal,
        user_liquidity: Decimal,
        total_veaero: Decimal,
        total_liquidity: Decimal
    ) -> Decimal:
        """
        Calculate emission boost multiplier for LP position.
        
        Formula: min(1 + 1.5 * (veAERO_share / liquidity_share), 2.5)
        """
        if user_liquidity == 0 or total_liquidity == 0:
            return self.min_boost
            
        veaero_share = user_veaero / total_veaero if total_veaero > 0 else Decimal("0")
        liquidity_share = user_liquidity / total_liquidity
        
        if liquidity_share == 0:
            return self.min_boost
            
        boost = self.min_boost + self.boost_factor * (veaero_share / liquidity_share)
        return min(boost, self.max_boost)
        
    async def analyze_veaero_strategy(
        self,
        capital: Decimal,
        time_horizon_days: int
    ) -> Dict[str, Any]:
        """Analyze optimal veAERO strategy for given capital."""
        # Calculate different lock scenarios
        scenarios = []
        
        lock_periods = [30, 180, 365, 365*2, 365*4]  # 1m, 6m, 1y, 2y, 4y
        
        for lock_days in lock_periods:
            if lock_days > time_horizon_days:
                continue
                
            veaero = self.calculate_veaero_from_lock(capital, lock_days)
            
            # Estimate returns (simplified)
            weekly_rebase = self.calculate_current_emission() * self.veaero_share
            user_rebase_share = veaero / self.total_veaero
            weekly_rebase_tokens = weekly_rebase * user_rebase_share
            
            # Estimate bribe income (market dependent)
            estimated_bribe_apr = self._estimate_bribe_apr(lock_days)
            weekly_bribe_income = capital * estimated_bribe_apr / 100 / 52
            
            # Total returns
            total_weekly = weekly_rebase_tokens + weekly_bribe_income
            apr = (total_weekly * 52 / capital) * 100
            
            scenarios.append({
                "lock_period_days": lock_days,
                "lock_period_text": self._format_period(lock_days),
                "veaero_received": float(veaero),
                "voting_power_percent": float(user_rebase_share * 100),
                "weekly_rebase": float(weekly_rebase_tokens),
                "weekly_bribes": float(weekly_bribe_income),
                "total_apr": float(apr),
                "capital_locked": float(capital),
                "breakeven_weeks": float(capital / total_weekly) if total_weekly > 0 else float('inf')
            })
            
        # Sort by APR
        scenarios.sort(key=lambda x: x["total_apr"], reverse=True)
        
        return {
            "optimal_strategy": scenarios[0] if scenarios else None,
            "all_scenarios": scenarios,
            "recommendations": self._get_veaero_recommendations(scenarios, time_horizon_days)
        }
        
    async def calculate_pool_emissions(
        self,
        pool_votes: Decimal,
        total_votes: Decimal
    ) -> Dict[str, Decimal]:
        """Calculate emissions for a pool based on votes."""
        if total_votes == 0:
            return {
                "weekly_emissions": Decimal("0"),
                "daily_emissions": Decimal("0"),
                "hourly_emissions": Decimal("0")
            }
            
        vote_share = pool_votes / total_votes
        weekly_emission = self.calculate_current_emission()
        pool_emission = weekly_emission * self.lp_share * vote_share
        
        return {
            "weekly_emissions": pool_emission,
            "daily_emissions": pool_emission / 7,
            "hourly_emissions": pool_emission / 7 / 24,
            "vote_share": vote_share * 100,  # As percentage
            "emission_apr": self._calculate_emission_apr(pool_emission, Decimal("1000000"))  # Assumes $1M TVL
        }
        
    async def analyze_pool_emissions(
        self,
        pool_address: str,
        gauge_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze emission sustainability for a pool."""
        # This would fetch real data in production
        # Using example data
        pool_votes = Decimal("5000000")  # 5M veAERO votes
        pool_tvl = Decimal("10000000")  # $10M TVL
        
        emissions = await self.calculate_pool_emissions(pool_votes, self.total_veaero)
        
        # Project future emissions with decay
        future_emissions = []
        for week in range(1, 13):  # 12 weeks projection
            future_week = self.current_week + week
            future_emission = self.initial_weekly_emission * (self.emission_decay ** future_week)
            future_pool_emission = future_emission * self.lp_share * (pool_votes / self.total_veaero)
            
            future_emissions.append({
                "week": week,
                "emissions": float(future_pool_emission),
                "emission_apr": float(self._calculate_emission_apr(future_pool_emission, pool_tvl))
            })
            
        return {
            "current_emissions": {
                "weekly": float(emissions["weekly_emissions"]),
                "daily": float(emissions["daily_emissions"]),
                "emission_apr": float(emissions["emission_apr"])
            },
            "vote_metrics": {
                "pool_votes": float(pool_votes),
                "vote_share": float(emissions["vote_share"]),
                "votes_per_million_tvl": float(pool_votes / (pool_tvl / 1000000))
            },
            "sustainability": {
                "emission_decay_impact": "2% weekly reduction",
                "projected_apr_12w": future_emissions,
                "breakeven_vote_share": self._calculate_breakeven_votes(pool_tvl)
            }
        }
        
    def calculate_bribe_efficiency(
        self,
        bribe_amount: Decimal,
        expected_votes: Decimal,
        pool_tvl: Decimal
    ) -> Dict[str, Any]:
        """Calculate bribe efficiency metrics."""
        # Cost per vote
        cost_per_vote = bribe_amount / expected_votes if expected_votes > 0 else Decimal("inf")
        
        # Expected emissions
        vote_share = expected_votes / self.total_veaero
        weekly_emissions = self.calculate_current_emission() * self.lp_share * vote_share
        
        # ROI calculation
        emission_value = weekly_emissions * Decimal("1.0")  # Assume $1 per AERO
        bribe_roi = ((emission_value - bribe_amount) / bribe_amount * 100) if bribe_amount > 0 else Decimal("0")
        
        # Efficiency score (0-100)
        target_cost_per_vote = Decimal("0.001")  # $0.001 per veAERO vote
        efficiency_score = min(100, float((target_cost_per_vote / cost_per_vote) * 100)) if cost_per_vote > 0 else 0
        
        return {
            "bribe_amount": float(bribe_amount),
            "expected_votes": float(expected_votes),
            "cost_per_vote": float(cost_per_vote),
            "expected_emissions": float(weekly_emissions),
            "emission_value": float(emission_value),
            "bribe_roi": float(bribe_roi),
            "efficiency_score": efficiency_score,
            "recommendation": self._get_bribe_recommendation(efficiency_score, bribe_roi)
        }
        
    def _estimate_bribe_apr(self, lock_days: int) -> Decimal:
        """Estimate bribe APR based on lock period."""
        # Longer locks get better bribes
        if lock_days >= 365 * 4:
            return Decimal("25")  # 25% APR for 4-year lock
        elif lock_days >= 365 * 2:
            return Decimal("20")  # 20% APR for 2-year lock
        elif lock_days >= 365:
            return Decimal("15")  # 15% APR for 1-year lock
        elif lock_days >= 180:
            return Decimal("10")  # 10% APR for 6-month lock
        else:
            return Decimal("5")   # 5% APR for shorter locks
            
    def _calculate_emission_apr(self, weekly_emissions: Decimal, tvl: Decimal) -> Decimal:
        """Calculate APR from emissions."""
        if tvl == 0:
            return Decimal("0")
            
        yearly_emissions = weekly_emissions * 52
        yearly_value = yearly_emissions * Decimal("1.0")  # Assume $1 per AERO
        return (yearly_value / tvl) * 100
        
    def _calculate_breakeven_votes(self, pool_tvl: Decimal) -> Dict[str, float]:
        """Calculate votes needed for various APR targets."""
        targets = {}
        
        for target_apr in [10, 20, 30, 50]:
            # Reverse calculate needed emissions
            yearly_value_needed = pool_tvl * Decimal(target_apr) / 100
            weekly_emissions_needed = yearly_value_needed / 52
            
            # Calculate votes needed
            current_emission = self.calculate_current_emission()
            lp_emissions = current_emission * self.lp_share
            
            if lp_emissions > 0:
                vote_share_needed = weekly_emissions_needed / lp_emissions
                votes_needed = vote_share_needed * self.total_veaero
                targets[f"{target_apr}%_apr"] = float(votes_needed)
                
        return targets
        
    def _format_period(self, days: int) -> str:
        """Format lock period for display."""
        if days >= 365:
            years = days / 365
            if years == int(years):
                return f"{int(years)} year{'s' if years > 1 else ''}"
            else:
                return f"{years:.1f} years"
        elif days >= 30:
            months = days / 30
            return f"{int(months)} month{'s' if months > 1 else ''}"
        else:
            return f"{days} days"
            
    def _get_veaero_recommendations(self, scenarios: List[Dict], time_horizon: int) -> List[str]:
        """Get veAERO strategy recommendations."""
        recommendations = []
        
        if not scenarios:
            recommendations.append("No viable veAERO strategies for your time horizon")
            return recommendations
            
        optimal = scenarios[0]
        
        if optimal["lock_period_days"] >= 365 * 4:
            recommendations.append("4-year lock maximizes returns but requires long commitment")
            recommendations.append("Consider splitting capital: 50% 4-year, 50% 1-year for flexibility")
            
        if optimal["total_apr"] > 20:
            recommendations.append(f"Strong {optimal['total_apr']:.1f}% APR justifies locking")
            
        if time_horizon < 365:
            recommendations.append("Short time horizon limits veAERO effectiveness")
            recommendations.append("Consider providing liquidity instead of locking")
            
        if optimal["breakeven_weeks"] < 26:
            recommendations.append(f"Quick breakeven in {optimal['breakeven_weeks']:.0f} weeks")
            
        return recommendations
        
    def _get_bribe_recommendation(self, efficiency_score: float, roi: Decimal) -> str:
        """Get bribe recommendation based on metrics."""
        if efficiency_score > 80 and roi > 50:
            return "Excellent bribe opportunity - execute immediately"
        elif efficiency_score > 60 and roi > 20:
            return "Good bribe opportunity - consider executing"
        elif efficiency_score > 40:
            return "Moderate opportunity - compare with alternatives"
        else:
            return "Poor efficiency - look for better opportunities"