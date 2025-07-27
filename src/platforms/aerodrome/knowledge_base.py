"""
Aerodrome Knowledge Base System

Manages platform-specific knowledge including:
- Protocol mechanics and concepts
- Best practices and strategies
- Risk factors and mitigations
- Historical patterns and insights
"""
import json
import logging
import os
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class AerodromeKnowledgeBase:
    """
    Knowledge management system for Aerodrome platform.
    
    Loads and provides access to structured knowledge about:
    - How Aerodrome works
    - Profitable strategies
    - Risk management
    - Optimization techniques
    """
    
    def __init__(self):
        """Initialize knowledge base."""
        self.knowledge_path = Path(__file__).parent.parent / "knowledge" / "aerodrome"
        self.knowledge_cache = {}
        self.concepts = {}
        self.strategies = {}
        self.risk_factors = {}
        
    async def load_all_knowledge(self) -> Dict[str, Any]:
        """Load all knowledge documents."""
        logger.info("Loading Aerodrome knowledge base...")
        
        knowledge = {
            "concepts": await self._load_concepts(),
            "strategies": await self._load_strategies(),
            "mechanics": await self._load_mechanics(),
            "risk_factors": await self._load_risk_factors(),
            "patterns": await self._load_patterns()
        }
        
        self.knowledge_cache = knowledge
        return knowledge
        
    async def _load_concepts(self) -> Dict[str, str]:
        """Load platform concepts and explanations."""
        concepts = {
            "ve_tokenomics": """
                Vote-escrowed tokenomics where AERO tokens are locked for voting power.
                Longer locks (up to 4 years) provide more veAERO and greater influence.
                veAERO holders direct emissions and earn protocol fees.
            """,
            
            "gauge_system": """
                Gauges measure liquidity provision and distribute AERO rewards.
                Each pool can have a gauge that receives votes from veAERO holders.
                More votes = more AERO emissions directed to that pool.
            """,
            
            "bribes": """
                External incentives offered to veAERO voters to influence their votes.
                Protocols pay bribes to attract emissions to their liquidity pools.
                Voters claim bribes after voting, creating a vote marketplace.
            """,
            
            "boost_mechanism": """
                veAERO holders providing liquidity get up to 2.5x boost on emissions.
                Boost = min(1 + 1.5 * (veAERO_voting_power / pool_liquidity_share), 2.5)
                Larger veAERO position relative to liquidity = higher boost.
            """,
            
            "epoch_system": """
                Weekly cycles starting Thursday 00:00 UTC.
                Votes are cast during the epoch and tallied at the end.
                Emissions for the following week are set based on vote results.
            """,
            
            "stable_vs_volatile": """
                Stable pools: For correlated assets (stablecoins), 0.01% fee, x³y+y³x=k formula.
                Volatile pools: For uncorrelated assets, 0.3% fee, x*y=k formula.
                Choose based on asset correlation and expected price movements.
            """,
            
            "concentrated_liquidity": """
                V3 feature allowing LPs to concentrate capital in specific price ranges.
                Higher capital efficiency but requires active management.
                Best for stable pairs or range-bound volatile pairs.
            """,
            
            "routing": """
                Aerodrome router finds optimal paths through multiple pools.
                Can split trades across routes for better execution.
                Always check for multi-hop opportunities.
            """,
            
            "impermanent_loss": """
                Temporary loss from providing liquidity vs just holding tokens.
                Minimal in stable pools, significant in volatile pools.
                Can be offset by fees + emissions if APR > IL.
            """,
            
            "rebase_mechanism": """
                Weekly rebase distributes new AERO emissions.
                veAERO holders receive rebase proportional to their share.
                Compounds voting power over time for long-term holders.
            """
        }
        
        self.concepts = concepts
        return concepts
        
    async def _load_strategies(self) -> Dict[str, Any]:
        """Load proven strategies."""
        strategies = {
            "emission_farming": {
                "description": "Provide liquidity to high-emission pools",
                "steps": [
                    "Identify pools with high gauge votes but low TVL",
                    "Add liquidity to capture outsized emission share",
                    "Compound rewards weekly for maximum returns",
                    "Monitor vote changes and rebalance as needed"
                ],
                "expected_apr_range": [30, 100],
                "risk_level": "medium",
                "minimum_capital": 1000,
                "optimal_conditions": {
                    "pool_tvl": "< $5M",
                    "emission_apr": "> 25%",
                    "vote_stability": "> 70%"
                }
            },
            
            "bribe_arbitrage": {
                "description": "Acquire veAERO to farm bribes efficiently",
                "steps": [
                    "Calculate bribe ROI: (annual_bribes / veAERO_cost)",
                    "Lock AERO when ROI > 20% annually",
                    "Vote for highest bribe/vote ratio pools",
                    "Claim and compound bribes weekly"
                ],
                "expected_apr_range": [15, 50],
                "risk_level": "low",
                "minimum_capital": 5000,
                "considerations": [
                    "4-year lock for maximum efficiency",
                    "Track bribe consistency over time",
                    "Diversify votes across 3-5 pools"
                ]
            },
            
            "stable_pool_farming": {
                "description": "Low-risk yield from stablecoin pools",
                "steps": [
                    "Focus on USDC/USDT, USDC/DAI pools",
                    "Monitor peg stability",
                    "Compound fees and emissions daily",
                    "Use leverage carefully (max 2x)"
                ],
                "expected_apr_range": [8, 25],
                "risk_level": "low",
                "minimum_capital": 500,
                "advantages": [
                    "Minimal impermanent loss",
                    "Predictable returns",
                    "Good for large positions"
                ]
            },
            
            "new_pool_alpha": {
                "description": "Early liquidity in promising new pools",
                "steps": [
                    "Monitor pool factory for new deployments",
                    "Research project fundamentals",
                    "Provide liquidity in first 24-48 hours",
                    "Exit when TVL > $10M or APR < 30%"
                ],
                "expected_apr_range": [50, 500],
                "risk_level": "high",
                "minimum_capital": 500,
                "red_flags": [
                    "Anonymous teams",
                    "No audits",
                    "Suspicious tokenomics",
                    "Artificial volume"
                ]
            },
            
            "vote_following": {
                "description": "Follow smart money veAERO votes",
                "steps": [
                    "Identify top veAERO holders (>100k veAERO)",
                    "Analyze their voting patterns",
                    "Provide liquidity to their targeted pools",
                    "Front-run emission changes"
                ],
                "expected_apr_range": [25, 60],
                "risk_level": "medium",
                "minimum_capital": 2000,
                "tools_needed": [
                    "On-chain vote tracking",
                    "Wallet analysis",
                    "Alert system for vote changes"
                ]
            },
            
            "correlated_asset_arb": {
                "description": "Arbitrage between correlated volatile assets",
                "steps": [
                    "Identify correlated pairs (ETH/stETH, USDC/USDbC)",
                    "Monitor price deviations > 0.5%",
                    "Execute swaps through router",
                    "Provide liquidity during stable periods"
                ],
                "expected_apr_range": [20, 80],
                "risk_level": "medium",
                "minimum_capital": 3000,
                "requirements": [
                    "Fast execution capability",
                    "Price feed monitoring",
                    "Gas optimization"
                ]
            }
        }
        
        self.strategies = strategies
        return strategies
        
    async def _load_mechanics(self) -> Dict[str, Any]:
        """Load detailed platform mechanics."""
        mechanics = {
            "emission_schedule": {
                "initial_supply": 500000000,  # 500M AERO
                "weekly_emissions": "2% decay from previous week",
                "distribution": {
                    "liquidity_providers": "67%",
                    "veAERO_holders": "33%"
                },
                "decay_formula": "emissions_week_n = emissions_week_(n-1) * 0.98"
            },
            
            "voting_mechanics": {
                "epoch_length": "1 week",
                "epoch_start": "Thursday 00:00 UTC",
                "vote_deadline": "Wednesday 23:59 UTC",
                "minimum_veAERO": "1 veAERO",
                "vote_allocation": "Must sum to 100%",
                "vote_reset": "Votes don't carry over between epochs"
            },
            
            "fee_structure": {
                "stable_pools": {
                    "base_fee": "0.01%",
                    "protocol_take": "0%",
                    "lp_share": "100%"
                },
                "volatile_pools": {
                    "base_fee": "0.3%",
                    "protocol_take": "0%",
                    "lp_share": "100%"
                },
                "concentrated_liquidity": {
                    "fee_tiers": ["0.01%", "0.05%", "0.3%", "1%"],
                    "dynamic_fees": "Based on volatility"
                }
            },
            
            "boost_calculation": {
                "formula": "boost = min(1 + 1.5 * (voting_power / liquidity_share), 2.5)",
                "max_boost": 2.5,
                "min_boost": 1.0,
                "variables": {
                    "voting_power": "User's veAERO balance",
                    "liquidity_share": "User's LP tokens / Total LP tokens"
                }
            },
            
            "bribe_system": {
                "claim_period": "After epoch ends",
                "distribution": "Pro-rata based on votes",
                "supported_tokens": "Any ERC20",
                "minimum_bribe": "$100 equivalent",
                "platform_fee": "0%"
            }
        }
        
        return mechanics
        
    async def _load_risk_factors(self) -> Dict[str, Any]:
        """Load risk factors and mitigation strategies."""
        risk_factors = {
            "impermanent_loss": {
                "description": "Loss from price divergence in volatile pools",
                "severity": "high",
                "affected_pools": "volatile",
                "mitigation": [
                    "Focus on correlated assets",
                    "Use concentrated liquidity near current price",
                    "Monitor and rebalance regularly",
                    "Ensure APR > expected IL"
                ]
            },
            
            "emission_dilution": {
                "description": "AERO token inflation reducing value",
                "severity": "medium",
                "timeline": "long-term",
                "mitigation": [
                    "Focus on fee-generating pools",
                    "Compound emissions immediately",
                    "Convert to stable assets periodically"
                ]
            },
            
            "vote_migration": {
                "description": "Votes moving away from your pool",
                "severity": "medium",
                "indicators": [
                    "Declining vote share",
                    "New competitive pools",
                    "Bribe reduction"
                ],
                "mitigation": [
                    "Diversify across multiple pools",
                    "Monitor vote trends",
                    "Be ready to migrate quickly"
                ]
            },
            
            "smart_contract_risk": {
                "description": "Potential bugs or exploits",
                "severity": "low",
                "audit_status": "Audited by Zellic",
                "mitigation": [
                    "Use established pools only",
                    "Avoid pools < 1 week old",
                    "Consider insurance (Nexus Mutual)",
                    "Don't concentrate entire portfolio"
                ]
            },
            
            "liquidity_risk": {
                "description": "Unable to exit position efficiently",
                "severity": "medium",
                "affected_pools": "Low TVL pools",
                "mitigation": [
                    "Prefer pools with TVL > $1M",
                    "Check daily volume",
                    "Plan exit strategy in advance",
                    "Use limit orders when possible"
                ]
            },
            
            "gas_spike_risk": {
                "description": "High gas making operations unprofitable",
                "severity": "low",
                "chain": "Base (typically low gas)",
                "mitigation": [
                    "Batch operations",
                    "Compound during low gas periods",
                    "Maintain minimum position sizes"
                ]
            }
        }
        
        self.risk_factors = risk_factors
        return risk_factors
        
    async def _load_patterns(self) -> Dict[str, Any]:
        """Load observed market patterns."""
        patterns = {
            "weekly_cycles": {
                "monday_tuesday": "Lower activity, good for entries",
                "wednesday": "Bribe activity peaks",
                "thursday": "New epoch, emission reallocation",
                "friday_weekend": "Reduced volume, wider spreads"
            },
            
            "tvl_apr_correlation": {
                "pattern": "Inverse relationship",
                "formula": "APR ≈ k / TVL^0.5",
                "sweet_spot": "$1M - $5M TVL",
                "exceptions": "New tokens with high bribes"
            },
            
            "bribe_efficiency": {
                "optimal_pools": "3-5 for diversification",
                "bribe_roi": "Typically 15-40% APR",
                "best_timing": "Vote 2-4 hours before epoch end",
                "avoid": "Pools with irregular bribes"
            },
            
            "new_pool_lifecycle": {
                "launch": "Very high APR (>100%)",
                "growth": "TVL increases, APR decreases",
                "maturity": "Stable APR around market rate",
                "decline": "Votes migrate, APR drops",
                "typical_timeline": "2-4 weeks to maturity"
            },
            
            "whale_behavior": {
                "entry_signals": "Large veAERO votes shifting",
                "exit_signals": "Gradual liquidity removal",
                "follow_threshold": ">100k veAERO holders",
                "lag_time": "Usually 1-2 epochs behind"
            }
        }
        
        return patterns
        
    async def query_strategy(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Query specific strategy details."""
        if not self.strategies:
            await self._load_strategies()
            
        return self.strategies.get(strategy_name)
        
    async def get_risk_assessment(self, pool_type: str) -> List[Dict[str, Any]]:
        """Get risk assessment for pool type."""
        if not self.risk_factors:
            await self._load_risk_factors()
            
        relevant_risks = []
        for risk_name, risk_data in self.risk_factors.items():
            if pool_type == "volatile" or risk_data.get("affected_pools") != "volatile":
                relevant_risks.append({
                    "risk": risk_name,
                    **risk_data
                })
                
        return relevant_risks
        
    async def search_knowledge(self, query: str) -> Dict[str, Any]:
        """Search knowledge base with natural language query."""
        if not self.knowledge_cache:
            await self.load_all_knowledge()
            
        results = {}
        query_lower = query.lower()
        
        # Search through all knowledge categories
        for category, content in self.knowledge_cache.items():
            if isinstance(content, dict):
                for key, value in content.items():
                    # Convert value to string for searching
                    value_str = json.dumps(value) if not isinstance(value, str) else value
                    
                    if query_lower in key.lower() or query_lower in value_str.lower():
                        if category not in results:
                            results[category] = {}
                        results[category][key] = value
                        
        return results
        
    async def get_concept_explanation(self, concept: str) -> str:
        """Get explanation for a specific concept."""
        if not self.concepts:
            await self._load_concepts()
            
        # Try exact match first
        if concept in self.concepts:
            return self.concepts[concept]
            
        # Try fuzzy match
        concept_lower = concept.lower()
        for key, explanation in self.concepts.items():
            if concept_lower in key.lower():
                return explanation
                
        return f"No explanation found for concept: {concept}"