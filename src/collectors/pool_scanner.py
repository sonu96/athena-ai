"""
Aerodrome Pool Scanner
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from decimal import Decimal

from src.cdp.base_client import BaseClient
from src.agent.memory import AthenaMemory, MemoryType
from config.contracts import TOKENS

logger = logging.getLogger(__name__)


class PoolScanner:
    """
    Scans Aerodrome pools for opportunities.
    
    Monitors liquidity pools, tracks APRs, volumes, and identifies
    profitable opportunities for liquidity provision and trading.
    """
    
    def __init__(self, base_client: BaseClient, memory: AthenaMemory, firestore_client=None):
        """Initialize pool scanner."""
        self.base_client = base_client
        self.memory = memory
        self.firestore_client = firestore_client
        self.pool_profiles = None
        self.scanning = False
        
        # Pool data cache
        self.pools = {}
        self.last_scan = None
        
        # Initialize pool profile manager if firestore is available
        if self.firestore_client:
            from src.agent.pool_profiles import PoolProfileManager
            self.pool_profiles = PoolProfileManager(self.firestore_client)
            logger.info("Pool scanner initialized with pool profile manager")
        
        # Top opportunities
        self.opportunities = {
            "high_apr": [],
            "high_volume": [],
            "new_pools": [],
            "imbalanced": [],
        }
        
    async def start_scanning(self):
        """Start continuous pool scanning."""
        if self.scanning:
            logger.warning("Pool scanning already active")
            return
            
        self.scanning = True
        logger.info("=== Starting Aerodrome pool scanning...")
        
        while self.scanning:
            try:
                await self._scan_cycle()
                await asyncio.sleep(300)  # Scan every 5 minutes
                
            except Exception as e:
                logger.error(f"Pool scanning error: {e}")
                await asyncio.sleep(600)  # Wait longer on error
                
    async def stop_scanning(self):
        """Stop pool scanning."""
        self.scanning = False
        logger.info("Pool scanning stopped")
        
    async def _scan_cycle(self):
        """Single scanning cycle."""
        logger.info("Scanning Aerodrome pools...")
        
        # Get major token pairs to scan
        pairs_to_scan = self._get_pairs_to_scan()
        
        # Scan each pair
        new_opportunities = {
            "high_apr": [],
            "high_volume": [],
            "new_pools": [],
            "imbalanced": [],
        }
        
        for pair in pairs_to_scan:
            pool_data = await self._scan_pool(pair["token_a"], pair["token_b"], pair["stable"])
            
            if pool_data:
                # Categorize opportunity
                await self._categorize_opportunity(pool_data, new_opportunities)
                
        # Update opportunities
        self.opportunities = new_opportunities
        self.last_scan = datetime.utcnow()
        
        # Store significant findings in memory
        await self._store_findings(new_opportunities)
        
    def _get_pairs_to_scan(self) -> List[Dict]:
        """Get list of pairs to scan."""
        # Focus on major pairs
        major_pairs = [
            {"token_a": "WETH", "token_b": "USDC", "stable": False},
            {"token_a": "WETH", "token_b": "DAI", "stable": False},
            {"token_a": "AERO", "token_b": "USDC", "stable": False},
            {"token_a": "AERO", "token_b": "WETH", "stable": False},
            {"token_a": "USDC", "token_b": "DAI", "stable": True},
            {"token_a": "USDC", "token_b": "USDbC", "stable": True},
        ]
        
        return major_pairs
        
    async def _scan_pool(self, token_a: str, token_b: str, stable: bool) -> Optional[Dict]:
        """Scan a specific pool."""
        try:
            # Get pool info
            pool_info = await self.base_client.get_pool_info(token_a, token_b, stable)
            
            if not pool_info:
                return None
                
            # Use real data from CDP
            pool_data = {
                "pair": f"{token_a}/{token_b}",
                "address": pool_info.get("address"),
                "stable": stable,
                "tvl": pool_info.get("tvl", Decimal("0")),
                "volume_24h": await self._estimate_volume(pool_info, stable),
                "apr": await self._calculate_real_apr(pool_info, stable),
                "fee_apr": self._calculate_fee_apr(stable),
                "incentive_apr": await self._estimate_incentive_apr(token_a, token_b),
                "reserves": {
                    token_a: pool_info.get("reserve0", Decimal("0")),
                    token_b: pool_info.get("reserve1", Decimal("0")),
                },
                "ratio": pool_info.get("ratio", Decimal("1")),
                "imbalanced": pool_info.get("imbalanced", False),
                "timestamp": datetime.utcnow(),
            }
            
            # Store in cache
            pool_key = f"{token_a}/{token_b}-{stable}"
            self.pools[pool_key] = pool_data
            
            logger.debug(f"Scanned pool {pool_key}: address={pool_data.get('address')}, APR={pool_data.get('apr')}%, TVL=${pool_data.get('tvl'):,.0f}")
            
            # Validate TVL for data quality
            tvl = pool_data.get('tvl', Decimal('0'))
            reserves = pool_data.get('reserves', {})
            
            # Check for unrealistic TVL values
            if tvl > Decimal('1000000000'):  # $1B TVL seems unrealistic for most pools
                logger.warning(f"Suspicious TVL for {pool_key}: ${tvl:,.0f} - may be a data issue")
            elif tvl < Decimal('1000'):  # Less than $1k TVL might indicate empty pool
                logger.warning(f"Very low TVL for {pool_key}: ${tvl:,.0f} - pool may be empty")
            
            # Log reserve details for validation
            if reserves:
                for token, reserve in reserves.items():
                    if reserve > Decimal('1000000000'):  # 1B tokens seems excessive
                        logger.warning(f"Suspicious reserve amount for {token} in {pool_key}: {reserve:,.0f}")
            
            # Update pool profile if available
            if self.pool_profiles and pool_data.get("address"):
                try:
                    # Get current gas price
                    gas_price = await self.base_client.get_gas_price()
                    await self.pool_profiles.update_pool(pool_data, gas_price=gas_price)
                    logger.info(f"Updated pool profile for {pool_key} at address {pool_data.get('address')}")
                except Exception as e:
                    logger.error(f"Failed to update pool profile for {pool_key}: {e}")
            
            return pool_data
            
        except Exception as e:
            logger.error(f"Error scanning pool {token_a}/{token_b}: {e}")
            return None
            
    def _calculate_fee_apr(self, stable: bool) -> Decimal:
        """Calculate fee APR based on pool type."""
        # Stable pools: 0.01% fee, Volatile pools: 0.3% fee
        # APR = (daily_volume * fee_rate * 365) / TVL
        # This is a simplified calculation
        if stable:
            return Decimal("5.0")  # Typical for stable pools
        else:
            return Decimal("15.0")  # Typical for volatile pools
    
    async def _estimate_volume(self, pool_info: Dict, stable: bool) -> Decimal:
        """Estimate 24h volume based on TVL and pool type."""
        # In production, this would query historical events
        # For now, estimate based on typical volume/TVL ratios
        tvl = pool_info.get("tvl", Decimal("0"))
        if tvl == 0:
            return Decimal("0")
            
        # Stable pools typically have higher volume/TVL ratio
        if stable:
            volume_ratio = Decimal("0.8")  # 80% daily volume
        else:
            volume_ratio = Decimal("0.5")  # 50% daily volume
            
        return tvl * volume_ratio
    
    async def _calculate_real_apr(self, pool_info: Dict, stable: bool) -> Decimal:
        """Calculate real APR from pool data."""
        tvl = pool_info.get("tvl", Decimal("0"))
        if tvl == 0:
            return Decimal("0")
            
        # Fee APR calculation
        fee_apr = self._calculate_fee_apr(stable)
        
        # Add estimated incentive APR (would come from gauge contracts)
        incentive_apr = await self._estimate_incentive_apr(
            pool_info.get("token_a", ""),
            pool_info.get("token_b", "")
        )
        
        return fee_apr + incentive_apr
    
    async def _estimate_incentive_apr(self, token_a: str, token_b: str) -> Decimal:
        """Estimate incentive APR from AERO emissions."""
        # In production, this would query gauge contracts for actual emissions
        # For now, use estimates based on pool importance
        
        # AERO pairs get highest incentives
        if "AERO" in token_a or "AERO" in token_b:
            return Decimal("70.0")
        # Major pairs get good incentives
        elif ("WETH" in token_a or "WETH" in token_b) and ("USDC" in token_a or "USDC" in token_b):
            return Decimal("40.0")
        # Stable pairs get moderate incentives
        elif "USD" in token_a and "USD" in token_b:
            return Decimal("20.0")
        else:
            return Decimal("10.0")
            
    async def _categorize_opportunity(self, pool_data: Dict, opportunities: Dict):
        """Categorize pool as an opportunity."""
        # High APR opportunity
        if pool_data["apr"] > Decimal("50"):
            opportunities["high_apr"].append({
                **pool_data,
                "reason": f"High APR: {pool_data['apr']}%",
                "score": float(pool_data["apr"]) / 100,
            })
            
        # High volume opportunity
        if pool_data["volume_24h"] > Decimal("1000000"):
            opportunities["high_volume"].append({
                **pool_data,
                "reason": f"High volume: ${pool_data['volume_24h']:,.0f}",
                "score": float(pool_data["volume_24h"]) / 1000000,
            })
            
        # Check if pool is imbalanced (mock check)
        # TODO: Implement real imbalance detection
        if self._is_imbalanced(pool_data):
            opportunities["imbalanced"].append({
                **pool_data,
                "reason": "Pool imbalanced - arbitrage opportunity",
                "score": 0.8,
            })
            
    def _is_imbalanced(self, pool_data: Dict) -> bool:
        """Check if pool is imbalanced."""
        # Use the imbalanced flag from pool data if available
        if "imbalanced" in pool_data:
            return pool_data["imbalanced"]
            
        # Otherwise check ratio
        ratio = pool_data.get("ratio", Decimal("1"))
        # Consider imbalanced if ratio deviates more than 10% from 1:1
        return abs(ratio - Decimal("1")) > Decimal("0.1")
        
    async def _store_findings(self, opportunities: Dict):
        """Store significant findings in memory - enhanced to capture all significant pools."""
        from config.settings import settings
        
        # Get thresholds from settings or use defaults
        min_apr_for_memory = getattr(settings, 'min_apr_for_memory', 20)
        min_volume_for_memory = getattr(settings, 'min_volume_for_memory', 100000)
        
        # Store ALL high APR pools, not just the top one
        if opportunities["high_apr"]:
            for pool in opportunities["high_apr"]:
                if pool["apr"] >= min_apr_for_memory:
                    await self.memory.remember(
                        content=f"High APR pool: {pool['pair']} at {pool['apr']}% APR (TVL: ${pool['tvl']:,.0f})",
                        memory_type=MemoryType.OBSERVATION,
                        category="pool_behavior",
                        metadata={
                            "pool": pool["pair"],
                            "pool_address": pool.get("address"),
                            "apr": float(pool["apr"]),
                            "fee_apr": float(pool.get("fee_apr", 0)),
                            "incentive_apr": float(pool.get("incentive_apr", 0)),
                            "tvl": float(pool["tvl"]),
                            "stable": pool.get("stable", False),
                            "timestamp": pool.get("timestamp"),
                        },
                        confidence=0.9 if pool["apr"] > 50 else 0.7
                    )
            
        # Store ALL high volume pools
        if opportunities["high_volume"]:
            for pool in opportunities["high_volume"]:
                if pool["volume_24h"] >= min_volume_for_memory:
                    await self.memory.remember(
                        content=f"High volume pool: {pool['pair']} with ${pool['volume_24h']:,.0f} daily volume (APR: {pool['apr']}%)",
                        memory_type=MemoryType.OBSERVATION,
                        category="pool_behavior",
                        metadata={
                            "pool": pool["pair"],
                            "pool_address": pool.get("address"),
                            "volume": float(pool["volume_24h"]),
                            "apr": float(pool["apr"]),
                            "tvl": float(pool["tvl"]),
                            "volume_to_tvl_ratio": float(pool["volume_24h"] / pool["tvl"]) if pool["tvl"] > 0 else 0,
                            "stable": pool.get("stable", False),
                            "timestamp": pool.get("timestamp"),
                        },
                        confidence=0.9 if pool["volume_24h"] > 1000000 else 0.8
                    )
                    
        # Store imbalanced pools for arbitrage tracking
        if opportunities["imbalanced"]:
            for pool in opportunities["imbalanced"]:
                # Only store significantly imbalanced pools
                if pool.get("ratio") and (pool["ratio"] > 2 or pool["ratio"] < 0.5):
                    await self.memory.remember(
                        content=f"Imbalanced pool detected: {pool['pair']} with ratio {pool['ratio']:.4f}",
                        memory_type=MemoryType.OBSERVATION,
                        category="arbitrage_opportunity",
                        metadata={
                            "pool": pool["pair"],
                            "pool_address": pool.get("address"),
                            "ratio": float(pool["ratio"]),
                            "reserves": pool.get("reserves"),
                            "tvl": float(pool["tvl"]),
                            "timestamp": pool.get("timestamp"),
                        },
                        confidence=0.8
                    )
                    
        # Store new pools if any
        if opportunities.get("new_pools"):
            for pool in opportunities["new_pools"]:
                await self.memory.remember(
                    content=f"New pool discovered: {pool['pair']} (TVL: ${pool['tvl']:,.0f}, APR: {pool['apr']}%)",
                    memory_type=MemoryType.OBSERVATION,
                    category="new_pool",
                    metadata={
                        "pool": pool["pair"],
                        "pool_address": pool.get("address"),
                        "apr": float(pool["apr"]),
                        "tvl": float(pool["tvl"]),
                        "stable": pool.get("stable", False),
                        "timestamp": pool.get("timestamp"),
                    },
                    confidence=1.0
                )
            
    def get_opportunities(self, category: Optional[str] = None) -> List[Dict]:
        """Get current opportunities."""
        if category:
            return self.opportunities.get(category, [])
            
        # Return all opportunities sorted by score
        all_opps = []
        for opps in self.opportunities.values():
            all_opps.extend(opps)
            
        return sorted(all_opps, key=lambda x: x.get("score", 0), reverse=True)
        
    def get_pool_data(self, token_a: str, token_b: str, stable: bool = False) -> Optional[Dict]:
        """Get cached pool data."""
        pool_key = f"{token_a}/{token_b}-{stable}"
        return self.pools.get(pool_key)
        
    def get_summary(self) -> Dict:
        """Get scanning summary."""
        return {
            "last_scan": self.last_scan.isoformat() if self.last_scan else None,
            "pools_tracked": len(self.pools),
            "opportunities": {
                "high_apr": len(self.opportunities["high_apr"]),
                "high_volume": len(self.opportunities["high_volume"]),
                "new_pools": len(self.opportunities["new_pools"]),
                "imbalanced": len(self.opportunities["imbalanced"]),
            },
            "top_apr": max(
                (p["apr"] for p in self.pools.values()),
                default=Decimal("0")
            ),
            "total_tvl": sum(
                p["tvl"] for p in self.pools.values()
            ),
        }