"""
Gauge Reader for Aerodrome Protocol

Reads emission rates and reward data from gauge contracts to calculate
real APR values instead of using hardcoded estimates.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta

from src.blockchain.rpc_reader import RPCReader
from config.contracts import CONTRACTS, TOKENS
from config.settings import settings

logger = logging.getLogger(__name__)


class GaugeReader:
    """Reads data from Aerodrome gauge contracts."""
    
    def __init__(self, rpc_reader: Optional[RPCReader] = None):
        """Initialize gauge reader."""
        self.rpc_reader = rpc_reader
        self.voter_address = CONTRACTS["voter"]["address"]
        
        # Cache for gauge addresses
        self._gauge_cache = {}
        self._cache_timestamp = None
        self._cache_ttl = 3600  # 1 hour cache
        
        # AERO token address
        self.aero_address = TOKENS["AERO"]
        
    async def __aenter__(self):
        """Enter async context."""
        if not self.rpc_reader:
            self.rpc_reader = RPCReader(settings.cdp_rpc_url)
            await self.rpc_reader.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        if self.rpc_reader and hasattr(self.rpc_reader, '__aexit__'):
            await self.rpc_reader.__aexit__(exc_type, exc_val, exc_tb)
            
    async def get_gauge_address(self, pool_address: str) -> Optional[str]:
        """Get gauge address for a pool from the voter contract."""
        try:
            # Check cache first
            if pool_address in self._gauge_cache:
                if self._cache_timestamp and (datetime.utcnow() - self._cache_timestamp).seconds < self._cache_ttl:
                    return self._gauge_cache[pool_address]
                    
            # Query voter contract for gauge address
            result = await self.rpc_reader.call_contract_function(
                self.voter_address,
                "gauges(address)",
                [pool_address]
            )
            
            if not result or result == "0x0":
                logger.debug(f"No gauge found for pool {pool_address}")
                return None
                
            # Extract address from result
            gauge_address = "0x" + result[-40:]
            
            # Check if it's zero address
            if gauge_address == "0x0000000000000000000000000000000000000000":
                return None
                
            # Cache the result
            self._gauge_cache[pool_address] = gauge_address
            if not self._cache_timestamp:
                self._cache_timestamp = datetime.utcnow()
                
            logger.info(f"Found gauge {gauge_address} for pool {pool_address}")
            return gauge_address
            
        except Exception as e:
            logger.error(f"Failed to get gauge address for pool {pool_address}: {e}")
            return None
            
    async def get_reward_rate(self, gauge_address: str, token_address: Optional[str] = None) -> Decimal:
        """Get reward rate from gauge for a specific token (default: AERO)."""
        try:
            if not token_address:
                token_address = self.aero_address
                
            # Call rewardRate(address) on gauge
            result = await self.rpc_reader.call_contract_function(
                gauge_address,
                "rewardRate(address)",
                [token_address]
            )
            
            if not result or result == "0x0":
                return Decimal("0")
                
            # Convert hex to decimal (rewards per second)
            reward_rate = Decimal(int(result, 16)) / Decimal(10**18)
            
            logger.debug(f"Gauge {gauge_address} reward rate: {reward_rate} tokens/sec")
            return reward_rate
            
        except Exception as e:
            logger.error(f"Failed to get reward rate from gauge {gauge_address}: {e}")
            return Decimal("0")
            
    async def get_gauge_total_supply(self, gauge_address: str) -> Decimal:
        """Get total LP tokens staked in gauge."""
        try:
            result = await self.rpc_reader.call_contract_function(
                gauge_address,
                "totalSupply()"
            )
            
            if not result or result == "0x0":
                return Decimal("0")
                
            # Convert hex to decimal
            total_supply = Decimal(int(result, 16)) / Decimal(10**18)
            
            logger.debug(f"Gauge {gauge_address} total supply: {total_supply}")
            return total_supply
            
        except Exception as e:
            logger.error(f"Failed to get gauge total supply: {e}")
            return Decimal("0")
            
    async def calculate_emission_apr(
        self,
        pool_address: str,
        pool_tvl: Decimal,
        aero_price: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate APR from AERO emissions for a pool.
        
        Args:
            pool_address: The pool contract address
            pool_tvl: Total value locked in the pool (in USD)
            aero_price: Price of AERO token (default: estimate)
            
        Returns:
            Emission APR as a percentage
        """
        try:
            # Get gauge address
            gauge_address = await self.get_gauge_address(pool_address)
            if not gauge_address:
                logger.debug(f"No gauge found for pool {pool_address}, emission APR = 0")
                return Decimal("0")
                
            # Get AERO reward rate (tokens per second)
            reward_rate = await self.get_reward_rate(gauge_address)
            if reward_rate == 0:
                return Decimal("0")
                
            # Get total LP tokens staked
            gauge_supply = await self.get_gauge_total_supply(gauge_address)
            
            # If no tokens staked, use pool's total supply
            if gauge_supply == 0:
                # Would need to get pool's total supply here
                # For now, assume all LP tokens are staked
                gauge_supply = pool_tvl / Decimal("1")  # Simplified
                
            # Estimate AERO price if not provided
            if not aero_price:
                # In production, fetch from price oracle
                # For now, use a reasonable estimate
                aero_price = Decimal("1.8")  # Approximate AERO price
                
            # Calculate annual rewards value
            seconds_per_year = Decimal("31536000")  # 365 days
            annual_rewards = reward_rate * seconds_per_year
            annual_rewards_usd = annual_rewards * aero_price
            
            # Calculate APR
            if pool_tvl > 0:
                apr = (annual_rewards_usd / pool_tvl) * Decimal("100")
                
                logger.info(
                    f"Pool {pool_address}: "
                    f"reward_rate={reward_rate:.6f} AERO/sec, "
                    f"annual_rewards={annual_rewards:.2f} AERO, "
                    f"TVL=${pool_tvl:,.2f}, "
                    f"emission_apr={apr:.2f}%"
                )
                
                return apr
            else:
                return Decimal("0")
                
        except Exception as e:
            logger.error(f"Failed to calculate emission APR for pool {pool_address}: {e}")
            return Decimal("0")
            
    async def get_all_gauge_addresses(self, pool_addresses: List[str]) -> Dict[str, str]:
        """Get gauge addresses for multiple pools efficiently."""
        gauge_map = {}
        
        for pool_address in pool_addresses:
            gauge_address = await self.get_gauge_address(pool_address)
            if gauge_address:
                gauge_map[pool_address] = gauge_address
                
        return gauge_map
        
    async def get_gauge_metrics(self, gauge_address: str) -> Dict:
        """Get comprehensive metrics for a gauge."""
        try:
            # Get reward rate
            reward_rate = await self.get_reward_rate(gauge_address)
            
            # Get total supply
            total_supply = await self.get_gauge_total_supply(gauge_address)
            
            # Could also get:
            # - Earned rewards for specific accounts
            # - Multiple reward tokens if gauge supports it
            # - Voting weight allocated to this gauge
            
            return {
                "gauge_address": gauge_address,
                "reward_rate": float(reward_rate),
                "total_supply": float(total_supply),
                "aero_per_day": float(reward_rate * Decimal("86400")),
                "has_emissions": reward_rate > 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get gauge metrics: {e}")
            return {}
            
    async def estimate_pool_apr(
        self,
        pool_address: str,
        pool_data: Dict,
        aero_price: Optional[Decimal] = None
    ) -> Dict[str, Decimal]:
        """
        Estimate complete APR for a pool (fees + emissions).
        
        Args:
            pool_address: Pool contract address
            pool_data: Pool data including TVL, volume, stable flag
            aero_price: AERO token price
            
        Returns:
            Dict with fee_apr, emission_apr, and total_apr
        """
        try:
            tvl = Decimal(str(pool_data.get("tvl", 0)))
            volume_24h = Decimal(str(pool_data.get("volume_24h", 0)))
            stable = pool_data.get("stable", False)
            
            # Calculate fee APR from volume
            if tvl > 0:
                # Fee rates: 0.01% for stable, 0.3% for volatile
                fee_rate = Decimal("0.0001") if stable else Decimal("0.003")
                daily_fees = volume_24h * fee_rate
                annual_fees = daily_fees * Decimal("365")
                fee_apr = (annual_fees / tvl) * Decimal("100")
            else:
                fee_apr = Decimal("0")
                
            # Calculate emission APR
            emission_apr = await self.calculate_emission_apr(
                pool_address,
                tvl,
                aero_price
            )
            
            # Total APR
            total_apr = fee_apr + emission_apr
            
            logger.info(
                f"Pool {pool_address} APR breakdown: "
                f"fee={fee_apr:.2f}%, emission={emission_apr:.2f}%, "
                f"total={total_apr:.2f}%"
            )
            
            return {
                "fee_apr": fee_apr,
                "emission_apr": emission_apr,
                "total_apr": total_apr
            }
            
        except Exception as e:
            logger.error(f"Failed to estimate pool APR: {e}")
            return {
                "fee_apr": Decimal("0"),
                "emission_apr": Decimal("0"),
                "total_apr": Decimal("0")
            }