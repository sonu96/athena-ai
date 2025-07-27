"""
Aerodrome Pool Scanner using QuickNode API

Uses the QuickNode Aerodrome API add-on for direct access to pool data
without needing contract calls.
"""
import logging
import aiohttp
from typing import List, Dict, Optional
from datetime import datetime
from decimal import Decimal

from src.agent.memory import AthenaMemory, MemoryType
from config.settings import settings

logger = logging.getLogger(__name__)


class AerodromeAPIScanner:
    """Scanner that uses QuickNode's Aerodrome API for pool data."""
    
    def __init__(self, memory: AthenaMemory):
        """Initialize the API scanner.
        
        Args:
            memory: Memory system for storing observations
        """
        self.memory = memory
        self.base_url = settings.quicknode_endpoint.rstrip('/')
        self.api_path = "/addon/1051/v1"  # Aerodrome API add-on path
        
    async def fetch_pools(self, limit: int = 50, min_apr: float = 20.0) -> List[Dict]:
        """Fetch pool data from Aerodrome API.
        
        Args:
            limit: Maximum number of pools to fetch
            min_apr: Minimum APR threshold
            
        Returns:
            List of pool data dictionaries
        """
        try:
            url = f"{self.base_url}{self.api_path}/pools/detailed"
            params = {
                "limit": limit,
                "sort": "tvl"  # Sort by TVL to get biggest pools
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Data is returned as a list directly
                        if isinstance(data, list):
                            pools = data
                        else:
                            pools = data.get("pools", [])
                        
                        # Process pools to calculate USD values
                        processed_pools = []
                        for pool in pools:
                            try:
                                # Convert TVL from wei to USD (assuming it's in wei)
                                tvl_raw = pool.get("liquidity", {}).get("tvl", 0)
                                # If TVL is huge (>1e20), it's likely in wei format
                                if tvl_raw > 1e20:
                                    tvl_usd = tvl_raw / 1e18  # Convert from wei
                                else:
                                    tvl_usd = tvl_raw
                                    
                                # Skip tiny pools
                                if tvl_usd < 1000:
                                    continue
                                    
                                # Get APR from trading data
                                apr = pool.get("trading", {}).get("apr", 0)
                                
                                # Process volume
                                volume_raw = pool.get("trading", {}).get("volume_24h", 0)
                                if volume_raw > 1e20:
                                    volume_usd = volume_raw / 1e18
                                else:
                                    volume_usd = volume_raw
                                    
                                processed_pool = {
                                    "address": pool.get("address"),
                                    "symbol": pool.get("symbol"),
                                    "tvl": tvl_usd,
                                    "apr": apr,
                                    "volume_24h": volume_usd,
                                    "is_stable": pool.get("type_info", {}).get("is_stable", False)
                                }
                                
                                # Only include if APR meets threshold or has significant volume
                                if apr >= min_apr or volume_usd > 100000:
                                    processed_pools.append(processed_pool)
                                    
                            except Exception as e:
                                logger.debug(f"Failed to process pool: {e}")
                                continue
                        
                        logger.info(f"Found {len(processed_pools)} significant pools")
                        return processed_pools
                    else:
                        logger.error(f"API request failed: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Failed to fetch pools from API: {e}")
            return []
            
    async def get_pool_details(self, pool_address: str) -> Optional[Dict]:
        """Get detailed information about a specific pool.
        
        Args:
            pool_address: Pool contract address
            
        Returns:
            Pool details or None if failed
        """
        try:
            url = f"{self.base_url}{self.api_path}/pools/{pool_address}"
            params = {"target": "base"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Failed to get pool details: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Failed to get pool details: {e}")
            return None
            
    async def scan_pools(self) -> List[Dict]:
        """Scan Aerodrome pools and store interesting observations.
        
        Returns:
            List of scanned pool data
        """
        logger.info("Starting Aerodrome pool scan via API...")
        
        # Fetch pools from API - using lower threshold to catch more opportunities
        pools = await self.fetch_pools(limit=100, min_apr=5.0)
        
        if not pools:
            logger.warning("No pools fetched from API")
            return []
            
        # Process and store observations
        observations = []
        
        for pool in pools:
            try:
                pool_data = {
                    "address": pool["address"],
                    "pair": pool["symbol"],
                    "stable": pool["is_stable"],
                    "tvl": pool["tvl"],
                    "apr_total": pool["apr"],
                    "volume_24h": pool["volume_24h"],
                    "timestamp": datetime.utcnow()
                }
                
                # Store high APR pools in memory
                if pool_data["apr_total"] >= 20:
                    await self.memory.remember(
                        content=f"High APR pool {pool_data['pair']}: {pool_data['apr_total']:.1f}% APR, ${pool_data['tvl']:,.0f} TVL",
                        metadata={
                            "pool_address": pool_data["address"],
                            "apr": pool_data["apr_total"],
                            "tvl": pool_data["tvl"],
                            "volume_24h": pool_data["volume_24h"]
                        },
                        memory_type=MemoryType.OBSERVATION,
                        category="pool_analysis"
                    )
                    
                # Store high volume pools
                if pool_data["volume_24h"] >= 100000:
                    await self.memory.remember(
                        content=f"High volume pool {pool_data['pair']}: ${pool_data['volume_24h']:,.0f} 24h volume",
                        metadata={
                            "pool_address": pool_data["address"],
                            "volume": pool_data["volume_24h"],
                            "apr": pool_data["apr_total"]
                        },
                        memory_type=MemoryType.OBSERVATION,
                        category="observations"
                    )
                    
                observations.append(pool_data)
                
                logger.info(f"Pool {pool_data['pair']}: TVL ${pool_data['tvl']:,.0f}, APR {pool_data['apr_total']:.1f}%, Volume ${pool_data['volume_24h']:,.0f}")
                
            except Exception as e:
                logger.error(f"Failed to process pool: {e}")
                continue
                
        logger.info(f"Scan complete. Processed {len(observations)} pools")
        return observations
        
    async def get_gas_price(self) -> Optional[float]:
        """Get current gas price from QuickNode RPC.
        
        Returns:
            Gas price in Gwei or None if failed
        """
        try:
            # Use standard RPC call for gas price
            async with aiohttp.ClientSession() as session:
                payload = {
                    "jsonrpc": "2.0",
                    "method": "eth_gasPrice",
                    "params": [],
                    "id": 1
                }
                
                async with session.post(self.base_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        gas_wei = int(data["result"], 16)
                        gas_gwei = gas_wei / 1e9
                        return gas_gwei
                    else:
                        logger.error(f"Failed to get gas price: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Failed to get gas price: {e}")
            return None