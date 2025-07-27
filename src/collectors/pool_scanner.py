"""
Aerodrome Pool Scanner - Enhanced with QuickNode MCP

Uses natural language queries to scan pools instead of complex RPC calls.
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from decimal import Decimal

from src.mcp.quicknode_mcp import QuickNodeMCP
from src.agent.memory import AthenaMemory, MemoryType
from config.settings import settings

logger = logging.getLogger(__name__)


class PoolScanner:
    """
    Scans Aerodrome pools using QuickNode MCP for simplified data access.
    
    Natural language queries replace hundreds of lines of RPC code.
    """
    
    def __init__(self, mcp_client: QuickNodeMCP, memory: AthenaMemory):
        """Initialize pool scanner with MCP client."""
        self.mcp = mcp_client
        self.memory = memory
        self.scanning = False
        
        # Pool data cache
        self.pools = {}
        self.last_scan = None
        
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
        logger.info("=== Starting Aerodrome pool scanning with MCP...")
        
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
        """Single scanning cycle using MCP."""
        logger.info("Scanning Aerodrome pools via MCP...")
        
        # Get comprehensive pool data with one query
        pools = await self.mcp.get_aerodrome_pools(
            min_apr=settings.min_apr_for_memory,
            min_tvl=50000,  # $50k minimum
            limit=100
        )
        
        logger.info(f"Found {len(pools)} pools meeting criteria")
        
        # Process and categorize pools
        self.opportunities = {
            "high_apr": [],
            "high_volume": [],
            "new_pools": [],
            "imbalanced": [],
        }
        
        memories_stored = 0
        
        for pool in pools:
            pool_data = {
                "address": pool.get("address"),
                "pair": pool.get("pair"),
                "apr": float(pool.get("apr", 0)),
                "tvl": float(pool.get("tvl", 0)),
                "volume_24h": float(pool.get("volume_24h", 0)),
                "fee_apr": float(pool.get("fee_apr", 0)),
                "incentive_apr": float(pool.get("incentive_apr", 0)),
                "stable": pool.get("stable", False),
                "reserves": pool.get("reserves", {}),
                "ratio": float(pool.get("ratio", 0)),
                "timestamp": datetime.utcnow()
            }
            
            # Store in cache
            self.pools[pool_data["address"]] = pool_data
            
            # Categorize opportunities
            if pool_data["apr"] >= 50:
                self.opportunities["high_apr"].append(pool_data)
                
            if pool_data["volume_24h"] >= settings.min_volume_for_memory:
                self.opportunities["high_volume"].append(pool_data)
                
            # Check if pool is imbalanced
            if abs(pool_data["ratio"] - 1.0) > 0.2:  # More than 20% imbalance
                self.opportunities["imbalanced"].append(pool_data)
                
            # Store high-value pools in memory
            if pool_data["apr"] >= settings.min_apr_for_memory or pool_data["volume_24h"] >= settings.min_volume_for_memory:
                await self._store_pool_memory(pool_data)
                memories_stored += 1
                
                # Prevent memory overflow
                if memories_stored >= settings.max_memories_per_cycle:
                    logger.warning(f"Memory limit reached ({settings.max_memories_per_cycle})")
                    break
                    
        # Store pattern observations
        if self.opportunities["high_apr"]:
            await self.memory.remember(
                content=f"Found {len(self.opportunities['high_apr'])} high APR pools (>50%), "
                       f"top pool: {self.opportunities['high_apr'][0]['pair']} at {self.opportunities['high_apr'][0]['apr']:.1f}%",
                memory_type=MemoryType.OBSERVATION,
                category="market_pattern",
                metadata={
                    "pattern_type": "high_apr_availability",
                    "count": len(self.opportunities["high_apr"]),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        # Get gas optimization insights
        gas_info = await self.mcp.optimize_gas_timing()
        if gas_info:
            await self.memory.remember(
                content=f"Gas optimization: {gas_info.get('recommended_window', 'No specific window')}",
                memory_type=MemoryType.OBSERVATION,
                category="gas_optimization",
                metadata=gas_info
            )
            
        self.last_scan = datetime.utcnow()
        logger.info(f"✅ Scan complete: {len(self.pools)} pools, {memories_stored} memories stored")
        
    async def _store_pool_memory(self, pool_data: Dict):
        """Store pool data as memory with enhanced metadata."""
        try:
            # Determine memory category based on pool characteristics
            if pool_data["apr"] >= 100:
                category = "apr_anomaly"
            elif pool_data["volume_24h"] >= 1000000:
                category = "high_volume_pool"
            elif pool_data["tvl"] < 100000 and pool_data["apr"] > 50:
                category = "new_pool"
            else:
                category = "pool_analysis"
                
            content = (
                f"Pool {pool_data['pair']}: "
                f"APR {pool_data['apr']:.1f}% "
                f"(Fee: {pool_data['fee_apr']:.1f}%, Incentive: {pool_data['incentive_apr']:.1f}%), "
                f"TVL ${pool_data['tvl']:,.0f}, "
                f"24h Volume ${pool_data['volume_24h']:,.0f}"
            )
            
            metadata = {
                "pool": pool_data["pair"],
                "pool_address": pool_data["address"],
                "apr": pool_data["apr"],
                "fee_apr": pool_data["fee_apr"],
                "incentive_apr": pool_data["incentive_apr"],
                "tvl": pool_data["tvl"],
                "volume_24h": pool_data["volume_24h"],
                "stable": pool_data["stable"],
                "ratio": pool_data["ratio"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.memory.remember(
                content=content,
                memory_type=MemoryType.OBSERVATION,
                category=category,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to store pool memory: {e}")
            
    async def get_pool_analysis(self, pool_address: str) -> Dict:
        """Get detailed analysis for a specific pool using MCP."""
        analysis = await self.mcp.analyze_pool(pool_address)
        
        # Store analysis in memory
        if analysis:
            await self.memory.remember(
                content=f"Detailed analysis for pool {pool_address}: {analysis.get('summary', 'No summary')}",
                memory_type=MemoryType.ANALYSIS,
                category="pool_deep_dive",
                metadata=analysis
            )
            
        return analysis
        
    async def find_arbitrage(self) -> List[Dict]:
        """Find arbitrage opportunities using MCP."""
        opportunities = await self.mcp.find_arbitrage_opportunities()
        
        # Store significant opportunities
        for opp in opportunities:
            if opp.get("expected_profit", 0) > 10:  # $10 minimum
                await self.memory.remember(
                    content=f"Arbitrage opportunity: {opp.get('path', 'Unknown')} for ${opp.get('expected_profit', 0):.2f} profit",
                    memory_type=MemoryType.OBSERVATION,
                    category="arbitrage_opportunity",
                    metadata=opp
                )
                
        return opportunities
        
    def get_top_pools(self, category: str = "apr", limit: int = 10) -> List[Dict]:
        """Get top pools by category."""
        pools = list(self.pools.values())
        
        if category == "apr":
            pools.sort(key=lambda x: x["apr"], reverse=True)
        elif category == "tvl":
            pools.sort(key=lambda x: x["tvl"], reverse=True)
        elif category == "volume":
            pools.sort(key=lambda x: x["volume_24h"], reverse=True)
            
        return pools[:limit]
        
    def get_opportunities(self) -> Dict[str, List[Dict]]:
        """Get current opportunities."""
        return self.opportunities