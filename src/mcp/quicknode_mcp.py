"""
QuickNode MCP Client for Natural Language Blockchain Queries

Provides a natural language interface to QuickNode's blockchain services,
including Aerodrome pool data, transaction analytics, and gas optimization.
"""
import asyncio
import subprocess
import json
import logging
import os
from typing import Dict, Any, List, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)


class QuickNodeMCP:
    """Natural language interface to QuickNode services via MCP."""
    
    def __init__(self, api_key: str):
        """Initialize QuickNode MCP client.
        
        Args:
            api_key: QuickNode API key
        """
        self.api_key = api_key
        self.process = None
        self._initialized = False
        
    async def initialize(self):
        """Start the MCP server process."""
        if self._initialized:
            return
            
        try:
            # Start MCP server with QuickNode credentials
            env = {
                "QUICKNODE_API_KEY": self.api_key,
                "PATH": os.environ.get("PATH", "")
            }
            
            self.process = await asyncio.create_subprocess_exec(
                "npx", "-y", "@quicknode/mcp",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            self._initialized = True
            logger.info("QuickNode MCP server initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize QuickNode MCP: {e}")
            raise
            
    async def query(self, prompt: str) -> Dict[str, Any]:
        """Execute a natural language query against blockchain data.
        
        Args:
            prompt: Natural language query
            
        Returns:
            Query results as dictionary
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Send query to MCP server
            request = {
                "method": "tools/call",
                "params": {
                    "name": "query_blockchain",
                    "arguments": {
                        "prompt": prompt
                    }
                }
            }
            
            # Write request
            self.process.stdin.write(json.dumps(request).encode() + b'\n')
            await self.process.stdin.drain()
            
            # Read response
            response_line = await self.process.stdout.readline()
            response = json.loads(response_line.decode())
            
            if "error" in response:
                logger.error(f"MCP query error: {response['error']}")
                return {}
                
            return response.get("result", {})
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {}
            
    async def get_aerodrome_pools(self, 
                                min_apr: float = 20,
                                min_tvl: float = 100000,
                                limit: int = 50) -> List[Dict]:
        """Get Aerodrome pools filtered by criteria.
        
        Args:
            min_apr: Minimum APR percentage
            min_tvl: Minimum TVL in USD
            limit: Maximum number of results
            
        Returns:
            List of pool dictionaries
        """
        prompt = f"""
        Find Aerodrome pools on Base network with:
        - APR greater than {min_apr}%
        - TVL greater than ${min_tvl:,.0f}
        - Include: address, pair, APR, TVL, 24h volume, reserves, emission rewards
        - Sort by APR descending
        - Limit to {limit} results
        - Format as structured data
        """
        
        result = await self.query(prompt)
        pools = result.get("pools", [])
        
        # Ensure decimal conversion for numeric fields
        for pool in pools:
            pool["apr"] = Decimal(str(pool.get("apr", 0)))
            pool["tvl"] = Decimal(str(pool.get("tvl", 0)))
            pool["volume_24h"] = Decimal(str(pool.get("volume_24h", 0)))
            
        return pools
        
    async def analyze_pool(self, pool_address: str) -> Dict[str, Any]:
        """Get comprehensive analysis of a specific pool.
        
        Args:
            pool_address: Pool contract address
            
        Returns:
            Detailed pool analytics
        """
        prompt = f"""
        Analyze Aerodrome pool at address {pool_address}:
        - Current metrics: APR, TVL, volume, reserves
        - Historical data: 7-day APR trend, volume trend
        - Emission rewards and fee structure
        - Impermanent loss estimation
        - Recommended actions based on current state
        """
        
        return await self.query(prompt)
        
    async def find_arbitrage_opportunities(self) -> List[Dict]:
        """Find arbitrage opportunities across Aerodrome pools."""
        prompt = """
        Find arbitrage opportunities on Aerodrome:
        - Price discrepancies between pools
        - Routing paths for profit
        - Consider gas costs on Base
        - Only show opportunities > $10 profit after gas
        - Include execution steps
        """
        
        result = await self.query(prompt)
        return result.get("opportunities", [])
        
    async def optimize_gas_timing(self) -> Dict[str, Any]:
        """Get optimal times for transactions based on gas patterns."""
        prompt = """
        Analyze Base network gas patterns:
        - Current gas price
        - 24-hour gas price trend
        - Predicted low-gas windows in next 6 hours
        - Recommended transaction timing
        - Historical patterns by hour and day
        """
        
        return await self.query(prompt)
        
    async def analyze_rebalance_opportunity(self, 
                                          current_pool: str,
                                          current_apr: float,
                                          position_value: float) -> Dict[str, Any]:
        """Analyze if rebalancing would be profitable.
        
        Args:
            current_pool: Current pool address or pair
            current_apr: Current position APR
            position_value: Position value in USD
            
        Returns:
            Rebalancing analysis and recommendations
        """
        prompt = f"""
        Analyze rebalancing opportunity for:
        - Current position: {current_pool} with {current_apr}% APR
        - Position value: ${position_value:,.2f}
        - Find better opportunities considering:
          - Gas costs for exit and entry
          - Slippage impact
          - APR improvement needed to justify costs
          - Risk assessment
        - Recommend: stay, rebalance, or wait
        """
        
        return await self.query(prompt)
        
    async def get_market_overview(self) -> Dict[str, Any]:
        """Get comprehensive Aerodrome market overview."""
        prompt = """
        Provide Aerodrome market overview on Base:
        - Total TVL across all pools
        - Average APRs by pool type (stable vs volatile)
        - Top 5 pools by TVL
        - Top 5 pools by APR
        - 24h volume leaders
        - New pools in last 24h
        - Market trends and notable changes
        """
        
        return await self.query(prompt)
        
    async def close(self):
        """Close the MCP server process."""
        if self.process:
            self.process.terminate()
            await self.process.wait()
            self._initialized = False
            logger.info("QuickNode MCP server closed")
            
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Convenience function for testing
async def test_mcp():
    """Test the QuickNode MCP integration."""
    import os
    from config.settings import settings
    
    async with QuickNodeMCP(settings.quicknode_api_key) as mcp:
        # Test market overview
        print("Testing market overview...")
        overview = await mcp.get_market_overview()
        print(f"Market Overview: {json.dumps(overview, indent=2)}")
        
        # Test pool search
        print("\nTesting pool search...")
        pools = await mcp.get_aerodrome_pools(min_apr=30, limit=5)
        for pool in pools:
            print(f"  {pool['pair']}: {pool['apr']}% APR, ${pool['tvl']:,.0f} TVL")
            
        # Test gas optimization
        print("\nTesting gas optimization...")
        gas_info = await mcp.optimize_gas_timing()
        print(f"Gas Optimization: {json.dumps(gas_info, indent=2)}")


if __name__ == "__main__":
    # For testing
    asyncio.run(test_mcp())