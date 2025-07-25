"""
RPC Reader for fetching blockchain data without web3 dependency
Uses direct JSON-RPC calls to read contract data
"""
import aiohttp
import json
import logging
from typing import Any, Dict, List, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)


class RPCReader:
    """Read blockchain data using JSON-RPC calls."""
    
    def __init__(self, rpc_url: str):
        """Initialize RPC reader."""
        self.rpc_url = rpc_url
        self.session = None
        logger.info(f"RPC Reader initialized with URL: {rpc_url[:50]}...")
        
    async def __aenter__(self):
        """Enter async context."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        if self.session:
            await self.session.close()
            
    async def call_contract_function(
        self,
        contract_address: str,
        function_signature: str,
        function_args: List[Any] = None,
        block: str = "latest"
    ) -> Any:
        """
        Call a contract function using eth_call.
        
        Args:
            contract_address: The contract address
            function_signature: The function signature (e.g., "balanceOf(address)")
            function_args: Function arguments
            block: Block to query (default: "latest")
            
        Returns:
            The decoded result
        """
        try:
            # Encode function call
            # Pre-encoded function selectors for common functions
            encoded_functions = {
                "getReserves()": "0x0902f1ac",
                "totalSupply()": "0x18160ddd",
                "getPool(address,address,bool)": "0x79bc57d5",
                "token0()": "0x0dfe1681",
                "token1()": "0xd21220a7",
                "decimals()": "0x313ce567",
                "reserve0()": "0x443cb4bc",
                "reserve1()": "0x5a76f25e",
            }
            
            # Handle getPool with arguments
            if function_signature.startswith("getPool") and function_args:
                # Function selector for getPool(address,address,bool)
                data = "0x79bc57d5"  # Correct selector for Aerodrome's getPool
                # Encode arguments (address, address, bool)
                # Remove 0x prefix and pad addresses to 32 bytes
                token0 = function_args[0].lower().replace("0x", "").zfill(64)
                token1 = function_args[1].lower().replace("0x", "").zfill(64)
                stable = "0000000000000000000000000000000000000000000000000000000000000001" if function_args[2] else "0000000000000000000000000000000000000000000000000000000000000000"
                data = data + token0 + token1 + stable
            else:
                data = encoded_functions.get(function_signature, "0x")
            
            # Prepare eth_call params
            params = [{
                "to": contract_address,
                "data": data
            }, block]
            
            # Make RPC call
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_call",
                "params": params,
                "id": 1
            }
            
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.post(
                self.rpc_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                result = await response.json()
                
                if "error" in result:
                    logger.error(f"RPC error: {result['error']}")
                    return None
                    
                return result.get("result", "0x0")
                
        except Exception as e:
            logger.error(f"Failed to call contract function: {e}")
            return None
            
    async def get_storage_at(self, address: str, slot: str) -> Optional[str]:
        """Get storage value at specific slot."""
        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_getStorageAt",
                "params": [address, slot, "latest"],
                "id": 1
            }
            
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.post(
                self.rpc_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                result = await response.json()
                
                if "error" in result:
                    logger.error(f"RPC error: {result['error']}")
                    return None
                    
                return result.get("result", "0x0")
                
        except Exception as e:
            logger.error(f"Failed to get storage: {e}")
            return None

    async def get_pool_reserves(self, pool_address: str) -> Optional[Dict[str, Decimal]]:
        """Get pool reserves using direct RPC call."""
        try:
            # First try standard getReserves()
            result = await self.call_contract_function(
                pool_address,
                "getReserves()"
            )
            
            if result and result != "0x0":
                # Decode result (reserves returns 3 uint256 values)
                hex_data = result[2:]
                
                # Each uint256 is 64 hex chars
                reserve0_hex = hex_data[0:64]
                reserve1_hex = hex_data[64:128]
                
                # Convert to decimal - return raw values
                # Decimals will be handled by the caller based on token info
                reserve0 = Decimal(int(reserve0_hex, 16))
                reserve1 = Decimal(int(reserve1_hex, 16))
                
                return {
                    "reserve0": reserve0,
                    "reserve1": reserve1
                }
            
            # If getReserves() fails, try reading from storage slot 0x8 (Uniswap V2 layout)
            logger.info(f"Trying storage slot reading for pool {pool_address}")
            
            # Uniswap V2 and many forks store reserves in slot 8
            # Format: reserve0 (112 bits) | reserve1 (112 bits) | blockTimestampLast (32 bits)
            storage_data = await self.get_storage_at(pool_address, "0x8")
            
            if storage_data and storage_data != "0x0":
                # Remove 0x prefix
                data = storage_data[2:]
                
                # Parse packed data (right to left):
                # - Last 8 chars (32 bits): blockTimestampLast
                # - Next 28 chars (112 bits): reserve1
                # - Next 28 chars (112 bits): reserve0
                
                if len(data) >= 64:
                    # Extract reserves (112 bits = 28 hex chars each)
                    reserve1_hex = data[-40:-8]  # Skip timestamp, get reserve1
                    reserve0_hex = data[-68:-40]  # Get reserve0
                    
                    reserve0 = Decimal(int(reserve0_hex, 16))
                    reserve1 = Decimal(int(reserve1_hex, 16))
                    
                    # Only return if values seem reasonable
                    if reserve0 > 0 and reserve1 > 0:
                        # Note: decimals will be handled by the caller based on token info
                        return {
                            "reserve0": reserve0,
                            "reserve1": reserve1
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get pool reserves: {e}")
            return None
            
    async def get_total_supply(self, token_address: str) -> Optional[Decimal]:
        """Get token total supply."""
        try:
            result = await self.call_contract_function(
                token_address,
                "totalSupply()"
            )
            
            if not result or result == "0x0":
                return None
                
            # Convert hex to decimal - return raw value
            # Decimals will be handled by the caller
            supply = Decimal(int(result, 16))
            return supply
            
        except Exception as e:
            logger.error(f"Failed to get total supply: {e}")
            return None
            
    async def get_pool_address(
        self,
        factory_address: str,
        token0: str,
        token1: str,
        stable: bool
    ) -> Optional[str]:
        """Get pool address from factory."""
        try:
            result = await self.call_contract_function(
                factory_address,
                "getPool(address,address,bool)",
                [token0, token1, stable]
            )
            
            if not result or result == "0x0":
                return None
                
            # Extract address from result (last 40 chars)
            address = "0x" + result[-40:]
            
            # Check if it's zero address
            if address == "0x0000000000000000000000000000000000000000":
                return None
                
            return address
            
        except Exception as e:
            logger.error(f"Failed to get pool address: {e}")
            return None
            
    async def get_token_info(self, pool_address: str) -> Optional[Dict[str, str]]:
        """Get token addresses from pool."""
        try:
            # Get token0
            token0_result = await self.call_contract_function(pool_address, "token0()")
            # Get token1  
            token1_result = await self.call_contract_function(pool_address, "token1()")
            
            if token0_result and token1_result:
                token0 = "0x" + token0_result[-40:]
                token1 = "0x" + token1_result[-40:]
                return {
                    "token0": token0,
                    "token1": token1
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to get token info: {e}")
            return None