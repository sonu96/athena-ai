"""
CDP SDK Wrapper for Base Chain Operations
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from decimal import Decimal

# Ensure we're using the correct CDP SDK version
from .version_check import check_cdp_version

from cdp import CdpClient
from config.settings import settings
from config.contracts import CONTRACTS, TOKENS, DEFAULT_SLIPPAGE

logger = logging.getLogger(__name__)


class BaseClient:
    """CDP client for interacting with Base blockchain and Aerodrome."""
    
    def __init__(self):
        """Initialize CDP client."""
        self.cdp = None
        self.wallet = None
        self._initialized = False
        self._wallet_secret = None
        
    async def initialize(self):
        """Initialize CDP SDK and wallet."""
        if self._initialized:
            return
            
        try:
            # Configure CDP from JSON file if available
            import os
            import json
            import secrets
            
            json_path = os.environ.get('CDP_API_KEY_JSON_PATH')
            if json_path and os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    cdp_data = json.load(f)
                api_key_id = cdp_data.get('id', cdp_data.get('api_key_name'))
                api_key_secret = cdp_data.get('privateKey', cdp_data.get('private_key'))
                logger.info(f"Loaded CDP credentials from JSON file: {json_path}")
            else:
                # Fall back to settings
                api_key_id = settings.cdp_api_key
                api_key_secret = settings.cdp_api_secret
            
            # Get or generate wallet secret
            self._wallet_secret = settings.cdp_wallet_secret or os.environ.get('CDP_WALLET_SECRET')
            if not self._wallet_secret:
                self._wallet_secret = secrets.token_hex(32)
                logger.info("Generated new wallet secret")
                logger.info(f"⚠️  Save this to .env: CDP_WALLET_SECRET={self._wallet_secret}")
                # Also save to secret manager for persistence
                try:
                    from src.gcp.secret_manager import create_or_update_secret
                    create_or_update_secret("cdp-wallet-secret", self._wallet_secret)
                    logger.info("✅ Saved wallet secret to Secret Manager")
                except Exception as e:
                    logger.warning(f"Could not save wallet secret to Secret Manager: {e}")
            
            # Create CDP client
            self.cdp = CdpClient(
                api_key_id=api_key_id,
                api_key_secret=api_key_secret,
                wallet_secret=self._wallet_secret
            )
            logger.info("CDP client created successfully")
            
            # Create or load account
            if settings.agent_wallet_id and settings.agent_wallet_id.strip():
                # Load existing account
                self.wallet = await self.cdp.evm.get_account(settings.agent_wallet_id)
                logger.info(f"Loaded existing account: {settings.agent_wallet_id}")
            else:
                # Create new account on Base chain
                self.wallet = await self.cdp.evm.create_account()
                # Get the wallet address - for CDP SDK v1.23.0, the wallet ID is the address
                wallet_address = self.wallet.addresses[0].address if hasattr(self.wallet, 'addresses') else str(self.wallet.address)
                logger.info(f"Created new wallet")
                logger.info(f"Address: {wallet_address}")
                logger.info(f"⚠️  Save this to .env: AGENT_WALLET_ID={wallet_address}")
                
            self._initialized = True
            logger.info("CDP client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CDP client: {e}")
            raise
        finally:
            # Close client if initialization fails
            if hasattr(self, 'cdp') and not self._initialized:
                await self.cdp.close()
            
    @property
    def address(self) -> str:
        """Get wallet address."""
        if not self.wallet:
            raise ValueError("Wallet not initialized")
        # Handle different SDK versions
        if hasattr(self.wallet, 'addresses') and self.wallet.addresses:
            return self.wallet.addresses[0].address
        elif hasattr(self.wallet, 'address'):
            return self.wallet.address
        else:
            raise ValueError("Unable to get wallet address")
        
    async def get_balance(self, token: str = "ETH") -> Decimal:
        """Get token balance."""
        try:
            # For CDP SDK v1.23.0, use the wallet's balances method
            if hasattr(self.wallet, 'balances'):
                balances = await self.wallet.balances()
                # balances is a dict like {"ETH": balance_value}
                if token in balances:
                    return Decimal(str(balances[token]))
            
            # Fallback to 0 if not found
            return Decimal("0")
        except Exception as e:
            logger.error(f"Failed to get balance for {token}: {e}")
            return Decimal("0")
            
    async def get_all_balances(self) -> Dict[str, Decimal]:
        """Get all token balances."""
        balances = {}
        
        # Get ETH balance
        balances["ETH"] = await self.get_balance("ETH")
        
        # Get token balances
        for token_name, token_address in TOKENS.items():
            balance = await self.get_balance(token_name)
            if balance > 0:
                balances[token_name] = balance
                
        return balances
        
    async def swap_tokens(
        self,
        token_in: str,
        token_out: str,
        amount_in: Decimal,
        slippage: float = DEFAULT_SLIPPAGE,
        stable: bool = False
    ) -> Optional[str]:
        """
        Swap tokens using Aerodrome router.
        
        Args:
            token_in: Input token symbol
            token_out: Output token symbol
            amount_in: Amount to swap
            slippage: Slippage tolerance (default 0.5%)
            stable: Whether to use stable pool
            
        Returns:
            Transaction hash if successful
        """
        try:
            # Get token addresses
            token_in_address = TOKENS.get(token_in, token_in)
            token_out_address = TOKENS.get(token_out, token_out)
            
            # Build swap route
            route = [{
                "from": token_in_address,
                "to": token_out_address,
                "stable": stable,
            }]
            
            # Get quote first
            quote = await self._get_quote(
                token_in_address,
                token_out_address,
                amount_in,
                stable
            )
            
            # Calculate minimum output with slippage
            min_amount_out = quote * Decimal(str(1 - slippage))
            
            # Build transaction
            deadline = int(asyncio.get_event_loop().time()) + 1200  # 20 minutes
            
            contract_invocation = self.wallet.invoke_contract(
                contract_address=CONTRACTS["router"]["address"],
                method="swapExactTokensForTokens",
                args={
                    "amountIn": str(int(amount_in * 10**18)),
                    "amountOutMin": str(int(min_amount_out * 10**18)),
                    "routes": route,
                    "to": self.address,
                    "deadline": deadline,
                }
            )
            
            # Wait for transaction
            contract_invocation.wait()
            
            logger.info(
                f"Swap successful: {amount_in} {token_in} -> {token_out} "
                f"(tx: {contract_invocation.transaction.transaction_hash})"
            )
            
            return contract_invocation.transaction.transaction_hash
            
        except Exception as e:
            logger.error(f"Swap failed: {e}")
            return None
            
    async def add_liquidity(
        self,
        token_a: str,
        token_b: str,
        amount_a: Decimal,
        amount_b: Decimal,
        stable: bool = False,
        slippage: float = DEFAULT_SLIPPAGE
    ) -> Optional[str]:
        """Add liquidity to Aerodrome pool."""
        try:
            # Get token addresses
            token_a_address = TOKENS.get(token_a, token_a)
            token_b_address = TOKENS.get(token_b, token_b)
            
            # Calculate minimum amounts with slippage
            min_amount_a = amount_a * Decimal(str(1 - slippage))
            min_amount_b = amount_b * Decimal(str(1 - slippage))
            
            deadline = int(asyncio.get_event_loop().time()) + 1200
            
            contract_invocation = self.wallet.invoke_contract(
                contract_address=CONTRACTS["router"]["address"],
                method="addLiquidity",
                args={
                    "tokenA": token_a_address,
                    "tokenB": token_b_address,
                    "stable": stable,
                    "amountADesired": str(int(amount_a * 10**18)),
                    "amountBDesired": str(int(amount_b * 10**18)),
                    "amountAMin": str(int(min_amount_a * 10**18)),
                    "amountBMin": str(int(min_amount_b * 10**18)),
                    "to": self.address,
                    "deadline": deadline,
                }
            )
            
            contract_invocation.wait()
            
            logger.info(
                f"Added liquidity: {amount_a} {token_a} + {amount_b} {token_b} "
                f"(tx: {contract_invocation.transaction.transaction_hash})"
            )
            
            return contract_invocation.transaction.transaction_hash
            
        except Exception as e:
            logger.error(f"Add liquidity failed: {e}")
            return None
            
    async def get_pool_info(
        self,
        token_a: str,
        token_b: str,
        stable: bool = False
    ) -> Dict:
        """Get pool information."""
        try:
            token_a_address = TOKENS.get(token_a, token_a)
            token_b_address = TOKENS.get(token_b, token_b)
            
            # Get pool address from factory
            pool_address = await self._get_pool_address(
                token_a_address,
                token_b_address,
                stable
            )
            
            if not pool_address:
                return {}
                
            # Use RPC reader to get real data
            from src.blockchain.rpc_reader import RPCReader
            
            # Use CDP's authenticated RPC endpoint
            async with RPCReader(settings.cdp_rpc_url) as reader:
                # Get token info first to determine decimals
                token_info = await reader.get_token_info(pool_address)
                if not token_info:
                    logger.error(f"Failed to read token info for pool {pool_address}")
                    return {}
                
                # Get reserves
                reserves_data = await reader.get_pool_reserves(pool_address)
                if not reserves_data:
                    logger.error(f"Failed to read reserves for pool {pool_address}")
                    return {}
                    
                reserve0 = reserves_data["reserve0"]
                reserve1 = reserves_data["reserve1"]
                
                # Get total supply
                total_supply_decimal = await reader.get_total_supply(pool_address)
                if not total_supply_decimal:
                    total_supply_decimal = Decimal("0")
                    
            # Determine decimals based on token addresses
            # Common Base tokens
            decimals_map = {
                "0x4200000000000000000000000000000000000006": 18,  # WETH
                "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913": 6,   # USDC
                "0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA": 6,   # USDbC
                "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb": 18,  # DAI
                "0x940181a94A35A4569E4529A3CDfB74e38FD98631": 18,  # AERO
            }
            
            # Get decimals for token0 and token1
            decimals0 = decimals_map.get(token_info["token0"].lower(), 18)
            decimals1 = decimals_map.get(token_info["token1"].lower(), 18)
            
            # Apply correct decimals if we got raw values from storage
            if reserve0 > Decimal(10**30):  # Likely raw value
                reserve0 = reserve0 / Decimal(10**decimals0)
            if reserve1 > Decimal(10**30):  # Likely raw value  
                reserve1 = reserve1 / Decimal(10**decimals1)
            
            # Calculate TVL using pool ratios and known stablecoin values
            # Stablecoins we can use as price anchors
            stablecoins = {
                "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # USDC
                "0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA",  # USDbC
                "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb",  # DAI
            }
            
            token0_addr = token_info["token0"].lower()
            token1_addr = token_info["token1"].lower()
            
            # If one token is a stablecoin, we can calculate the other token's price
            if token0_addr in stablecoins:
                # token0 is $1, so token1 price = reserve0 / reserve1
                token1_price = reserve0 / reserve1 if reserve1 > 0 else Decimal("0")
                tvl = reserve0 + (reserve1 * token1_price)
            elif token1_addr in stablecoins:
                # token1 is $1, so token0 price = reserve1 / reserve0
                token0_price = reserve1 / reserve0 if reserve0 > 0 else Decimal("0")
                tvl = (reserve0 * token0_price) + reserve1
            else:
                # Neither is stablecoin, need to get price from another source
                # For now, just sum reserves (will be fixed later)
                tvl = reserve0 + reserve1
                logger.warning(f"No stablecoin in pool {token_a}/{token_b}, TVL calculation may be incorrect")
            
            # Calculate pool token ratio
            ratio = reserve0 / reserve1 if reserve1 > 0 else Decimal(0)
            
            # Map reserves to the correct tokens based on token0/token1 ordering
            if token_a_address.lower() == token0_addr:
                reserve_a = reserve0
                reserve_b = reserve1
            else:
                reserve_a = reserve1
                reserve_b = reserve0
            
            return {
                "address": pool_address,
                "token_a": token_a,
                "token_b": token_b,
                "stable": stable,
                "reserve0": reserve0,
                "reserve1": reserve1,
                "reserve_a": reserve_a,
                "reserve_b": reserve_b,
                "total_supply": total_supply_decimal,
                "tvl": tvl,
                "ratio": ratio,
                "imbalanced": abs(ratio - 1) > Decimal("0.1")  # More than 10% imbalance
            }
            
        except Exception as e:
            logger.error(f"Failed to get pool info: {e}")
            return {}
            
    async def _get_quote(
        self,
        token_in: str,
        token_out: str,
        amount_in: Decimal,
        stable: bool
    ) -> Decimal:
        """Get swap quote from Aerodrome."""
        try:
            # Build route for getAmountsOut
            route = [{
                "from": token_in,
                "to": token_out,
                "stable": stable,
            }]
            
            # Use wallet's read_contract method
            amounts_result = await self.wallet.read_contract(
                contract_address=CONTRACTS["router"]["address"],
                method="getAmountsOut",
                args={
                    "amountIn": str(int(amount_in * 10**18)),
                    "routes": route
                },
                abi=CONTRACTS["router"]["abi"]
            )
            
            # The result is an array where the last element is the output amount
            if amounts_result and len(amounts_result) > 1:
                amount_out = Decimal(str(amounts_result[-1])) / Decimal(10**18)
                return amount_out
            else:
                logger.warning("No quote received from router")
                return Decimal(0)
                
        except Exception as e:
            logger.error(f"Failed to get quote: {e}")
            # Fallback to a conservative estimate
            return amount_in * Decimal("0.95")
        
    async def _get_pool_address(
        self,
        token_a: str,
        token_b: str,
        stable: bool
    ) -> Optional[str]:
        """Get pool address from factory."""
        try:
            # Get token addresses
            token_a_address = TOKENS.get(token_a, token_a)
            token_b_address = TOKENS.get(token_b, token_b)
            
            # Try to get from factory first using CDP RPC
            from src.blockchain.rpc_reader import RPCReader
            
            try:
                async with RPCReader(settings.cdp_rpc_url) as reader:
                    pool_address = await reader.get_pool_address(
                        CONTRACTS["factory"]["address"],
                        token_a_address,
                        token_b_address,
                        stable
                    )
                    
                if pool_address:
                    logger.info(f"Found pool at {pool_address} for {token_a}/{token_b} stable={stable}")
                    return pool_address
            except Exception as e:
                logger.warning(f"Failed to query factory: {e}")
            
            # Fallback to known pools
            known_pools = {
                # WETH-USDC volatile (Standard AMM) - verified working
                (TOKENS["WETH"].lower(), TOKENS["USDC"].lower(), False): "0xcDAc0d6c6C59727a65F871236188350531885C43",
                (TOKENS["USDC"].lower(), TOKENS["WETH"].lower(), False): "0xcDAc0d6c6C59727a65F871236188350531885C43",
                
                # Note: SlipStream pool 0xb2cc224c1c9fee385f8ad6a55b4d94e92359dc59 uses different interface
                
                # AERO-USDC volatile (verified working)
                ("0x940181a94a35a4569e4529a3cdfb74e38fd98631".lower(), TOKENS["USDC"].lower(), False): "0x6cDcb1C4A4D1C3C6d054b27AC5B77e89eAFb971d",
                (TOKENS["USDC"].lower(), "0x940181a94a35a4569e4529a3cdfb74e38fd98631".lower(), False): "0x6cDcb1C4A4D1C3C6d054b27AC5B77e89eAFb971d",
                
                # Add more verified pools as needed
            }
            
            # Check known pools
            pool_key = (token_a_address.lower(), token_b_address.lower(), stable)
            pool_address = known_pools.get(pool_key)
            if not pool_address:
                # Check reverse order
                pool_key = (token_b_address.lower(), token_a_address.lower(), stable)
                pool_address = known_pools.get(pool_key)
            
            if pool_address:
                logger.info(f"Using known pool at {pool_address} for {token_a}/{token_b} stable={stable}")
                return pool_address
            else:
                logger.warning(f"No pool found for {token_a}/{token_b} stable={stable}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get pool address: {e}")
            return None
            
    async def estimate_gas(self, method: str, **kwargs) -> int:
        """Estimate gas for a transaction."""
        try:
            # Different gas estimates based on method type
            gas_estimates = {
                "swap": 250000,
                "addLiquidity": 350000,
                "removeLiquidity": 300000,
                "approve": 50000,
                "transfer": 65000,
            }
            
            # Get base estimate
            base_estimate = gas_estimates.get(method, 200000)
            
            # Apply gas buffer from settings
            from config.contracts import GAS_BUFFER
            return int(base_estimate * GAS_BUFFER)
            
        except Exception as e:
            logger.error(f"Failed to estimate gas: {e}")
            return 300000  # Conservative default
        
    async def get_gas_price(self) -> Decimal:
        """Get current gas price."""
        try:
            # CDP SDK should provide gas price estimation
            # For Base chain, gas is typically very low
            # This would be fetched from the network in production
            
            # Base mainnet typically has gas prices around 0.001-0.1 gwei
            # During high congestion it might go up to 1-5 gwei
            # For now, return a reasonable estimate
            return Decimal("0.05")  # 0.05 gwei is typical for Base
            
        except Exception as e:
            logger.error(f"Failed to get gas price: {e}")
            return Decimal("0.1")  # Conservative fallback