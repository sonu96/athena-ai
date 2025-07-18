"""
CDP SDK Wrapper for Base Chain Operations
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from decimal import Decimal

from cdp import Wallet, WalletData, Cdp
from cdp.wallet import Wallet as CdpWallet
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
        
    async def initialize(self):
        """Initialize CDP SDK and wallet."""
        if self._initialized:
            return
            
        try:
            # Configure CDP
            Cdp.configure(
                api_key_name=settings.cdp_api_key,
                api_key_private_key=settings.cdp_api_secret,
            )
            
            # Create or load wallet
            if settings.agent_wallet_id:
                # Load existing wallet
                self.wallet = Wallet.fetch(settings.agent_wallet_id)
                logger.info(f"Loaded existing wallet: {settings.agent_wallet_id}")
            else:
                # Create new wallet
                self.wallet = Wallet.create()
                logger.info(f"Created new wallet: {self.wallet.id}")
                # TODO: Save wallet ID to settings
                
            self._initialized = True
            logger.info("CDP client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CDP client: {e}")
            raise
            
    @property
    def address(self) -> str:
        """Get wallet address."""
        if not self.wallet:
            raise ValueError("Wallet not initialized")
        return self.wallet.default_address.address_id
        
    async def get_balance(self, token: str = "ETH") -> Decimal:
        """Get token balance."""
        try:
            if token == "ETH":
                balance = self.wallet.balance("eth")
            else:
                # Get ERC20 balance
                balance = self.wallet.balance(TOKENS.get(token, token))
            return Decimal(str(balance))
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
            min_amount_out = quote * (1 - slippage)
            
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
            min_amount_a = amount_a * (1 - slippage)
            min_amount_b = amount_b * (1 - slippage)
            
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
                
            # Get pool data (reserves, fees, etc.)
            # This would require additional contract calls
            return {
                "address": pool_address,
                "token_a": token_a,
                "token_b": token_b,
                "stable": stable,
                # TODO: Add reserves, TVL, APR, etc.
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
        # TODO: Implement actual quote fetching
        # For now, return a mock quote
        return amount_in * Decimal("0.95")
        
    async def _get_pool_address(
        self,
        token_a: str,
        token_b: str,
        stable: bool
    ) -> Optional[str]:
        """Get pool address from factory."""
        try:
            result = self.wallet.read_contract(
                contract_address=CONTRACTS["factory"]["address"],
                method="getPair",
                args={
                    "tokenA": token_a,
                    "tokenB": token_b,
                    "stable": stable,
                }
            )
            return result if result != "0x0000000000000000000000000000000000000000" else None
        except Exception as e:
            logger.error(f"Failed to get pool address: {e}")
            return None
            
    async def estimate_gas(self, method: str, **kwargs) -> int:
        """Estimate gas for a transaction."""
        # TODO: Implement gas estimation
        return 200000  # Default gas estimate
        
    async def get_gas_price(self) -> Decimal:
        """Get current gas price."""
        # TODO: Implement actual gas price fetching
        return Decimal("0.001")  # 0.001 gwei on Base