"""
Coinbase AgentKit Client for Athena AI

Provides AI-native blockchain transaction capabilities using existing CDP credentials.
Replaces complex CDP SDK implementation with simple, powerful AgentKit tools.
"""
import logging
from typing import Dict, List, Optional, Any
from decimal import Decimal
import json
import asyncio

from coinbase_agentkit import (
    AgentKit,
    AgentKitConfig,
    CdpEvmWalletProvider,
    CdpEvmWalletProviderConfig,
    wallet_action_provider,
    cdp_api_action_provider
)

from langchain_core.tools import Tool
from config.settings import settings

logger = logging.getLogger(__name__)


class AthenaAgentKit:
    """AI-native blockchain interface using Coinbase AgentKit."""
    
    def __init__(self, api_key: str = None, api_secret: str = None, 
                 wallet_data: str = None, network: str = "base-mainnet"):
        """Initialize AgentKit with existing CDP credentials.
        
        Args:
            api_key: CDP API key (defaults to settings)
            api_secret: CDP API secret (defaults to settings)
            wallet_data: Existing wallet data for recovery
            network: Network to operate on (base-mainnet, base-sepolia)
        """
        self.api_key = api_key or settings.cdp_api_key
        self.api_secret = api_secret or settings.cdp_api_secret
        self.wallet_data = wallet_data or settings.agent_wallet_id
        self.network = network
        
        self.agent = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the AgentKit with wallet."""
        if self._initialized:
            return
            
        try:
            # Create wallet provider
            wallet_config = CdpEvmWalletProviderConfig(
                api_key=self.api_key,
                api_secret=self.api_secret,
                network=self.network
            )
            
            wallet_provider = CdpEvmWalletProvider(wallet_config)
            
            # Initialize with existing wallet or create new
            if self.wallet_data:
                logger.info(f"Loading existing wallet: {self.wallet_data}")
                # Wallet recovery will be handled by the provider
            else:
                logger.info("Creating new wallet...")
                
            # Create AgentKit instance with providers
            config = AgentKitConfig(
                wallet_provider=wallet_provider,
                action_providers=[wallet_action_provider, cdp_api_action_provider]
            )
            
            self.agent = AgentKit(config)
            self._initialized = True
            logger.info("AgentKit initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AgentKit: {e}")
            raise
            
    @property
    def address(self) -> str:
        """Get wallet address."""
        if not self.agent or not self.agent.wallet:
            return None
        return self.agent.wallet.default_address.address_id
        
    async def get_balance(self, asset: str = "eth") -> Decimal:
        """Get balance for an asset.
        
        Args:
            asset: Asset to check (eth, usdc, etc.)
            
        Returns:
            Balance as Decimal
        """
        try:
            action = GetBalanceAction(asset=asset)
            result = await self.agent.run_action(action)
            return Decimal(str(result.get("balance", 0)))
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return Decimal("0")
            
    async def swap(self, from_asset: str, to_asset: str, amount: Decimal, 
                   slippage: float = 0.5) -> Dict[str, Any]:
        """Execute a token swap.
        
        Args:
            from_asset: Source token symbol
            to_asset: Target token symbol  
            amount: Amount to swap
            slippage: Max slippage percentage
            
        Returns:
            Transaction result
        """
        try:
            # Use natural language for complex operations
            prompt = (
                f"Swap {amount} {from_asset} for {to_asset} "
                f"with maximum {slippage}% slippage on Aerodrome"
            )
            
            result = await self.agent.run(prompt)
            logger.info(f"Swap executed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Swap failed: {e}")
            return {"error": str(e)}
            
    async def add_liquidity(self, token_a: str, token_b: str,
                          amount_a: Decimal, amount_b: Decimal,
                          pool_type: str = "volatile") -> Dict[str, Any]:
        """Add liquidity to an Aerodrome pool.
        
        Args:
            token_a: First token symbol
            token_b: Second token symbol
            amount_a: Amount of first token
            amount_b: Amount of second token
            pool_type: Pool type (volatile or stable)
            
        Returns:
            Transaction result
        """
        try:
            prompt = (
                f"Add liquidity to Aerodrome {pool_type} pool {token_a}/{token_b} "
                f"with {amount_a} {token_a} and {amount_b} {token_b}"
            )
            
            result = await self.agent.run(prompt)
            logger.info(f"Liquidity added: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Add liquidity failed: {e}")
            return {"error": str(e)}
            
    async def remove_liquidity(self, pool_address: str, 
                             lp_amount: Decimal) -> Dict[str, Any]:
        """Remove liquidity from a pool.
        
        Args:
            pool_address: Pool contract address
            lp_amount: Amount of LP tokens to remove
            
        Returns:
            Transaction result
        """
        try:
            prompt = (
                f"Remove {lp_amount} LP tokens from Aerodrome pool at {pool_address}"
            )
            
            result = await self.agent.run(prompt)
            logger.info(f"Liquidity removed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Remove liquidity failed: {e}")
            return {"error": str(e)}
            
    async def claim_rewards(self, pool_address: str) -> Dict[str, Any]:
        """Claim rewards from a pool.
        
        Args:
            pool_address: Pool contract address
            
        Returns:
            Transaction result
        """
        try:
            prompt = f"Claim all pending rewards from Aerodrome pool at {pool_address}"
            
            result = await self.agent.run(prompt)
            logger.info(f"Rewards claimed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Claim rewards failed: {e}")
            return {"error": str(e)}
            
    def get_tools(self) -> List[Tool]:
        """Get LangChain-compatible tools for the agent.
        
        Returns:
            List of Tool objects for LangChain integration
        """
        tools = []
        
        # Wallet tools
        tools.append(Tool(
            name="get_balance",
            description="Get balance of a token in the wallet",
            func=lambda asset: asyncio.run(self.get_balance(asset))
        ))
        
        tools.append(Tool(
            name="get_wallet_address", 
            description="Get the wallet address",
            func=lambda: self.address
        ))
        
        # Trading tools
        tools.append(Tool(
            name="swap_tokens",
            description="Swap one token for another on Aerodrome",
            func=lambda args: asyncio.run(self.swap(**json.loads(args)))
        ))
        
        # Liquidity tools
        tools.append(Tool(
            name="add_liquidity",
            description="Add liquidity to an Aerodrome pool",
            func=lambda args: asyncio.run(self.add_liquidity(**json.loads(args)))
        ))
        
        tools.append(Tool(
            name="remove_liquidity",
            description="Remove liquidity from an Aerodrome pool",
            func=lambda args: asyncio.run(self.remove_liquidity(**json.loads(args)))
        ))
        
        tools.append(Tool(
            name="claim_rewards",
            description="Claim pending rewards from a pool",
            func=lambda pool: asyncio.run(self.claim_rewards(pool))
        ))
        
        # Natural language tool for complex operations
        tools.append(Tool(
            name="execute_defi_action",
            description="Execute any DeFi action using natural language",
            func=lambda prompt: asyncio.run(self.agent.run(prompt))
        ))
        
        return tools
        
    async def execute_natural_language(self, prompt: str) -> Dict[str, Any]:
        """Execute any blockchain operation using natural language.
        
        Args:
            prompt: Natural language instruction
            
        Returns:
            Execution result
        """
        try:
            result = await self.agent.run(prompt)
            return result
        except Exception as e:
            logger.error(f"Natural language execution failed: {e}")
            return {"error": str(e)}
            
    async def simulate_transaction(self, prompt: str) -> Dict[str, Any]:
        """Simulate a transaction before execution.
        
        Args:
            prompt: Natural language instruction
            
        Returns:
            Simulation result
        """
        try:
            # Add simulation flag to prompt
            simulation_prompt = f"SIMULATE ONLY: {prompt}"
            result = await self.agent.run(simulation_prompt)
            return result
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {"error": str(e)}


# Convenience function for testing
async def test_agentkit():
    """Test the AgentKit integration."""
    agent = AthenaAgentKit()
    await agent.initialize()
    
    print(f"Wallet address: {agent.address}")
    
    # Test balance
    eth_balance = await agent.get_balance("eth")
    print(f"ETH balance: {eth_balance}")
    
    # Test natural language
    result = await agent.execute_natural_language(
        "What is my current USDC balance?"
    )
    print(f"Natural language result: {result}")
    

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agentkit())