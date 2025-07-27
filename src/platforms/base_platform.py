"""
Base DeFi Platform Interface

Abstract base class that defines the interface for all DeFi platform integrations.
This ensures consistent access patterns and enables platform-agnostic agent logic.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime


class BaseDeFiPlatform(ABC):
    """
    Abstract base class for DeFi platform integrations.
    
    Each platform implementation must provide methods for:
    - Understanding platform mechanics
    - Calculating yields and rewards
    - Identifying opportunities
    - Executing platform-specific strategies
    """
    
    def __init__(self, name: str):
        """Initialize platform with its name."""
        self.name = name
        self.knowledge_base = {}
        self.last_knowledge_update = None
        
    @abstractmethod
    async def load_knowledge_base(self) -> Dict[str, Any]:
        """
        Load platform-specific knowledge from documentation.
        
        Returns:
            Dictionary containing platform mechanics, tokenomics, strategies
        """
        pass
        
    @abstractmethod
    async def get_platform_mechanics(self) -> Dict[str, Any]:
        """
        Get detailed explanation of how the platform works.
        
        Returns:
            Dictionary with sections like:
            - liquidity_provision
            - reward_distribution
            - governance
            - risks
        """
        pass
        
    @abstractmethod
    async def get_tokenomics(self) -> Dict[str, Any]:
        """
        Get platform token economics and distribution mechanics.
        
        Returns:
            Dictionary containing:
            - token_supply
            - emission_schedule
            - distribution_rules
            - utility_functions
        """
        pass
        
    @abstractmethod
    async def calculate_pool_rewards(
        self,
        pool_address: str,
        liquidity_amount: Decimal,
        time_period_days: int = 1
    ) -> Dict[str, Decimal]:
        """
        Calculate expected rewards for providing liquidity.
        
        Args:
            pool_address: Address of the liquidity pool
            liquidity_amount: Amount of liquidity to provide (in USD)
            time_period_days: Time period for calculation
            
        Returns:
            Dictionary with reward breakdowns:
            - trading_fees
            - token_incentives
            - additional_rewards
            - total_apr
        """
        pass
        
    @abstractmethod
    async def find_opportunities(
        self,
        min_apr: float = 20,
        max_risk_score: float = 0.7,
        capital_available: Decimal = Decimal("1000")
    ) -> List[Dict[str, Any]]:
        """
        Find profitable opportunities on the platform.
        
        Args:
            min_apr: Minimum APR threshold
            max_risk_score: Maximum acceptable risk (0-1)
            capital_available: Available capital in USD
            
        Returns:
            List of opportunity dictionaries with:
            - pool_address
            - expected_apr
            - risk_score
            - required_capital
            - strategy_type
        """
        pass
        
    @abstractmethod
    async def analyze_pool_dynamics(self, pool_address: str) -> Dict[str, Any]:
        """
        Analyze a specific pool's behavior and characteristics.
        
        Args:
            pool_address: Pool to analyze
            
        Returns:
            Analysis including:
            - historical_performance
            - volatility_metrics
            - whale_concentration
            - sustainability_score
        """
        pass
        
    @abstractmethod
    async def get_platform_strategies(self) -> List[Dict[str, Any]]:
        """
        Get documented strategies for this platform.
        
        Returns:
            List of strategies with:
            - name
            - description
            - expected_return
            - risk_level
            - capital_requirements
            - execution_steps
        """
        pass
        
    @abstractmethod
    async def validate_action(
        self,
        action_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate if an action is valid according to platform rules.
        
        Args:
            action_type: Type of action (add_liquidity, stake, vote, etc.)
            parameters: Action parameters
            
        Returns:
            Validation result:
            - is_valid: bool
            - warnings: List of warnings
            - estimated_outcome: Expected results
        """
        pass
        
    async def explain_concept(self, concept: str) -> str:
        """
        Explain a platform-specific concept in simple terms.
        
        Args:
            concept: Concept to explain (e.g., "impermanent_loss", "gauge_voting")
            
        Returns:
            Human-readable explanation
        """
        if not self.knowledge_base:
            await self.load_knowledge_base()
            
        concepts = self.knowledge_base.get("concepts", {})
        return concepts.get(concept, f"No explanation found for '{concept}'")
        
    async def get_risk_factors(self) -> List[Dict[str, Any]]:
        """
        Get platform-specific risk factors.
        
        Returns:
            List of risks with severity and mitigation strategies
        """
        if not self.knowledge_base:
            await self.load_knowledge_base()
            
        return self.knowledge_base.get("risk_factors", [])
        
    async def query_knowledge(self, query: str) -> Dict[str, Any]:
        """
        Natural language query against platform knowledge base.
        
        Args:
            query: Natural language question about the platform
            
        Returns:
            Relevant knowledge and context
        """
        if not self.knowledge_base:
            await self.load_knowledge_base()
            
        # Simple keyword matching for now
        # Could be enhanced with vector search
        query_lower = query.lower()
        results = {}
        
        for category, content in self.knowledge_base.items():
            if isinstance(content, dict):
                for key, value in content.items():
                    if key.lower() in query_lower or (isinstance(value, str) and query_lower in value.lower()):
                        results[f"{category}.{key}"] = value
                        
        return results
        
    def get_platform_config(self) -> Dict[str, Any]:
        """
        Get platform-specific configuration parameters.
        
        Returns:
            Configuration including addresses, fees, limits
        """
        return {
            "name": self.name,
            "last_knowledge_update": self.last_knowledge_update,
            "supported_features": self._get_supported_features()
        }
        
    def _get_supported_features(self) -> List[str]:
        """Get list of supported features for this platform."""
        # Override in subclasses to specify features
        return ["liquidity_provision", "yield_farming"]