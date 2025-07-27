"""
Risk Management Framework for Athena AI

Implements portfolio-level risk management with dynamic position sizing,
circuit breakers, and gas manipulation protection.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
import numpy as np

from src.gcp.firestore_client import FirestoreClient
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreaker:
    """Circuit breaker configuration and state."""
    name: str
    threshold: float
    window: timedelta
    cooldown: timedelta
    triggers: deque = field(default_factory=deque)
    tripped_until: Optional[datetime] = None
    activation_count: int = 0
    
    def check(self, value: float) -> bool:
        """Check if circuit breaker should trip."""
        if self.is_tripped():
            return True
            
        # Add new trigger
        self.triggers.append({
            "value": value,
            "timestamp": datetime.utcnow()
        })
        
        # Remove old triggers outside window
        cutoff = datetime.utcnow() - self.window
        while self.triggers and self.triggers[0]["timestamp"] < cutoff:
            self.triggers.popleft()
            
        # Check if threshold exceeded
        if self._evaluate_triggers():
            self.trip()
            return True
            
        return False
        
    def _evaluate_triggers(self) -> bool:
        """Evaluate if triggers exceed threshold."""
        if not self.triggers:
            return False
            
        # For loss-based breakers, sum the losses
        if self.name in ["rapid_loss", "portfolio_drawdown"]:
            total_loss = sum(t["value"] for t in self.triggers)
            return total_loss <= self.threshold  # Negative threshold
            
        # For spike-based breakers, check multiplier
        elif self.name == "gas_manipulation":
            if len(self.triggers) < 2:
                return False
            recent_avg = np.mean([t["value"] for t in list(self.triggers)[-5:]])
            return self.triggers[-1]["value"] > recent_avg * self.threshold
            
        # For count-based breakers
        else:
            return len(self.triggers) >= self.threshold
            
    def trip(self):
        """Activate circuit breaker."""
        self.tripped_until = datetime.utcnow() + self.cooldown
        self.activation_count += 1
        logger.critical(
            f"Circuit breaker '{self.name}' tripped! "
            f"Cooldown until {self.tripped_until.isoformat()}"
        )
        
    def is_tripped(self) -> bool:
        """Check if currently tripped."""
        if self.tripped_until:
            if datetime.utcnow() < self.tripped_until:
                return True
            else:
                self.tripped_until = None
        return False
        
    def reset(self):
        """Manually reset circuit breaker."""
        self.triggers.clear()
        self.tripped_until = None
        logger.info(f"Circuit breaker '{self.name}' manually reset")


@dataclass
class PortfolioState:
    """Current portfolio state for risk calculations."""
    total_value: Decimal
    positions: List[Dict[str, Any]]
    token_exposures: Dict[str, Decimal]
    stable_pool_ratio: float
    correlation_matrix: Optional[np.ndarray] = None
    

class RiskManager:
    """
    Comprehensive risk management system.
    
    Features:
    - Dynamic position sizing
    - Portfolio-level risk controls
    - Circuit breaker system
    - Gas manipulation protection
    """
    
    def __init__(self, firestore: FirestoreClient):
        """Initialize risk manager."""
        self.firestore = firestore
        
        # Initialize circuit breakers
        self.circuit_breakers = {
            "rapid_loss": CircuitBreaker(
                "rapid_loss",
                settings.circuit_breaker_loss_threshold,
                timedelta(hours=1),
                timedelta(hours=settings.circuit_breaker_cooldown_hours)
            ),
            "portfolio_drawdown": CircuitBreaker(
                "portfolio_drawdown",
                settings.circuit_breaker_drawdown_threshold,
                timedelta(hours=24),
                timedelta(hours=4)
            ),
            "gas_manipulation": CircuitBreaker(
                "gas_manipulation",
                settings.circuit_breaker_gas_multiplier,
                timedelta(minutes=10),
                timedelta(minutes=30)
            ),
            "api_failures": CircuitBreaker(
                "api_failures",
                5,  # 5 failures
                timedelta(minutes=5),
                timedelta(minutes=10)
            ),
            "memory_corruption": CircuitBreaker(
                "memory_corruption",
                1,  # Any corruption
                timedelta(seconds=1),
                timedelta(hours=24)  # Manual reset required
            )
        }
        
        # Gas price history for manipulation detection
        self.gas_history = deque(maxlen=1000)
        
        # Portfolio tracking
        self.portfolio_state = None
        self.position_history = deque(maxlen=100)
        
    async def calculate_position_size(self, pool_data: Dict[str, Any],
                                    portfolio_state: PortfolioState) -> Decimal:
        """
        Calculate risk-adjusted position size.
        
        Args:
            pool_data: Pool information including volatility, age, etc.
            portfolio_state: Current portfolio state
            
        Returns:
            Risk-adjusted position size in USD
        """
        # Base calculation
        base_size = Decimal(str(settings.agent_max_position_size))
        
        # Calculate risk multipliers
        volatility_factor = self._calculate_volatility_discount(pool_data)
        portfolio_factor = self._portfolio_concentration_limit(portfolio_state)
        gas_factor = self._gas_cost_adjustment(pool_data)
        correlation_factor = self._correlation_penalty(pool_data, portfolio_state)
        
        # Apply all factors
        adjusted_size = (
            base_size * 
            Decimal(str(volatility_factor)) * 
            Decimal(str(portfolio_factor)) * 
            Decimal(str(gas_factor)) * 
            Decimal(str(correlation_factor))
        )
        
        # Apply hard limits
        max_position = portfolio_state.total_value * Decimal(str(settings.risk_portfolio_limit_percent))
        max_new_pool = Decimal(str(settings.risk_new_pool_limit_usd))
        
        # Check pool age
        pool_age_days = pool_data.get("age_days", 0)
        if pool_age_days < 7:
            adjusted_size = min(adjusted_size, max_new_pool)
            
        # Final size
        final_size = min(adjusted_size, max_position)
        
        logger.info(
            f"Position size calculation: base=${base_size}, "
            f"adjusted=${adjusted_size:.2f}, final=${final_size:.2f}"
        )
        
        return final_size
        
    def _calculate_volatility_discount(self, pool_data: Dict[str, Any]) -> float:
        """Reduce position size for volatile pools."""
        apr_variance = pool_data.get("apr_variance_30d", 0.5)
        
        if apr_variance < 0.1:  # Low volatility
            return 1.0
        elif apr_variance < 0.3:  # Medium volatility
            return 0.8
        elif apr_variance < 0.5:  # High volatility
            return 0.6
        else:  # Extreme volatility
            return 0.5
            
    def _portfolio_concentration_limit(self, portfolio_state: PortfolioState) -> float:
        """Prevent overconcentration in similar pools."""
        # Check token exposure
        if portfolio_state.token_exposures:
            max_token_exposure = max(portfolio_state.token_exposures.values())
            if max_token_exposure > Decimal(str(settings.risk_max_token_concentration)):
                return 0.3
                
        # Check pool type concentration
        stable_ratio = portfolio_state.stable_pool_ratio
        if stable_ratio > 0.3 or stable_ratio < 0.1:  # Imbalanced
            return 0.7
            
        return 1.0
        
    def _gas_cost_adjustment(self, pool_data: Dict[str, Any]) -> float:
        """Reduce size if gas costs are significant."""
        estimated_gas_cost = Decimal(str(pool_data.get("estimated_entry_gas", 10)))
        position_size = Decimal(str(pool_data.get("target_position_size", 1000)))
        
        if position_size == 0:
            return 1.0
            
        gas_percentage = estimated_gas_cost / position_size
        
        if gas_percentage > Decimal("0.05"):  # Gas > 5% of position
            return 0.5
        elif gas_percentage > Decimal("0.03"):  # Gas > 3%
            return 0.7
        elif gas_percentage > Decimal("0.01"):  # Gas > 1%
            return 0.9
            
        return 1.0
        
    def _correlation_penalty(self, pool_data: Dict[str, Any],
                           portfolio_state: PortfolioState) -> float:
        """Reduce size for correlated positions."""
        if not portfolio_state.positions:
            return 1.0
            
        # Check if pool is correlated with existing positions
        pool_tokens = set(pool_data.get("tokens", []))
        correlation_count = 0
        
        for position in portfolio_state.positions:
            position_tokens = set(position.get("tokens", []))
            if pool_tokens & position_tokens:  # Common tokens
                correlation_count += 1
                
        if correlation_count >= 3:
            return 0.5
        elif correlation_count >= 2:
            return 0.7
        elif correlation_count >= 1:
            return 0.85
            
        return 1.0
        
    async def check_circuit_breakers(self) -> Dict[str, bool]:
        """Check all circuit breakers and return status."""
        status = {}
        
        for name, breaker in self.circuit_breakers.items():
            status[name] = breaker.is_tripped()
            
        # Log any tripped breakers
        tripped = [name for name, is_tripped in status.items() if is_tripped]
        if tripped:
            logger.warning(f"Active circuit breakers: {', '.join(tripped)}")
            
        return status
        
    async def record_loss(self, loss_amount: Decimal, time_window: str = "1h"):
        """Record a loss event for circuit breaker evaluation."""
        loss_percentage = float(loss_amount / self.portfolio_state.total_value)
        
        if time_window == "1h":
            self.circuit_breakers["rapid_loss"].check(loss_percentage)
        elif time_window == "24h":
            self.circuit_breakers["portfolio_drawdown"].check(loss_percentage)
            
    async def validate_gas_price(self, current_gas: float) -> Dict[str, Any]:
        """Validate gas price for manipulation detection."""
        # Add to history
        self.gas_history.append({
            "price": current_gas,
            "timestamp": datetime.utcnow()
        })
        
        # Check circuit breaker
        if self.circuit_breakers["gas_manipulation"].check(current_gas):
            return {
                "action": "halt",
                "reason": "Circuit breaker tripped",
                "severity": "critical"
            }
            
        # Not enough history
        if len(self.gas_history) < 100:
            return {"action": "proceed", "reason": "insufficient data"}
            
        # Calculate statistics
        gas_prices = [h["price"] for h in self.gas_history]
        avg_gas = np.mean(gas_prices)
        std_gas = np.std(gas_prices)
        percentile_95 = np.percentile(gas_prices, 95)
        
        # Detection rules
        if current_gas > avg_gas * settings.gas_manipulation_threshold:
            return {
                "action": "wait",
                "reason": "gas spike detected",
                "severity": "high"
            }
            
        if current_gas > percentile_95:
            return {
                "action": "postpone",
                "reason": "gas above 95th percentile",
                "severity": "medium"
            }
            
        # Check for rapid changes
        if self._detect_rapid_gas_changes(current_gas):
            return {
                "action": "delay",
                "reason": "rapid gas changes",
                "severity": "low"
            }
            
        return {"action": "proceed", "reason": "gas normal"}
        
    def _detect_rapid_gas_changes(self, current_gas: float) -> bool:
        """Detect rapid gas price changes."""
        if len(self.gas_history) < 10:
            return False
            
        recent_prices = [h["price"] for h in list(self.gas_history)[-10:]]
        price_changes = np.diff(recent_prices)
        
        # Check for sudden spikes
        max_change = np.max(np.abs(price_changes))
        avg_price = np.mean(recent_prices)
        
        return max_change > avg_price * 0.5  # 50% change
        
    async def calculate_portfolio_risk_score(self) -> Tuple[str, float]:
        """
        Calculate overall portfolio risk score.
        
        Returns:
            Risk level (LOW/MEDIUM/HIGH/CRITICAL) and numeric score
        """
        if not self.portfolio_state:
            return "LOW", 0.0
            
        # Calculate risk components
        concentration_risk = self._calculate_concentration_risk()
        volatility_risk = self._calculate_volatility_risk()
        liquidity_risk = self._calculate_liquidity_risk()
        correlation_risk = self._calculate_correlation_risk()
        
        # Weighted sum
        total_risk = (
            concentration_risk * 0.3 +
            volatility_risk * 0.3 +
            liquidity_risk * 0.2 +
            correlation_risk * 0.2
        )
        
        # Determine level
        if total_risk > 75:
            level = "CRITICAL"
        elif total_risk > 50:
            level = "HIGH"
        elif total_risk > 25:
            level = "MEDIUM"
        else:
            level = "LOW"
            
        logger.info(
            f"Portfolio risk score: {total_risk:.1f} ({level}). "
            f"Components - Concentration: {concentration_risk:.1f}, "
            f"Volatility: {volatility_risk:.1f}, "
            f"Liquidity: {liquidity_risk:.1f}, "
            f"Correlation: {correlation_risk:.1f}"
        )
        
        return level, total_risk
        
    def _calculate_concentration_risk(self) -> float:
        """Calculate concentration risk (0-25)."""
        if not self.portfolio_state.positions:
            return 0.0
            
        # Check largest position
        position_values = [p["value_usd"] for p in self.portfolio_state.positions]
        max_position = max(position_values)
        max_concentration = max_position / self.portfolio_state.total_value
        
        # Score based on concentration
        if max_concentration > 0.3:
            return 25.0
        elif max_concentration > 0.2:
            return 20.0
        elif max_concentration > 0.15:
            return 15.0
        elif max_concentration > 0.1:
            return 10.0
        else:
            return 5.0
            
    def _calculate_volatility_risk(self) -> float:
        """Calculate volatility risk (0-25)."""
        if not self.portfolio_state.positions:
            return 0.0
            
        # Average volatility of positions
        volatilities = [p.get("apr_variance", 0.3) for p in self.portfolio_state.positions]
        avg_volatility = np.mean(volatilities) if volatilities else 0.0
        
        # Score based on volatility
        if avg_volatility > 0.5:
            return 25.0
        elif avg_volatility > 0.3:
            return 20.0
        elif avg_volatility > 0.2:
            return 15.0
        elif avg_volatility > 0.1:
            return 10.0
        else:
            return 5.0
            
    def _calculate_liquidity_risk(self) -> float:
        """Calculate liquidity risk (0-25)."""
        if not self.portfolio_state.positions:
            return 0.0
            
        # Check positions in low liquidity pools
        low_liquidity_value = sum(
            p["value_usd"] for p in self.portfolio_state.positions
            if p.get("tvl", float('inf')) < 100000
        )
        
        low_liquidity_ratio = low_liquidity_value / self.portfolio_state.total_value
        
        # Score based on low liquidity exposure
        if low_liquidity_ratio > 0.3:
            return 25.0
        elif low_liquidity_ratio > 0.2:
            return 20.0
        elif low_liquidity_ratio > 0.15:
            return 15.0
        elif low_liquidity_ratio > 0.1:
            return 10.0
        else:
            return 5.0
            
    def _calculate_correlation_risk(self) -> float:
        """Calculate correlation risk (0-25)."""
        if not self.portfolio_state.positions or len(self.portfolio_state.positions) < 2:
            return 0.0
            
        # Count correlated positions
        correlated_pairs = 0
        positions = self.portfolio_state.positions
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                tokens_i = set(positions[i].get("tokens", []))
                tokens_j = set(positions[j].get("tokens", []))
                if tokens_i & tokens_j:  # Common tokens
                    correlated_pairs += 1
                    
        # Score based on correlation
        correlation_ratio = correlated_pairs / (len(positions) * (len(positions) - 1) / 2)
        
        if correlation_ratio > 0.5:
            return 25.0
        elif correlation_ratio > 0.3:
            return 20.0
        elif correlation_ratio > 0.2:
            return 15.0
        elif correlation_ratio > 0.1:
            return 10.0
        else:
            return 5.0
            
    async def update_portfolio_state(self, positions: List[Dict[str, Any]],
                                   total_value: Decimal):
        """Update the current portfolio state."""
        # Calculate token exposures
        token_exposures = defaultdict(Decimal)
        stable_value = Decimal("0")
        
        for position in positions:
            # Update token exposures
            for token in position.get("tokens", []):
                token_exposures[token] += Decimal(str(position["value_usd"]))
                
            # Track stable pools
            if position.get("stable", False):
                stable_value += Decimal(str(position["value_usd"]))
                
        # Normalize exposures
        for token in token_exposures:
            token_exposures[token] = token_exposures[token] / total_value
            
        # Calculate stable ratio
        stable_ratio = float(stable_value / total_value) if total_value > 0 else 0.0
        
        # Update state
        self.portfolio_state = PortfolioState(
            total_value=total_value,
            positions=positions,
            token_exposures=dict(token_exposures),
            stable_pool_ratio=stable_ratio
        )
        
    async def emergency_response(self, emergency_type: str, context: Dict[str, Any]):
        """Execute automated emergency response."""
        logger.critical(f"Emergency response triggered: {emergency_type}")
        
        response = {
            "type": emergency_type,
            "timestamp": datetime.utcnow(),
            "actions": []
        }
        
        if emergency_type == "RAPID_LOSS":
            response["actions"].extend([
                "trading_paused",
                "risky_positions_identified",
                "alert_sent"
            ])
            
        elif emergency_type == "MEMORY_CORRUPTION":
            response["actions"].extend([
                "system_halted",
                "recovery_initiated",
                "manual_intervention_required"
            ])
            
        elif emergency_type == "GAS_MANIPULATION":
            response["actions"].extend([
                "transactions_postponed",
                "waiting_for_normal_gas",
                "cautious_mode_activated"
            ])
            
        # Store emergency event
        await self.firestore.set_document(
            "emergency_events",
            f"emergency_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            response
        )
        
        return response
        
    def get_status(self) -> Dict[str, Any]:
        """Get current risk management status."""
        breaker_status = {}
        for name, breaker in self.circuit_breakers.items():
            breaker_status[name] = {
                "tripped": breaker.is_tripped(),
                "activation_count": breaker.activation_count,
                "cooldown_until": breaker.tripped_until.isoformat() if breaker.tripped_until else None
            }
            
        return {
            "circuit_breakers": breaker_status,
            "portfolio_state": {
                "total_value": float(self.portfolio_state.total_value) if self.portfolio_state else 0,
                "position_count": len(self.portfolio_state.positions) if self.portfolio_state else 0,
                "risk_score": asyncio.run(self.calculate_portfolio_risk_score()) if self.portfolio_state else ("LOW", 0)
            },
            "gas_history_size": len(self.gas_history)
        }