# Risk Management Framework

## Overview

Athena AI implements a comprehensive risk management system that protects against financial losses, system failures, and market manipulation. The framework operates at both position and portfolio levels with dynamic adjustments based on market conditions.

## Position Sizing Algorithm

### Dynamic Position Calculation

```python
def calculate_position_size(pool_data, portfolio_state):
    """
    Calculate risk-adjusted position size for a pool.
    
    Factors considered:
    - Pool volatility (APR variance)
    - Portfolio concentration
    - Gas costs relative to position
    - Pool age and liquidity depth
    - Correlation with existing positions
    """
    
    # Base calculation
    base_size = settings.agent_max_position_size  # Default: $1000
    
    # Risk multipliers (0.0 - 1.0)
    volatility_factor = calculate_volatility_discount(pool_data)
    portfolio_factor = portfolio_concentration_limit(portfolio_state)
    gas_factor = gas_cost_adjustment(pool_data)
    correlation_factor = correlation_penalty(pool_data, portfolio_state)
    
    # Apply all factors
    adjusted_size = (
        base_size * 
        volatility_factor * 
        portfolio_factor * 
        gas_factor * 
        correlation_factor
    )
    
    # Hard limits
    max_position = portfolio_state.total_value * 0.2  # 20% max per position
    max_new_pool = 500  # $500 for pools < 7 days old
    
    if pool_data.age_days < 7:
        return min(adjusted_size, max_new_pool)
    
    return min(adjusted_size, max_position)
```

### Risk Adjustment Factors

#### 1. Volatility Factor (0.5 - 1.0)
```python
def calculate_volatility_discount(pool_data):
    """Reduce position size for volatile pools"""
    apr_variance = pool_data.apr_variance_30d
    
    if apr_variance < 0.1:  # Low volatility
        return 1.0
    elif apr_variance < 0.3:  # Medium volatility
        return 0.8
    elif apr_variance < 0.5:  # High volatility
        return 0.6
    else:  # Extreme volatility
        return 0.5
```

#### 2. Portfolio Concentration (0.3 - 1.0)
```python
def portfolio_concentration_limit(portfolio_state):
    """Prevent overconcentration in similar pools"""
    # Check token exposure
    max_token_exposure = max(portfolio_state.token_exposures.values())
    if max_token_exposure > 0.4:  # 40% in single token
        return 0.3
    
    # Check pool type concentration
    stable_ratio = portfolio_state.stable_pool_ratio
    if stable_ratio > 0.3 or stable_ratio < 0.1:  # Imbalanced
        return 0.7
    
    return 1.0
```

#### 3. Gas Cost Factor (0.5 - 1.0)
```python
def gas_cost_adjustment(pool_data):
    """Reduce size if gas costs are significant"""
    estimated_gas_cost = pool_data.estimated_entry_gas
    position_size = pool_data.target_position_size
    
    gas_percentage = estimated_gas_cost / position_size
    
    if gas_percentage > 0.05:  # Gas > 5% of position
        return 0.5
    elif gas_percentage > 0.03:  # Gas > 3%
        return 0.7
    elif gas_percentage > 0.01:  # Gas > 1%
        return 0.9
    
    return 1.0
```

## Circuit Breakers

### Financial Circuit Breakers

| Trigger | Threshold | Time Window | Cooldown | Action |
|---------|-----------|-------------|----------|---------|
| Rapid Loss | -5% | 1 hour | 2 hours | Pause all trading |
| Portfolio Drawdown | -10% | 24 hours | 4 hours | Emergency positions only |
| Gas Spike | 3x average | 10 minutes | 30 minutes | Postpone non-critical |
| Slippage High | > 2% | 5 trades | 1 hour | Reduce trade sizes |

### System Circuit Breakers

| Trigger | Threshold | Detection | Cooldown | Action |
|---------|-----------|-----------|----------|---------|
| Memory Corruption | Any | Checksum fail | Manual | Full system halt |
| API Failures | 5 failures | 5 minutes | 10 minutes | Fallback mode |
| LLM Timeout | 3 timeouts | 5 minutes | 5 minutes | Rule-based decisions |
| Position Mismatch | > $1000 | On reconcile | Until fixed | Trading pause |

### Circuit Breaker Implementation

```python
class CircuitBreaker:
    def __init__(self, threshold, window, cooldown):
        self.threshold = threshold
        self.window = window
        self.cooldown = cooldown
        self.triggers = deque()
        self.tripped_until = None
    
    def check(self, value):
        """Check if circuit breaker should trip"""
        if self.is_tripped():
            return True
            
        # Add new trigger
        self.triggers.append({
            "value": value,
            "timestamp": datetime.utcnow()
        })
        
        # Remove old triggers
        cutoff = datetime.utcnow() - self.window
        self.triggers = deque(
            t for t in self.triggers 
            if t["timestamp"] > cutoff
        )
        
        # Check threshold
        if self._evaluate_triggers():
            self.trip()
            return True
            
        return False
    
    def trip(self):
        """Activate circuit breaker"""
        self.tripped_until = datetime.utcnow() + self.cooldown
        logger.critical(f"Circuit breaker tripped: {self.name}")
        
    def is_tripped(self):
        """Check if currently tripped"""
        if self.tripped_until:
            return datetime.utcnow() < self.tripped_until
        return False
```

## Portfolio Risk Limits

### Exposure Limits

```yaml
portfolio_limits:
  # Token concentration
  max_single_token_exposure: 50%
  max_stable_exposure: 30%
  max_volatile_exposure: 70%
  
  # Pool characteristics
  max_new_pool_exposure: 20%  # Pools < 7 days
  max_low_liquidity_exposure: 15%  # TVL < $100k
  
  # Correlation limits
  max_correlated_positions: 3  # Pools with > 0.7 correlation
  max_correlation_exposure: 40%  # Total in correlated pools
```

### Risk Scoring System

```python
def calculate_portfolio_risk_score():
    """
    Calculate overall portfolio risk (0-100)
    Lower is better
    """
    risk_components = {
        "concentration_risk": calculate_concentration_risk(),  # 0-25
        "volatility_risk": calculate_volatility_risk(),       # 0-25
        "liquidity_risk": calculate_liquidity_risk(),         # 0-25
        "correlation_risk": calculate_correlation_risk()      # 0-25
    }
    
    total_risk = sum(risk_components.values())
    
    # Risk thresholds
    if total_risk > 75:
        return "CRITICAL", total_risk
    elif total_risk > 50:
        return "HIGH", total_risk
    elif total_risk > 25:
        return "MEDIUM", total_risk
    else:
        return "LOW", total_risk
```

## Gas Protection

### Gas Manipulation Detection

```python
class GasProtection:
    def __init__(self):
        self.gas_history = deque(maxlen=1000)
        self.manipulation_threshold = 3.0  # 3x average
        
    def validate_gas_price(self, current_gas):
        """Detect potential gas manipulation"""
        # Calculate statistics
        if len(self.gas_history) < 100:
            return {"action": "proceed", "reason": "insufficient data"}
            
        avg_gas = np.mean(self.gas_history)
        std_gas = np.std(self.gas_history)
        percentile_95 = np.percentile(self.gas_history, 95)
        
        # Detection rules
        if current_gas > avg_gas * self.manipulation_threshold:
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
        if self._detect_rapid_changes(current_gas):
            return {
                "action": "delay",
                "reason": "rapid gas changes",
                "severity": "low"
            }
            
        return {"action": "proceed", "reason": "gas normal"}
```

### Gas Budget Management

```python
class GasBudget:
    def __init__(self, daily_limit_usd=50):
        self.daily_limit = daily_limit_usd
        self.spent_today = 0
        self.last_reset = datetime.utcnow().date()
        
    def can_spend(self, estimated_gas_usd):
        """Check if gas budget allows transaction"""
        self._reset_if_new_day()
        
        if self.spent_today + estimated_gas_usd > self.daily_limit:
            return False, f"Would exceed daily limit: ${self.daily_limit}"
            
        if estimated_gas_usd > self.daily_limit * 0.2:  # Single tx > 20%
            return False, "Single transaction too expensive"
            
        return True, "Within budget"
```

## Emergency Procedures

### Automated Emergency Response

```python
async def handle_emergency(emergency_type, context):
    """Automated emergency response system"""
    
    if emergency_type == "RAPID_LOSS":
        # 1. Pause all trading
        await trading_engine.pause()
        
        # 2. Close risky positions
        risky_positions = await identify_risky_positions()
        for position in risky_positions:
            await emergency_exit(position)
            
        # 3. Notify operators
        await send_alert("Rapid loss detected - trading paused")
        
    elif emergency_type == "MEMORY_CORRUPTION":
        # 1. Halt system
        await system.emergency_halt()
        
        # 2. Activate recovery
        await disaster_recovery.start_memory_recovery()
        
        # 3. Require manual intervention
        await send_critical_alert("Memory corruption - manual intervention required")
        
    elif emergency_type == "GAS_MANIPULATION":
        # 1. Postpone all transactions
        await transaction_queue.postpone_all()
        
        # 2. Wait for normal conditions
        await gas_monitor.wait_for_normal()
        
        # 3. Resume with caution
        await trading_engine.resume_cautious_mode()
```

### Manual Override Controls

```yaml
# API endpoints for emergency control
POST /risk/emergency-stop
  - Immediately halt all trading
  - Close all positions at market
  - Requires admin authentication

POST /risk/override-limits
  - Temporarily adjust risk limits
  - Requires reason and duration
  - Logged for audit

GET /risk/emergency-status
  - Current system state
  - Active circuit breakers
  - Recent emergency events
```

## Risk Monitoring

### Real-time Risk Metrics

```python
risk_metrics = {
    # Portfolio metrics
    "portfolio_var_95": calculate_var(confidence=0.95),
    "portfolio_risk_score": calculate_portfolio_risk_score(),
    "max_position_concentration": get_max_concentration(),
    
    # Market risk
    "gas_manipulation_score": gas_protection.get_manipulation_score(),
    "market_volatility": calculate_market_volatility(),
    
    # System risk
    "circuit_breakers_active": count_active_breakers(),
    "memory_health_score": memory_system.get_health_score(),
    "api_reliability": calculate_api_reliability()
}
```

### Risk Dashboard

The risk management system provides real-time visibility through:

1. **Portfolio Overview**
   - Current positions and exposures
   - Risk score with breakdown
   - Active circuit breakers

2. **Historical Analysis**
   - Loss events and recoveries
   - Circuit breaker activations
   - Risk score trends

3. **Predictive Alerts**
   - Approaching risk limits
   - Unusual market conditions
   - System health warnings

## Configuration

### Risk Parameters

```python
# config/settings.py additions
class RiskSettings(BaseSettings):
    # Position limits
    agent_max_position_size: float = 1000.0
    risk_portfolio_limit_percent: float = 0.2
    risk_new_pool_limit_usd: float = 500.0
    
    # Circuit breakers
    circuit_breaker_loss_threshold: float = -0.05
    circuit_breaker_drawdown_threshold: float = -0.10
    circuit_breaker_gas_multiplier: float = 3.0
    circuit_breaker_cooldown_hours: int = 2
    
    # Gas protection
    max_gas_price_gwei: float = 100.0
    gas_daily_budget_usd: float = 50.0
    gas_manipulation_threshold: float = 3.0
    
    # Portfolio limits
    max_token_concentration: float = 0.5
    max_correlation_exposure: float = 0.4
    max_new_pool_exposure: float = 0.2
```

## Best Practices

1. **Regular Reviews**
   - Weekly risk parameter review
   - Monthly circuit breaker test
   - Quarterly strategy assessment

2. **Conservative Defaults**
   - Start with small positions
   - Increase limits gradually
   - Monitor performance closely

3. **Emergency Preparedness**
   - Test recovery procedures monthly
   - Keep emergency contacts updated
   - Document all overrides

4. **Continuous Improvement**
   - Analyze all loss events
   - Update risk models
   - Learn from circuit breaker triggers