# Platform Knowledge System

## Overview

Athena now incorporates a sophisticated platform knowledge system that enables deep understanding of DeFi protocols from their documentation, whitepapers, and mechanics. This allows the agent to make informed decisions based on protocol design rather than just empirical observation.

## Architecture

### 1. Base Platform Interface

Located at `src/platforms/base_platform.py`, this abstract base class defines the interface that all platform implementations must follow:

```python
class BaseDeFiPlatform(ABC):
    async def load_knowledge_base() -> Dict[str, Any]
    async def get_platform_mechanics() -> Dict[str, Any]
    async def get_tokenomics() -> Dict[str, Any]
    async def calculate_pool_rewards() -> Dict[str, Decimal]
    async def find_opportunities() -> List[Dict[str, Any]]
    async def analyze_pool_dynamics() -> Dict[str, Any]
    async def get_platform_strategies() -> List[Dict[str, Any]]
    async def validate_action() -> Dict[str, Any]
```

### 2. Aerodrome Implementation

The Aerodrome platform implementation (`src/platforms/aerodrome/`) includes:

#### Knowledge Base (`knowledge_base.py`)
- Loads and manages platform documentation
- Provides concept explanations
- Stores strategies and risk factors
- Enables natural language queries

#### Tokenomics Model (`tokenomics.py`)
- Models AERO emission schedules
- Calculates veAERO voting power
- Predicts optimal lock durations
- Analyzes bribe efficiency

#### Platform Strategies (`strategies.py`)
- Implements platform-specific strategies
- Validates opportunities against knowledge
- Calculates optimal timing
- Risk-adjusted decision making

## Knowledge Documents

### Whitepaper Summary (`whitepaper_summary.md`)
- Executive summary of Aerodrome's ve(3,3) model
- Core innovations and differentiators
- Economic flywheel explanation
- Risk factors and mitigation

### Pool Mechanics (`pool_mechanics.json`)
```json
{
  "pool_types": {
    "stable": {
      "formula": "x³y + y³x = k",
      "fee": 0.0001,
      "use_cases": ["Stablecoin pairs", "Pegged assets"]
    },
    "volatile": {
      "formula": "x × y = k",
      "fee": 0.003,
      "use_cases": ["Uncorrelated pairs", "Volatile assets"]
    }
  }
}
```

### Gauge System (`gauge_system.json`)
- Voting mechanics and epoch schedules
- Reward distribution formulas
- Boost calculations
- Bribe ecosystem dynamics

### Tokenomics (`tokenomics.json`)
- Token supply and emission model
- veAERO mechanics
- Value accrual mechanisms
- Game theory analysis

### Strategies (`strategies.json`)
- Proven profitable strategies
- Risk levels and capital requirements
- Execution steps
- Expected returns

## Integration with Agent

### 1. Knowledge Loading
The agent loads platform knowledge on startup:
```python
# In agent initialization
self.platform = AerodromePlatform()
await self.platform.load_knowledge_base()
```

### 2. Enhanced Analysis
During the ANALYZE state, the agent queries platform knowledge:
```python
platform_concepts = await self.platform.explain_concept("gauge_system")
platform_risks = await self.platform.get_risk_factors()
```

### 3. Strategy Validation
Before executing decisions, actions are validated:
```python
validation = await self.platform.validate_action(
    "add_liquidity",
    {"pool_address": pool, "amount_a": 100, "amount_b": 100}
)
```

### 4. Opportunity Discovery
Platform knowledge enhances opportunity finding:
```python
opportunities = await self.platform.find_opportunities(
    min_apr=20,
    max_risk_score=0.7,
    capital_available=Decimal("5000")
)
```

## Benefits

### 1. Deep Understanding
- Agent understands WHY strategies work, not just THAT they work
- Can reason about tokenomics and incentive structures
- Predicts protocol behavior based on design

### 2. Better Decisions
- Validates actions against protocol rules
- Optimizes for platform-specific mechanics
- Exploits knowledge of bribes, votes, and emissions

### 3. Risk Awareness
- Understands platform-specific risks
- Knows mitigation strategies
- Can avoid common pitfalls

### 4. Future Extensibility
- Easy to add new platforms (Aave, Compound, etc.)
- Consistent interface for all protocols
- Knowledge can be updated without code changes

## Adding New Platforms

To add support for a new DeFi platform:

1. Create platform directory: `src/platforms/newplatform/`
2. Implement platform class extending `BaseDeFiPlatform`
3. Create knowledge documents in `src/platforms/knowledge/newplatform/`
4. Add platform-specific strategies and tokenomics
5. Update agent to load new platform when needed

Example structure:
```
src/platforms/aave/
├── __init__.py
├── platform.py          # Aave platform implementation
├── knowledge_base.py    # Aave-specific knowledge
├── tokenomics.py        # AAVE token model
└── strategies.py        # Aave strategies

src/platforms/knowledge/aave/
├── whitepaper_summary.md
├── lending_mechanics.json
├── tokenomics.json
└── strategies.json
```

## Memory Integration

Platform knowledge enhances memory formation:

```python
# Memories now include platform context
await self.memory.remember(
    content=f"Pool {pool} offering {apr}% APR. Sustainability: {sustainability_score}",
    memory_type=MemoryType.OBSERVATION,
    category="pool_analysis",
    metadata={
        "pool": pool,
        "apr": apr,
        "sustainability_score": sustainability_score,
        "emission_analysis": emission_analysis
    }
)
```

## Testing

Test platform knowledge with:
```bash
python test_platform_knowledge.py
```

This verifies:
- Knowledge loading
- Concept explanations
- Strategy recommendations
- Action validation
- Risk assessment

## Future Enhancements

### 1. Multi-Platform Arbitrage
- Cross-platform opportunity detection
- Unified liquidity management
- Protocol bridging strategies

### 2. Dynamic Knowledge Updates
- Real-time documentation parsing
- Community-sourced strategies
- Automated knowledge extraction

### 3. ML-Enhanced Understanding
- Pattern recognition in documentation
- Strategy success prediction
- Risk factor discovery

## Conclusion

The platform knowledge system transforms Athena from a reactive agent that learns through trial and error to a proactive agent that understands protocol mechanics deeply. This results in better decisions, lower risk, and higher returns.