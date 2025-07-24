# Athena AI - System Architecture

## Overview

Athena is designed as a modular, event-driven system that operates autonomously 24/7. The architecture emphasizes learning, scalability, and reliability.

## System Components

### 1. Core Agent (LangGraph)

The brain of Athena, implemented as a state machine with observation mode and five primary states:

```python
StateMachine:
  [Observation Mode: 3 days]
        ↓
  OBSERVE → ANALYZE → DECIDE → EXECUTE → LEARN
     ↑                                      ↓
     └──────────────────────────────────────┘
```

**State Descriptions:**

- **OBSERVE**: Collects data from blockchain, monitors positions, tracks gas prices
- **ANALYZE**: Processes observations with historical context, identifies patterns
- **DECIDE**: Evaluates opportunities, calculates risk/reward, selects strategies
- **EXECUTE**: Performs on-chain actions via CDP SDK (disabled during observation)
- **LEARN**: Updates memory with outcomes, refines strategies

**Observation Mode:**
- First 3 days: Pattern learning without trading
- Builds confidence in market patterns
- Transitions to full trading after observation period

### 2. Enhanced Memory System (v2.0)

Comprehensive memory architecture with tiered storage and pool profiling:

```
Tiered Memory Architecture
├── L1: Raw Observations (All significant pools)
│   ├── APR >= 20% pools
│   ├── Volume >= $100k pools
│   └── Imbalanced pools (ratio deviation > 100%)
├── L2: Pool Profiles (Individual behavior tracking)
│   ├── Historical ranges (APR, TVL, volume)
│   ├── Time patterns (hourly, daily)
│   └── Behavioral metrics (volatility, correlations)
└── L3: Pattern Correlations (Cross-pool insights)
    ├── Inter-pool relationships
    ├── Gas price correlations
    └── Predictive patterns

Mem0 Cloud API (Primary)
├── Vector-based semantic search
├── Pool-specific queries
├── Cross-pool correlations
└── Confidence scoring
    ↓ (persistence)
Firestore Collections
├── agent_state/         # Current operational state
├── cycles/              # Reasoning cycle history
├── positions/           # Active trading positions
├── performance/         # Aggregated metrics
├── observed_patterns/   # Discovered patterns
├── pattern_confidence/  # Pattern success tracking
├── observation_metrics/ # Learning phase metrics
├── pool_profiles/       # Individual pool behaviors (NEW)
├── pool_metrics/        # Time-series pool data (NEW)
└── pattern_correlations/# Cross-pool patterns (NEW)
```

**Enhanced Memory Operations:**

```python
# Store ALL significant pools (not just top performers)
for pool in scanned_pools:
    if pool["apr"] >= 20 or pool["volume"] >= 100000:
        memory.add(
            category="pool_behavior",
            content=f"Pool {pool['pair']}: APR {pool['apr']}%",
            metadata={
                "pool_address": pool["address"],
                "apr": pool["apr"],
                "tvl": pool["tvl"],
                "timestamp": pool["timestamp"]
            }
        )

# Pool-specific queries
pool_memories = await memory.recall_pool_memories(
    pool_pair="WETH/USDC",
    time_window_hours=24
)

# Cross-pool correlations
correlations = await memory.get_cross_pool_correlations()

# Pool profile predictions
predictions = pool_profiles.predict_opportunities(
    timestamp=datetime.utcnow() + timedelta(hours=1)
)
```

**Memory Storage Improvements:**
- **v1.0**: 2 memories per scan (top APR, top volume)
- **v2.0**: 10-15 memories per scan (all significant activity)
- **Result**: 5-7x more data for pattern recognition

### 3. CDP Integration Layer

Handles all blockchain interactions through CDP SDK v1.24.0:

```python
CDPClient
├── Wallet Management
│   ├── create_wallet()     # Ed25519 support
│   ├── get_balance()
│   └── estimate_gas()
├── Aerodrome Operations  
│   ├── swap_tokens()
│   ├── add_liquidity()
│   ├── stake_in_gauge()
│   ├── vote_for_pools()
│   └── claim_rewards()
├── Blockchain RPC Reader   # NEW
│   ├── read_pool_reserves() # Direct storage reading
│   ├── get_pool_data()     # Aerodrome V2 support
│   └── fallback_reading()  # Storage slot access
└── Monitoring
    ├── watch_position()
    ├── track_tx_status()
    └── get_pool_info()
```

### 4. Data Collection Pipeline

Continuous monitoring of blockchain state:

```
Collectors → Pub/Sub → Processing → Storage
    ↓           ↓          ↓           ↓
GasMonitor  EventBus   Analyzer   Firestore
PoolScanner
PriceOracle
```

**Collector Types:**

- **GasMonitor**: Tracks gas prices every 30 seconds
- **PoolScanner**: Monitors Aerodrome pools for opportunities
- **PriceOracle**: Maintains accurate price feeds
- **EventMonitor**: Listens for important on-chain events

### 5. API Layer (FastAPI)

RESTful API with WebSocket support:

```
FastAPI Application
├── Health & Monitoring
│   ├── GET /health
│   └── GET /metrics
├── Performance
│   ├── GET /performance/{period}
│   └── GET /profits/breakdown
├── Positions
│   ├── GET /positions/current
│   └── GET /positions/history
├── Strategies
│   ├── GET /strategies/active
│   ├── POST /strategies/override
│   └── GET /strategies/performance
├── Memory
│   ├── GET /memories/recent
│   └── GET /memories/search
└── Real-time
    └── WS /live
```

### 6. Infrastructure (Google Cloud)

```
Google Cloud Platform
├── Cloud Run (Always-on)
│   ├── Agent Container
│   ├── API Container
│   └── Collector Container
├── Firestore
│   ├── memories/
│   ├── positions/
│   ├── transactions/
│   └── strategies/
├── Pub/Sub Topics
│   ├── market-events
│   ├── execution-commands
│   └── learning-updates
├── Secret Manager
│   ├── API Keys
│   ├── RPC Endpoints
│   └── Wallet Seeds
└── Cloud Scheduler
    ├── Claim Rewards (4h)
    ├── Rebalance Check (1h)
    └── Memory Cleanup (24h)
```

## Data Flow

### 1. Observation Flow
```
Blockchain → Collectors → Pub/Sub → Agent.OBSERVE → Memory
```

### 2. Decision Flow
```
Agent.ANALYZE → Memory.search() → Agent.DECIDE → Execution Queue
```

### 3. Execution Flow
```
Execution Queue → CDP SDK → Blockchain → Result → Agent.LEARN
```

### 4. Learning Flow
```
Result + Context → Pattern Extraction → Memory.update() → Strategy Refinement
```

## Key Design Principles

### 1. Event-Driven Architecture
- All components communicate via events
- Enables parallel processing
- Improves fault tolerance

### 2. Stateless Services
- Agent state persisted in Firestore
- Enables horizontal scaling
- Simplifies deployment

### 3. Observability First
- Every decision traced in LangSmith
- Comprehensive logging
- Real-time monitoring

### 4. Fail-Safe Mechanisms
- Transaction simulation before execution
- Risk limits enforced at multiple levels
- Automatic position unwinding if needed

## Security Architecture

### 1. Key Management
```
Google Secret Manager
├── CDP API Keys (rotated monthly)
├── Google AI Keys (Gemini 1.5 Flash)
└── RPC Endpoints (multiple for redundancy)
```

### 2. Access Control
```
IAM Roles
├── Agent: Read/Write to specific Firestore collections
├── API: Read-only access to data
└── Admin: Full access for maintenance
```

### 3. Transaction Security
- All transactions simulated first
- Slippage protection enforced
- Maximum position size limits
- Emergency shutdown capability

## Performance Optimization

### 1. Caching Strategy
```
Redis Cache
├── Recent gas prices (TTL: 30s)
├── Pool data (TTL: 60s)
├── Price feeds (TTL: 10s)
└── Agent decisions (TTL: 5m)
```

### 2. Batch Operations
- Claim all rewards in single transaction
- Batch memory updates every 100 operations
- Aggregate similar swaps when possible

### 3. Parallel Processing
- Multiple collectors run concurrently
- Strategy evaluation parallelized
- Independent execution threads

## Monitoring & Alerting

### 1. Key Metrics
```
Business Metrics
├── Total Profit (24h, 7d, 30d)
├── Win Rate by Strategy
├── Active Position Value
└── Learning Rate

System Metrics
├── Agent Uptime
├── Transaction Success Rate
├── API Response Time
└── Memory Usage
```

### 2. Alert Conditions
- Profit below threshold
- High failure rate
- System errors
- Unusual market conditions

## Development Workflow

### 1. Local Development
```bash
# Run agent locally
python main.py --mode=development

# Test specific strategy
python -m pytest tests/strategies/test_arbitrage.py

# Simulate market conditions
python scripts/market_simulator.py
```

### 2. Testing Strategy
```
Unit Tests → Integration Tests → Simulation → Paper Trading → Production
```

### 3. Deployment Pipeline
```
Code Push → Cloud Build → Run Tests → Deploy Staging → Validate → Deploy Prod
```

## Future Enhancements

### Phase 2
- Multi-chain support (Ethereum, Arbitrum)
- Advanced ML models for prediction
- Social sentiment analysis

### Phase 3
- Decentralized deployment
- DAO governance integration
- Custom strategy marketplace

## Related Documentation

- [Database Architecture](DATABASE_ARCHITECTURE.md) - Detailed memory and storage design
- [API Documentation](API.md) - Complete endpoint reference
- [Deployment Guide](DEPLOYMENT.md) - Production deployment procedures

---

This architecture is designed to be modular, scalable, and continuously improving. Each component can be upgraded independently without affecting the system's overall operation.