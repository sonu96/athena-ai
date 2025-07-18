# Athena AI - System Architecture

## Overview

Athena is designed as a modular, event-driven system that operates autonomously 24/7. The architecture emphasizes learning, scalability, and reliability.

## System Components

### 1. Core Agent (LangGraph)

The brain of Athena, implemented as a state machine with five primary states:

```python
StateMachine:
  OBSERVE → ANALYZE → DECIDE → EXECUTE → LEARN
     ↑                                      ↓
     └──────────────────────────────────────┘
```

**State Descriptions:**

- **OBSERVE**: Collects data from blockchain, monitors positions, tracks gas prices
- **ANALYZE**: Processes observations with historical context, identifies patterns
- **DECIDE**: Evaluates opportunities, calculates risk/reward, selects strategies
- **EXECUTE**: Performs on-chain actions via CDP SDK
- **LEARN**: Updates memory with outcomes, refines strategies

### 2. Memory System (Mem0)

Hierarchical memory architecture with three levels:

```
Short-term Memory (Redis)
├── Current market state
├── Recent transactions
└── Active positions

Long-term Memory (Firestore)
├── Historical patterns
├── Strategy performance
└── Learned behaviors

Semantic Memory (Vector DB)
├── Market relationships
├── Protocol knowledge
└── Trading strategies
```

**Memory Operations:**

```python
# Store observation
memory.add(
    category="market_pattern",
    content="Gas drops 40% at 3 AM UTC",
    confidence=0.87,
    observations=150
)

# Retrieve relevant memories
context = memory.search(
    query="optimal gas timing",
    categories=["market_pattern", "execution_strategy"],
    limit=5
)
```

### 3. CDP Integration Layer

Handles all blockchain interactions through CDP SDK:

```python
CDPClient
├── Wallet Management
│   ├── create_wallet()
│   ├── get_balance()
│   └── estimate_gas()
├── Aerodrome Operations  
│   ├── swap_tokens()
│   ├── add_liquidity()
│   ├── stake_in_gauge()
│   ├── vote_for_pools()
│   └── claim_rewards()
└── Monitoring
    ├── watch_position()
    ├── track_tx_status()
    └── get_pool_data()
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
├── OpenAI Keys (monitored for usage)
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

---

This architecture is designed to be modular, scalable, and continuously improving. Each component can be upgraded independently without affecting the system's overall operation.