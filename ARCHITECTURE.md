# Athena AI - System Architecture

## Overview

Athena AI is an autonomous DeFi agent that learns from market patterns and manages liquidity positions on the Base blockchain. The system combines memory-driven intelligence with AI-native blockchain integration through QuickNode MCP and Coinbase AgentKit, enabling 24/7 autonomous operation with continuous learning and optimization through natural language interfaces.

## Core Philosophy

- **Memory-First**: Every decision is informed by past experiences and learned patterns
- **AI-Native Integration**: Natural language interfaces replace complex blockchain code
- **Continuous Learning**: The agent improves with every cycle, discovering new patterns
- **Risk-Aware**: Conservative approach with pattern validation before trading
- **Minimal Complexity**: 90% less code through intelligent tool selection

## System Components

### 1. Core Agent (LangGraph State Machine)

The brain of Athena, implemented as an enhanced state machine with nine distinct states:

```python
StateMachine:
  [Observation Mode: 3 days]
        ↓
  OBSERVE → REMEMBER → ANALYZE → THEORIZE → STRATEGIZE → DECIDE → EXECUTE → LEARN → REFLECT
     ↑                                                                                    ↓
     └────────────────────────────────────────────────────────────────────────────────┘
```

**Enhanced State Descriptions:**

- **OBSERVE**: Collects real-time market data, gas prices, pool states via CDP and QuickNode
- **REMEMBER**: Retrieves relevant memories and patterns from Mem0 vector database
- **ANALYZE**: Processes observations with historical context using Gemini 1.5 Flash
- **THEORIZE**: Forms hypotheses about market behavior and opportunities
- **STRATEGIZE**: Plans rebalancing and compound timing using SmartRebalancer
- **DECIDE**: Evaluates all options and selects best actions based on confidence
- **EXECUTE**: Performs on-chain transactions via CDP SDK (disabled during observation)
- **LEARN**: Stores outcomes and extracts patterns for future use
- **REFLECT**: Self-evaluates performance and adjusts emotional state

**Observation Mode:**
- First 3 days: Pattern discovery without trading
- Stores 10-15 memories per cycle (5-7x more than v1)
- Builds confidence scores for discovered patterns
- Transitions to trading with high-confidence patterns

### 2. Memory System (Mem0 + Firestore)

Advanced memory architecture with vector search and pattern recognition:

```
Memory Architecture
├── Mem0 Cloud (Vector Database)
│   ├── Semantic search with embeddings
│   ├── Pool-specific memory retrieval
│   ├── Pattern matching and correlation
│   └── Confidence-weighted results
│
├── Memory Types
│   ├── OBSERVATION: Raw market data and events
│   ├── PATTERN: Discovered market behaviors
│   ├── STRATEGY: Successful strategy configurations
│   ├── OUTCOME: Trade results and performance
│   ├── LEARNING: Extracted insights and rules
│   └── ERROR: Failed attempts for learning
│
└── Firestore Collections
    ├── agent_state/         # Current operational state
    ├── cycles/              # Complete reasoning cycles
    ├── positions/           # Active LP positions
    ├── pool_profiles/       # Individual pool behavior models
    ├── pattern_confidence/  # Pattern validation tracking
    └── rebalance_history/   # Historical rebalancing data
```

**Key Memory Categories:**

```python
MEMORY_CATEGORIES = [
    # Market Intelligence
    "market_pattern",           # General market trends
    "pool_behavior",           # Individual pool characteristics
    "cross_pool_correlation",  # Inter-pool relationships
    
    # Optimization Patterns
    "apr_degradation_patterns",    # How yields decay over time
    "gas_optimization_windows",    # Best transaction times
    "compound_roi_patterns",       # Optimal compound frequency
    "pool_lifecycle_patterns",     # New pool behavior
    
    # Performance Tracking
    "rebalance_success_metrics",   # Rebalancing outcomes
    "strategy_performance",        # Strategy effectiveness
    "profit_source",              # Where profits come from
]
```

**Memory Operations:**

```python
# Intelligent memory storage with metadata
await memory.remember(
    content="WETH/USDC pool APR dropped from 45% to 32% after TVL exceeded $10M",
    memory_type=MemoryType.PATTERN,
    category="apr_degradation_patterns",
    metadata={
        "pool": "WETH/USDC",
        "apr_before": 45,
        "apr_after": 32,
        "tvl_threshold": 10000000,
        "time_elapsed_hours": 48
    },
    confidence=0.85
)

# Context-aware recall
relevant_memories = await memory.recall(
    query="optimal rebalancing time for high APR pools",
    category="rebalance_timing",
    min_confidence=0.7,
    limit=10
)
```

### 3. Smart Rebalancer

Memory-driven position optimization system:

```python
SmartRebalancer
├── Position Analysis
│   ├── analyze_positions()      # Evaluate all positions
│   ├── predict_apr()           # ML-based APR prediction
│   └── should_rebalance()      # Pattern-based decisions
│
├── Opportunity Discovery
│   ├── find_better_pool()      # Search for upgrades
│   ├── score_opportunity()     # Rank by patterns
│   └── analyze_profitability() # Cost/benefit analysis
│
├── Timing Optimization
│   ├── predict_optimal_gas()   # Gas pattern analysis
│   ├── optimize_compound()     # Compound timing
│   └── estimate_best_time()    # Transaction scheduling
│
└── Execution
    ├── execute_rebalance()     # Multi-step rebalancing
    ├── compound_rewards()      # Reinvest earnings
    └── learn_from_outcomes()   # Update patterns
```

**Key Features:**
- APR prediction using historical patterns
- Gas cost optimization with learned windows
- Risk-adjusted rebalancing decisions
- Continuous learning from outcomes

### 4. Blockchain Integration

**Coinbase AgentKit (Transaction Execution):**
```python
AthenaAgentKit
├── AI-Native Operations
│   ├── Natural language execution
│   ├── Automatic error handling
│   └── Transaction simulation
│
├── DeFi Operations
│   ├── swap()              # Natural language swaps
│   ├── add_liquidity()     # Smart LP provision
│   ├── remove_liquidity()  # Easy LP withdrawal
│   └── claim_rewards()     # Auto-compound capable
│
└── Wallet Management
    ├── Uses existing CDP credentials
    ├── Secure non-custodial wallet
    └── LangChain tool integration
```

**QuickNode MCP (Data & Analytics):**
```python
QuickNodeMCP
├── Natural Language Queries
│   ├── "Find high APR pools on Aerodrome"
│   ├── "Analyze rebalancing opportunities"
│   └── "Get optimal gas timing"
│
├── Pool Analytics
│   ├── get_aerodrome_pools()     # Filtered pool data
│   ├── analyze_pool()            # Deep pool analysis
│   └── find_arbitrage()          # Opportunity detection
│
└── Market Intelligence
    ├── get_market_overview()      # Comprehensive stats
    ├── optimize_gas_timing()      # Gas predictions
    └── analyze_rebalance()        # Position optimization
```

### 5. Data Collection Pipeline

Streamlined with natural language interfaces:

```
Data Sources
├── QuickNode MCP (All Data)
│   ├── Natural language queries
│   ├── Real-time pool analytics
│   ├── Gas optimization insights
│   └── Market intelligence
│
├── AgentKit (Execution Only)
│   ├── Transaction execution
│   ├── Wallet operations
│   └── Position management
│
└── Collectors (Simplified)
    ├── PoolScanner
    │   ├── Single MCP query replaces complex logic
    │   ├── Automatic categorization
    │   └── Memory formation
    │
    └── Market Monitor
        ├── Comprehensive overview
        ├── Pattern detection
        └── Opportunity alerts
```

**Collection Strategy:**
- Focus on actionable data (APR > 20%, Volume > $100k)
- Store patterns, not just snapshots
- Correlate across multiple data points

### 6. API Layer (FastAPI)

RESTful API with real-time WebSocket support:

```
API Endpoints
├── Monitoring
│   ├── GET /health              # System status
│   ├── GET /performance/{period} # Profit metrics
│   └── GET /metrics             # Detailed stats
│
├── Intelligence
│   ├── GET /observations        # Recent market data
│   ├── GET /theories           # Current hypotheses
│   ├── GET /memories/recent    # Latest learnings
│   └── GET /patterns           # Discovered patterns
│
├── Portfolio
│   ├── GET /positions          # Current holdings
│   ├── GET /rebalance/recommendations
│   └── GET /compound/status
│
├── Control
│   ├── POST /strategies/override  # Emergency control
│   └── POST /pause              # Pause trading
│
└── Real-time
    └── WS /live                 # Live updates
        ├── Emotional state
        ├── Current thinking
        ├── Performance updates
        └── Action notifications
```

### 7. Infrastructure (Google Cloud Platform)

```
GCP Architecture
├── Cloud Run Services
│   ├── athena-agent         # Main agent (always-on)
│   ├── athena-api          # FastAPI backend
│   └── athena-dashboard    # React frontend
│
├── Firestore Database
│   ├── Collections
│   │   ├── agent_state/     # Operational state
│   │   ├── pool_profiles/   # Pool behavior models
│   │   ├── patterns/        # Discovered patterns
│   │   └── performance/     # Historical metrics
│   │
│   └── Indexes
│       ├── timestamp + type
│       ├── pool + confidence
│       └── category + score
│
├── Secret Manager
│   ├── cdp-api-key         # CDP authentication
│   ├── cdp-wallet-secret   # Wallet seed (encrypted)
│   ├── google-api-key      # Gemini AI access
│   ├── mem0-api-key        # Memory service
│   ├── quicknode-api-key   # QuickNode access
│   └── langsmith-api-key   # Observability
│
└── Monitoring
    ├── Cloud Logging       # Structured logs
    ├── Cloud Monitoring    # Metrics & alerts
    └── LangSmith          # LLM observability
```

## Data Flow Architecture

### 1. Observation & Memory Flow
```
Market Data Sources
    ↓
[QuickNode API + CDP SDK]
    ↓
Agent.OBSERVE (Raw data collection)
    ↓
Agent.REMEMBER (Context retrieval from Mem0)
    ↓
Pattern Recognition & Storage
    ↓
[Mem0 Vector DB + Firestore]
```

### 2. Intelligence & Decision Flow
```
Observations + Memories
    ↓
Agent.ANALYZE (Gemini 1.5 Flash)
    ↓
Agent.THEORIZE (Hypothesis formation)
    ↓
Agent.STRATEGIZE (SmartRebalancer)
    ↓
Agent.DECIDE (Confidence-weighted selection)
    ↓
Execution Queue
```

### 3. Execution & Learning Flow
```
Execution Queue
    ↓
[CDP SDK Transaction]
    ↓
Blockchain Confirmation
    ↓
Agent.LEARN (Outcome analysis)
    ↓
Memory Update (Success/Failure patterns)
    ↓
Agent.REFLECT (Strategy refinement)
```

### 4. Rebalancing Decision Flow
```
Current Positions
    ↓
SmartRebalancer.analyze_positions()
    ↓
Memory.recall(pool patterns)
    ↓
APR Prediction & Opportunity Scoring
    ↓
Gas Optimization Timing
    ↓
Profitability Analysis
    ↓
Execution or Wait Decision
```

## Key Design Principles

### 1. Memory-Driven Intelligence
- Every decision informed by past experiences
- Pattern recognition over hard-coded rules
- Continuous learning and adaptation
- Confidence-based action selection

### 2. Simplified Integration
- QuickNode API for complex blockchain data
- CDP SDK for secure transaction execution
- Minimal RPC calls, maximum reliability
- Focus on business logic, not infrastructure

### 3. Observable & Debuggable
- LangSmith integration for LLM tracing
- Structured logging at every state
- Real-time monitoring via API/WebSocket
- Clear decision reasoning in logs

### 4. Risk Management
- Conservative observation period before trading
- Pattern confidence thresholds
- Gas cost optimization
- Position size limits
- Slippage protection

## Security Architecture

### 1. Key Management
```
Google Secret Manager (Encrypted at rest)
├── Authentication Keys
│   ├── CDP API credentials
│   ├── QuickNode API key
│   ├── Google AI API key
│   └── Mem0 API key
│
├── Wallet Security
│   ├── CDP wallet seed (never exposed)
│   ├── Encrypted with KMS
│   └── Auto-rotation capability
│
└── Access Patterns
    ├── Runtime-only access
    ├── No local key storage
    └── Audit logging enabled
```

### 2. Transaction Security
- CDP SDK handles key management
- Transaction simulation before execution
- Slippage limits enforced (0.5% default)
- Gas price sanity checks
- Maximum position size limits ($1000)
- Rate limiting on executions

### 3. Access Control
```
IAM Configuration
├── Service Accounts
│   ├── agent-sa: Firestore write, Secrets read
│   ├── api-sa: Firestore read, metrics write
│   └── dashboard-sa: Read-only access
│
└── API Security
    ├── CORS configuration
    ├── Rate limiting
    └── Optional API key auth
```

## Performance Optimization

### 1. Memory Optimization
```
Mem0 Strategies
├── Semantic search for relevance
├── Category-based filtering
├── Confidence thresholds
├── Time-window queries
└── Limited metadata size

Firestore Optimization
├── Composite indexes for common queries
├── Batch writes for efficiency
├── TTL on old observations
└── Aggregated metrics collection
```

### 2. API Response Caching
- QuickNode responses: 30s cache
- Pool analytics: 60s cache
- Gas prices: 30s cache
- Pattern confidence: 5m cache

### 3. Execution Optimization
- Gas price prediction for timing
- Batched reward claims
- Multi-hop swap routing
- Parallel memory queries

## Monitoring & Observability

### 1. Business Metrics
```
Performance Tracking
├── Financial
│   ├── Total profit (USD)
│   ├── APR achieved
│   ├── Gas costs
│   └── Position values
│
├── Operational
│   ├── Rebalance frequency
│   ├── Compound efficiency
│   ├── Pattern accuracy
│   └── Memory utilization
│
└── Learning
    ├── Patterns discovered
    ├── Confidence improvements
    ├── Strategy adaptations
    └── Error recoveries
```

### 2. System Monitoring
```
Health Checks
├── Service availability
├── API response times
├── Memory query latency
├── Blockchain connectivity
└── Error rates

LangSmith Tracing
├── Full reasoning traces
├── Decision pathways
├── LLM performance
└── Token usage
```

### 3. Alerting Rules
- Agent offline > 5 minutes
- Profit < expected for 24h
- Pattern confidence dropping
- High transaction failures
- Memory storage errors

## Development & Deployment

### 1. Local Development
```bash
# Setup environment
cp .env.example .env
# Add API keys to .env

# Run agent locally
python run.py

# Run with dashboard
python run.py &
cd frontend && npm run dev

# Monitor logs
tail -f athena.log
```

### 2. Testing Strategy
```
Development Flow
├── Local testing with mock data
├── Testnet deployment (Base Sepolia)
├── Observation mode (3 days)
├── Limited trading (small positions)
└── Full autonomous operation
```

### 3. Deployment Pipeline
```
GitHub Push
    ↓
Cloud Build Trigger
    ↓
Docker Image Build
    ↓
Automated Tests
    ↓
Deploy to Cloud Run
    ↓
Health Check Verification
    ↓
Production Active
```

### 4. Configuration Management
```
.env (Local Development)
    ↓
Google Secret Manager (Production)
    ↓
Runtime Environment Variables
    ↓
Application Configuration
```

## System Evolution

### Current State (v2.0)
- ✅ Enhanced state machine with 9 states
- ✅ Smart rebalancing with memory
- ✅ QuickNode integration
- ✅ Pattern-based learning
- ✅ Observation mode
- ✅ Gas optimization

### Near-term Enhancements
- Advanced ML for APR prediction
- Cross-pool arbitrage detection
- Impermanent loss prediction
- Multi-position portfolio optimization
- Social sentiment integration

### Long-term Vision
- Multi-chain deployment (Ethereum, Arbitrum)
- Decentralized memory network
- DAO governance for strategies
- User-specific agent instances
- Strategy marketplace

## Architecture Summary

Athena AI represents a new paradigm in DeFi automation:

1. **Learning-First**: Unlike traditional bots, Athena learns and improves
2. **Memory-Driven**: Decisions based on experience, not just current data
3. **Simplified Stack**: QuickNode APIs eliminate blockchain complexity
4. **Risk-Aware**: Conservative approach with pattern validation
5. **Fully Autonomous**: 24/7 operation without human intervention

The architecture prioritizes:
- **Reliability** over complexity
- **Learning** over hard-coded rules
- **Patterns** over point-in-time decisions
- **Safety** over aggressive returns

## Related Documentation

- [Quick Start Guide](QUICK_START.md) - Get running in 5 minutes
- [API Documentation](API.md) - Complete endpoint reference
- [Memory System](MEMORY_SYSTEM.md) - Deep dive into memory architecture
- [Deployment Guide](DEPLOYMENT.md) - Production deployment procedures
- [Database Architecture](DATABASE_ARCHITECTURE.md) - Storage design details

---

*Athena AI - Autonomous DeFi intelligence that learns, adapts, and profits.*