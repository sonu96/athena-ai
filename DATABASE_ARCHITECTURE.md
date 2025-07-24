# Database Architecture - Athena AI

## Overview

Athena AI employs a sophisticated multi-layered database architecture designed to support autonomous DeFi operations, pattern learning, and memory formation. The system combines vector-based semantic memory (Mem0), structured document storage (Google Firestore), and in-memory caching for optimal performance.

## Core Components

### 1. Mem0 - Semantic Memory System

**Purpose**: Long-term semantic memory storage with vector similarity search capabilities

**Implementation**: `src/agent/memory.py`

**Key Features**:
- Vector-based storage for semantic similarity search
- Supports both cloud (API-based) and local fallback modes
- Structured memory types with confidence scoring
- Reference linking between related memories

**Memory Types**:
```python
class MemoryType(str, Enum):
    OBSERVATION = "observation"    # Raw market observations
    PATTERN = "pattern"           # Discovered patterns
    STRATEGY = "strategy"         # Strategy configurations
    OUTCOME = "outcome"           # Execution results
    LEARNING = "learning"         # Derived insights
    ERROR = "error"              # Error tracking
```

**Memory Categories** (from `config/settings.py`):
- `observations`: Market state observations
- `pool_analysis`: Liquidity pool insights
- `gas_patterns`: Gas price patterns
- `strategies`: Trading strategies
- `decisions`: Decision history
- `emotions`: Emotional state tracking

**Storage Format**:
```python
MemoryEntry = {
    "id": "unique_id",
    "type": MemoryType,
    "category": str,
    "content": str,
    "metadata": {
        "confidence": float,
        "timestamp": datetime,
        "references": List[str],
        # Additional context-specific metadata
    }
}
```

### 2. Google Firestore - Structured Data

**Purpose**: Persistent structured storage for state, metrics, and operational data

**Implementation**: `src/gcp/firestore_client.py`

**Collections**:

#### `agent_state`
- Document: `current`
- Purpose: Current agent operational state
- Fields:
  - `cycle_number`: Current reasoning cycle
  - `emotions`: Emotional state metrics
  - `last_decision`: Most recent decision
  - `last_update`: Timestamp

#### `cycles`
- Documents: `cycle_{number}`
- Purpose: Historical reasoning cycle records
- Fields:
  - `cycle_number`: Cycle identifier
  - `state`: Complete state snapshot
  - `decisions`: Decisions made
  - `timestamp`: Execution time

#### `positions`
- Purpose: Active trading positions
- Fields:
  - `token_pair`: Trading pair
  - `type`: Position type (LP, stake, etc.)
  - `amount`: Position size
  - `entry_price`: Entry price
  - `created_at`: Creation timestamp
  - `status`: active/closed

#### `performance`
- Document: `summary`
- Purpose: Aggregated performance metrics
- Fields:
  - `total_profit`: Cumulative profit
  - `winning_trades`: Successful trade count
  - `losing_trades`: Failed trade count
  - `win_rate`: Success percentage
  - `last_update`: Update timestamp

#### `observed_patterns`
- Purpose: Discovered market patterns during observation
- Fields:
  - `pattern_type`: Type of pattern
  - `description`: Pattern details
  - `confidence`: Initial confidence
  - `discovered_at`: Discovery timestamp
  - `context`: Market conditions

#### `pattern_confidence`
- Purpose: Pattern success tracking
- Fields:
  - `pattern_id`: Reference to pattern
  - `confidence`: Current confidence score
  - `occurrences`: Total observations
  - `successes`: Successful predictions
  - `last_update`: Latest update

#### `observation_metrics`
- Document: `current`
- Purpose: Observation phase metrics
- Fields:
  - `patterns_discovered`: Pattern count
  - `observations_made`: Total observations
  - `confidence_levels`: Pattern confidence map
  - `last_update`: Update timestamp

### 3. In-Memory Caches

**Gas Monitor Cache** (`src/collectors/gas_monitor.py`):
```python
price_history = [
    {
        "price": Decimal,
        "timestamp": datetime,
        "hour": int,
        "day_of_week": int
    }
]
# Maintains 24-hour rolling window (2880 data points)
```

**Pool Scanner Cache** (`src/collectors/pool_scanner.py`):
```python
pools = {
    "TOKEN_A/TOKEN_B-stable": {
        "pair": str,
        "address": str,
        "tvl": Decimal,
        "volume_24h": Decimal,
        "apr": Decimal,
        "reserves": Dict[str, Decimal],
        "timestamp": datetime
    }
}
```

## Data Flow Architecture

### 1. Observation Flow
```
Market Data → Collectors → Memory System → Pattern Detection
     ↓            ↓              ↓                ↓
CDP Client   Gas Monitor    Mem0 Storage    Firestore
             Pool Scanner                  (patterns)
```

### 2. Decision Flow
```
Agent State → Memory Recall → Analysis → Decision
     ↓            ↓             ↓          ↓
Firestore      Mem0 Search   LangGraph  Firestore
(state)        (context)     (process)   (cycles)
```

### 3. Learning Flow
```
Execution → Outcome → Learning → Memory Update
    ↓          ↓         ↓            ↓
CDP SDK    Firestore   Agent      Mem0 + Firestore
          (positions)  Logic    (memories + confidence)
```

## Memory Recall Strategy

### Semantic Search (Mem0)
1. **Query Construction**: Natural language queries based on current context
2. **Vector Similarity**: Finds semantically similar memories
3. **Confidence Filtering**: Only returns memories above threshold
4. **Category Scoping**: Can filter by specific memory categories

### Pattern Matching
1. **Time-Based Patterns**: Hourly/daily patterns from observations
2. **Market Conditions**: Patterns linked to specific market states
3. **Success Tracking**: Confidence updates based on outcomes

## Data Persistence & Recovery

### Backup Strategy
- **Mem0**: Automatic cloud persistence (when using API)
- **Firestore**: Google Cloud automatic backups
- **Local Fallback**: In-memory storage for Mem0 if API unavailable

### Recovery Process
1. Restore agent state from Firestore
2. Reload high-confidence patterns
3. Resume memory formation from last checkpoint
4. Reconstruct in-memory caches from recent data

## Performance Optimization

### Caching Strategy
- **Gas Prices**: 30-second update interval, 24-hour window
- **Pool Data**: 5-minute update interval, full cache
- **Memory Recall**: Results cached within reasoning cycle

### Query Optimization
- **Firestore**: Compound indexes on frequently queried fields
- **Mem0**: Vector indices for fast similarity search
- **Batch Operations**: Group writes to reduce API calls

## Security Considerations

### Data Sanitization
- Decimal values converted to float for storage
- Datetime objects converted to ISO format
- Complex objects serialized to strings
- No private keys or sensitive data in memory

### Access Control
- Firestore: Service account with minimal permissions
- Mem0: API key authentication
- Read-only access for monitoring endpoints

## Monitoring & Metrics

### Key Metrics Tracked
- Total memories stored
- Pattern discovery rate
- Memory recall performance
- Storage utilization
- Query response times

### Health Indicators
- Memory formation rate
- Pattern confidence trends
- Successful recall percentage
- Storage growth rate

## Future Enhancements

1. **Multi-Agent Memory Sharing**: Enable memory synchronization across multiple agents
2. **Advanced Pattern Recognition**: ML-based pattern discovery from raw memories
3. **Memory Compression**: Automatic summarization of old memories
4. **Cross-Chain Patterns**: Pattern recognition across multiple blockchains
5. **Distributed Storage**: IPFS integration for decentralized memory storage