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
- `market_pattern`: Market state observations and patterns
- `gas_optimization`: Gas price patterns and optimal timing
- `strategy_performance`: Trading strategy results
- `pool_behavior`: Liquidity pool insights and behaviors
- `user_preference`: User-defined preferences
- `error_learning`: Error tracking and learning
- `profit_source`: Sources of profitable opportunities
- `gauge_emissions`: AERO emission rates and patterns (NEW)
- `volume_tracking`: Real swap volumes from events (NEW)
- `arbitrage_opportunity`: Detected price imbalances (NEW)
- `new_pool`: New pool discoveries (NEW)
- `apr_anomaly`: Unusual APR changes (NEW)
- `fee_collection`: Fee event tracking (NEW)

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

#### `pool_profiles` (NEW - v2.0)
- Documents: Pool address as document ID
- Purpose: Individual pool behavior profiles
- Fields:
  - `pool_address`: Pool contract address
  - `pair`: Token pair (e.g., "WETH/USDC")
  - `stable`: Whether it's a stable pool
  - `apr_range`: [min, max] APR observed
  - `tvl_range`: [min, max] TVL observed
  - `volume_range`: [min, max] volume observed
  - `hourly_patterns`: Hour → average metrics
  - `daily_patterns`: Day → average metrics
  - `typical_volume_to_tvl`: Average ratio
  - `volatility_score`: APR volatility measure
  - `correlation_with_gas`: Gas price correlation
  - `observations_count`: Total observations
  - `confidence_score`: Profile reliability score
  - `last_updated`: Last update timestamp

#### `pool_metrics` (NEW - v2.0)
- Purpose: Time-series pool data
- Fields:
  - `pool_address`: Pool contract address
  - `timestamp`: Observation time
  - `apr`: Current APR
  - `tvl`: Total value locked
  - `volume_24h`: 24-hour volume
  - `fee_apr`: Fee component of APR
  - `incentive_apr`: Incentive component
  - `reserves`: Token reserves
  - `ratio`: Reserve ratio
  - `gas_price`: Gas price at observation

#### `pattern_correlations` (NEW - v2.0)
- Purpose: Cross-pool pattern relationships
- Fields:
  - `pool_a`: First pool pair
  - `pool_b`: Second pool pair
  - `correlation_type`: Type (volume, apr, liquidity)
  - `correlation_strength`: -1 to 1 correlation value
  - `discovered_at`: Discovery timestamp
  - `occurrences`: Number of observations
  - `confidence`: Correlation confidence

#### `gauge_data` (NEW - Real Data Collection)
- Documents: Gauge address as document ID
- Purpose: Gauge emission data and rewards
- Fields:
  - `gauge_address`: Gauge contract address
  - `pool_address`: Associated pool address
  - `reward_rate`: AERO tokens per second
  - `total_supply`: Total LP tokens staked
  - `aero_per_day`: Daily AERO emissions
  - `last_update`: Update timestamp
  - `historical_rates`: Time series of reward rates

#### `event_volumes` (NEW - Real Data Collection)
- Purpose: Real volume data from on-chain events
- Fields:
  - `pool_address`: Pool contract address
  - `hour_key`: Hour identifier (YYYY-MM-DD-HH)
  - `swap_volume`: Total swap volume in hour
  - `fee_volume`: Total fees collected
  - `swap_count`: Number of swaps
  - `unique_traders`: Unique addresses
  - `largest_swap`: Biggest single swap
  - `timestamp`: Hour timestamp

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
        "volume_24h": Decimal,  # Real volume from events
        "apr": Decimal,         # Real total APR
        "fee_apr": Decimal,    # Calculated from volume
        "incentive_apr": Decimal,  # From gauge emissions
        "reserves": Dict[str, Decimal],
        "timestamp": datetime
    }
}
```

**Gauge Reader Cache** (`src/aerodrome/gauge_reader.py`):
```python
_gauge_cache = {
    "pool_address": "gauge_address"
}
# TTL: 1 hour to reduce RPC calls
```

**Event Monitor Cache** (`src/aerodrome/event_monitor.py`):
```python
hourly_volumes = {
    "pool_address": {
        "YYYY-MM-DD-HH": Decimal  # Volume in that hour
    }
}
daily_volumes = {
    "pool_address": {
        "YYYY-MM-DD": Decimal  # Daily total
    }
}
```

## Data Flow Architecture

### 1. Observation Flow (Enhanced with Real Data)
```
Market Data → Collectors → Memory System → Pattern Detection
     ↓            ↓              ↓                ↓
CDP Client   Gas Monitor    Mem0 Storage    Firestore
     ↓       Pool Scanner                  (patterns)
     ↓            ↓                              ↓
RPC Reader   Gauge Reader                 pool_profiles
     ↓       Event Monitor                gauge_data
Blockchain                               event_volumes
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

## Real Data Collection Architecture (NEW)

### Data Sources
1. **Gauge Contracts**: Direct reads for AERO emission rates
2. **Event Logs**: Swap and Fee events for volume tracking
3. **Pool Contracts**: Reserve data and pool parameters
4. **Voter Contract**: Gauge addresses for pools

### Collection Pipeline
```
1. Pool Scanner identifies pools to monitor
2. Gauge Reader fetches emission data for each pool
3. Event Monitor tracks swap/fee events in real-time
4. Real APR calculation combines fee and emission data
5. Data stored in appropriate Firestore collections
6. Memory system indexes significant patterns
```

### APR Calculation Methodology
- **Fee APR** = (24h_volume × fee_rate × 365) / TVL × 100
- **Emission APR** = (reward_rate × seconds_per_year × AERO_price) / TVL × 100
- **Total APR** = Fee APR + Emission APR

### Event Monitoring Strategy
- Query blocks in 1000-block chunks to prevent timeouts
- Cache block timestamps to reduce RPC calls
- Maintain hourly aggregates for efficient 24h calculations
- Store raw event data for detailed analysis

### Data Quality & Validation
1. **Volume Verification**:
   - Cross-reference event volumes with pool state changes
   - Detect and filter wash trading patterns
   - Validate against known pool constraints

2. **APR Sanity Checks**:
   - Flag APRs > 1000% for manual review
   - Compare calculated vs reported values
   - Track sudden APR changes as anomalies

3. **Emission Validation**:
   - Verify gauge exists for pool before reading
   - Confirm reward rate changes align with epochs
   - Track total AERO emissions vs protocol limits

4. **Real-time Monitoring**:
   - Alert on data collection failures
   - Track RPC response times and errors
   - Monitor memory usage and storage growth

## Future Enhancements

1. **Multi-Agent Memory Sharing**: Enable memory synchronization across multiple agents
2. **Advanced Pattern Recognition**: ML-based pattern discovery from raw memories
3. **Memory Compression**: Automatic summarization of old memories
4. **Cross-Chain Patterns**: Pattern recognition across multiple blockchains
5. **Distributed Storage**: IPFS integration for decentralized memory storage
6. **Historical Event Replay**: Backfill historical data for pattern learning
7. **Price Oracle Integration**: Real-time token prices for accurate USD calculations
8. **Voting Power Analysis**: Track veAERO voting patterns and bribe efficiency