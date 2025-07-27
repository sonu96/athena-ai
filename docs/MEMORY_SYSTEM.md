# Athena AI Enhanced Memory System

## Overview

The Enhanced Memory System transforms Athena from storing only top-performing pools to comprehensively learning from all market activity. This enables better pattern recognition, cross-pool correlations, and predictive capabilities.

## Architecture

### 1. Tiered Memory Storage

The system implements a multi-tier approach to memory storage:

```
┌─────────────────────────────────────────────────────────┐
│                    L1: Raw Observations                  │
│         All pools meeting minimum thresholds             │
│    (APR >= 20% OR Volume >= $100k OR Imbalanced)       │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                 L2: Pool Profiles                        │
│      Individual behavior tracking per pool               │
│    (Hourly patterns, daily patterns, ranges)            │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│              L3: Pattern Correlations                    │
│        Cross-pool relationships and predictions          │
└─────────────────────────────────────────────────────────┘
```

### 2. Pool Profiles System

Each pool maintains a comprehensive profile tracking:

- **Historical Ranges**: APR, TVL, and volume ranges
- **Time Patterns**: Hourly and daily behavioral patterns
- **Behavioral Metrics**: 
  - Typical volume-to-TVL ratio
  - Volatility score
  - Gas price correlation
- **Confidence Score**: Based on observations and consistency

### 3. Memory Categories

#### Observation Mode (Learning Phase)
- `pool_behavior`: Individual pool metrics and behaviors
- `arbitrage_opportunity`: Imbalanced pool states
- `new_pool`: Newly discovered pools
- `market_pattern`: Overall market conditions
- `gas_patterns`: Gas price correlations

#### Trading Mode (Execution Phase)
- `strategy_performance`: Outcome tracking
- `cross_pool_correlation`: Inter-pool relationships
- `time_based_pattern`: Temporal patterns
- `anomaly_detection`: Unusual market behavior

## Configuration

### Environment Variables

```bash
# Minimum thresholds for memory storage
MIN_APR_FOR_MEMORY=20              # Store pools with APR >= 20%
MIN_VOLUME_FOR_MEMORY=100000       # Store pools with volume >= $100k
MAX_MEMORIES_PER_CYCLE=50          # Prevent memory overflow
POOL_PROFILE_UPDATE_INTERVAL=3600  # Update profiles every hour

# Observation mode settings
OBSERVATION_MODE=true              # Start in learning mode
OBSERVATION_DAYS=3                 # Days to observe before trading
MIN_PATTERN_CONFIDENCE=0.7         # Minimum confidence to act
```

## Memory Storage Improvements

### Before (v1.0)
```python
# Only stored the top performer
top_apr = max(opportunities["high_apr"], key=lambda x: x["apr"])
await memory.remember(top_apr)  # Only 1 memory stored
```

### After (v2.0)
```python
# Store all significant pools
for pool in opportunities["high_apr"]:
    if pool["apr"] >= min_apr_for_memory:
        await memory.remember(pool)  # 10-15 memories per scan
```

## Pool-Specific Memory Queries

The enhanced system provides specialized query methods:

```python
# Get memories for a specific pool
memories = await memory.recall_pool_memories(
    pool_pair="WETH/USDC",
    time_window_hours=24
)

# Get discovered patterns for a pool
patterns = await memory.get_pool_patterns("AERO/USDC")

# Get cross-pool correlations
correlations = await memory.get_cross_pool_correlations()

# Get chronological timeline
timeline = await memory.get_pool_timeline("WETH/DAI", hours=48)
```

## Firestore Collections

### pool_profiles
Stores comprehensive profiles for each pool:
```json
{
  "pool_address": "0x...",
  "pair": "WETH/USDC",
  "stable": false,
  "apr_range": [25.0, 85.0],
  "tvl_range": [100000, 5000000],
  "hourly_patterns": {...},
  "confidence_score": 0.85
}
```

### pool_metrics
Time-series data for pool metrics:
```json
{
  "pool_address": "0x...",
  "timestamp": "2025-01-23T10:00:00Z",
  "apr": 65.0,
  "tvl": 2500000,
  "volume_24h": 1250000,
  "reserves": {...},
  "gas_price": 0.05
}
```

### pattern_correlations
Cross-pool pattern relationships:
```json
{
  "pool_a": "WETH/USDC",
  "pool_b": "WETH/DAI",
  "correlation_type": "volume",
  "correlation_strength": 0.82,
  "discovered_at": "2025-01-23T10:00:00Z"
}
```

## Pattern Recognition

### Time-Based Patterns
The system identifies:
- Hourly patterns (e.g., high activity at specific hours)
- Daily patterns (e.g., weekend vs weekday behavior)
- Gas correlation patterns (e.g., volume changes with gas prices)

### Predictive Capabilities
Pool profiles enable predictions:
```python
predictions = pool_profiles.predict_opportunities(
    timestamp=datetime.utcnow() + timedelta(hours=1)
)
# Returns predicted APR and volume based on historical patterns
```

### Anomaly Detection
Identifies unusual behavior:
```python
anomalies = profile.get_anomalies()
# Returns deviations > 2 standard deviations from normal
```

## Benefits

### 1. Comprehensive Learning
- **Before**: 2 memories per scan (top APR, top volume)
- **After**: 10-15 memories per scan (all significant activity)
- **Result**: 5-7x more data for pattern recognition

### 2. Pool-Specific Intelligence
- Individual behavior tracking per pool
- Identifies pool-specific opportunities
- Predicts future behavior based on patterns

### 3. Cross-Pool Insights
- Discovers correlations between pools
- Identifies arbitrage opportunities
- Tracks liquidity flows between pools

### 4. Time-Aware Decisions
- Leverages temporal patterns
- Predicts optimal entry/exit times
- Adapts to market regime changes

## Usage Examples

### During Observation Mode
```python
# Agent observes all pools
for pool in scanned_pools:
    # Update profile with new metrics
    await pool_profiles.update_pool(pool_data, gas_price)
    
    # Store memories for all significant pools
    if pool["apr"] >= 20 or pool["volume"] >= 100000:
        await memory.remember(pool_observation)
```

### During Trading Mode
```python
# Use high-confidence patterns
patterns = firestore.get_high_confidence_patterns(0.7)

# Get pool predictions
predictions = pool_profiles.predict_opportunities(next_hour)

# Make decisions based on comprehensive data
if prediction["confidence"] == "high":
    execute_strategy(prediction)
```

## Memory Performance Optimization

### Caching Strategy
The memory system implements multi-level caching for optimal performance:

```python
# Cache Configuration
CACHE_CONFIG = {
    "pattern_cache": {
        "size": 1000,
        "ttl": 3600,  # 1 hour
        "hit_rate_target": 0.7
    },
    "query_cache": {
        "size": 500,
        "ttl": 300,   # 5 minutes
        "hit_rate_target": 0.6
    },
    "pool_cache": {
        "size": 200,
        "ttl": 600,   # 10 minutes
        "hit_rate_target": 0.8
    }
}
```

### Query Optimization
```python
# Optimized memory recall flow
async def optimized_recall(query, category=None, pool=None):
    # 1. Check bloom filter (O(1))
    if not bloom_filter.might_contain(query):
        return []
    
    # 2. Try cache first
    cache_key = f"{category}:{pool}:{hash(query)}"
    if cached := cache.get(cache_key):
        return cached
    
    # 3. Query with optimizations
    results = await memory.recall(
        query=query,
        category=category,
        filters={"pool": pool} if pool else None,
        use_parallel=True,
        batch_size=10
    )
    
    # 4. Cache results
    cache.set(cache_key, results, ttl=300)
    return results
```

### Scalability Limits
To prevent memory explosion, the system enforces strict limits:

| Category | Max Memories | Retention | Pruning Strategy |
|----------|-------------|-----------|------------------|
| observation | 1,000 | 7 days | Low confidence removal |
| pattern | 500 | 30 days | Similarity merging |
| pool_behavior | 200 | 30 days | Least recently used |
| cross_pool_correlation | 100 | 14 days | Weak correlation removal |
| strategy_performance | 300 | 90 days | Aggregation |
| error_learning | 50 | 7 days | Resolution tracking |

### Memory Pruning Algorithm
```python
async def prune_memories():
    """Hourly memory pruning job"""
    # 1. Remove low confidence memories
    await memory.delete_where(
        confidence__lt=0.3,
        age__gt=timedelta(hours=48)
    )
    
    # 2. Merge similar patterns
    patterns = await memory.get_patterns()
    for p1, p2 in find_similar_pairs(patterns, threshold=0.9):
        await memory.merge_patterns(p1, p2)
    
    # 3. Enforce category quotas
    for category, limit in CATEGORY_LIMITS.items():
        excess = await memory.count(category=category) - limit
        if excess > 0:
            await memory.delete_oldest(category, count=excess)
    
    # 4. Compress metadata
    await memory.compress_metadata(age__gt=timedelta(days=1))
```

## Monitoring

### Performance Metrics
Track memory system health with these key metrics:

```python
# Memory system metrics
metrics = {
    "memory_count": await memory.count(),
    "cache_hit_rate": cache.get_hit_rate(),
    "query_latency_p95": query_monitor.get_percentile(95),
    "pruning_effectiveness": pruner.get_effectiveness(),
    "storage_usage_mb": memory.get_storage_usage() / 1024 / 1024
}

# Pool profile metrics
profile_metrics = {
    "total_profiles": pool_profiles.count(),
    "high_confidence_pools": pool_profiles.count_high_confidence(0.8),
    "avg_observations_per_pool": pool_profiles.get_avg_observations(),
    "prediction_accuracy": pool_profiles.get_prediction_accuracy()
}
```

### Alert Thresholds
```yaml
alerts:
  - name: MemoryGrowthHigh
    condition: memory_growth_rate > 200/hour
    severity: warning
    
  - name: CacheHitRateLow
    condition: cache_hit_rate < 0.5
    severity: warning
    
  - name: QueryLatencyHigh
    condition: query_latency_p95 > 200ms
    severity: critical
    
  - name: MemoryQuotaExceeded
    condition: category_count > category_limit
    severity: error
```

## Future Enhancements

1. **Machine Learning Integration**
   - Use pool profiles for ML training data
   - Implement neural network for pattern recognition
   - Advanced correlation analysis

2. **Strategy Optimization**
   - Backtest strategies using historical profiles
   - Optimize parameters per pool
   - Dynamic strategy selection

3. **Risk Management**
   - Pool-specific risk scoring
   - Correlation-based portfolio management
   - Anomaly-based circuit breakers

## Conclusion

The Enhanced Memory System transforms Athena from a reactive agent that only remembers "winners" to a proactive learner that builds comprehensive market intelligence. By storing 5-7x more data and maintaining individual pool profiles, Athena can identify subtle patterns, predict opportunities, and make more informed decisions.