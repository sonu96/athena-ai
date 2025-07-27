# Performance Optimization Guide

## Overview

This guide details performance optimization strategies for Athena AI, focusing on reducing latency, minimizing costs, and maximizing throughput. All optimizations maintain system reliability while improving efficiency.

## Performance Targets

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Memory Query P95 | < 100ms | > 200ms |
| LLM Response Time | < 2s | > 5s |
| Firestore Writes/min | < 20 | > 50 |
| Cache Hit Rate | > 70% | < 50% |
| Agent Cycle Time | < 30s | > 60s |
| API Response P95 | < 200ms | > 500ms |

## LLM Optimization

### Response Caching System

```python
class LLMCache:
    """Intelligent caching for LLM responses"""
    
    def __init__(self):
        self.response_cache = TTLCache(maxsize=1000, ttl=86400)  # 24h
        self.semantic_hasher = SemanticHasher()
        self.hit_count = 0
        self.miss_count = 0
        
    async def get_cached_response(self, prompt, context):
        """Get cached response or generate new"""
        
        # 1. Generate semantic hash
        cache_key = self.semantic_hasher.hash(prompt, context)
        
        # 2. Check cache
        if cache_key in self.response_cache:
            self.hit_count += 1
            logger.debug(f"Cache hit: {self.get_hit_rate():.2%}")
            return self.response_cache[cache_key]
        
        # 3. Generate new response
        self.miss_count += 1
        response = await self._generate_response(prompt, context)
        
        # 4. Cache if successful
        if response and not response.get("error"):
            self.response_cache[cache_key] = response
            
        return response
```

### Pattern Template Matching

```python
# Common patterns that bypass LLM
PATTERN_TEMPLATES = {
    "gas_high": {
        "condition": lambda ctx: ctx.gas_price > ctx.avg_gas * 2,
        "response": "Gas prices elevated. Postponing non-critical operations."
    },
    "pool_new": {
        "condition": lambda ctx: ctx.pool_age_days < 1,
        "response": "New pool detected. Observing for 24h before engaging."
    },
    "apr_spike": {
        "condition": lambda ctx: ctx.apr > ctx.historical_apr * 3,
        "response": "Abnormal APR spike. Likely temporary - monitoring closely."
    }
}

async def fast_decision_path(context):
    """Skip LLM for known patterns"""
    for pattern_name, pattern in PATTERN_TEMPLATES.items():
        if pattern["condition"](context):
            return {
                "decision": pattern["response"],
                "pattern_matched": pattern_name,
                "confidence": 0.9
            }
    return None  # No pattern matched, use LLM
```

### Batch Analysis Operations

```python
async def batch_llm_analysis(observations):
    """Combine multiple analyses into single LLM call"""
    
    # Group similar observations
    grouped = group_by_type(observations)
    
    # Create batch prompt
    batch_prompt = f"""
    Analyze these market observations together:
    
    Gas Prices: {grouped.get('gas', [])}
    High APR Pools: {grouped.get('high_apr', [])}
    New Pools: {grouped.get('new_pools', [])}
    
    Provide:
    1. Overall market assessment
    2. Correlation insights
    3. Recommended focus areas
    4. Risk warnings
    """
    
    # Single LLM call instead of multiple
    response = await llm.generate(batch_prompt)
    
    # Parse and distribute results
    return parse_batch_response(response)
```

## Query Performance Optimization

### Memory Query Optimization

```python
class OptimizedMemoryQuery:
    """High-performance memory retrieval"""
    
    def __init__(self):
        self.bloom_filters = {}
        self.query_cache = TTLCache(maxsize=500, ttl=300)
        self.index_cache = {}
        
    async def fast_recall(self, query_params):
        """Optimized memory recall with multiple strategies"""
        
        # 1. Bloom filter check (O(1))
        category = query_params.get("category")
        if category and not self._bloom_check(category, query_params):
            return []  # Category doesn't contain query
            
        # 2. Cache check
        cache_key = self._generate_cache_key(query_params)
        if cached := self.query_cache.get(cache_key):
            return cached
            
        # 3. Use indices for metadata filtering
        candidates = await self._index_filter(query_params)
        
        # 4. Parallel vector search
        if len(candidates) > 10:
            results = await self._parallel_search(candidates, query_params)
        else:
            results = await self._sequential_search(candidates, query_params)
            
        # 5. Cache results
        self.query_cache[cache_key] = results
        
        return results
        
    async def _parallel_search(self, candidates, query_params):
        """Execute searches in parallel for better performance"""
        
        # Split candidates into chunks
        chunk_size = 20
        chunks = [candidates[i:i+chunk_size] 
                 for i in range(0, len(candidates), chunk_size)]
        
        # Parallel execution
        tasks = [
            self._search_chunk(chunk, query_params)
            for chunk in chunks
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Merge and sort results
        merged = []
        for chunk_results in results:
            merged.extend(chunk_results)
            
        return sorted(merged, key=lambda x: x.score, reverse=True)
```

### Query Plan Optimization

```python
class QueryPlanner:
    """Intelligent query planning for complex searches"""
    
    def create_plan(self, query_params):
        """Create optimized query execution plan"""
        
        plan = QueryPlan()
        
        # 1. Analyze query complexity
        complexity = self._analyze_complexity(query_params)
        
        # 2. Choose strategy
        if complexity.is_simple:
            plan.add_step("direct_query", query_params)
            
        elif complexity.has_time_range:
            # Time-based partitioning
            plan.add_step("partition_by_time", {
                "start": query_params["start_time"],
                "end": query_params["end_time"],
                "bucket_size": timedelta(hours=6)
            })
            
        elif complexity.has_multiple_pools:
            # Pool-based partitioning
            plan.add_step("partition_by_pool", {
                "pools": query_params["pools"],
                "parallel": True
            })
            
        # 3. Add caching strategy
        if complexity.is_frequent:
            plan.cache_strategy = "aggressive"
            plan.cache_ttl = 600  # 10 minutes
        else:
            plan.cache_strategy = "normal"
            plan.cache_ttl = 60   # 1 minute
            
        return plan
```

## Firestore Cost Optimization

### Write Batching System

```python
class FirestoreBatcher:
    """Intelligent write batching to reduce costs"""
    
    def __init__(self):
        self.write_buffer = []
        self.last_flush = datetime.utcnow()
        self.flush_interval = timedelta(seconds=60)
        self.max_batch_size = 500  # Firestore limit
        
    async def write(self, collection, document_id, data):
        """Buffer writes for batching"""
        
        # Add to buffer
        self.write_buffer.append({
            "collection": collection,
            "id": document_id,
            "data": data,
            "timestamp": datetime.utcnow()
        })
        
        # Check if flush needed
        if self._should_flush():
            await self.flush()
            
    def _should_flush(self):
        """Determine if buffer should be flushed"""
        
        # Size limit
        if len(self.write_buffer) >= self.max_batch_size:
            return True
            
        # Time limit
        if datetime.utcnow() - self.last_flush > self.flush_interval:
            return True
            
        # Priority writes
        if any(w.get("priority") == "high" for w in self.write_buffer):
            return True
            
        return False
        
    async def flush(self):
        """Execute batched writes"""
        
        if not self.write_buffer:
            return
            
        # Group by collection for efficiency
        grouped = self._group_by_collection(self.write_buffer)
        
        # Execute batches
        batch = self.firestore.batch()
        write_count = 0
        
        for collection, documents in grouped.items():
            for doc in documents:
                if write_count >= 500:
                    # Commit current batch
                    await batch.commit()
                    batch = self.firestore.batch()
                    write_count = 0
                    
                ref = self.firestore.collection(collection).document(doc["id"])
                batch.set(ref, doc["data"], merge=True)
                write_count += 1
                
        # Final commit
        if write_count > 0:
            await batch.commit()
            
        # Track costs
        self._track_write_costs(len(self.write_buffer))
        
        # Clear buffer
        self.write_buffer = []
        self.last_flush = datetime.utcnow()
```

### Aggregation Before Storage

```python
class DataAggregator:
    """Aggregate data before writing to reduce storage"""
    
    def aggregate_pool_metrics(self, metrics_list):
        """Aggregate pool metrics to reduce writes"""
        
        if not metrics_list:
            return None
            
        # Group by time bucket (15 minutes)
        buckets = {}
        for metric in metrics_list:
            bucket_time = self._round_to_15min(metric.timestamp)
            if bucket_time not in buckets:
                buckets[bucket_time] = []
            buckets[bucket_time].append(metric)
            
        # Aggregate each bucket
        aggregated = []
        for bucket_time, bucket_metrics in buckets.items():
            aggregated.append({
                "timestamp": bucket_time,
                "apr": {
                    "min": min(m.apr for m in bucket_metrics),
                    "max": max(m.apr for m in bucket_metrics),
                    "avg": sum(m.apr for m in bucket_metrics) / len(bucket_metrics)
                },
                "tvl": {
                    "min": min(m.tvl for m in bucket_metrics),
                    "max": max(m.tvl for m in bucket_metrics),
                    "last": bucket_metrics[-1].tvl
                },
                "volume": sum(m.volume for m in bucket_metrics),
                "sample_count": len(bucket_metrics)
            })
            
        return aggregated
```

## Monitoring & Profiling

### Performance Monitoring

```python
class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self):
        self.metrics = {
            "memory_query_latency": Histogram(
                "memory_query_latency_seconds",
                "Time to query memory system"
            ),
            "llm_response_time": Histogram(
                "llm_response_time_seconds",
                "Time for LLM to respond"
            ),
            "firestore_operations": Counter(
                "firestore_operations_total",
                "Total Firestore operations",
                ["operation_type"]
            ),
            "cache_hit_rate": Gauge(
                "cache_hit_rate_ratio",
                "Cache hit rate percentage"
            )
        }
        
    @contextmanager
    def measure(self, metric_name):
        """Context manager for timing operations"""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.metrics[metric_name].observe(duration)
            
            # Log slow operations
            if duration > self.thresholds.get(metric_name, float('inf')):
                logger.warning(f"Slow {metric_name}: {duration:.2f}s")
```

### Profiling Tools

```python
# Profile memory allocations
@profile_memory
async def memory_intensive_operation():
    # Your code here
    pass

# Profile execution time
@profile_time(threshold=0.1)  # Log if > 100ms
async def time_critical_operation():
    # Your code here
    pass

# Profile database queries
@profile_queries
async def database_heavy_operation():
    # Your code here
    pass
```

## Scaling Guidelines

### When to Scale

| Metric | Scale Trigger | Scaling Action |
|--------|---------------|----------------|
| Memory count | > 50,000 | Add memory pruning workers |
| Query latency | > 200ms P95 | Increase cache size |
| LLM queue depth | > 10 | Add LLM instances |
| Firestore writes | > 50/min | Increase batch size |
| CPU usage | > 80% | Horizontal scaling |

### Horizontal Scaling Strategy

```yaml
# Cloud Run scaling configuration
scaling:
  minInstances: 1
  maxInstances: 10
  
  metrics:
    - type: cpu
      target: 70
    - type: memory
      target: 80
    - type: concurrency
      target: 100
      
  scaleDownControl:
    maxScaleDownRate: 2  # Max 2 instances removed per minute
    timeWindow: 60s
```

### Cache Scaling

```python
def auto_scale_cache():
    """Dynamically adjust cache sizes based on hit rates"""
    
    for cache_name, cache in caches.items():
        hit_rate = cache.get_hit_rate()
        
        if hit_rate < 0.5 and cache.maxsize < 2000:
            # Poor hit rate, increase size
            cache.resize(cache.maxsize * 1.5)
            logger.info(f"Increased {cache_name} size to {cache.maxsize}")
            
        elif hit_rate > 0.9 and cache.maxsize > 100:
            # Excellent hit rate, can reduce size
            cache.resize(cache.maxsize * 0.8)
            logger.info(f"Reduced {cache_name} size to {cache.maxsize}")
```

## Best Practices

### 1. Measure First
- Profile before optimizing
- Use production-like data
- Consider worst-case scenarios

### 2. Cache Wisely
- Cache expensive operations
- Set appropriate TTLs
- Monitor hit rates

### 3. Batch Operations
- Group similar operations
- Use bulk APIs
- Respect service limits

### 4. Async Everything
- Use asyncio throughout
- Avoid blocking operations
- Parallelize when possible

### 5. Monitor Continuously
- Set up alerts
- Track trends
- Review weekly

## Performance Debugging

### Slow Query Checklist

1. **Check indices**
   ```python
   # Verify Firestore indices
   for collection in collections:
       indices = firestore.collection(collection).list_indexes()
       print(f"{collection}: {len(indices)} indices")
   ```

2. **Analyze query plan**
   ```python
   # Use explain() when available
   query.explain()
   ```

3. **Check cache effectiveness**
   ```python
   for cache_name, cache in caches.items():
       print(f"{cache_name}: {cache.get_hit_rate():.2%} hit rate")
   ```

### Memory Leak Detection

```python
import tracemalloc
import gc

def check_memory_growth():
    """Detect potential memory leaks"""
    
    tracemalloc.start()
    
    # Take snapshots
    snapshot1 = tracemalloc.take_snapshot()
    
    # Run operations
    for _ in range(100):
        await agent.run_cycle()
        
    gc.collect()
    snapshot2 = tracemalloc.take_snapshot()
    
    # Compare snapshots
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print("Top 10 memory increases:")
    for stat in top_stats[:10]:
        print(stat)
```

## Configuration Tuning

### Optimal Settings

```python
# Performance-optimized configuration
PERFORMANCE_CONFIG = {
    # Memory settings
    "memory_cache_size": 1000,
    "memory_cache_ttl": 3600,
    "memory_query_batch_size": 20,
    "memory_parallel_queries": True,
    
    # LLM settings
    "llm_cache_size": 1000,
    "llm_cache_ttl": 86400,
    "llm_timeout": 30,
    "llm_max_retries": 2,
    
    # Firestore settings
    "firestore_batch_size": 500,
    "firestore_flush_interval": 60,
    "firestore_read_cache_ttl": 300,
    "firestore_max_concurrent_writes": 10,
    
    # Query settings
    "query_default_limit": 100,
    "query_max_limit": 1000,
    "query_timeout": 5,
    
    # System settings
    "health_check_interval": 60,
    "metrics_export_interval": 30,
    "gc_interval": 3600
}
```

### Environment-Specific Tuning

```python
# Development (optimize for debugging)
if environment == "development":
    config["memory_cache_ttl"] = 60  # Shorter for testing
    config["llm_cache_size"] = 100   # Smaller for cost
    
# Production (optimize for performance)
elif environment == "production":
    config["memory_cache_ttl"] = 3600  # Longer for performance
    config["llm_cache_size"] = 2000    # Larger for hit rate
    config["firestore_batch_size"] = 500  # Maximum batching
```

## Continuous Improvement

### Weekly Performance Review

1. **Analyze metrics trends**
2. **Identify bottlenecks**
3. **Test optimizations**
4. **Deploy improvements**
5. **Monitor impact**

### Performance Regression Testing

```python
# Run after each deployment
async def performance_regression_test():
    """Ensure performance hasn't degraded"""
    
    results = {}
    
    # Test memory queries
    start = time.time()
    for _ in range(100):
        await memory.recall("test query")
    results["memory_query_avg"] = (time.time() - start) / 100
    
    # Test LLM responses
    start = time.time()
    for _ in range(10):
        await llm.generate("test prompt")
    results["llm_response_avg"] = (time.time() - start) / 10
    
    # Compare with baselines
    for metric, value in results.items():
        baseline = PERFORMANCE_BASELINES[metric]
        if value > baseline * 1.2:  # 20% regression
            raise PerformanceRegression(f"{metric}: {value} > {baseline}")
    
    return results
```