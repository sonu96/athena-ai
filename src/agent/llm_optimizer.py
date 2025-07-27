"""
LLM Response Caching and Optimization System

Implements semantic caching, response templates, and batch operations
to reduce LLM calls and improve response times.
"""
import asyncio
import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.gcp.firestore_client import FirestoreClient
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """Cached LLM response with metadata."""
    query_hash: str
    response: str
    semantic_hash: str
    timestamp: datetime
    hit_count: int = 0
    tokens_saved: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class ResponseTemplate:
    """Pre-defined response template for common queries."""
    pattern: str
    template: str
    variables: List[str]
    usage_count: int = 0
    avg_tokens_saved: int = 0
    

class SemanticCache:
    """Semantic similarity-based cache for LLM responses."""
    
    def __init__(self, max_size: int = 1000, similarity_threshold: float = 0.85):
        """Initialize semantic cache."""
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[str, CachedResponse] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.query_texts: List[str] = []
        
    def _compute_semantic_hash(self, text: str) -> str:
        """Compute semantic hash of text."""
        # Normalize and extract key concepts
        normalized = text.lower().strip()
        # Simple semantic hash - in production use sentence embeddings
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
        
    async def get(self, query: str) -> Optional[CachedResponse]:
        """Get cached response for similar query."""
        # Direct hash lookup
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        if query_hash in self.cache:
            response = self.cache[query_hash]
            response.hit_count += 1
            logger.debug(f"Cache hit (direct): {query[:50]}...")
            return response
            
        # Semantic similarity search
        if len(self.query_texts) < 2:
            return None
            
        try:
            # Compute query embedding
            query_vec = self.vectorizer.transform([query])
            
            # Find most similar cached query
            similarities = cosine_similarity(query_vec, self.vectorizer.transform(self.query_texts))
            max_sim_idx = np.argmax(similarities[0])
            max_similarity = similarities[0][max_sim_idx]
            
            if max_similarity > self.similarity_threshold:
                # Find corresponding cache entry
                similar_query = self.query_texts[max_sim_idx]
                similar_hash = hashlib.sha256(similar_query.encode()).hexdigest()
                
                if similar_hash in self.cache:
                    response = self.cache[similar_hash]
                    response.hit_count += 1
                    logger.debug(
                        f"Cache hit (semantic, sim={max_similarity:.2f}): "
                        f"{query[:30]}... -> {similar_query[:30]}..."
                    )
                    return response
                    
        except Exception as e:
            logger.error(f"Semantic cache search error: {e}")
            
        return None
        
    async def put(self, query: str, response: str, tokens_used: int = 0):
        """Add response to cache."""
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        semantic_hash = self._compute_semantic_hash(query)
        
        cached = CachedResponse(
            query_hash=query_hash,
            response=response,
            semantic_hash=semantic_hash,
            timestamp=datetime.utcnow(),
            tokens_saved=tokens_used
        )
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].timestamp
            )
            del self.cache[oldest_key]
            # Remove from query texts
            self.query_texts = [q for q in self.query_texts if hashlib.sha256(q.encode()).hexdigest() != oldest_key]
            
        self.cache[query_hash] = cached
        self.query_texts.append(query)
        
        # Update vectorizer if enough samples
        if len(self.query_texts) >= 10:
            try:
                self.vectorizer.fit(self.query_texts)
            except Exception as e:
                logger.error(f"Vectorizer update error: {e}")
                
                
class LLMOptimizer:
    """
    Comprehensive LLM optimization system.
    
    Features:
    - Semantic response caching
    - Pattern-based templates
    - Batch request processing
    - Token usage optimization
    """
    
    def __init__(self, firestore: FirestoreClient):
        """Initialize LLM optimizer."""
        self.firestore = firestore
        
        # Caching systems
        self.semantic_cache = SemanticCache(
            max_size=settings.llm_cache_size,
            similarity_threshold=0.85
        )
        self.direct_cache: Dict[str, CachedResponse] = {}
        
        # Response templates
        self.templates = self._initialize_templates()
        
        # Batch processing
        self.batch_queue: List[Dict[str, Any]] = []
        self.batch_lock = asyncio.Lock()
        self._batch_processor_task = None
        
        # Statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "tokens_saved": 0,
            "template_uses": 0,
            "batch_requests": 0
        }
        
    def _initialize_templates(self) -> Dict[str, ResponseTemplate]:
        """Initialize common response templates."""
        return {
            "pool_analysis": ResponseTemplate(
                pattern="analyze pool .* apr .* tvl .*",
                template="Pool Analysis:\n- APR: {apr}%\n- TVL: ${tvl}\n- Risk: {risk}\n- Recommendation: {recommendation}",
                variables=["apr", "tvl", "risk", "recommendation"]
            ),
            "gas_check": ResponseTemplate(
                pattern="check gas price.*",
                template="Current gas: {gas_price} gwei\nStatus: {status}\nRecommendation: {action}",
                variables=["gas_price", "status", "action"]
            ),
            "position_update": ResponseTemplate(
                pattern="update position .* pool .*",
                template="Position Update:\n- Pool: {pool}\n- Current Value: ${current}\n- Change: {change}%\n- Action: {action}",
                variables=["pool", "current", "change", "action"]
            ),
            "market_summary": ResponseTemplate(
                pattern="market summary|overall market.*",
                template="Market Summary:\n- Total TVL: ${tvl}\n- Avg APR: {apr}%\n- Active Pools: {pools}\n- Trend: {trend}",
                variables=["tvl", "apr", "pools", "trend"]
            )
        }
        
    async def start_batch_processor(self):
        """Start the batch processing service."""
        if self._batch_processor_task:
            return
            
        self._batch_processor_task = asyncio.create_task(self._process_batches())
        logger.info("Started LLM batch processor")
        
    async def stop_batch_processor(self):
        """Stop the batch processing service."""
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass
            self._batch_processor_task = None
            logger.info("Stopped LLM batch processor")
            
    async def _process_batches(self):
        """Process queued requests in batches."""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second
                
                async with self.batch_lock:
                    if not self.batch_queue:
                        continue
                        
                    # Process up to 10 requests at once
                    batch = self.batch_queue[:10]
                    self.batch_queue = self.batch_queue[10:]
                    
                if batch:
                    await self._execute_batch(batch)
                    self.stats["batch_requests"] += len(batch)
                    
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(5)
                
    async def _execute_batch(self, batch: List[Dict[str, Any]]):
        """Execute a batch of LLM requests."""
        # Group by request type for better batching
        grouped = {}
        for request in batch:
            req_type = request.get("type", "general")
            if req_type not in grouped:
                grouped[req_type] = []
            grouped[req_type].append(request)
            
        # Process each group
        for req_type, requests in grouped.items():
            if req_type == "pool_analysis":
                await self._batch_pool_analysis(requests)
            elif req_type == "pattern_detection":
                await self._batch_pattern_detection(requests)
            else:
                # Process individually
                for req in requests:
                    await self._process_single_request(req)
                    
    async def _batch_pool_analysis(self, requests: List[Dict[str, Any]]):
        """Batch process pool analysis requests."""
        # Combine into single prompt
        pools = [req["data"]["pool"] for req in requests]
        
        combined_prompt = f"""Analyze these {len(pools)} pools efficiently:
        
{json.dumps(pools, indent=2)}

For each pool provide: APR assessment, risk level, and recommendation.
"""
        
        # Single LLM call for all pools
        # In production, call actual LLM here
        logger.info(f"Batch analyzing {len(pools)} pools")
        
        # Distribute results
        for i, req in enumerate(requests):
            if "callback" in req:
                await req["callback"](f"Analysis for pool {i+1}")
                
    async def _batch_pattern_detection(self, requests: List[Dict[str, Any]]):
        """Batch process pattern detection requests."""
        # Combine observations
        observations = [req["data"]["observation"] for req in requests]
        
        combined_prompt = f"""Detect patterns in these observations:
        
{json.dumps(observations, indent=2)}

Identify correlations, trends, and actionable insights.
"""
        
        # Single LLM call
        logger.info(f"Batch detecting patterns in {len(observations)} observations")
        
        # Distribute results
        for req in requests:
            if "callback" in req:
                await req["callback"]("Pattern detected")
                
    async def _process_single_request(self, request: Dict[str, Any]):
        """Process a single request."""
        # Check cache first
        cached = await self.get_cached_response(request["prompt"])
        if cached:
            if "callback" in request:
                await request["callback"](cached.response)
            return
            
        # Process normally
        logger.debug(f"Processing single request: {request.get('type')}")
        response = f"Processed: {request['prompt'][:50]}..."
        
        if "callback" in request:
            await request["callback"](response)
            
    async def optimize_prompt(self, prompt: str, context: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Optimize prompt before sending to LLM.
        
        Returns:
            Optimized prompt and optimization metadata
        """
        metadata = {
            "original_length": len(prompt),
            "optimizations": []
        }
        
        # 1. Check cache first
        cached = await self.get_cached_response(prompt)
        if cached:
            self.stats["cache_hits"] += 1
            self.stats["tokens_saved"] += cached.tokens_saved
            metadata["cache_hit"] = True
            metadata["tokens_saved"] = cached.tokens_saved
            return cached.response, metadata
            
        self.stats["cache_misses"] += 1
        
        # 2. Try template matching
        template_result = self._match_template(prompt, context)
        if template_result:
            self.stats["template_uses"] += 1
            metadata["template_used"] = template_result["template_name"]
            metadata["optimizations"].append("template")
            return template_result["response"], metadata
            
        # 3. Compress prompt
        optimized = self._compress_prompt(prompt)
        if len(optimized) < len(prompt):
            metadata["optimizations"].append("compression")
            metadata["compressed_length"] = len(optimized)
            
        # 4. Add to batch queue if enabled
        if settings.llm_batch_operations and context and context.get("allow_batch", True):
            metadata["optimizations"].append("batched")
            await self._add_to_batch(optimized, context)
            return "[BATCHED]", metadata
            
        return optimized, metadata
        
    def _match_template(self, prompt: str, context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Try to match prompt to a template."""
        import re
        
        for name, template in self.templates.items():
            if re.search(template.pattern, prompt.lower()):
                # Extract variables from context
                if context and all(var in context for var in template.variables):
                    response = template.template
                    for var in template.variables:
                        response = response.replace(f"{{{var}}}", str(context[var]))
                        
                    template.usage_count += 1
                    
                    return {
                        "response": response,
                        "template_name": name,
                        "tokens_saved": len(prompt) // 4  # Rough estimate
                    }
                    
        return None
        
    def _compress_prompt(self, prompt: str) -> str:
        """Compress prompt to reduce tokens."""
        # Remove redundant whitespace
        compressed = " ".join(prompt.split())
        
        # Common abbreviations
        abbreviations = {
            "annual percentage rate": "APR",
            "total value locked": "TVL",
            "impermanent loss": "IL",
            "liquidity provider": "LP",
            "automated market maker": "AMM"
        }
        
        for full, abbr in abbreviations.items():
            compressed = compressed.replace(full, abbr)
            compressed = compressed.replace(full.title(), abbr)
            
        return compressed
        
    async def _add_to_batch(self, prompt: str, context: Dict[str, Any]):
        """Add request to batch queue."""
        async with self.batch_lock:
            self.batch_queue.append({
                "prompt": prompt,
                "context": context,
                "timestamp": datetime.utcnow(),
                "type": context.get("request_type", "general")
            })
            
    async def get_cached_response(self, query: str) -> Optional[CachedResponse]:
        """
        Get cached response for query.
        
        Checks both direct and semantic caches.
        """
        # Check direct cache first (exact match)
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        if query_hash in self.direct_cache:
            cached = self.direct_cache[query_hash]
            
            # Check TTL
            if datetime.utcnow() - cached.timestamp < timedelta(hours=settings.llm_cache_ttl_hours):
                cached.hit_count += 1
                return cached
            else:
                # Expired
                del self.direct_cache[query_hash]
                
        # Check semantic cache
        return await self.semantic_cache.get(query)
        
    async def cache_response(self, query: str, response: str, tokens_used: int = 0):
        """Cache LLM response."""
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        
        cached = CachedResponse(
            query_hash=query_hash,
            response=response,
            semantic_hash=self.semantic_cache._compute_semantic_hash(query),
            timestamp=datetime.utcnow(),
            tokens_saved=tokens_used
        )
        
        # Add to both caches
        self.direct_cache[query_hash] = cached
        await self.semantic_cache.put(query, response, tokens_used)
        
        # Persist important caches to Firestore
        if tokens_used > 100:  # Only persist expensive queries
            await self._persist_cache_entry(cached)
            
    async def _persist_cache_entry(self, cached: CachedResponse):
        """Persist cache entry to Firestore."""
        try:
            await self.firestore.set_document(
                "llm_cache",
                cached.query_hash,
                {
                    "response": cached.response,
                    "semantic_hash": cached.semantic_hash,
                    "timestamp": cached.timestamp.isoformat(),
                    "tokens_saved": cached.tokens_saved,
                    "hit_count": cached.hit_count
                }
            )
        except Exception as e:
            logger.error(f"Failed to persist cache entry: {e}")
            
    async def load_cache_from_storage(self):
        """Load cache entries from Firestore on startup."""
        try:
            # Load recent cache entries
            cutoff = datetime.utcnow() - timedelta(hours=settings.llm_cache_ttl_hours)
            
            cache_entries = await self.firestore.query_documents(
                "llm_cache",
                [("timestamp", ">=", cutoff.isoformat())],
                limit=settings.llm_cache_size // 2  # Load half from storage
            )
            
            loaded = 0
            for entry in cache_entries:
                cached = CachedResponse(
                    query_hash=entry.get("id"),
                    response=entry.get("response"),
                    semantic_hash=entry.get("semantic_hash"),
                    timestamp=datetime.fromisoformat(entry.get("timestamp")),
                    tokens_saved=entry.get("tokens_saved", 0),
                    hit_count=entry.get("hit_count", 0)
                )
                
                self.direct_cache[cached.query_hash] = cached
                loaded += 1
                
            logger.info(f"Loaded {loaded} cache entries from storage")
            
        except Exception as e:
            logger.error(f"Failed to load cache from storage: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = self.stats["cache_hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hit_rate": cache_hit_rate,
            "total_requests": total_requests,
            "tokens_saved": self.stats["tokens_saved"],
            "template_uses": self.stats["template_uses"],
            "batch_requests": self.stats["batch_requests"],
            "cache_sizes": {
                "direct": len(self.direct_cache),
                "semantic": len(self.semantic_cache.cache)
            },
            "template_stats": {
                name: {
                    "usage_count": template.usage_count,
                    "avg_tokens_saved": template.avg_tokens_saved
                }
                for name, template in self.templates.items()
            }
        }
        
    async def clear_expired_cache(self):
        """Clear expired cache entries."""
        current_time = datetime.utcnow()
        ttl = timedelta(hours=settings.llm_cache_ttl_hours)
        
        # Clear direct cache
        expired_keys = [
            key for key, cached in self.direct_cache.items()
            if current_time - cached.timestamp > ttl
        ]
        
        for key in expired_keys:
            del self.direct_cache[key]
            
        # Clear semantic cache
        semantic_expired = [
            key for key, cached in self.semantic_cache.cache.items()
            if current_time - cached.timestamp > ttl
        ]
        
        for key in semantic_expired:
            del self.semantic_cache.cache[key]
            
        if expired_keys or semantic_expired:
            logger.info(
                f"Cleared {len(expired_keys)} direct and "
                f"{len(semantic_expired)} semantic expired cache entries"
            )
