"""
Firestore Write Optimization System

Implements batching, aggregation, and rate limiting to reduce Firestore
write costs and improve performance.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json

from google.cloud import firestore
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class WriteOperation:
    """Represents a pending write operation."""
    collection: str
    document_id: str
    data: Dict[str, Any]
    operation_type: str  # "set", "update", "delete"
    timestamp: datetime
    priority: int = 0  # Higher priority = process sooner
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class BatchStats:
    """Statistics for batch operations."""
    total_writes: int = 0
    batched_writes: int = 0
    bytes_saved: int = 0
    cost_saved: float = 0.0
    avg_batch_size: float = 0.0
    

class WriteBatcher:
    """Batches multiple write operations together."""
    
    def __init__(self, firestore_client: firestore.Client, 
                 batch_size: int = 500, flush_interval: int = 60):
        """Initialize write batcher."""
        self.client = firestore_client
        self.batch_size = batch_size
        self.flush_interval = timedelta(seconds=flush_interval)
        
        # Pending operations by collection
        self.pending_ops: Dict[str, List[WriteOperation]] = defaultdict(list)
        self.last_flush: Dict[str, datetime] = {}
        self.stats = BatchStats()
        
        # Rate limiting
        self.write_history = deque(maxlen=settings.firestore_max_writes_per_minute)
        
        # Locks
        self.batch_lock = asyncio.Lock()
        self._is_running = False
        self._flush_task = None
        
    async def start(self):
        """Start the batch flushing service."""
        if self._is_running:
            return
            
        self._is_running = True
        self._flush_task = asyncio.create_task(self._flush_periodically())
        logger.info("Started Firestore write batcher")
        
    async def stop(self):
        """Stop the batch flushing service."""
        self._is_running = False
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
                
        # Flush any remaining operations
        await self.flush_all()
        logger.info("Stopped Firestore write batcher")
        
    async def _flush_periodically(self):
        """Periodically flush pending operations."""
        while self._is_running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Find collections that need flushing
                collections_to_flush = []
                current_time = datetime.utcnow()
                
                async with self.batch_lock:
                    for collection, ops in self.pending_ops.items():
                        if not ops:
                            continue
                            
                        last_flush = self.last_flush.get(collection, datetime.min)
                        time_since_flush = current_time - last_flush
                        
                        # Flush if time limit reached or batch is full
                        if (time_since_flush >= self.flush_interval or 
                            len(ops) >= self.batch_size):
                            collections_to_flush.append(collection)
                            
                # Flush collections outside lock
                for collection in collections_to_flush:
                    await self._flush_collection(collection)
                    
            except Exception as e:
                logger.error(f"Error in batch flush service: {e}")
                await asyncio.sleep(10)
                
    async def add_write(self, operation: WriteOperation) -> bool:
        """
        Add write operation to batch.
        
        Returns:
            True if added to batch, False if rate limited
        """
        # Check rate limit
        if not self._check_rate_limit():
            logger.warning("Write rate limit exceeded")
            return False
            
        async with self.batch_lock:
            self.pending_ops[operation.collection].append(operation)
            
            # Check if immediate flush needed
            if len(self.pending_ops[operation.collection]) >= self.batch_size:
                await self._flush_collection(operation.collection)
                
        self.stats.total_writes += 1
        return True
        
    def _check_rate_limit(self) -> bool:
        """Check if write rate is within limits."""
        current_time = datetime.utcnow()
        
        # Clean old entries
        cutoff = current_time - timedelta(minutes=1)
        while self.write_history and self.write_history[0] < cutoff:
            self.write_history.popleft()
            
        # Check limit
        if len(self.write_history) >= settings.firestore_max_writes_per_minute:
            return False
            
        self.write_history.append(current_time)
        return True
        
    async def _flush_collection(self, collection: str):
        """Flush all pending operations for a collection."""
        async with self.batch_lock:
            ops = self.pending_ops[collection]
            if not ops:
                return
                
            # Take operations and clear
            operations_to_flush = ops.copy()
            self.pending_ops[collection] = []
            self.last_flush[collection] = datetime.utcnow()
            
        # Execute batch outside lock
        await self._execute_batch(collection, operations_to_flush)
        
    async def _execute_batch(self, collection: str, operations: List[WriteOperation]):
        """Execute a batch of write operations."""
        if not operations:
            return
            
        logger.debug(f"Flushing {len(operations)} operations for {collection}")
        
        try:
            # Group by operation type for efficiency
            sets = [op for op in operations if op.operation_type == "set"]
            updates = [op for op in operations if op.operation_type == "update"]
            deletes = [op for op in operations if op.operation_type == "delete"]
            
            # Execute in batches of max 500 (Firestore limit)
            for i in range(0, len(operations), 500):
                batch = self.client.batch()
                batch_ops = operations[i:i+500]
                
                for op in batch_ops:
                    doc_ref = self.client.collection(op.collection).document(op.document_id)
                    
                    if op.operation_type == "set":
                        batch.set(doc_ref, op.data)
                    elif op.operation_type == "update":
                        batch.update(doc_ref, op.data)
                    elif op.operation_type == "delete":
                        batch.delete(doc_ref)
                        
                # Commit batch
                await asyncio.get_event_loop().run_in_executor(
                    None, batch.commit
                )
                
                self.stats.batched_writes += len(batch_ops)
                
            # Update statistics
            self._update_stats(operations)
            
        except Exception as e:
            logger.error(f"Batch execution failed for {collection}: {e}")
            # In production, implement retry logic or dead letter queue
            
    def _update_stats(self, operations: List[WriteOperation]):
        """Update batch statistics."""
        # Estimate bytes saved by batching
        total_size = sum(len(json.dumps(op.data)) for op in operations)
        overhead_saved = len(operations) * 100  # Rough estimate of per-write overhead
        
        self.stats.bytes_saved += overhead_saved
        self.stats.cost_saved += overhead_saved * 0.000001  # Rough cost estimate
        
        # Update average batch size
        if self.stats.batched_writes > 0:
            self.stats.avg_batch_size = (
                self.stats.batched_writes / 
                max(1, self.stats.total_writes - self.stats.batched_writes)
            )
            
    async def flush_all(self):
        """Flush all pending operations."""
        collections = list(self.pending_ops.keys())
        for collection in collections:
            await self._flush_collection(collection)
            
            
class WriteAggregator:
    """Aggregates multiple updates to the same document."""
    
    def __init__(self):
        """Initialize write aggregator."""
        self.pending_updates: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.update_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.lock = asyncio.Lock()
        
    async def add_update(self, collection: str, document_id: str, 
                        updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add updates to aggregator.
        
        Returns:
            Aggregated updates for the document
        """
        key = (collection, document_id)
        
        async with self.lock:
            if key not in self.pending_updates:
                self.pending_updates[key] = {}
                
            # Merge updates
            for field, value in updates.items():
                if field.startswith("__"):  # Special aggregation fields
                    self._aggregate_field(key, field, value)
                else:
                    self.pending_updates[key][field] = value
                    
            self.update_counts[key] += 1
            
            return self.pending_updates[key].copy()
            
    def _aggregate_field(self, key: Tuple[str, str], field: str, value: Any):
        """Handle special aggregation fields."""
        if field == "__increment":
            # Increment numeric fields
            for subfield, increment in value.items():
                current = self.pending_updates[key].get(subfield, 0)
                self.pending_updates[key][subfield] = current + increment
                
        elif field == "__append":
            # Append to array fields
            for subfield, items in value.items():
                current = self.pending_updates[key].get(subfield, [])
                if not isinstance(current, list):
                    current = []
                current.extend(items)
                self.pending_updates[key][subfield] = current
                
        elif field == "__merge":
            # Deep merge objects
            for subfield, obj in value.items():
                current = self.pending_updates[key].get(subfield, {})
                if isinstance(current, dict) and isinstance(obj, dict):
                    current.update(obj)
                    self.pending_updates[key][subfield] = current
                else:
                    self.pending_updates[key][subfield] = obj
                    
    async def get_and_clear(self, collection: str, document_id: str) -> Optional[Dict[str, Any]]:
        """Get aggregated updates and clear."""
        key = (collection, document_id)
        
        async with self.lock:
            if key in self.pending_updates:
                updates = self.pending_updates[key]
                del self.pending_updates[key]
                
                count = self.update_counts[key]
                del self.update_counts[key]
                
                logger.debug(
                    f"Aggregated {count} updates for {collection}/{document_id}"
                )
                
                return updates
                
        return None
        
        
class FirestoreOptimizer:
    """
    Main Firestore optimization coordinator.
    
    Features:
    - Write batching and aggregation
    - Read caching
    - Query optimization
    - Cost tracking
    """
    
    def __init__(self, firestore_client: firestore.Client):
        """Initialize Firestore optimizer."""
        self.client = firestore_client
        
        # Components
        self.write_batcher = WriteBatcher(
            firestore_client,
            batch_size=settings.firestore_batch_size,
            flush_interval=settings.firestore_flush_interval_seconds
        )
        self.write_aggregator = WriteAggregator()
        
        # Read cache
        self.read_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(seconds=settings.firestore_read_cache_ttl)
        
        # Query optimization
        self.query_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "avg_time": 0.0,
            "last_used": datetime.min
        })
        
    async def start(self):
        """Start optimization services."""
        await self.write_batcher.start()
        logger.info("Started Firestore optimizer")
        
    async def stop(self):
        """Stop optimization services."""
        await self.write_batcher.stop()
        logger.info("Stopped Firestore optimizer")
        
    async def set_document(self, collection: str, document_id: str, 
                          data: Dict[str, Any], priority: int = 0) -> bool:
        """
        Optimized document set operation.
        
        Args:
            collection: Collection name
            document_id: Document ID
            data: Document data
            priority: Write priority (higher = sooner)
            
        Returns:
            True if queued successfully
        """
        # Create write operation
        operation = WriteOperation(
            collection=collection,
            document_id=document_id,
            data=data,
            operation_type="set",
            timestamp=datetime.utcnow(),
            priority=priority
        )
        
        # Invalidate cache
        cache_key = f"{collection}:{document_id}"
        if cache_key in self.read_cache:
            del self.read_cache[cache_key]
            
        # Add to batch
        return await self.write_batcher.add_write(operation)
        
    async def update_document(self, collection: str, document_id: str,
                            updates: Dict[str, Any], aggregate: bool = True) -> bool:
        """
        Optimized document update operation.
        
        Args:
            collection: Collection name
            document_id: Document ID
            updates: Fields to update
            aggregate: Whether to aggregate with pending updates
            
        Returns:
            True if queued successfully
        """
        # Aggregate if enabled
        if aggregate:
            aggregated = await self.write_aggregator.add_update(
                collection, document_id, updates
            )
            # Don't write yet - wait for more updates
            return True
            
        # Create write operation
        operation = WriteOperation(
            collection=collection,
            document_id=document_id,
            data=updates,
            operation_type="update",
            timestamp=datetime.utcnow()
        )
        
        # Invalidate cache
        cache_key = f"{collection}:{document_id}"
        if cache_key in self.read_cache:
            del self.read_cache[cache_key]
            
        # Add to batch
        return await self.write_batcher.add_write(operation)
        
    async def flush_aggregated_updates(self, collection: str, document_id: str) -> bool:
        """
        Flush any aggregated updates for a document.
        
        Returns:
            True if updates were flushed
        """
        updates = await self.write_aggregator.get_and_clear(collection, document_id)
        
        if updates:
            return await self.update_document(
                collection, document_id, updates, aggregate=False
            )
            
        return False
        
    async def get_document(self, collection: str, document_id: str,
                          use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get document with caching.
        
        Args:
            collection: Collection name
            document_id: Document ID
            use_cache: Whether to use cache
            
        Returns:
            Document data or None
        """
        cache_key = f"{collection}:{document_id}"
        
        # Check cache
        if use_cache and cache_key in self.read_cache:
            data, timestamp = self.read_cache[cache_key]
            if datetime.utcnow() - timestamp < self.cache_ttl:
                logger.debug(f"Cache hit for {cache_key}")
                return data
            else:
                # Expired
                del self.read_cache[cache_key]
                
        # Fetch from Firestore
        try:
            doc_ref = self.client.collection(collection).document(document_id)
            doc = await asyncio.get_event_loop().run_in_executor(
                None, doc_ref.get
            )
            
            if doc.exists:
                data = doc.to_dict()
                # Cache result
                self.read_cache[cache_key] = (data, datetime.utcnow())
                return data
                
        except Exception as e:
            logger.error(f"Error getting document {cache_key}: {e}")
            
        return None
        
    async def query_documents(self, collection: str, filters: List[Tuple[str, str, Any]],
                            order_by: Optional[List[Tuple[str, str]]] = None,
                            limit: Optional[int] = None,
                            optimize: bool = True) -> List[Dict[str, Any]]:
        """
        Optimized query execution.
        
        Args:
            collection: Collection name
            filters: List of (field, operator, value) tuples
            order_by: List of (field, direction) tuples
            limit: Maximum results
            optimize: Whether to apply optimizations
            
        Returns:
            List of documents
        """
        # Build query signature for stats
        query_sig = f"{collection}:{filters}:{order_by}:{limit}"
        query_hash = hashlib.sha256(query_sig.encode()).hexdigest()[:8]
        
        # Track query stats
        start_time = datetime.utcnow()
        
        try:
            # Build query
            query = self.client.collection(collection)
            
            # Apply filters
            for field, op, value in filters:
                query = query.where(field, op, value)
                
            # Apply ordering
            if order_by:
                for field, direction in order_by:
                    query = query.order_by(
                        field,
                        direction=firestore.Query.ASCENDING if direction == "asc" 
                        else firestore.Query.DESCENDING
                    )
                    
            # Apply limit
            if limit:
                query = query.limit(limit)
                
            # Execute query
            results = await asyncio.get_event_loop().run_in_executor(
                None, lambda: list(query.stream())
            )
            
            documents = [doc.to_dict() for doc in results]
            
            # Update query stats
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            stats = self.query_stats[query_hash]
            stats["count"] += 1
            stats["avg_time"] = (
                (stats["avg_time"] * (stats["count"] - 1) + elapsed) / 
                stats["count"]
            )
            stats["last_used"] = datetime.utcnow()
            
            # Log slow queries
            if elapsed > 1.0:
                logger.warning(
                    f"Slow query detected ({elapsed:.2f}s): {query_sig[:100]}..."
                )
                
            return documents
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
            
    async def bulk_get_documents(self, references: List[Tuple[str, str]]) -> Dict[str, Dict[str, Any]]:
        """
        Efficiently get multiple documents.
        
        Args:
            references: List of (collection, document_id) tuples
            
        Returns:
            Dict mapping document path to data
        """
        results = {}
        uncached_refs = []
        
        # Check cache first
        for collection, doc_id in references:
            cache_key = f"{collection}:{doc_id}"
            
            if cache_key in self.read_cache:
                data, timestamp = self.read_cache[cache_key]
                if datetime.utcnow() - timestamp < self.cache_ttl:
                    results[cache_key] = data
                else:
                    uncached_refs.append((collection, doc_id))
            else:
                uncached_refs.append((collection, doc_id))
                
        # Fetch uncached documents
        if uncached_refs:
            # In production, use getAll() for efficiency
            for collection, doc_id in uncached_refs:
                data = await self.get_document(collection, doc_id, use_cache=False)
                if data:
                    cache_key = f"{collection}:{doc_id}"
                    results[cache_key] = data
                    
        return results
        
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "write_stats": {
                "total_writes": self.write_batcher.stats.total_writes,
                "batched_writes": self.write_batcher.stats.batched_writes,
                "batch_efficiency": (
                    self.write_batcher.stats.batched_writes / 
                    max(1, self.write_batcher.stats.total_writes)
                ),
                "bytes_saved": self.write_batcher.stats.bytes_saved,
                "cost_saved": self.write_batcher.stats.cost_saved
            },
            "cache_stats": {
                "read_cache_size": len(self.read_cache),
                "cache_hit_rate": self._calculate_cache_hit_rate()
            },
            "query_stats": {
                "unique_queries": len(self.query_stats),
                "slow_queries": sum(
                    1 for stats in self.query_stats.values() 
                    if stats["avg_time"] > 1.0
                )
            }
        }
        
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified)."""
        # In production, track actual hits/misses
        return 0.7  # Placeholder