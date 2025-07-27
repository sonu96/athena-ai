"""
Memory Management System for Athena AI

Implements tiered storage, automatic pruning, and performance optimization
to prevent memory explosion and maintain fast query performance.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from src.agent.memory import AthenaMemory, MemoryType, MemoryEntry
from src.gcp.firestore_client import FirestoreClient
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class MemoryTier:
    """Configuration for a memory storage tier."""
    name: str
    ttl: timedelta
    storage: str
    compression: bool = False
    
    
class MemoryStats:
    """Track memory system statistics."""
    
    def __init__(self):
        self.memories_pruned = 0
        self.patterns_merged = 0
        self.memories_archived = 0
        self.compression_ratio = 0.0
        self.last_prune_time = None
        

class MemoryManager:
    """
    Unified memory management with retention, pruning, and caching.
    
    Implements:
    - Tiered storage (Hot/Warm/Cold/Archive)
    - Automatic pruning and deduplication
    - Category quotas enforcement
    - Performance caching
    """
    
    def __init__(self, memory: AthenaMemory, firestore: FirestoreClient):
        """Initialize memory manager with dependencies."""
        self.memory = memory
        self.firestore = firestore
        self.stats = MemoryStats()
        
        # Storage tiers configuration
        self.tiers = {
            "hot": MemoryTier("hot", timedelta(hours=24), "mem0"),
            "warm": MemoryTier("warm", timedelta(days=7), "firestore_aggregated", True),
            "cold": MemoryTier("cold", timedelta(days=30), "firestore_summary", True),
            "archive": MemoryTier("archive", None, "gcs", True)
        }
        
        # Category limits
        self.category_limits = {
            "observation": settings.max_memories_observation,
            "pattern": settings.max_memories_pattern,
            "pool_behavior": settings.max_memories_pool_behavior,
            "cross_pool_correlation": settings.max_memories_correlation,
            "strategy_performance": 300,
            "error_learning": 50
        }
        
        # Performance caches (using dict for simplicity, can upgrade to TTLCache)
        self.pattern_cache = {}
        self.query_cache = {}
        
        # For similarity detection
        self.vectorizer = TfidfVectorizer(max_features=100)
        self._is_running = False
        
    async def start_pruning_service(self):
        """Start the hourly pruning service."""
        if self._is_running:
            logger.warning("Pruning service already running")
            return
            
        self._is_running = True
        logger.info("Starting memory pruning service")
        
        while self._is_running:
            try:
                await self.prune_memories()
                await asyncio.sleep(settings.memory_prune_interval_hours * 3600)
            except Exception as e:
                logger.error(f"Error in pruning service: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes
                
    async def stop_pruning_service(self):
        """Stop the pruning service."""
        self._is_running = False
        logger.info("Stopping memory pruning service")
        
    async def prune_memories(self):
        """Execute comprehensive memory pruning."""
        logger.info("Starting memory pruning cycle")
        start_time = datetime.utcnow()
        
        try:
            # 1. Remove low-confidence memories
            await self._prune_low_confidence()
            
            # 2. Merge similar patterns
            await self._merge_similar_patterns()
            
            # 3. Compress metadata
            await self._compress_metadata()
            
            # 4. Archive old memories
            await self._archive_old_memories()
            
            # 5. Enforce category quotas
            await self._enforce_quotas()
            
            # 6. Update statistics
            self.stats.last_prune_time = datetime.utcnow()
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(
                f"Memory pruning completed in {duration:.2f}s. "
                f"Pruned: {self.stats.memories_pruned}, "
                f"Merged: {self.stats.patterns_merged}, "
                f"Archived: {self.stats.memories_archived}"
            )
            
        except Exception as e:
            logger.error(f"Error during memory pruning: {e}")
            raise
            
    async def _prune_low_confidence(self):
        """Remove memories with low confidence after 48 hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=48)
        
        # Get all memories (simplified - in production, use batching)
        all_memories = await self.memory.get_all_memories()
        
        to_delete = []
        for memory in all_memories:
            if (memory.confidence < settings.memory_confidence_threshold and
                memory.timestamp < cutoff_time):
                to_delete.append(memory.id)
                
        if to_delete:
            logger.info(f"Pruning {len(to_delete)} low-confidence memories")
            for memory_id in to_delete:
                await self.memory.delete_memory(memory_id)
                self.stats.memories_pruned += 1
                
    async def _merge_similar_patterns(self):
        """Merge patterns with high similarity."""
        # Get all pattern memories
        patterns = await self.memory.recall(
            query="",
            memory_type=MemoryType.PATTERN,
            limit=1000
        )
        
        if len(patterns) < 2:
            return
            
        # Extract pattern contents
        pattern_texts = [p.get("content", "") for p in patterns]
        
        # Calculate similarity matrix
        try:
            tfidf_matrix = self.vectorizer.fit_transform(pattern_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except Exception as e:
            logger.error(f"Error calculating pattern similarity: {e}")
            return
            
        # Find similar pairs
        merged_count = 0
        merged_indices = set()
        
        for i in range(len(patterns)):
            if i in merged_indices:
                continue
                
            for j in range(i + 1, len(patterns)):
                if j in merged_indices:
                    continue
                    
                if similarity_matrix[i][j] > settings.memory_similarity_threshold:
                    # Merge patterns
                    await self._merge_patterns(patterns[i], patterns[j])
                    merged_indices.add(j)
                    merged_count += 1
                    
        self.stats.patterns_merged += merged_count
        logger.info(f"Merged {merged_count} similar patterns")
        
    async def _merge_patterns(self, pattern1: Dict, pattern2: Dict):
        """Merge two similar patterns into one."""
        # Combine metadata
        merged_metadata = {
            **pattern1.get("metadata", {}),
            **pattern2.get("metadata", {})
        }
        
        # Update occurrence count
        merged_metadata["occurrences"] = (
            pattern1.get("metadata", {}).get("occurrences", 1) +
            pattern2.get("metadata", {}).get("occurrences", 1)
        )
        
        # Use higher confidence
        merged_confidence = max(
            pattern1.get("confidence", 0.5),
            pattern2.get("confidence", 0.5)
        )
        
        # Update first pattern
        await self.memory.update_memory(
            memory_id=pattern1["id"],
            metadata=merged_metadata,
            confidence=merged_confidence
        )
        
        # Delete second pattern
        await self.memory.delete_memory(pattern2["id"])
        
    async def _compress_metadata(self):
        """Compress metadata for memories older than 24 hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        # Get memories to compress
        memories = await self.memory.get_memories_by_age(min_age=cutoff_time)
        
        compressed_count = 0
        for memory in memories:
            metadata = memory.get("metadata", {})
            if not metadata:
                continue
                
            # Compress metadata
            compressed = self._compress_memory_metadata(metadata)
            
            if len(json.dumps(compressed)) < len(json.dumps(metadata)):
                await self.memory.update_memory(
                    memory_id=memory["id"],
                    metadata=compressed
                )
                compressed_count += 1
                
        logger.info(f"Compressed metadata for {compressed_count} memories")
        
    def _compress_memory_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Compress metadata to essential fields only."""
        essential_fields = [
            "pool", "apr", "tvl", "volume", "timestamp",
            "pattern_type", "confidence", "occurrences"
        ]
        
        compressed = {}
        for field in essential_fields:
            if field in metadata:
                # Use abbreviated keys
                key = self._abbreviate_key(field)
                compressed[key] = metadata[field]
                
        return compressed
        
    def _abbreviate_key(self, key: str) -> str:
        """Abbreviate metadata keys to save space."""
        abbreviations = {
            "pool": "p",
            "apr": "a",
            "tvl": "t",
            "volume": "v",
            "timestamp": "ts",
            "pattern_type": "pt",
            "confidence": "c",
            "occurrences": "o"
        }
        return abbreviations.get(key, key[:3])
        
    async def _archive_old_memories(self):
        """Archive memories older than 30 days."""
        archive_cutoff = datetime.utcnow() - timedelta(days=30)
        
        # Get old memories
        old_memories = await self.memory.get_memories_by_age(max_age=archive_cutoff)
        
        if not old_memories:
            return
            
        # Extract patterns and learnings
        patterns_to_archive = []
        for memory in old_memories:
            if memory.get("type") in [MemoryType.PATTERN, MemoryType.LEARNING]:
                patterns_to_archive.append({
                    "content": memory.get("content"),
                    "metadata": self._compress_memory_metadata(memory.get("metadata", {})),
                    "confidence": memory.get("confidence", 0.5),
                    "created_at": memory.get("timestamp")
                })
                
        # Archive to GCS (simplified - would use actual GCS client)
        if patterns_to_archive:
            archive_doc = {
                "archived_at": datetime.utcnow(),
                "patterns": patterns_to_archive,
                "count": len(patterns_to_archive)
            }
            
            await self.firestore.set_document(
                "memory_archives",
                f"archive_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                archive_doc
            )
            
            self.stats.memories_archived += len(patterns_to_archive)
            
        # Delete old memories
        for memory in old_memories:
            await self.memory.delete_memory(memory["id"])
            
        logger.info(f"Archived {len(patterns_to_archive)} old memories")
        
    async def _enforce_quotas(self):
        """Enforce memory quotas per category."""
        for category, limit in self.category_limits.items():
            # Count memories in category
            count = await self.memory.count_by_category(category)
            
            if count > limit:
                excess = count - limit
                logger.warning(
                    f"Category '{category}' exceeds limit: {count}/{limit}. "
                    f"Removing {excess} oldest memories."
                )
                
                # Get oldest memories in category
                oldest = await self.memory.get_oldest_in_category(
                    category=category,
                    limit=excess
                )
                
                # Delete excess memories
                for memory in oldest:
                    await self.memory.delete_memory(memory["id"])
                    self.stats.memories_pruned += 1
                    
    async def optimize_memory_query(self, query: str, category: Optional[str] = None,
                                  pool: Optional[str] = None) -> List[Dict]:
        """Optimized memory query with caching and performance enhancements."""
        # Generate cache key
        cache_key = f"{category}:{pool}:{hash(query)}"
        
        # Check cache
        if cache_key in self.query_cache:
            logger.debug(f"Query cache hit for: {cache_key}")
            return self.query_cache[cache_key]
            
        # Check if category exists (bloom filter simulation)
        if category and not await self._category_might_exist(category):
            return []
            
        # Execute optimized query
        results = await self.memory.recall(
            query=query,
            category=category,
            limit=50,
            min_confidence=settings.min_pattern_confidence
        )
        
        # Filter by pool if specified
        if pool:
            results = [
                r for r in results
                if r.get("metadata", {}).get("pool") == pool
            ]
            
        # Cache results
        self.query_cache[cache_key] = results
        
        # Implement simple cache eviction (keep last 500 entries)
        if len(self.query_cache) > settings.query_cache_size:
            # Remove oldest entries
            keys_to_remove = list(self.query_cache.keys())[:-settings.query_cache_size]
            for key in keys_to_remove:
                del self.query_cache[key]
                
        return results
        
    async def _category_might_exist(self, category: str) -> bool:
        """Simple bloom filter simulation for category existence."""
        # In production, implement actual bloom filter
        count = await self.memory.count_by_category(category)
        return count > 0
        
    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        return {
            "memories_pruned": self.stats.memories_pruned,
            "patterns_merged": self.stats.patterns_merged,
            "memories_archived": self.stats.memories_archived,
            "last_prune_time": self.stats.last_prune_time.isoformat() if self.stats.last_prune_time else None,
            "cache_sizes": {
                "pattern_cache": len(self.pattern_cache),
                "query_cache": len(self.query_cache)
            }
        }
        
    async def migrate_to_tier(self, memory_id: str, from_tier: str, to_tier: str):
        """Migrate memory between storage tiers."""
        logger.info(f"Migrating memory {memory_id} from {from_tier} to {to_tier}")
        
        # Get memory from source
        memory = await self.memory.get_memory_by_id(memory_id)
        if not memory:
            logger.error(f"Memory {memory_id} not found")
            return
            
        # Apply compression if needed
        if self.tiers[to_tier].compression:
            memory["metadata"] = self._compress_memory_metadata(
                memory.get("metadata", {})
            )
            
        # Store in destination
        if to_tier == "firestore_aggregated":
            await self.firestore.set_document(
                "memories_warm",
                memory_id,
                memory
            )
        elif to_tier == "firestore_summary":
            # Store only summary
            summary = {
                "id": memory_id,
                "type": memory.get("type"),
                "category": memory.get("category"),
                "confidence": memory.get("confidence"),
                "created_at": memory.get("timestamp")
            }
            await self.firestore.set_document(
                "memories_cold",
                memory_id,
                summary
            )
            
        # Remove from source if different storage
        if from_tier == "mem0" and to_tier != "mem0":
            await self.memory.delete_memory(memory_id)