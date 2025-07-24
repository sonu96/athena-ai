"""
Athena's Memory System using Mem0
"""
import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from enum import Enum

from mem0 import Memory, MemoryClient
from pydantic import BaseModel, Field
from config.settings import settings, MEMORY_CATEGORIES

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of memories Athena can form."""
    OBSERVATION = "observation"
    PATTERN = "pattern"
    STRATEGY = "strategy"
    OUTCOME = "outcome"
    LEARNING = "learning"
    ERROR = "error"


class MemoryEntry(BaseModel):
    """Schema for a memory entry."""
    id: Optional[str] = None
    type: MemoryType
    category: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    references: List[str] = Field(default_factory=list)  # IDs of related memories
    

class AthenaMemory:
    """
    Athena's memory system for learning and pattern recognition.
    
    Uses Mem0 for vector storage and retrieval with additional
    structure for DeFi-specific memories.
    """
    
    def __init__(self):
        """Initialize memory system."""
        # Initialize Mem0
        # Initialize memory based on configuration
        if settings.mem0_api_key:
            # Use Mem0 cloud with API key
            self.memory = MemoryClient(api_key=settings.mem0_api_key)
        else:
            # Use simple in-memory storage for now
            self.memory = None
            self._local_memories = []
        self.user_id = "athena_agent"
        
        # Memory statistics
        self.stats = {
            "total_memories": 0,
            "patterns_discovered": 0,
            "successful_strategies": 0,
            "failed_strategies": 0,
        }
        
    async def remember(
        self,
        content: str,
        memory_type: MemoryType,
        category: str,
        metadata: Optional[Dict] = None,
        confidence: float = 1.0,
        references: Optional[List[str]] = None
    ) -> str:
        """
        Store a new memory.
        
        Args:
            content: The memory content
            memory_type: Type of memory
            category: Category (from MEMORY_CATEGORIES)
            metadata: Additional metadata
            confidence: Confidence score (0-1)
            references: Related memory IDs
            
        Returns:
            Memory ID
        """
        try:
            # Validate category
            if category not in MEMORY_CATEGORIES:
                logger.warning(f"Unknown category: {category}")
                category = "general"
                
            # Create memory entry
            entry = MemoryEntry(
                type=memory_type,
                category=category,
                content=content,
                metadata=metadata or {},
                confidence=confidence,
                references=references or []
            )
            
            # Add to Mem0
            # Ensure content is JSON serializable
            if isinstance(content, dict):
                import json
                # Custom JSON encoder for Decimal and other types
                def decimal_default(obj):
                    if isinstance(obj, Decimal):
                        return float(obj)
                    elif isinstance(obj, datetime):
                        return obj.isoformat()
                    elif hasattr(obj, '__dict__'):
                        return str(obj)
                    return str(obj)
                content_str = json.dumps(content, default=decimal_default)
            else:
                content_str = str(content)
                
            messages = [{
                "role": "assistant",
                "content": f"[{memory_type.value}] {content_str}"
            }]
            
            if self.memory:
                # Ensure all metadata values are JSON serializable
                def convert_value(obj):
                    """Recursively convert values for JSON serialization."""
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    elif isinstance(obj, Decimal):
                        return float(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_value(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_value(item) for item in obj]
                    elif hasattr(obj, '__dict__'):
                        return str(obj)
                    else:
                        return obj
                
                safe_metadata = {}
                for k, v in entry.metadata.items():
                    safe_metadata[k] = convert_value(v)
                
                # Prepare full metadata
                full_metadata = {
                    **safe_metadata,
                    "type": memory_type.value,
                    "category": category,
                    "confidence": confidence,
                    "timestamp": entry.timestamp.isoformat(),
                }
                
                # Check metadata size and limit if necessary
                import json
                metadata_str = json.dumps(full_metadata)
                if len(metadata_str) > 1900:  # Mem0 has 2000 char limit, leave buffer
                    # Keep only essential fields
                    limited_metadata = {
                        "type": memory_type.value,
                        "category": category,
                        "confidence": confidence,
                        "timestamp": entry.timestamp.isoformat(),
                    }
                    # Add most important custom fields if they exist
                    for key in ["pool", "apr", "tvl", "volume", "pattern_type"]:
                        if key in safe_metadata:
                            limited_metadata[key] = safe_metadata[key]
                    full_metadata = limited_metadata
                
                result = self.memory.add(
                    messages=messages,
                    user_id=self.user_id,
                    metadata=full_metadata
                )
                
                # Handle different response formats from Mem0
                if isinstance(result, list) and len(result) > 0:
                    # New format returns a list
                    first_item = result[0]
                    if isinstance(first_item, dict):
                        memory_id = first_item.get('id', '')
                    else:
                        # If it's a string ID directly
                        memory_id = str(first_item) if first_item else ''
                elif isinstance(result, dict):
                    # Old format
                    memory_id = result.get('id', result.get('results', [{}])[0].get('id', ''))
                else:
                    logger.warning(f"Unexpected Mem0 response format: {type(result)}")
                    memory_id = ''
            else:
                # Use local storage
                import uuid
                memory_id = str(uuid.uuid4())
                self._local_memories.append({
                    "id": memory_id,
                    "entry": entry,
                    "messages": messages,
                    "timestamp": datetime.utcnow()
                })
            entry.id = memory_id
            
            # Update stats
            self.stats["total_memories"] += 1
            if memory_type == MemoryType.PATTERN:
                self.stats["patterns_discovered"] += 1
                
            logger.info(f"Stored memory {memory_id}: {content[:50]}...")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return ""
            
    async def recall(
        self,
        query: str,
        category: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 5,
        min_confidence: float = 0.5
    ) -> List[Dict]:
        """
        Recall relevant memories.
        
        Args:
            query: Search query
            category: Filter by category
            memory_type: Filter by type
            limit: Maximum memories to return
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of relevant memories
        """
        try:
            # Build filters
            filters = {}
            if category:
                filters["category"] = category
            if memory_type:
                filters["type"] = memory_type.value
                
            # Search memories
            if self.memory:
                results = self.memory.search(
                    query=query,
                    user_id=self.user_id,
                    limit=limit * 2,  # Get extra to filter by confidence
                    filters=filters
                )
            else:
                # Simple search in local storage
                results = []
                for mem in self._local_memories:
                    entry = mem["entry"]
                    # Check filters
                    if category and entry.category != category:
                        continue
                    if memory_type and entry.type != memory_type:
                        continue
                    # Simple text match
                    if query.lower() in mem["messages"][0]["content"].lower():
                        results.append({
                            "id": mem["id"],
                            "metadata": {
                                "confidence": entry.confidence,
                                "category": entry.category,
                                "type": entry.type.value
                            },
                            "content": mem["messages"][0]["content"]
                        })
            
            # Filter by confidence and limit
            filtered_results = []
            for result in results:
                if result.get("metadata", {}).get("confidence", 1.0) >= min_confidence:
                    filtered_results.append(result)
                if len(filtered_results) >= limit:
                    break
                    
            return filtered_results
            
        except Exception as e:
            logger.error(f"Failed to recall memories: {e}")
            return []
            
    async def find_patterns(
        self,
        observations: List[str],
        min_occurrences: int = 3
    ) -> List[Dict]:
        """
        Analyze observations to find patterns.
        
        Args:
            observations: List of observations
            min_occurrences: Minimum occurrences to consider a pattern
            
        Returns:
            Discovered patterns
        """
        patterns = []
        
        try:
            # Group similar observations
            observation_groups = {}
            
            for obs in observations:
                # Search for similar past observations
                similar = await self.recall(
                    query=obs,
                    memory_type=MemoryType.OBSERVATION,
                    limit=10,
                    min_confidence=0.7
                )
                
                # Group by similarity
                for sim in similar:
                    key = sim.get("content", "")[:50]  # Use first 50 chars as key
                    if key not in observation_groups:
                        observation_groups[key] = []
                    observation_groups[key].append(sim)
                    
            # Identify patterns
            for key, group in observation_groups.items():
                if len(group) >= min_occurrences:
                    # Extract pattern
                    pattern = {
                        "pattern": f"Repeated observation: {key}",
                        "occurrences": len(group),
                        "confidence": min(0.9, len(group) / 10),  # Higher occurrences = higher confidence
                        "examples": group[:3]  # Keep first 3 examples
                    }
                    patterns.append(pattern)
                    
                    # Store as pattern memory
                    await self.remember(
                        content=pattern["pattern"],
                        memory_type=MemoryType.PATTERN,
                        category="market_pattern",
                        metadata={
                            "occurrences": pattern["occurrences"],
                            "discovered_at": datetime.utcnow().isoformat()
                        },
                        confidence=pattern["confidence"]
                    )
                    
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to find patterns: {e}")
            return []
            
    async def learn_from_outcome(
        self,
        strategy: str,
        outcome: Dict,
        success: bool
    ) -> None:
        """
        Learn from strategy execution outcome.
        
        Args:
            strategy: Strategy description
            outcome: Execution results
            success: Whether it was successful
        """
        try:
            # Custom JSON encoder for Decimal and other types
            def decimal_default(obj):
                if isinstance(obj, Decimal):
                    return float(obj)
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                elif hasattr(obj, '__dict__'):
                    return str(obj)
                return str(obj)
            
            # Store outcome
            await self.remember(
                content=f"Strategy '{strategy}' {'succeeded' if success else 'failed'}: {json.dumps(outcome, default=decimal_default)}",
                memory_type=MemoryType.OUTCOME,
                category="strategy_performance",
                metadata={
                    "strategy": strategy,
                    "success": success,
                    "profit": outcome.get("profit", 0),
                    "gas_used": outcome.get("gas_used", 0),
                },
                confidence=1.0
            )
            
            # Update stats
            if success:
                self.stats["successful_strategies"] += 1
            else:
                self.stats["failed_strategies"] += 1
                
            # Extract learnings
            if success:
                learning = f"Strategy '{strategy}' is effective under current conditions"
                confidence = 0.8
            else:
                learning = f"Strategy '{strategy}' needs adjustment: {outcome.get('error', 'Unknown error')}"
                confidence = 0.9
                
            await self.remember(
                content=learning,
                memory_type=MemoryType.LEARNING,
                category="strategy_performance",
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Failed to learn from outcome: {e}")
            
    async def get_strategy_performance(self, strategy: str) -> Dict:
        """Get performance metrics for a strategy."""
        try:
            # Search for strategy outcomes
            outcomes = await self.recall(
                query=strategy,
                memory_type=MemoryType.OUTCOME,
                category="strategy_performance",
                limit=100
            )
            
            if not outcomes:
                return {
                    "total_executions": 0,
                    "success_rate": 0,
                    "avg_profit": 0,
                    "total_profit": 0,
                }
                
            # Calculate metrics
            total = len(outcomes)
            successful = sum(1 for o in outcomes if o.get("metadata", {}).get("success", False))
            total_profit = sum(o.get("metadata", {}).get("profit", 0) for o in outcomes)
            
            return {
                "total_executions": total,
                "success_rate": successful / total if total > 0 else 0,
                "avg_profit": total_profit / total if total > 0 else 0,
                "total_profit": total_profit,
                "recent_outcomes": outcomes[:5]  # Last 5 outcomes
            }
            
        except Exception as e:
            logger.error(f"Failed to get strategy performance: {e}")
            return {}
            
    async def recall_pool_memories(
        self, 
        pool_pair: str,
        memory_type: Optional[MemoryType] = None,
        time_window_hours: Optional[int] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Recall memories specific to a pool.
        
        Args:
            pool_pair: Pool pair identifier (e.g., "WETH/USDC")
            memory_type: Filter by memory type
            time_window_hours: Only get memories from last N hours
            limit: Maximum memories to return
            
        Returns:
            List of pool-specific memories
        """
        # Build query that includes pool pair
        query = f"pool {pool_pair}"
        
        # Get memories
        memories = await self.recall(
            query=query,
            memory_type=memory_type,
            limit=limit * 2  # Get extra to filter
        )
        
        # Filter by pool metadata
        pool_memories = []
        for mem in memories:
            metadata = mem.get("metadata", {})
            if metadata.get("pool") == pool_pair:
                # Check time window if specified
                if time_window_hours:
                    try:
                        timestamp = datetime.fromisoformat(metadata.get("timestamp", ""))
                        if (datetime.utcnow() - timestamp).total_seconds() / 3600 > time_window_hours:
                            continue
                    except:
                        pass
                        
                pool_memories.append(mem)
                
        return pool_memories[:limit]
        
    async def get_pool_patterns(self, pool_pair: Optional[str] = None) -> List[Dict]:
        """
        Get discovered patterns for a specific pool or all pools.
        
        Args:
            pool_pair: Optional pool pair to filter by
            
        Returns:
            List of pattern memories
        """
        query = f"pool {pool_pair} pattern" if pool_pair else "pattern"
        
        patterns = await self.recall(
            query=query,
            memory_type=MemoryType.PATTERN,
            limit=100
        )
        
        # If pool specified, filter to exact matches
        if pool_pair:
            patterns = [
                p for p in patterns 
                if p.get("metadata", {}).get("pool") == pool_pair
            ]
            
        return patterns
        
    async def get_cross_pool_correlations(self) -> List[Dict]:
        """
        Get memories about correlations between different pools.
        
        Returns:
            List of correlation memories
        """
        # Search for correlation patterns
        correlations = await self.recall(
            query="correlation between pools",
            memory_type=MemoryType.PATTERN,
            category="cross_pool_correlation",
            limit=50
        )
        
        return correlations
        
    async def remember_pool_correlation(
        self,
        pool_a: str,
        pool_b: str,
        correlation_type: str,
        correlation_strength: float,
        metadata: Optional[Dict] = None
    ):
        """
        Remember a correlation between two pools.
        
        Args:
            pool_a: First pool pair
            pool_b: Second pool pair
            correlation_type: Type of correlation (e.g., "volume", "apr", "liquidity")
            correlation_strength: Strength of correlation (-1 to 1)
            metadata: Additional metadata
        """
        content = f"Correlation discovered between {pool_a} and {pool_b}: {correlation_type} correlation of {correlation_strength:.2f}"
        
        await self.remember(
            content=content,
            memory_type=MemoryType.PATTERN,
            category="cross_pool_correlation",
            metadata={
                "pool_a": pool_a,
                "pool_b": pool_b,
                "correlation_type": correlation_type,
                "correlation_strength": correlation_strength,
                "pools": [pool_a, pool_b],
                **(metadata or {})
            },
            confidence=abs(correlation_strength)
        )
        
    async def get_pool_timeline(self, pool_pair: str, hours: int = 24) -> List[Dict]:
        """
        Get chronological timeline of memories for a pool.
        
        Args:
            pool_pair: Pool pair identifier
            hours: How many hours back to look
            
        Returns:
            List of memories sorted by timestamp
        """
        memories = await self.recall_pool_memories(
            pool_pair=pool_pair,
            time_window_hours=hours,
            limit=200
        )
        
        # Sort by timestamp
        for mem in memories:
            try:
                timestamp_str = mem.get("metadata", {}).get("timestamp", "")
                if timestamp_str:
                    mem["_timestamp"] = datetime.fromisoformat(timestamp_str)
                else:
                    mem["_timestamp"] = datetime.min
            except:
                mem["_timestamp"] = datetime.min
                
        memories.sort(key=lambda x: x["_timestamp"])
        
        # Remove temporary timestamp
        for mem in memories:
            mem.pop("_timestamp", None)
            
        return memories
        
    async def forget_old_memories(self, days: int = 30) -> int:
        """Remove memories older than specified days."""
        # TODO: Implement memory cleanup
        return 0
        
    async def export_memories(self) -> Dict:
        """Export all memories for backup."""
        try:
            if self.memory:
                all_memories = self.memory.get_all(user_id=self.user_id)
            else:
                all_memories = [{"id": m["id"], "content": m["messages"][0]["content"], 
                                "metadata": {"type": m["entry"].type.value, 
                                           "category": m["entry"].category}}
                               for m in self._local_memories]
            return {
                "memories": all_memories,
                "stats": self.stats,
                "exported_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to export memories: {e}")
            return {}