"""
Disaster Recovery System for Athena AI

Implements state checkpoints, automated recovery procedures,
and system restoration capabilities.
"""
import asyncio
import logging
import json
import hashlib
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import shutil

from src.agent.memory import AthenaMemory
from src.agentkit.agent_client import AthenaAgentKit
from src.gcp.firestore_client import FirestoreClient
from src.mcp.quicknode_mcp import QuickNodeMCP
from config.settings import settings

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages system state checkpoints with triple redundancy."""
    
    def __init__(self, firestore: FirestoreClient, local_path: str = "./checkpoints"):
        """Initialize checkpoint manager."""
        self.firestore = firestore
        self.local_path = Path(local_path)
        self.local_path.mkdir(exist_ok=True)
        
        # Checkpoint configuration
        self.checkpoint_interval = timedelta(hours=settings.checkpoint_interval_hours)
        self.retention_days = settings.checkpoint_retention_days
        self.last_checkpoint_time = None
        
    async def create_checkpoint(self, state: Dict[str, Any]) -> str:
        """
        Create a comprehensive system checkpoint.
        
        Args:
            state: Complete system state to checkpoint
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = f"ckpt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Creating checkpoint: {checkpoint_id}")
        
        try:
            # Create checkpoint structure
            checkpoint = {
                "id": checkpoint_id,
                "timestamp": datetime.utcnow().isoformat(),
                "version": "2.0",
                "state": state,
                "metadata": {
                    "agent_version": "2.0",
                    "memory_count": state.get("memory_stats", {}).get("total_count", 0),
                    "position_count": len(state.get("positions", [])),
                    "total_value": state.get("portfolio", {}).get("total_value", 0)
                }
            }
            
            # Calculate checksum
            checkpoint["checksum"] = self._calculate_checksum(checkpoint)
            
            # Compress checkpoint
            compressed = self._compress_checkpoint(checkpoint)
            
            # Store in all locations (parallel for speed)
            await asyncio.gather(
                self._store_firestore(checkpoint_id, checkpoint),
                self._store_gcs(checkpoint_id, compressed),
                self._store_local(checkpoint_id, compressed)
            )
            
            # Update last checkpoint time
            self.last_checkpoint_time = datetime.utcnow()
            
            # Clean old checkpoints
            await self._cleanup_old_checkpoints()
            
            logger.info(
                f"Checkpoint {checkpoint_id} created successfully. "
                f"Size: {len(compressed) / 1024 / 1024:.2f} MB"
            )
            
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise
            
    def _calculate_checksum(self, checkpoint: Dict[str, Any]) -> str:
        """Calculate SHA256 checksum of checkpoint data."""
        # Remove checksum field for calculation
        data = {k: v for k, v in checkpoint.items() if k != "checksum"}
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
        
    def _compress_checkpoint(self, checkpoint: Dict[str, Any]) -> bytes:
        """Compress checkpoint data using gzip."""
        json_bytes = json.dumps(checkpoint).encode('utf-8')
        return gzip.compress(json_bytes, compresslevel=9)
        
    async def _store_firestore(self, checkpoint_id: str, checkpoint: Dict[str, Any]):
        """Store checkpoint in Firestore."""
        # Store metadata separately for querying
        await self.firestore.set_document(
            "checkpoints",
            checkpoint_id,
            {
                "id": checkpoint_id,
                "timestamp": checkpoint["timestamp"],
                "checksum": checkpoint["checksum"],
                "metadata": checkpoint["metadata"],
                # Store state separately due to size limits
                "state_ref": f"checkpoint_states/{checkpoint_id}"
            }
        )
        
        # Store actual state
        await self.firestore.set_document(
            "checkpoint_states",
            checkpoint_id,
            checkpoint["state"]
        )
        
    async def _store_gcs(self, checkpoint_id: str, compressed_data: bytes):
        """Store checkpoint in Google Cloud Storage."""
        # In production, use actual GCS client
        # For now, simulate with Firestore binary storage
        await self.firestore.set_document(
            "checkpoint_backups",
            checkpoint_id,
            {
                "data": compressed_data.hex(),  # Store as hex string
                "size_bytes": len(compressed_data),
                "compressed": True
            }
        )
        
    async def _store_local(self, checkpoint_id: str, compressed_data: bytes):
        """Store checkpoint locally."""
        checkpoint_file = self.local_path / f"{checkpoint_id}.gz"
        
        try:
            with open(checkpoint_file, 'wb') as f:
                f.write(compressed_data)
            logger.debug(f"Local checkpoint saved: {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save local checkpoint: {e}")
            
    async def _cleanup_old_checkpoints(self):
        """Remove checkpoints older than retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        
        # Clean Firestore
        old_checkpoints = await self.firestore.query_documents(
            "checkpoints",
            [("timestamp", "<", cutoff_date.isoformat())]
        )
        
        for checkpoint in old_checkpoints:
            checkpoint_id = checkpoint["id"]
            logger.info(f"Removing old checkpoint: {checkpoint_id}")
            
            # Remove from all locations
            await self.firestore.delete_document("checkpoints", checkpoint_id)
            await self.firestore.delete_document("checkpoint_states", checkpoint_id)
            await self.firestore.delete_document("checkpoint_backups", checkpoint_id)
            
            # Remove local file
            local_file = self.local_path / f"{checkpoint_id}.gz"
            if local_file.exists():
                local_file.unlink()
                
    async def restore_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Restore system state from checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to restore
            
        Returns:
            Restored state
        """
        logger.info(f"Restoring checkpoint: {checkpoint_id}")
        
        try:
            # Try to load from Firestore first
            checkpoint_meta = await self.firestore.get_document("checkpoints", checkpoint_id)
            if checkpoint_meta:
                state = await self.firestore.get_document(
                    "checkpoint_states",
                    checkpoint_id
                )
                
                checkpoint = {
                    **checkpoint_meta,
                    "state": state
                }
                
                # Verify checksum
                if self._verify_checkpoint(checkpoint):
                    logger.info("Checkpoint restored from Firestore")
                    return checkpoint["state"]
                else:
                    logger.error("Firestore checkpoint failed verification")
                    
            # Try GCS backup
            backup = await self.firestore.get_document("checkpoint_backups", checkpoint_id)
            if backup:
                compressed_data = bytes.fromhex(backup["data"])
                checkpoint = self._decompress_checkpoint(compressed_data)
                
                if self._verify_checkpoint(checkpoint):
                    logger.info("Checkpoint restored from GCS backup")
                    return checkpoint["state"]
                else:
                    logger.error("GCS checkpoint failed verification")
                    
            # Try local backup
            local_file = self.local_path / f"{checkpoint_id}.gz"
            if local_file.exists():
                with open(local_file, 'rb') as f:
                    compressed_data = f.read()
                    
                checkpoint = self._decompress_checkpoint(compressed_data)
                
                if self._verify_checkpoint(checkpoint):
                    logger.info("Checkpoint restored from local backup")
                    return checkpoint["state"]
                else:
                    logger.error("Local checkpoint failed verification")
                    
            raise Exception(f"Checkpoint {checkpoint_id} not found or corrupted")
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            raise
            
    def _decompress_checkpoint(self, compressed_data: bytes) -> Dict[str, Any]:
        """Decompress checkpoint data."""
        json_bytes = gzip.decompress(compressed_data)
        return json.loads(json_bytes.decode('utf-8'))
        
    def _verify_checkpoint(self, checkpoint: Dict[str, Any]) -> bool:
        """Verify checkpoint integrity."""
        expected_checksum = checkpoint.get("checksum")
        if not expected_checksum:
            return False
            
        actual_checksum = self._calculate_checksum(checkpoint)
        return actual_checksum == expected_checksum
        
    async def get_latest_checkpoint(self) -> Optional[str]:
        """Get the ID of the latest valid checkpoint."""
        # Query latest checkpoints
        checkpoints = await self.firestore.query_documents(
            "checkpoints",
            order_by=[("timestamp", "desc")],
            limit=10
        )
        
        # Find first valid checkpoint
        for checkpoint in checkpoints:
            checkpoint_id = checkpoint["id"]
            try:
                # Quick validation
                state = await self.firestore.get_document(
                    "checkpoint_states",
                    checkpoint_id
                )
                if state:
                    return checkpoint_id
            except:
                continue
                
        return None


class StateRecovery:
    """Handles disaster recovery and state restoration."""
    
    def __init__(self, memory: AthenaMemory, agentkit: AthenaAgentKit,
                 firestore: FirestoreClient, mcp: QuickNodeMCP):
        """Initialize recovery system."""
        self.memory = memory
        self.agentkit = agentkit
        self.firestore = firestore
        self.mcp = mcp
        self.checkpoint_manager = CheckpointManager(firestore)
        
    async def detect_memory_corruption(self) -> List[Dict[str, Any]]:
        """Detect corrupted memories using checksums and validation."""
        logger.info("Detecting memory corruption...")
        
        corrupted_memories = []
        
        # Get all memories
        all_memories = await self.memory.get_all_memories()
        
        for memory in all_memories:
            # Check structure
            if not self._validate_memory_structure(memory):
                corrupted_memories.append({
                    "id": memory.get("id"),
                    "type": "structure_invalid",
                    "memory": memory
                })
                continue
                
            # Check metadata
            metadata = memory.get("metadata", {})
            if metadata and not self._validate_metadata(metadata):
                corrupted_memories.append({
                    "id": memory.get("id"),
                    "type": "metadata_invalid",
                    "memory": memory
                })
                
        logger.info(f"Found {len(corrupted_memories)} corrupted memories")
        return corrupted_memories
        
    def _validate_memory_structure(self, memory: Dict[str, Any]) -> bool:
        """Validate memory has required fields."""
        required_fields = ["id", "type", "content", "timestamp"]
        return all(field in memory for field in required_fields)
        
    def _validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate metadata integrity."""
        # Check for corrupted values
        try:
            json.dumps(metadata)  # Should be JSON serializable
            return True
        except:
            return False
            
    async def recover_memory_system(self) -> bool:
        """
        Full memory recovery procedure.
        
        Returns:
            Success status
        """
        logger.info("Starting memory system recovery...")
        
        try:
            # 1. Detect corruption
            corrupted = await self.detect_memory_corruption()
            
            if not corrupted:
                logger.info("No memory corruption detected")
                return True
                
            # 2. Backup current state
            backup_path = await self._backup_corrupted_state()
            logger.info(f"Corrupted state backed up to: {backup_path}")
            
            # 3. Quarantine corrupted memories
            await self._quarantine_memories(corrupted)
            
            # 4. Restore from Firestore
            clean_memories = await self._restore_memories_from_firestore()
            
            # 5. Rebuild Mem0 database
            await self._rebuild_memory_database(clean_memories)
            
            # 6. Verify integrity
            if await self._verify_memory_integrity():
                logger.info("Memory recovery completed successfully")
                return True
            else:
                logger.error("Memory recovery verification failed")
                return False
                
        except Exception as e:
            logger.error(f"Memory recovery failed: {e}")
            return False
            
    async def _backup_corrupted_state(self) -> str:
        """Backup corrupted state before recovery."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_id = f"corrupted_backup_{timestamp}"
        
        # Get all memories
        all_memories = await self.memory.get_all_memories()
        
        # Store backup
        await self.firestore.set_document(
            "corrupted_backups",
            backup_id,
            {
                "timestamp": datetime.utcnow().isoformat(),
                "memories": all_memories,
                "count": len(all_memories)
            }
        )
        
        return backup_id
        
    async def _quarantine_memories(self, corrupted: List[Dict[str, Any]]):
        """Move corrupted memories to quarantine."""
        for item in corrupted:
            memory_id = item["id"]
            
            # Store in quarantine
            await self.firestore.set_document(
                "quarantined_memories",
                memory_id,
                {
                    "quarantined_at": datetime.utcnow().isoformat(),
                    "corruption_type": item["type"],
                    "memory": item["memory"]
                }
            )
            
            # Remove from active memory
            await self.memory.delete_memory(memory_id)
            
    async def _restore_memories_from_firestore(self) -> List[Dict[str, Any]]:
        """Restore clean memories from Firestore backup."""
        # Get latest memory backup
        backups = await self.firestore.query_documents(
            "memory_backups",
            order_by=[("timestamp", "desc")],
            limit=1
        )
        
        if not backups:
            logger.warning("No memory backups found in Firestore")
            return []
            
        backup = backups[0]
        memories = backup.get("memories", [])
        
        logger.info(f"Restoring {len(memories)} memories from Firestore backup")
        return memories
        
    async def _rebuild_memory_database(self, memories: List[Dict[str, Any]]):
        """Rebuild Mem0 database with clean memories."""
        logger.info(f"Rebuilding memory database with {len(memories)} memories")
        
        # Clear existing memories
        await self.memory.clear_all()
        
        # Re-add memories
        for memory in memories:
            try:
                await self.memory.remember(
                    content=memory["content"],
                    memory_type=memory.get("type", "observation"),
                    category=memory.get("category", "general"),
                    metadata=memory.get("metadata", {}),
                    confidence=memory.get("confidence", 0.5)
                )
            except Exception as e:
                logger.error(f"Failed to restore memory {memory.get('id')}: {e}")
                
    async def _verify_memory_integrity(self) -> bool:
        """Verify memory system integrity after recovery."""
        try:
            # Test memory operations
            test_id = await self.memory.remember(
                content="Recovery test memory",
                memory_type="observation",
                category="system_test"
            )
            
            # Try to recall
            results = await self.memory.recall("Recovery test", limit=1)
            
            # Clean up test
            await self.memory.delete_memory(test_id)
            
            return len(results) > 0
            
        except Exception as e:
            logger.error(f"Memory integrity check failed: {e}")
            return False
            
    async def reconcile_positions(self) -> List[Dict[str, Any]]:
        """
        Compare and reconcile position state between blockchain and database.
        
        Returns:
            List of discrepancies found
        """
        logger.info("Starting position reconciliation...")
        
        discrepancies = []
        
        try:
            # Get blockchain positions (source of truth)
            blockchain_positions = await self._get_blockchain_positions()
            
            # Get database positions
            db_positions = await self._get_database_positions()
            
            # Compare positions
            discrepancies = self._compare_positions(blockchain_positions, db_positions)
            
            if discrepancies:
                logger.warning(f"Found {len(discrepancies)} position discrepancies")
                
                # Update database to match blockchain
                await self._update_positions_from_blockchain(
                    blockchain_positions,
                    discrepancies
                )
            else:
                logger.info("No position discrepancies found")
                
            return discrepancies
            
        except Exception as e:
            logger.error(f"Position reconciliation failed: {e}")
            raise
            
    async def _get_blockchain_positions(self) -> List[Dict[str, Any]]:
        """Get positions from blockchain using MCP."""
        query = f"Get all liquidity positions for wallet {self.agentkit.address}"
        result = await self.mcp.query(query)
        
        positions = result.get("positions", [])
        logger.info(f"Found {len(positions)} positions on blockchain")
        
        return positions
        
    async def _get_database_positions(self) -> List[Dict[str, Any]]:
        """Get positions from database."""
        positions = await self.firestore.query_documents(
            "positions",
            [("status", "==", "active")]
        )
        
        logger.info(f"Found {len(positions)} positions in database")
        return positions
        
    def _compare_positions(self, blockchain: List[Dict[str, Any]],
                          database: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compare blockchain and database positions."""
        discrepancies = []
        
        # Create lookup maps
        blockchain_map = {p["pool_address"]: p for p in blockchain}
        database_map = {p["pool_address"]: p for p in database}
        
        # Check for missing positions
        for pool_address in blockchain_map:
            if pool_address not in database_map:
                discrepancies.append({
                    "type": "missing_in_db",
                    "pool_address": pool_address,
                    "blockchain_position": blockchain_map[pool_address]
                })
                
        # Check for extra positions
        for pool_address in database_map:
            if pool_address not in blockchain_map:
                discrepancies.append({
                    "type": "extra_in_db",
                    "pool_address": pool_address,
                    "database_position": database_map[pool_address]
                })
                
        # Check for value mismatches
        for pool_address in blockchain_map:
            if pool_address in database_map:
                bc_value = blockchain_map[pool_address]["value_usd"]
                db_value = database_map[pool_address]["value_usd"]
                
                if abs(bc_value - db_value) > 100:  # $100 threshold
                    discrepancies.append({
                        "type": "value_mismatch",
                        "pool_address": pool_address,
                        "blockchain_value": bc_value,
                        "database_value": db_value,
                        "difference": bc_value - db_value
                    })
                    
        return discrepancies
        
    async def _update_positions_from_blockchain(self, blockchain_positions: List[Dict[str, Any]],
                                              discrepancies: List[Dict[str, Any]]):
        """Update database positions to match blockchain state."""
        for discrepancy in discrepancies:
            if discrepancy["type"] == "missing_in_db":
                # Add missing position
                position = discrepancy["blockchain_position"]
                await self.firestore.set_document(
                    "positions",
                    position["pool_address"],
                    position
                )
                
            elif discrepancy["type"] == "extra_in_db":
                # Mark as inactive instead of deleting
                await self.firestore.update_document(
                    "positions",
                    discrepancy["pool_address"],
                    {
                        "status": "inactive",
                        "updated_at": datetime.utcnow().isoformat(),
                        "reason": "not_found_on_blockchain"
                    }
                )
                
            elif discrepancy["type"] == "value_mismatch":
                # Update value
                pool_address = discrepancy["pool_address"]
                blockchain_value = discrepancy["blockchain_value"]
                
                await self.firestore.update_document(
                    "positions",
                    pool_address,
                    {
                        "value_usd": blockchain_value,
                        "updated_at": datetime.utcnow().isoformat(),
                        "reconciled": True
                    }
                )
                
    async def perform_full_recovery(self, checkpoint_id: Optional[str] = None) -> bool:
        """
        Perform full system recovery from checkpoint.
        
        Args:
            checkpoint_id: Specific checkpoint to restore (uses latest if None)
            
        Returns:
            Success status
        """
        logger.info("Starting full system recovery...")
        
        try:
            # 1. Verify wallet integrity
            if not await self._verify_wallet_integrity():
                logger.critical("Wallet integrity check failed!")
                return False
                
            # 2. Get checkpoint to restore
            if not checkpoint_id:
                checkpoint_id = await self.checkpoint_manager.get_latest_checkpoint()
                
            if not checkpoint_id:
                logger.error("No valid checkpoint found")
                return False
                
            logger.info(f"Restoring from checkpoint: {checkpoint_id}")
            
            # 3. Restore checkpoint
            state = await self.checkpoint_manager.restore_checkpoint(checkpoint_id)
            
            # 4. Restore components
            await self._restore_memory_state(state.get("memories", {}))
            await self._restore_position_state(state.get("positions", []))
            await self._restore_pattern_state(state.get("patterns", {}))
            
            # 5. Replay recent events
            await self._replay_recent_events(checkpoint_id)
            
            # 6. Validate recovery
            if await self._validate_recovery():
                logger.info("Full system recovery completed successfully")
                return True
            else:
                logger.error("Recovery validation failed")
                return False
                
        except Exception as e:
            logger.error(f"Full system recovery failed: {e}")
            return False
            
    async def _verify_wallet_integrity(self) -> bool:
        """Verify wallet is accessible and functional."""
        try:
            # Check wallet address
            address = self.agentkit.address
            if not address:
                return False
                
            # Try to get balance
            balance = await self.agentkit.get_balance("eth")
            
            logger.info(f"Wallet {address} verified. Balance: {balance} ETH")
            return True
            
        except Exception as e:
            logger.error(f"Wallet verification failed: {e}")
            return False
            
    async def _restore_memory_state(self, memory_state: Dict[str, Any]):
        """Restore memory system state."""
        logger.info("Restoring memory state...")
        
        # Clear and rebuild
        await self.memory.clear_all()
        
        memories = memory_state.get("memories", [])
        for memory in memories:
            await self.memory.remember(
                content=memory["content"],
                memory_type=memory.get("type"),
                category=memory.get("category"),
                metadata=memory.get("metadata"),
                confidence=memory.get("confidence", 0.5)
            )
            
    async def _restore_position_state(self, positions: List[Dict[str, Any]]):
        """Restore position state."""
        logger.info(f"Restoring {len(positions)} positions...")
        
        for position in positions:
            await self.firestore.set_document(
                "positions",
                position["pool_address"],
                position
            )
            
    async def _restore_pattern_state(self, patterns: Dict[str, Any]):
        """Restore discovered patterns."""
        logger.info("Restoring pattern state...")
        
        for pattern_id, pattern in patterns.items():
            await self.firestore.set_document(
                "observed_patterns",
                pattern_id,
                pattern
            )
            
    async def _replay_recent_events(self, checkpoint_id: str):
        """Replay events since checkpoint to catch up."""
        # Get checkpoint timestamp
        checkpoint = await self.firestore.get_document("checkpoints", checkpoint_id)
        checkpoint_time = datetime.fromisoformat(checkpoint["timestamp"])
        
        # Query recent events
        hours_elapsed = (datetime.utcnow() - checkpoint_time).total_seconds() / 3600
        
        if hours_elapsed > 24:
            logger.warning(f"Checkpoint is {hours_elapsed:.1f} hours old. Skipping replay.")
            return
            
        logger.info(f"Replaying {hours_elapsed:.1f} hours of events...")
        
        # Get recent blockchain events using MCP
        query = f"Get all transactions for {self.agentkit.address} in the last {int(hours_elapsed)} hours"
        events = await self.mcp.query(query)
        
        # Process events (simplified)
        for event in events.get("transactions", []):
            logger.debug(f"Replaying event: {event['type']} at {event['timestamp']}")
            
    async def _validate_recovery(self) -> bool:
        """Validate system after recovery."""
        checks = {
            "memory_functional": await self._verify_memory_integrity(),
            "positions_reconciled": len(await self.reconcile_positions()) == 0,
            "wallet_accessible": await self._verify_wallet_integrity()
        }
        
        all_passed = all(checks.values())
        
        logger.info(f"Recovery validation: {checks}")
        return all_passed