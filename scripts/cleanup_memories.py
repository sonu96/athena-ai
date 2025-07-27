#!/usr/bin/env python3
"""
Memory and Database Cleanup Script
Cleans up all Mem0 memories and Firestore collections
"""
import asyncio
import logging
from google.cloud import firestore
from mem0 import MemoryClient
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def cleanup_mem0_memories():
    """Clean up all Mem0 memories."""
    try:
        if not settings.mem0_api_key:
            logger.info("No Mem0 API key configured, skipping Mem0 cleanup")
            return
        
        logger.info("🧹 Cleaning up Mem0 memories...")
        memory_client = MemoryClient(api_key=settings.mem0_api_key)
        user_id = "athena_agent"
        
        # Get all memories for the user
        memories = memory_client.get_all(user_id=user_id)
        logger.info(f"Found {len(memories)} memories to delete")
        
        # Delete all memories
        for memory in memories:
            try:
                memory_client.delete(memory_id=memory['id'], user_id=user_id)
                logger.info(f"Deleted memory: {memory['id']}")
            except Exception as e:
                logger.error(f"Failed to delete memory {memory['id']}: {e}")
        
        logger.info("✅ Mem0 cleanup completed")
        
    except Exception as e:
        logger.error(f"Failed to cleanup Mem0 memories: {e}")

def cleanup_firestore_collections():
    """Clean up all Firestore collections."""
    try:
        logger.info("🧹 Cleaning up Firestore collections...")
        
        # Initialize Firestore client
        db = firestore.Client(
            project=settings.gcp_project_id,
            database=settings.firestore_database
        )
        
        # List of collections to clean up
        collections_to_clean = [
            'agent_memories',
            'agent_state',
            'pool_profiles',
            'pool_metrics',
            'pattern_correlations',
            'observation_metrics',
            'observed_patterns',
            'pattern_confidence',
            'gas_prices',
            'market_data',
            'agent_decisions',
            'agent_emotions',
            'strategy_outcomes',
            'error_logs',
            'risk_assessments',
            'circuit_breaker_events',
            'checkpoints',
            'performance_metrics'
        ]
        
        total_deleted = 0
        
        for collection_name in collections_to_clean:
            try:
                collection = db.collection(collection_name)
                docs = collection.stream()
                
                batch = db.batch()
                batch_count = 0
                
                for doc in docs:
                    batch.delete(doc.reference)
                    batch_count += 1
                    total_deleted += 1
                    
                    # Commit batch every 500 documents
                    if batch_count >= 500:
                        batch.commit()
                        batch = db.batch()
                        batch_count = 0
                        logger.info(f"Deleted batch from {collection_name}")
                
                # Commit remaining documents
                if batch_count > 0:
                    batch.commit()
                    
                logger.info(f"Cleaned collection: {collection_name}")
                
            except Exception as e:
                logger.error(f"Failed to clean collection {collection_name}: {e}")
        
        logger.info(f"✅ Firestore cleanup completed. Deleted {total_deleted} documents")
        
    except Exception as e:
        logger.error(f"Failed to cleanup Firestore: {e}")

async def main():
    """Main cleanup function."""
    logger.info("🚀 Starting memory and database cleanup...")
    
    # Cleanup Mem0 memories
    await cleanup_mem0_memories()
    
    # Cleanup Firestore collections
    cleanup_firestore_collections()
    
    logger.info("🎉 All cleanup operations completed!")

if __name__ == "__main__":
    asyncio.run(main())