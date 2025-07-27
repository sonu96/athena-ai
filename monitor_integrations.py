"""
Monitor Athena AI Integrations in Real-Time

This script monitors all key integrations:
- LangGraph state transitions
- QuickNode data collection
- Mem0 memory operations
- Firestore database operations
"""
import asyncio
import logging
from datetime import datetime
from google.cloud import firestore
from google.cloud import logging as cloud_logging
import google.generativeai as genai
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegrationMonitor:
    def __init__(self):
        self.firestore = firestore.Client(project=settings.gcp_project_id)
        self.cloud_logger = cloud_logging.Client(project=settings.gcp_project_id).logger("athena-monitor")
        self.start_time = datetime.utcnow()
        
    async def check_langraph_agent(self):
        """Monitor LangGraph agent state transitions."""
        try:
            # Check agent state in Firestore
            doc = self.firestore.collection('agent_state').document('current').get()
            if doc.exists:
                state = doc.to_dict()
                logger.info(f"✅ LangGraph Agent State: {state.get('current_state', 'Unknown')}")
                logger.info(f"   Last update: {state.get('last_update', 'Never')}")
                logger.info(f"   Emotions: {state.get('emotions', {})}")
                return True
            else:
                logger.warning("⚠️ No agent state found in Firestore")
                return False
        except Exception as e:
            logger.error(f"❌ LangGraph check failed: {e}")
            return False
            
    async def check_quicknode_data(self):
        """Monitor QuickNode data collection."""
        try:
            # Check recent pool scans
            pools_ref = self.firestore.collection('pool_metrics')
            recent_pools = pools_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(5).get()
            
            if recent_pools:
                logger.info(f"✅ QuickNode Pool Scanner: {len(recent_pools)} recent pools found")
                for pool in recent_pools[:3]:
                    data = pool.to_dict()
                    logger.info(f"   - {data.get('pair', 'Unknown')}: APR {data.get('apr_total', 0):.1f}%, TVL ${data.get('tvl', 0):,.0f}")
                return True
            else:
                logger.warning("⚠️ No recent pool data from QuickNode")
                return False
        except Exception as e:
            logger.error(f"❌ QuickNode check failed: {e}")
            return False
            
    async def check_mem0_memories(self):
        """Monitor Mem0 memory operations."""
        try:
            # Check memory collections in Firestore
            memories_ref = self.firestore.collection('agent_memories')
            memory_count = len(list(memories_ref.limit(100).get()))
            
            if memory_count > 0:
                logger.info(f"✅ Mem0 Memories: {memory_count} memories stored")
                
                # Check recent memories
                recent = memories_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(3).get()
                for mem in recent:
                    data = mem.to_dict()
                    logger.info(f"   - {data.get('category', 'unknown')}: {data.get('content', '')[:50]}...")
                return True
            else:
                logger.warning("⚠️ No memories found in storage")
                return False
        except Exception as e:
            logger.error(f"❌ Mem0 check failed: {e}")
            return False
            
    async def check_firestore_tables(self):
        """Monitor Firestore table operations."""
        try:
            collections = [
                'agent_state',
                'pool_profiles', 
                'pattern_confidence',
                'observed_patterns',
                'agent_memories',
                'pool_metrics'
            ]
            
            logger.info("✅ Firestore Collections Status:")
            all_good = True
            
            for collection in collections:
                try:
                    # Check if collection has documents
                    docs = list(self.firestore.collection(collection).limit(1).get())
                    if docs:
                        count = len(list(self.firestore.collection(collection).limit(100).get()))
                        logger.info(f"   ✓ {collection}: {count} documents")
                    else:
                        logger.info(f"   ⚠️ {collection}: Empty")
                        all_good = False
                except Exception as e:
                    logger.error(f"   ❌ {collection}: Error - {e}")
                    all_good = False
                    
            return all_good
        except Exception as e:
            logger.error(f"❌ Firestore check failed: {e}")
            return False
            
    async def check_gemini_api(self):
        """Check if Gemini API is working."""
        try:
            if settings.google_api_key:
                genai.configure(api_key=settings.google_api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content("Say 'API Working' if you can read this")
                logger.info(f"✅ Gemini API: Connected and responding")
                return True
            else:
                logger.warning("⚠️ Gemini API: No API key configured")
                return False
        except Exception as e:
            logger.error(f"❌ Gemini API check failed: {e}")
            return False
            
    async def check_cloud_logs(self):
        """Check recent Cloud Logging entries."""
        try:
            # Get recent logs
            filter_str = f'resource.type="cloud_run_revision" AND severity>=WARNING AND timestamp>="{self.start_time.isoformat()}Z"'
            entries = list(self.cloud_logger.client.list_entries(filter_=filter_str, max_results=5))
            
            if entries:
                logger.info(f"⚠️ Recent warnings/errors in Cloud Logging: {len(entries)}")
                for entry in entries[:3]:
                    logger.info(f"   - [{entry.severity}] {entry.payload.get('message', str(entry.payload))[:100]}...")
            else:
                logger.info("✅ Cloud Logging: No recent warnings/errors")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Cloud Logging check skipped: {e}")
            return True  # Don't fail on logging issues
            
    async def run_health_check(self):
        """Run complete health check of all integrations."""
        logger.info("=" * 70)
        logger.info("🏥 ATHENA AI INTEGRATION HEALTH CHECK")
        logger.info("=" * 70)
        
        checks = [
            ("LangGraph Agent", self.check_langraph_agent()),
            ("QuickNode Data", self.check_quicknode_data()),
            ("Mem0 Memories", self.check_mem0_memories()),
            ("Firestore Tables", self.check_firestore_tables()),
            ("Gemini API", self.check_gemini_api()),
            ("Cloud Logging", self.check_cloud_logs())
        ]
        
        results = []
        for name, check in checks:
            logger.info(f"\n🔍 Checking {name}...")
            result = await check
            results.append((name, result))
            await asyncio.sleep(0.5)  # Brief pause between checks
            
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("📊 HEALTH CHECK SUMMARY")
        logger.info("=" * 70)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status} - {name}")
            
        logger.info(f"\nOverall: {passed}/{total} checks passed")
        
        if passed == total:
            logger.info("🎉 All systems operational!")
        elif passed >= total * 0.7:
            logger.info("⚠️ Some issues detected, but system is functional")
        else:
            logger.info("🚨 Multiple failures detected - troubleshooting needed")
            
        return passed, total


async def main():
    """Run the integration monitor."""
    monitor = IntegrationMonitor()
    
    # Run initial health check
    await monitor.run_health_check()
    
    # Continue monitoring
    logger.info("\n📡 Starting continuous monitoring (Ctrl+C to stop)...")
    logger.info("Will check every 30 seconds...\n")
    
    try:
        while True:
            await asyncio.sleep(30)
            logger.info(f"\n⏰ Periodic check at {datetime.utcnow().strftime('%H:%M:%S')} UTC")
            await monitor.run_health_check()
    except KeyboardInterrupt:
        logger.info("\n👋 Monitoring stopped")


if __name__ == "__main__":
    asyncio.run(main())