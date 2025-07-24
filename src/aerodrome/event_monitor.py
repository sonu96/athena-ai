"""
Event Monitor for Aerodrome Protocol

Monitors pool events (Swap, Sync, Fees) to track real volumes and calculate
accurate fee-based APRs.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Set
from decimal import Decimal
from datetime import datetime, timedelta
from collections import defaultdict

from src.blockchain.rpc_reader import RPCReader
from config.contracts import CONTRACTS
from config.settings import settings

logger = logging.getLogger(__name__)


class EventMonitor:
    """Monitors Aerodrome pool events for volume and fee tracking."""
    
    def __init__(self, rpc_reader: Optional[RPCReader] = None):
        """Initialize event monitor."""
        self.rpc_reader = rpc_reader
        
        # Volume tracking
        self.hourly_volumes = defaultdict(lambda: defaultdict(Decimal))  # pool -> hour -> volume
        self.daily_volumes = defaultdict(lambda: defaultdict(Decimal))   # pool -> date -> volume
        
        # Fee tracking
        self.hourly_fees = defaultdict(lambda: defaultdict(Decimal))     # pool -> hour -> fees
        self.daily_fees = defaultdict(lambda: defaultdict(Decimal))       # pool -> date -> fees
        
        # Event signatures
        self.event_signatures = {
            "Swap": "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822",
            "Sync": "0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1",
            "Fees": "0x112c256902bf554b6ed882d2936687aaeb4225e8cd5b51303c90ca6cf43a8602"
        }
        
        # Monitoring state
        self.monitoring = False
        self.monitored_pools: Set[str] = set()
        self.last_block_processed = {}  # pool -> block number
        
    async def __aenter__(self):
        """Enter async context."""
        if not self.rpc_reader:
            self.rpc_reader = RPCReader(settings.cdp_rpc_url)
            await self.rpc_reader.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        if self.rpc_reader and hasattr(self.rpc_reader, '__aexit__'):
            await self.rpc_reader.__aexit__(exc_type, exc_val, exc_tb)
            
    async def add_pool(self, pool_address: str):
        """Add a pool to monitor."""
        self.monitored_pools.add(pool_address.lower())
        logger.info(f"Added pool {pool_address} to event monitoring")
        
    async def remove_pool(self, pool_address: str):
        """Remove a pool from monitoring."""
        self.monitored_pools.discard(pool_address.lower())
        logger.info(f"Removed pool {pool_address} from event monitoring")
        
    async def start_monitoring(self, pools: List[str]):
        """Start monitoring events for specified pools."""
        if self.monitoring:
            logger.warning("Event monitoring already active")
            return
            
        # Add pools to monitor
        for pool in pools:
            await self.add_pool(pool)
            
        self.monitoring = True
        logger.info(f"Starting event monitoring for {len(pools)} pools")
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
        
    async def stop_monitoring(self):
        """Stop event monitoring."""
        self.monitoring = False
        logger.info("Event monitoring stopped")
        
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Process events for each pool
                for pool_address in list(self.monitored_pools):
                    await self._process_pool_events(pool_address)
                    
                # Wait before next cycle
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Event monitoring error: {e}")
                await asyncio.sleep(300)  # Wait longer on error
                
    async def _process_pool_events(self, pool_address: str):
        """Process events for a specific pool."""
        try:
            # Get current block number
            current_block = await self._get_current_block()
            if not current_block:
                return
                
            # Get last processed block
            from_block = self.last_block_processed.get(pool_address, current_block - 1000)
            
            # Don't process if we're caught up
            if from_block >= current_block:
                return
                
            logger.debug(f"Processing events for pool {pool_address} from block {from_block} to {current_block}")
            
            # Get Swap events
            await self._process_swap_events(pool_address, from_block, current_block)
            
            # Get Fee events if available
            await self._process_fee_events(pool_address, from_block, current_block)
            
            # Update last processed block
            self.last_block_processed[pool_address] = current_block
            
        except Exception as e:
            logger.error(f"Failed to process events for pool {pool_address}: {e}")
            
    async def _get_current_block(self) -> Optional[int]:
        """Get current block number."""
        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_blockNumber",
                "params": [],
                "id": 1
            }
            
            async with self.rpc_reader.session.post(
                self.rpc_reader.rpc_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                result = await response.json()
                
                if "error" in result:
                    logger.error(f"RPC error getting block number: {result['error']}")
                    return None
                    
                return int(result.get("result", "0x0"), 16)
                
        except Exception as e:
            logger.error(f"Failed to get current block: {e}")
            return None
            
    async def _process_swap_events(self, pool_address: str, from_block: int, to_block: int):
        """Process Swap events to calculate volume."""
        try:
            # Get swap events
            events = await self._get_events(
                pool_address,
                self.event_signatures["Swap"],
                from_block,
                to_block
            )
            
            for event in events:
                await self._process_swap_event(pool_address, event)
                
        except Exception as e:
            logger.error(f"Failed to process swap events: {e}")
            
    async def _process_fee_events(self, pool_address: str, from_block: int, to_block: int):
        """Process Fee events to track fees collected."""
        try:
            # Get fee events
            events = await self._get_events(
                pool_address,
                self.event_signatures["Fees"],
                from_block,
                to_block
            )
            
            for event in events:
                await self._process_fee_event(pool_address, event)
                
        except Exception as e:
            logger.error(f"Failed to process fee events: {e}")
            
    async def _get_events(
        self,
        contract_address: str,
        event_signature: str,
        from_block: int,
        to_block: int
    ) -> List[Dict]:
        """Get events from the blockchain."""
        try:
            # Limit range to prevent timeout
            max_range = 1000
            if to_block - from_block > max_range:
                to_block = from_block + max_range
                
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_getLogs",
                "params": [{
                    "fromBlock": hex(from_block),
                    "toBlock": hex(to_block),
                    "address": contract_address,
                    "topics": [event_signature]
                }],
                "id": 1
            }
            
            async with self.rpc_reader.session.post(
                self.rpc_reader.rpc_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                result = await response.json()
                
                if "error" in result:
                    logger.error(f"RPC error getting events: {result['error']}")
                    return []
                    
                return result.get("result", [])
                
        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return []
            
    async def _process_swap_event(self, pool_address: str, event: Dict):
        """Process a single swap event."""
        try:
            # Parse event data
            data = event.get("data", "0x")
            if len(data) < 258:  # 0x + 4 * 64 chars
                return
                
            # Extract amounts (each uint256 is 64 hex chars)
            amount0_in = int(data[2:66], 16)
            amount1_in = int(data[66:130], 16)
            amount0_out = int(data[130:194], 16)
            amount1_out = int(data[194:258], 16)
            
            # Calculate volume (sum of in and out amounts)
            # Note: This is simplified - in production you'd convert to USD
            volume = Decimal(amount0_in + amount1_in + amount0_out + amount1_out) / Decimal(10**18)
            
            # Get timestamp from block
            block_number = int(event.get("blockNumber", "0x0"), 16)
            timestamp = await self._get_block_timestamp(block_number)
            if not timestamp:
                timestamp = datetime.utcnow()
                
            # Track volume
            hour_key = timestamp.strftime("%Y-%m-%d-%H")
            day_key = timestamp.strftime("%Y-%m-%d")
            
            self.hourly_volumes[pool_address][hour_key] += volume
            self.daily_volumes[pool_address][day_key] += volume
            
            logger.debug(f"Swap event: pool={pool_address}, volume={volume:.6f}, hour={hour_key}")
            
        except Exception as e:
            logger.error(f"Failed to process swap event: {e}")
            
    async def _process_fee_event(self, pool_address: str, event: Dict):
        """Process a single fee event."""
        try:
            # Parse event data
            data = event.get("data", "0x")
            if len(data) < 130:  # 0x + 2 * 64 chars
                return
                
            # Extract fee amounts
            amount0 = int(data[2:66], 16)
            amount1 = int(data[66:130], 16)
            
            # Calculate total fees (simplified - convert to USD in production)
            fees = Decimal(amount0 + amount1) / Decimal(10**18)
            
            # Get timestamp
            block_number = int(event.get("blockNumber", "0x0"), 16)
            timestamp = await self._get_block_timestamp(block_number)
            if not timestamp:
                timestamp = datetime.utcnow()
                
            # Track fees
            hour_key = timestamp.strftime("%Y-%m-%d-%H")
            day_key = timestamp.strftime("%Y-%m-%d")
            
            self.hourly_fees[pool_address][hour_key] += fees
            self.daily_fees[pool_address][day_key] += fees
            
            logger.debug(f"Fee event: pool={pool_address}, fees={fees:.6f}, hour={hour_key}")
            
        except Exception as e:
            logger.error(f"Failed to process fee event: {e}")
            
    async def _get_block_timestamp(self, block_number: int) -> Optional[datetime]:
        """Get timestamp for a block."""
        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_getBlockByNumber",
                "params": [hex(block_number), False],
                "id": 1
            }
            
            async with self.rpc_reader.session.post(
                self.rpc_reader.rpc_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                result = await response.json()
                
                if "error" in result or not result.get("result"):
                    return None
                    
                timestamp_hex = result["result"].get("timestamp", "0x0")
                timestamp = int(timestamp_hex, 16)
                
                return datetime.fromtimestamp(timestamp)
                
        except Exception as e:
            logger.error(f"Failed to get block timestamp: {e}")
            return None
            
    def get_24h_volume(self, pool_address: str) -> Decimal:
        """Get 24-hour volume for a pool."""
        pool_address = pool_address.lower()
        now = datetime.utcnow()
        volume = Decimal("0")
        
        # Sum up last 24 hours
        for i in range(24):
            hour = now - timedelta(hours=i)
            hour_key = hour.strftime("%Y-%m-%d-%H")
            volume += self.hourly_volumes[pool_address].get(hour_key, Decimal("0"))
            
        return volume
        
    def get_daily_volume(self, pool_address: str, date: Optional[datetime] = None) -> Decimal:
        """Get daily volume for a pool."""
        pool_address = pool_address.lower()
        if not date:
            date = datetime.utcnow()
            
        day_key = date.strftime("%Y-%m-%d")
        return self.daily_volumes[pool_address].get(day_key, Decimal("0"))
        
    def get_24h_fees(self, pool_address: str) -> Decimal:
        """Get 24-hour fees for a pool."""
        pool_address = pool_address.lower()
        now = datetime.utcnow()
        fees = Decimal("0")
        
        # Sum up last 24 hours
        for i in range(24):
            hour = now - timedelta(hours=i)
            hour_key = hour.strftime("%Y-%m-%d-%H")
            fees += self.hourly_fees[pool_address].get(hour_key, Decimal("0"))
            
        return fees
        
    def get_volume_summary(self) -> Dict:
        """Get summary of all tracked volumes."""
        summary = {}
        
        for pool_address in self.monitored_pools:
            volume_24h = self.get_24h_volume(pool_address)
            if volume_24h > 0:
                summary[pool_address] = {
                    "volume_24h": float(volume_24h),
                    "last_update": self.last_block_processed.get(pool_address, 0),
                    "hourly_data": {
                        k: float(v) for k, v in 
                        self.hourly_volumes[pool_address].items()
                    }
                }
                
        return summary