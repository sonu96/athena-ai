"""
Gas Price Monitor for Base Chain
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional
from decimal import Decimal

from src.cdp.base_client import BaseClient
from src.agent.memory import AthenaMemory, MemoryType

logger = logging.getLogger(__name__)


class GasMonitor:
    """
    Monitors gas prices on Base chain and identifies patterns.
    
    Tracks gas price changes, identifies optimal execution windows,
    and learns patterns over time.
    """
    
    def __init__(self, base_client: BaseClient, memory: AthenaMemory):
        """Initialize gas monitor."""
        self.base_client = base_client
        self.memory = memory
        self.monitoring = False
        
        # Gas price history (last 24 hours)
        self.price_history = []
        self.max_history = 2880  # 24 hours at 30-second intervals
        
        # Statistics
        self.stats = {
            "current_price": Decimal("0"),
            "24h_average": Decimal("0"),
            "24h_min": Decimal("0"),
            "24h_max": Decimal("0"),
            "optimal_windows": [],
        }
        
    async def start_monitoring(self):
        """Start continuous gas monitoring."""
        if self.monitoring:
            logger.warning("Gas monitoring already active")
            return
            
        self.monitoring = True
        logger.info("Starting gas price monitoring...")
        
        while self.monitoring:
            try:
                await self._monitor_cycle()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Gas monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def stop_monitoring(self):
        """Stop gas monitoring."""
        self.monitoring = False
        logger.info("Gas monitoring stopped")
        
    async def _monitor_cycle(self):
        """Single monitoring cycle."""
        # Get current gas price
        gas_price = await self.base_client.get_gas_price()
        
        # Record observation
        observation = {
            "price": gas_price,
            "timestamp": datetime.utcnow(),
            "hour": datetime.utcnow().hour,
            "day_of_week": datetime.utcnow().weekday(),
        }
        
        # Add to history
        self.price_history.append(observation)
        if len(self.price_history) > self.max_history:
            self.price_history.pop(0)
            
        # Update statistics
        self._update_statistics()
        
        # Check for patterns
        await self._check_patterns(observation)
        
        # Store in memory if significant change
        if self._is_significant_change(gas_price):
            await self.memory.remember(
                content=f"Gas price: {gas_price} gwei at {observation['hour']}:00 UTC",
                memory_type=MemoryType.OBSERVATION,
                category="gas_optimization",
                metadata=observation
            )
            
    def _update_statistics(self):
        """Update gas statistics."""
        if not self.price_history:
            return
            
        prices = [obs["price"] for obs in self.price_history]
        
        self.stats["current_price"] = prices[-1]
        self.stats["24h_average"] = sum(prices) / len(prices)
        self.stats["24h_min"] = min(prices)
        self.stats["24h_max"] = max(prices)
        
        # Find optimal windows (bottom 20% of prices)
        threshold = self.stats["24h_min"] + (self.stats["24h_max"] - self.stats["24h_min"]) * Decimal("0.2")
        
        optimal_hours = set()
        for obs in self.price_history:
            if obs["price"] <= threshold:
                optimal_hours.add(obs["hour"])
                
        self.stats["optimal_windows"] = sorted(list(optimal_hours))
        
    async def _check_patterns(self, observation: Dict):
        """Check for gas price patterns."""
        # Pattern 1: Time-based patterns
        if len(self.price_history) > 48:  # At least 24 hours of data
            await self._check_hourly_patterns()
            
        # Pattern 2: Spike detection
        if self._is_spike(observation["price"]):
            await self.memory.remember(
                content=f"Gas spike detected: {observation['price']} gwei (avg: {self.stats['24h_average']})",
                memory_type=MemoryType.PATTERN,
                category="gas_optimization",
                metadata={
                    "spike_ratio": float(observation["price"] / self.stats["24h_average"]),
                    "timestamp": observation["timestamp"].isoformat()
                },
                confidence=0.9
            )
            
    async def _check_hourly_patterns(self):
        """Analyze hourly gas patterns."""
        hourly_averages = {}
        
        # Calculate average gas price by hour
        for obs in self.price_history:
            hour = obs["hour"]
            if hour not in hourly_averages:
                hourly_averages[hour] = []
            hourly_averages[hour].append(obs["price"])
            
        # Find consistently cheap hours
        cheap_hours = []
        for hour, prices in hourly_averages.items():
            avg_price = sum(prices) / len(prices)
            if avg_price < self.stats["24h_average"] * Decimal("0.8"):
                cheap_hours.append((hour, avg_price))
                
        if cheap_hours:
            cheap_hours.sort(key=lambda x: x[1])  # Sort by price
            pattern = f"Gas consistently cheaper at hours: {[h[0] for h in cheap_hours[:3]]} UTC"
            
            await self.memory.remember(
                content=pattern,
                memory_type=MemoryType.PATTERN,
                category="gas_optimization",
                metadata={
                    "cheap_hours": cheap_hours[:3],
                    "average_savings": float((self.stats["24h_average"] - cheap_hours[0][1]) / self.stats["24h_average"])
                },
                confidence=0.85
            )
            
    def _is_significant_change(self, current_price: Decimal) -> bool:
        """Check if price change is significant."""
        if not self.price_history:
            return True
            
        last_price = self.price_history[-2]["price"] if len(self.price_history) > 1 else current_price
        change_ratio = abs(current_price - last_price) / last_price if last_price > 0 else 0
        
        return change_ratio > Decimal("0.1")  # 10% change
        
    def _is_spike(self, current_price: Decimal) -> bool:
        """Check if current price is a spike."""
        if not self.stats["24h_average"]:
            return False
            
        return current_price > self.stats["24h_average"] * Decimal("1.5")  # 50% above average
        
    def get_gas_recommendation(self) -> Dict:
        """Get gas price recommendation."""
        return {
            "current_price": float(self.stats["current_price"]),
            "24h_average": float(self.stats["24h_average"]),
            "optimal_hours": self.stats["optimal_windows"],
            "recommendation": self._get_recommendation_text(),
            "confidence": self._calculate_confidence(),
        }
        
    def _get_recommendation_text(self) -> str:
        """Generate recommendation text."""
        current = self.stats["current_price"]
        avg = self.stats["24h_average"]
        
        if current < avg * Decimal("0.8"):
            return "=� Excellent time to execute - gas is 20% below average"
        elif current < avg:
            return "=� Good time to execute - gas is below average"
        elif current < avg * Decimal("1.2"):
            return "=� Average gas prices - consider waiting if not urgent"
        else:
            return "=4 High gas prices - wait if possible"
            
    def _calculate_confidence(self) -> float:
        """Calculate confidence in recommendation."""
        # More data = higher confidence
        data_points = len(self.price_history)
        max_points = self.max_history
        
        return min(0.95, data_points / max_points)