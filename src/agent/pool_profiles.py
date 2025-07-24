"""
Pool Profiles System for Athena AI

Tracks individual pool behaviors and patterns over time to enable
better decision making and pattern recognition.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass, field, asdict
import json

logger = logging.getLogger(__name__)


@dataclass
class PoolMetrics:
    """Point-in-time metrics for a pool."""
    timestamp: datetime
    apr: Decimal
    tvl: Decimal
    volume_24h: Decimal
    fee_apr: Decimal
    incentive_apr: Decimal
    reserves: Dict[str, Decimal]
    ratio: Decimal
    gas_price: Optional[Decimal] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "apr": float(self.apr),
            "tvl": float(self.tvl),
            "volume_24h": float(self.volume_24h),
            "fee_apr": float(self.fee_apr),
            "incentive_apr": float(self.incentive_apr),
            "reserves": {k: float(v) for k, v in self.reserves.items()},
            "ratio": float(self.ratio),
            "gas_price": float(self.gas_price) if self.gas_price else None,
        }


@dataclass
class PoolProfile:
    """Comprehensive profile for a single pool."""
    pool_address: str
    pair: str
    stable: bool
    created_at: datetime
    
    # Historical ranges
    apr_range: Tuple[Decimal, Decimal] = (Decimal("0"), Decimal("0"))
    tvl_range: Tuple[Decimal, Decimal] = (Decimal("0"), Decimal("0"))
    volume_range: Tuple[Decimal, Decimal] = (Decimal("0"), Decimal("0"))
    
    # Patterns
    hourly_patterns: Dict[int, Dict] = field(default_factory=dict)  # hour -> metrics
    daily_patterns: Dict[str, Dict] = field(default_factory=dict)   # day_name -> metrics
    
    # Behaviors
    typical_volume_to_tvl: Decimal = Decimal("0")
    volatility_score: Decimal = Decimal("0")
    correlation_with_gas: Decimal = Decimal("0")
    
    # Statistics
    observations_count: int = 0
    last_updated: Optional[datetime] = None
    confidence_score: Decimal = Decimal("0")
    
    # Recent metrics for comparison
    recent_metrics: List[PoolMetrics] = field(default_factory=list)
    max_recent_metrics: int = 100
    
    def update_with_metrics(self, metrics: PoolMetrics):
        """Update profile with new metrics observation."""
        self.observations_count += 1
        self.last_updated = datetime.utcnow()
        
        # Update ranges
        self.apr_range = (
            min(self.apr_range[0], metrics.apr) if self.apr_range[0] > 0 else metrics.apr,
            max(self.apr_range[1], metrics.apr)
        )
        self.tvl_range = (
            min(self.tvl_range[0], metrics.tvl) if self.tvl_range[0] > 0 else metrics.tvl,
            max(self.tvl_range[1], metrics.tvl)
        )
        self.volume_range = (
            min(self.volume_range[0], metrics.volume_24h) if self.volume_range[0] > 0 else metrics.volume_24h,
            max(self.volume_range[1], metrics.volume_24h)
        )
        
        # Add to recent metrics
        self.recent_metrics.append(metrics)
        if len(self.recent_metrics) > self.max_recent_metrics:
            self.recent_metrics.pop(0)
            
        # Update patterns
        self._update_time_patterns(metrics)
        
        # Update behavioral metrics
        self._update_behaviors()
        
        # Update confidence
        self._update_confidence()
        
    def _update_time_patterns(self, metrics: PoolMetrics):
        """Update time-based patterns."""
        hour = metrics.timestamp.hour
        day_name = metrics.timestamp.strftime("%A")
        
        # Update hourly pattern
        if hour not in self.hourly_patterns:
            self.hourly_patterns[hour] = {
                "avg_apr": Decimal("0"),
                "avg_volume": Decimal("0"),
                "count": 0
            }
        
        hour_data = self.hourly_patterns[hour]
        hour_data["count"] += 1
        hour_data["avg_apr"] = (
            (hour_data["avg_apr"] * (hour_data["count"] - 1) + metrics.apr) / 
            hour_data["count"]
        )
        hour_data["avg_volume"] = (
            (hour_data["avg_volume"] * (hour_data["count"] - 1) + metrics.volume_24h) / 
            hour_data["count"]
        )
        
        # Update daily pattern
        if day_name not in self.daily_patterns:
            self.daily_patterns[day_name] = {
                "avg_apr": Decimal("0"),
                "avg_volume": Decimal("0"),
                "count": 0
            }
        
        day_data = self.daily_patterns[day_name]
        day_data["count"] += 1
        day_data["avg_apr"] = (
            (day_data["avg_apr"] * (day_data["count"] - 1) + metrics.apr) / 
            day_data["count"]
        )
        day_data["avg_volume"] = (
            (day_data["avg_volume"] * (day_data["count"] - 1) + metrics.volume_24h) / 
            day_data["count"]
        )
        
    def _update_behaviors(self):
        """Update behavioral metrics based on recent data."""
        if len(self.recent_metrics) < 10:
            return
            
        # Calculate typical volume to TVL ratio
        ratios = [
            m.volume_24h / m.tvl for m in self.recent_metrics 
            if m.tvl > 0
        ]
        if ratios:
            self.typical_volume_to_tvl = sum(ratios) / len(ratios)
            
        # Calculate volatility score (standard deviation of APR)
        aprs = [m.apr for m in self.recent_metrics]
        if len(aprs) > 1:
            mean_apr = sum(aprs) / len(aprs)
            variance = sum((apr - mean_apr) ** 2 for apr in aprs) / len(aprs)
            self.volatility_score = variance.sqrt()
            
        # Calculate correlation with gas prices
        if all(m.gas_price for m in self.recent_metrics[-20:]):
            self._calculate_gas_correlation()
            
    def _calculate_gas_correlation(self):
        """Calculate correlation between pool metrics and gas prices."""
        # Simple correlation calculation
        metrics_with_gas = [m for m in self.recent_metrics if m.gas_price]
        if len(metrics_with_gas) < 10:
            return
            
        volumes = [float(m.volume_24h) for m in metrics_with_gas]
        gas_prices = [float(m.gas_price) for m in metrics_with_gas]
        
        # Pearson correlation coefficient
        n = len(volumes)
        sum_x = sum(gas_prices)
        sum_y = sum(volumes)
        sum_xy = sum(x * y for x, y in zip(gas_prices, volumes))
        sum_x2 = sum(x ** 2 for x in gas_prices)
        sum_y2 = sum(y ** 2 for y in volumes)
        
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
        if denominator > 0:
            correlation = (n * sum_xy - sum_x * sum_y) / denominator
            self.correlation_with_gas = Decimal(str(correlation))
            
    def _update_confidence(self):
        """Update confidence score based on observations and consistency."""
        # Base confidence on number of observations
        obs_confidence = min(self.observations_count / 100, 1.0)
        
        # Adjust for data recency
        if self.recent_metrics:
            hours_since_update = (datetime.utcnow() - self.recent_metrics[-1].timestamp).total_seconds() / 3600
            recency_factor = max(0, 1 - (hours_since_update / 24))
        else:
            recency_factor = 0
            
        # Adjust for pattern consistency
        pattern_factor = 0.5  # Default
        if len(self.hourly_patterns) >= 12:  # At least half day covered
            pattern_factor = 0.8
        if len(self.daily_patterns) >= 4:  # At least 4 days covered
            pattern_factor = 1.0
            
        self.confidence_score = Decimal(str(
            (obs_confidence * 0.4 + recency_factor * 0.3 + pattern_factor * 0.3)
        ))
        
    def predict_metrics(self, timestamp: datetime) -> Optional[Dict]:
        """Predict likely metrics for a given timestamp based on patterns."""
        if self.confidence_score < Decimal("0.5"):
            return None
            
        hour = timestamp.hour
        day_name = timestamp.strftime("%A")
        
        predictions = {}
        
        # Use hourly pattern if available
        if hour in self.hourly_patterns and self.hourly_patterns[hour]["count"] > 5:
            predictions["apr"] = float(self.hourly_patterns[hour]["avg_apr"])
            predictions["volume"] = float(self.hourly_patterns[hour]["avg_volume"])
            predictions["confidence"] = "high" if self.hourly_patterns[hour]["count"] > 20 else "medium"
        # Fall back to daily pattern
        elif day_name in self.daily_patterns and self.daily_patterns[day_name]["count"] > 2:
            predictions["apr"] = float(self.daily_patterns[day_name]["avg_apr"])
            predictions["volume"] = float(self.daily_patterns[day_name]["avg_volume"])
            predictions["confidence"] = "medium"
        # Use overall average
        elif self.recent_metrics:
            predictions["apr"] = float(sum(m.apr for m in self.recent_metrics) / len(self.recent_metrics))
            predictions["volume"] = float(sum(m.volume_24h for m in self.recent_metrics) / len(self.recent_metrics))
            predictions["confidence"] = "low"
        else:
            return None
            
        predictions["timestamp"] = timestamp.isoformat()
        predictions["based_on_observations"] = self.observations_count
        
        return predictions
        
    def get_anomalies(self) -> List[Dict]:
        """Identify anomalous behavior in recent metrics."""
        if len(self.recent_metrics) < 20:
            return []
            
        anomalies = []
        
        # Calculate normal ranges (mean ± 2 * std dev)
        aprs = [m.apr for m in self.recent_metrics[:-5]]  # Exclude most recent
        volumes = [m.volume_24h for m in self.recent_metrics[:-5]]
        
        mean_apr = sum(aprs) / len(aprs)
        std_apr = (sum((apr - mean_apr) ** 2 for apr in aprs) / len(aprs)).sqrt()
        
        mean_volume = sum(volumes) / len(volumes)
        std_volume = (sum((v - mean_volume) ** 2 for v in volumes) / len(volumes)).sqrt()
        
        # Check recent metrics for anomalies
        for metric in self.recent_metrics[-5:]:
            apr_deviation = abs(metric.apr - mean_apr) / std_apr if std_apr > 0 else 0
            volume_deviation = abs(metric.volume_24h - mean_volume) / std_volume if std_volume > 0 else 0
            
            if apr_deviation > 2:
                anomalies.append({
                    "type": "apr_anomaly",
                    "timestamp": metric.timestamp.isoformat(),
                    "value": float(metric.apr),
                    "expected_range": (float(mean_apr - 2 * std_apr), float(mean_apr + 2 * std_apr)),
                    "severity": "high" if apr_deviation > 3 else "medium"
                })
                
            if volume_deviation > 2:
                anomalies.append({
                    "type": "volume_anomaly",
                    "timestamp": metric.timestamp.isoformat(),
                    "value": float(metric.volume_24h),
                    "expected_range": (float(mean_volume - 2 * std_volume), float(mean_volume + 2 * std_volume)),
                    "severity": "high" if volume_deviation > 3 else "medium"
                })
                
        return anomalies
        
    def to_dict(self) -> Dict:
        """Convert profile to dictionary for storage."""
        return {
            "pool_address": self.pool_address,
            "pair": self.pair,
            "stable": self.stable,
            "created_at": self.created_at.isoformat(),
            "apr_range": [float(self.apr_range[0]), float(self.apr_range[1])],
            "tvl_range": [float(self.tvl_range[0]), float(self.tvl_range[1])],
            "volume_range": [float(self.volume_range[0]), float(self.volume_range[1])],
            "hourly_patterns": {
                str(k): {
                    "avg_apr": float(v["avg_apr"]),
                    "avg_volume": float(v["avg_volume"]),
                    "count": v["count"]
                } for k, v in self.hourly_patterns.items()
            },
            "daily_patterns": {
                k: {
                    "avg_apr": float(v["avg_apr"]),
                    "avg_volume": float(v["avg_volume"]),
                    "count": v["count"]
                } for k, v in self.daily_patterns.items()
            },
            "typical_volume_to_tvl": float(self.typical_volume_to_tvl),
            "volatility_score": float(self.volatility_score),
            "correlation_with_gas": float(self.correlation_with_gas),
            "observations_count": self.observations_count,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "confidence_score": float(self.confidence_score),
        }


class PoolProfileManager:
    """Manages pool profiles and provides analysis capabilities."""
    
    def __init__(self, firestore_client=None):
        self.profiles: Dict[str, PoolProfile] = {}
        self.firestore = firestore_client
        logger.info(f"PoolProfileManager initialized with firestore_client: {firestore_client is not None}")
        
    async def update_pool(self, pool_data: Dict, gas_price: Optional[Decimal] = None):
        """Update or create pool profile with new data."""
        pool_address = pool_data.get("address", "")
        if not pool_address:
            logger.warning(f"Pool data missing address. Available keys: {list(pool_data.keys())}")
            return
            
        logger.debug(f"Updating pool profile for {pool_address} - pair: {pool_data.get('pair', 'unknown')}")
            
        # Create metrics from pool data
        metrics = PoolMetrics(
            timestamp=datetime.fromisoformat(pool_data.get("timestamp", datetime.utcnow().isoformat())),
            apr=Decimal(str(pool_data.get("apr", 0))),
            tvl=Decimal(str(pool_data.get("tvl", 0))),
            volume_24h=Decimal(str(pool_data.get("volume_24h", 0))),
            fee_apr=Decimal(str(pool_data.get("fee_apr", 0))),
            incentive_apr=Decimal(str(pool_data.get("incentive_apr", 0))),
            reserves=pool_data.get("reserves", {}),
            ratio=Decimal(str(pool_data.get("ratio", 1))),
            gas_price=gas_price
        )
        
        # Get or create profile
        if pool_address not in self.profiles:
            logger.info(f"Creating new pool profile for {pool_address} - {pool_data.get('pair', '')}")
            self.profiles[pool_address] = PoolProfile(
                pool_address=pool_address,
                pair=pool_data.get("pair", ""),
                stable=pool_data.get("stable", False),
                created_at=datetime.utcnow()
            )
            
        # Update profile
        profile = self.profiles[pool_address]
        profile.update_with_metrics(metrics)
        
        # Save to Firestore if available
        if self.firestore:
            await self._save_profile(profile)
            
    async def _save_profile(self, profile: PoolProfile):
        """Save profile to Firestore."""
        try:
            # Run sync Firestore operation in thread pool to avoid blocking
            await asyncio.to_thread(
                self.firestore.save_pool_profile,
                profile.pool_address,
                profile.to_dict()
            )
            logger.info(f"Pool profile saved for {profile.pool_address}")
        except Exception as e:
            logger.error(f"Failed to save pool profile for {profile.pool_address}: {e}")
            
    async def load_profiles(self):
        """Load profiles from Firestore."""
        if not self.firestore:
            return
            
        try:
            profiles_data = self.firestore.get_all_pool_profiles()
            for address, data in profiles_data.items():
                # Reconstruct profile from data
                profile = PoolProfile(
                    pool_address=address,
                    pair=data["pair"],
                    stable=data["stable"],
                    created_at=datetime.fromisoformat(data["created_at"])
                )
                # Update with stored data
                profile.apr_range = tuple(Decimal(str(x)) for x in data["apr_range"])
                profile.tvl_range = tuple(Decimal(str(x)) for x in data["tvl_range"])
                profile.volume_range = tuple(Decimal(str(x)) for x in data["volume_range"])
                profile.observations_count = data["observations_count"]
                profile.confidence_score = Decimal(str(data["confidence_score"]))
                
                self.profiles[address] = profile
                
        except Exception as e:
            logger.error(f"Failed to load pool profiles: {e}")
            
    def get_profile(self, pool_address: str) -> Optional[PoolProfile]:
        """Get profile for a specific pool."""
        return self.profiles.get(pool_address)
        
    def get_high_confidence_pools(self, min_confidence: Decimal = Decimal("0.7")) -> List[PoolProfile]:
        """Get pools with high confidence scores."""
        return [
            profile for profile in self.profiles.values()
            if profile.confidence_score >= min_confidence
        ]
        
    def get_correlated_pools(self, min_correlation: Decimal = Decimal("0.5")) -> List[Tuple[str, PoolProfile]]:
        """Get pools that correlate with gas prices."""
        return [
            (address, profile) for address, profile in self.profiles.items()
            if abs(profile.correlation_with_gas) >= min_correlation
        ]
        
    def predict_opportunities(self, timestamp: datetime) -> List[Dict]:
        """Predict opportunities based on time patterns."""
        opportunities = []
        
        for address, profile in self.profiles.items():
            if profile.confidence_score < Decimal("0.6"):
                continue
                
            prediction = profile.predict_metrics(timestamp)
            if prediction and prediction.get("confidence") in ["high", "medium"]:
                opportunities.append({
                    "pool": profile.pair,
                    "address": address,
                    "predicted_apr": prediction["apr"],
                    "predicted_volume": prediction["volume"],
                    "confidence": prediction["confidence"],
                    "profile_confidence": float(profile.confidence_score)
                })
                
        return sorted(opportunities, key=lambda x: x["predicted_apr"], reverse=True)
        
    def get_summary(self) -> Dict:
        """Get summary of all profiles."""
        return {
            "total_profiles": len(self.profiles),
            "high_confidence_pools": len(self.get_high_confidence_pools()),
            "gas_correlated_pools": len(self.get_correlated_pools()),
            "avg_observations": sum(p.observations_count for p in self.profiles.values()) / len(self.profiles) if self.profiles else 0,
            "pools_with_patterns": sum(1 for p in self.profiles.values() if len(p.hourly_patterns) > 12),
        }