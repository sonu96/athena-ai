"""
System Performance Monitoring for Athena AI

Implements comprehensive monitoring for all system components including
memory, risk, performance metrics, and health checks.
"""
import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json

from src.agent.memory_manager import MemoryManager
from src.agent.risk_manager import RiskManager
from src.agent.llm_optimizer import LLMOptimizer
from src.gcp.firestore_optimizer import FirestoreOptimizer
from src.gcp.firestore_client import FirestoreClient
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    

class MetricCollector:
    """Collects and stores system metrics."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize metric collector."""
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.aggregated: Dict[str, Dict[str, float]] = defaultdict(dict)
        
    def record(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Record a metric value."""
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            metadata=metadata or {}
        )
        self.metrics[metric_name].append(point)
        
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for a metric."""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1].value
        return None
        
    def get_average(self, metric_name: str, window_minutes: int = 60) -> Optional[float]:
        """Get average value over time window."""
        if metric_name not in self.metrics:
            return None
            
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        values = [
            point.value for point in self.metrics[metric_name]
            if point.timestamp > cutoff
        ]
        
        return sum(values) / len(values) if values else None
        
    def get_percentile(self, metric_name: str, percentile: float, 
                      window_minutes: int = 60) -> Optional[float]:
        """Get percentile value over time window."""
        if metric_name not in self.metrics:
            return None
            
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        values = sorted([
            point.value for point in self.metrics[metric_name]
            if point.timestamp > cutoff
        ])
        
        if not values:
            return None
            
        idx = int(len(values) * percentile / 100)
        return values[min(idx, len(values) - 1)]
        
        
class SystemMonitor:
    """
    Comprehensive system monitoring and health checks.
    
    Monitors:
    - Memory system performance
    - Risk management status
    - LLM optimization metrics
    - Firestore performance
    - System resources
    - Application health
    """
    
    def __init__(self, memory_manager: MemoryManager, risk_manager: RiskManager,
                 llm_optimizer: LLMOptimizer, firestore_optimizer: FirestoreOptimizer,
                 firestore: FirestoreClient):
        """Initialize system monitor."""
        self.memory_manager = memory_manager
        self.risk_manager = risk_manager
        self.llm_optimizer = llm_optimizer
        self.firestore_optimizer = firestore_optimizer
        self.firestore = firestore
        
        # Metric collectors
        self.metrics = MetricCollector()
        self.alerts: List[Dict[str, Any]] = []
        
        # Component health
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Monitoring state
        self._is_running = False
        self._monitor_task = None
        self._health_check_task = None
        
    async def start(self):
        """Start monitoring services."""
        if self._is_running:
            return
            
        self._is_running = True
        self._monitor_task = asyncio.create_task(self._collect_metrics())
        self._health_check_task = asyncio.create_task(self._run_health_checks())
        
        logger.info("Started system monitoring")
        
    async def stop(self):
        """Stop monitoring services."""
        self._is_running = False
        
        for task in [self._monitor_task, self._health_check_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        logger.info("Stopped system monitoring")
        
    async def _collect_metrics(self):
        """Continuously collect system metrics."""
        while self._is_running:
            try:
                # Collect all metrics
                await self._collect_memory_metrics()
                await self._collect_risk_metrics()
                await self._collect_llm_metrics()
                await self._collect_firestore_metrics()
                await self._collect_system_metrics()
                
                # Check for alerts
                await self._check_alerts()
                
                # Wait before next collection
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(30)
                
    async def _collect_memory_metrics(self):
        """Collect memory system metrics."""
        try:
            stats = self.memory_manager.get_stats()
            
            # Memory counts
            memory_count = await self.memory_manager.memory.count_all()
            self.metrics.record("memory.total_count", memory_count)
            
            # Pruning metrics
            self.metrics.record("memory.pruned_count", stats["memories_pruned"])
            self.metrics.record("memory.patterns_merged", stats["patterns_merged"])
            self.metrics.record("memory.archived_count", stats["memories_archived"])
            
            # Cache metrics
            self.metrics.record("memory.pattern_cache_size", stats["cache_sizes"]["pattern_cache"])
            self.metrics.record("memory.query_cache_size", stats["cache_sizes"]["query_cache"])
            
            # Category distribution
            for category, limit in self.memory_manager.category_limits.items():
                count = await self.memory_manager.memory.count_by_category(category)
                usage = count / limit if limit > 0 else 0
                self.metrics.record(f"memory.category.{category}.usage", usage)
                
        except Exception as e:
            logger.error(f"Error collecting memory metrics: {e}")
            
    async def _collect_risk_metrics(self):
        """Collect risk management metrics."""
        try:
            status = self.risk_manager.get_status()
            
            # Circuit breaker status
            for name, breaker_status in status["circuit_breakers"].items():
                self.metrics.record(
                    f"risk.circuit_breaker.{name}.tripped",
                    1.0 if breaker_status["tripped"] else 0.0
                )
                self.metrics.record(
                    f"risk.circuit_breaker.{name}.activations",
                    breaker_status["activation_count"]
                )
                
            # Portfolio metrics
            portfolio = status["portfolio_state"]
            self.metrics.record("risk.portfolio.total_value", portfolio["total_value"])
            self.metrics.record("risk.portfolio.position_count", portfolio["position_count"])
            
            risk_level, risk_score = portfolio["risk_score"]
            self.metrics.record("risk.portfolio.risk_score", risk_score)
            self.metrics.record(
                "risk.portfolio.risk_level",
                {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}.get(risk_level, 0)
            )
            
            # Gas history
            self.metrics.record("risk.gas_history_size", status["gas_history_size"])
            
        except Exception as e:
            logger.error(f"Error collecting risk metrics: {e}")
            
    async def _collect_llm_metrics(self):
        """Collect LLM optimization metrics."""
        try:
            stats = self.llm_optimizer.get_stats()
            
            # Cache performance
            self.metrics.record("llm.cache_hit_rate", stats["cache_hit_rate"])
            self.metrics.record("llm.total_requests", stats["total_requests"])
            self.metrics.record("llm.tokens_saved", stats["tokens_saved"])
            
            # Template usage
            self.metrics.record("llm.template_uses", stats["template_uses"])
            
            # Batch processing
            self.metrics.record("llm.batch_requests", stats["batch_requests"])
            
            # Cache sizes
            self.metrics.record("llm.direct_cache_size", stats["cache_sizes"]["direct"])
            self.metrics.record("llm.semantic_cache_size", stats["cache_sizes"]["semantic"])
            
        except Exception as e:
            logger.error(f"Error collecting LLM metrics: {e}")
            
    async def _collect_firestore_metrics(self):
        """Collect Firestore optimization metrics."""
        try:
            stats = self.firestore_optimizer.get_stats()
            
            # Write performance
            write_stats = stats["write_stats"]
            self.metrics.record("firestore.total_writes", write_stats["total_writes"])
            self.metrics.record("firestore.batch_efficiency", write_stats["batch_efficiency"])
            self.metrics.record("firestore.bytes_saved", write_stats["bytes_saved"])
            self.metrics.record("firestore.cost_saved", write_stats["cost_saved"])
            
            # Cache performance
            cache_stats = stats["cache_stats"]
            self.metrics.record("firestore.read_cache_size", cache_stats["read_cache_size"])
            self.metrics.record("firestore.cache_hit_rate", cache_stats["cache_hit_rate"])
            
            # Query performance
            query_stats = stats["query_stats"]
            self.metrics.record("firestore.unique_queries", query_stats["unique_queries"])
            self.metrics.record("firestore.slow_queries", query_stats["slow_queries"])
            
        except Exception as e:
            logger.error(f"Error collecting Firestore metrics: {e}")
            
    async def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.record("system.cpu_percent", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics.record("system.memory_percent", memory.percent)
            self.metrics.record("system.memory_available_mb", memory.available / 1024 / 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics.record("system.disk_percent", disk.percent)
            
            # Process metrics
            process = psutil.Process()
            self.metrics.record("process.cpu_percent", process.cpu_percent())
            self.metrics.record("process.memory_mb", process.memory_info().rss / 1024 / 1024)
            self.metrics.record("process.threads", process.num_threads())
            
            # Event loop metrics
            loop = asyncio.get_event_loop()
            pending_tasks = len([task for task in asyncio.all_tasks(loop) if not task.done()])
            self.metrics.record("asyncio.pending_tasks", pending_tasks)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            
    async def _run_health_checks(self):
        """Run periodic health checks."""
        while self._is_running:
            try:
                # Run all health checks
                await self._check_memory_health()
                await self._check_risk_health()
                await self._check_llm_health()
                await self._check_firestore_health()
                await self._check_system_health()
                
                # Store health check results
                await self._persist_health_status()
                
                # Wait before next check
                await asyncio.sleep(settings.health_check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error running health checks: {e}")
                await asyncio.sleep(60)
                
    async def _check_memory_health(self):
        """Check memory system health."""
        try:
            # Check if pruning is working
            last_prune = self.memory_manager.stats.last_prune_time
            if last_prune:
                hours_since_prune = (datetime.utcnow() - last_prune).total_seconds() / 3600
                if hours_since_prune > settings.memory_prune_interval_hours * 2:
                    status = "unhealthy"
                    message = f"Pruning overdue by {hours_since_prune:.1f} hours"
                elif hours_since_prune > settings.memory_prune_interval_hours * 1.5:
                    status = "degraded"
                    message = "Pruning slightly delayed"
                else:
                    status = "healthy"
                    message = "Pruning on schedule"
            else:
                status = "degraded"
                message = "No pruning history"
                
            # Check memory growth
            memory_count = await self.memory_manager.memory.count_all()
            growth_rate = self.metrics.get_average("memory.total_count", 60)
            
            details = {
                "memory_count": memory_count,
                "growth_rate": growth_rate,
                "last_prune": last_prune.isoformat() if last_prune else None
            }
            
            self.health_checks["memory"] = HealthCheck(
                component="memory",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                details=details
            )
            
        except Exception as e:
            self.health_checks["memory"] = HealthCheck(
                component="memory",
                status="unhealthy",
                message=f"Health check failed: {e}",
                timestamp=datetime.utcnow()
            )
            
    async def _check_risk_health(self):
        """Check risk management health."""
        try:
            # Check circuit breakers
            breaker_status = await self.risk_manager.check_circuit_breakers()
            tripped_count = sum(1 for tripped in breaker_status.values() if tripped)
            
            if tripped_count >= 3:
                status = "unhealthy"
                message = f"{tripped_count} circuit breakers tripped"
            elif tripped_count >= 1:
                status = "degraded"
                message = f"{tripped_count} circuit breaker(s) tripped"
            else:
                status = "healthy"
                message = "All circuit breakers operational"
                
            # Check portfolio risk
            risk_level, risk_score = await self.risk_manager.calculate_portfolio_risk_score()
            
            details = {
                "tripped_breakers": [name for name, tripped in breaker_status.items() if tripped],
                "risk_level": risk_level,
                "risk_score": risk_score
            }
            
            self.health_checks["risk"] = HealthCheck(
                component="risk",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                details=details
            )
            
        except Exception as e:
            self.health_checks["risk"] = HealthCheck(
                component="risk",
                status="unhealthy",
                message=f"Health check failed: {e}",
                timestamp=datetime.utcnow()
            )
            
    async def _check_llm_health(self):
        """Check LLM optimization health."""
        try:
            stats = self.llm_optimizer.get_stats()
            cache_hit_rate = stats["cache_hit_rate"]
            
            if cache_hit_rate < 0.3:
                status = "degraded"
                message = f"Low cache hit rate: {cache_hit_rate:.1%}"
            elif cache_hit_rate < settings.cache_hit_rate_target:
                status = "healthy"
                message = f"Cache hit rate below target: {cache_hit_rate:.1%}"
            else:
                status = "healthy"
                message = f"Cache performing well: {cache_hit_rate:.1%}"
                
            details = {
                "cache_hit_rate": cache_hit_rate,
                "total_requests": stats["total_requests"],
                "tokens_saved": stats["tokens_saved"]
            }
            
            self.health_checks["llm"] = HealthCheck(
                component="llm",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                details=details
            )
            
        except Exception as e:
            self.health_checks["llm"] = HealthCheck(
                component="llm",
                status="unhealthy",
                message=f"Health check failed: {e}",
                timestamp=datetime.utcnow()
            )
            
    async def _check_firestore_health(self):
        """Check Firestore optimization health."""
        try:
            stats = self.firestore_optimizer.get_stats()
            batch_efficiency = stats["write_stats"]["batch_efficiency"]
            
            if batch_efficiency < 0.5:
                status = "degraded"
                message = f"Low batch efficiency: {batch_efficiency:.1%}"
            else:
                status = "healthy"
                message = f"Batching effective: {batch_efficiency:.1%}"
                
            # Check write rate
            writes_per_min = self.metrics.get_average("firestore.total_writes", 1) * 60
            if writes_per_min and writes_per_min > settings.firestore_max_writes_per_minute:
                status = "degraded"
                message += f" - High write rate: {writes_per_min:.0f}/min"
                
            details = {
                "batch_efficiency": batch_efficiency,
                "writes_per_minute": writes_per_min,
                "cost_saved": stats["write_stats"]["cost_saved"]
            }
            
            self.health_checks["firestore"] = HealthCheck(
                component="firestore",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                details=details
            )
            
        except Exception as e:
            self.health_checks["firestore"] = HealthCheck(
                component="firestore",
                status="unhealthy",
                message=f"Health check failed: {e}",
                timestamp=datetime.utcnow()
            )
            
    async def _check_system_health(self):
        """Check system resource health."""
        try:
            cpu_percent = self.metrics.get_latest("system.cpu_percent") or 0
            memory_percent = self.metrics.get_latest("system.memory_percent") or 0
            disk_percent = self.metrics.get_latest("system.disk_percent") or 0
            
            issues = []
            if cpu_percent > 80:
                issues.append(f"High CPU: {cpu_percent:.0f}%")
            if memory_percent > 85:
                issues.append(f"High memory: {memory_percent:.0f}%")
            if disk_percent > 90:
                issues.append(f"Low disk: {disk_percent:.0f}% used")
                
            if len(issues) >= 2:
                status = "unhealthy"
                message = "; ".join(issues)
            elif issues:
                status = "degraded"
                message = "; ".join(issues)
            else:
                status = "healthy"
                message = "System resources normal"
                
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent
            }
            
            self.health_checks["system"] = HealthCheck(
                component="system",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                details=details
            )
            
        except Exception as e:
            self.health_checks["system"] = HealthCheck(
                component="system",
                status="unhealthy",
                message=f"Health check failed: {e}",
                timestamp=datetime.utcnow()
            )
            
    async def _check_alerts(self):
        """Check metrics for alert conditions."""
        try:
            # Memory growth alert
            memory_growth = self.metrics.get_average("memory.total_count", 60)
            if memory_growth and memory_growth > 100:  # 100 memories/hour
                await self._create_alert(
                    "high_memory_growth",
                    f"Memory growing rapidly: {memory_growth:.0f}/hour",
                    "warning"
                )
                
            # Risk alert
            risk_score = self.metrics.get_latest("risk.portfolio.risk_score")
            if risk_score and risk_score > 75:
                await self._create_alert(
                    "high_portfolio_risk",
                    f"Portfolio risk critical: {risk_score:.0f}",
                    "critical"
                )
                
            # Performance alerts
            cache_hit_rate = self.metrics.get_latest("llm.cache_hit_rate")
            if cache_hit_rate and cache_hit_rate < 0.3:
                await self._create_alert(
                    "low_cache_performance",
                    f"LLM cache hit rate low: {cache_hit_rate:.1%}",
                    "warning"
                )
                
            # System alerts
            cpu_95th = self.metrics.get_percentile("system.cpu_percent", 95, 10)
            if cpu_95th and cpu_95th > 90:
                await self._create_alert(
                    "sustained_high_cpu",
                    f"CPU sustained high: {cpu_95th:.0f}% (95th percentile)",
                    "warning"
                )
                
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
            
    async def _create_alert(self, alert_type: str, message: str, severity: str):
        """Create and store an alert."""
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.utcnow(),
            "resolved": False
        }
        
        # Check if alert already exists
        for existing in self.alerts:
            if (existing["type"] == alert_type and 
                not existing["resolved"] and
                (datetime.utcnow() - existing["timestamp"]).total_seconds() < 3600):
                return  # Don't duplicate alerts within an hour
                
        self.alerts.append(alert)
        logger.warning(f"Alert created: [{severity}] {message}")
        
        # Store in Firestore
        await self.firestore.set_document(
            "system_alerts",
            f"alert_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{alert_type}",
            alert
        )
        
    async def _persist_health_status(self):
        """Persist health check results."""
        try:
            # Aggregate health status
            overall_status = "healthy"
            unhealthy_components = []
            
            for component, check in self.health_checks.items():
                if check.status == "unhealthy":
                    overall_status = "unhealthy"
                    unhealthy_components.append(component)
                elif check.status == "degraded" and overall_status == "healthy":
                    overall_status = "degraded"
                    
            # Store health summary
            await self.firestore.set_document(
                "system_health",
                "current",
                {
                    "overall_status": overall_status,
                    "timestamp": datetime.utcnow(),
                    "unhealthy_components": unhealthy_components,
                    "checks": {
                        name: {
                            "status": check.status,
                            "message": check.message,
                            "details": check.details
                        }
                        for name, check in self.health_checks.items()
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to persist health status: {e}")
            
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        overall = "healthy"
        issues = []
        
        for component, check in self.health_checks.items():
            if check.status == "unhealthy":
                overall = "unhealthy"
                issues.append(f"{component}: {check.message}")
            elif check.status == "degraded":
                if overall == "healthy":
                    overall = "degraded"
                issues.append(f"{component}: {check.message}")
                
        return {
            "status": overall,
            "timestamp": datetime.utcnow(),
            "issues": issues,
            "components": {
                name: {
                    "status": check.status,
                    "message": check.message,
                    "last_check": check.timestamp
                }
                for name, check in self.health_checks.items()
            }
        }
        
    def get_metrics_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get summary of key metrics."""
        return {
            "memory": {
                "total_count": self.metrics.get_latest("memory.total_count"),
                "growth_rate": self.metrics.get_average("memory.total_count", window_minutes),
                "pruned_count": self.metrics.get_latest("memory.pruned_count")
            },
            "risk": {
                "portfolio_value": self.metrics.get_latest("risk.portfolio.total_value"),
                "risk_score": self.metrics.get_latest("risk.portfolio.risk_score"),
                "circuit_breakers_tripped": sum(
                    1 for name in ["rapid_loss", "portfolio_drawdown", "gas_manipulation"]
                    if self.metrics.get_latest(f"risk.circuit_breaker.{name}.tripped") == 1.0
                )
            },
            "performance": {
                "llm_cache_hit_rate": self.metrics.get_latest("llm.cache_hit_rate"),
                "firestore_batch_efficiency": self.metrics.get_latest("firestore.batch_efficiency"),
                "avg_cpu_percent": self.metrics.get_average("system.cpu_percent", window_minutes)
            },
            "alerts": [
                {
                    "type": alert["type"],
                    "message": alert["message"],
                    "severity": alert["severity"],
                    "age_minutes": (datetime.utcnow() - alert["timestamp"]).total_seconds() / 60
                }
                for alert in self.alerts
                if not alert["resolved"] and 
                   (datetime.utcnow() - alert["timestamp"]).total_seconds() < window_minutes * 60
            ]
        }