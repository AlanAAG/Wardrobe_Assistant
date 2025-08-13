import asyncio
import json
import time
import psutil
import structlog
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from collections import defaultdict, deque
import statistics
import aiofiles
import aioredis
from abc import ABC, abstractmethod

# Enhanced monitoring system for AI Wardrobe Management
# Provides intelligent monitoring, alerting, and performance optimization

class MonitoringLevel(Enum):
    """Monitoring detail levels"""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertSeverity(Enum):
    """Alert severity levels with escalation"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics tracked"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class MetricValue:
    """Single metric measurement"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    component: str = "unknown"

@dataclass
class PerformanceSnapshot:
    """System performance snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    active_connections: int
    cache_hit_rate: float
    response_times: Dict[str, float]

@dataclass
class ComponentHealth:
    """Health status of a system component"""
    component_name: str
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    response_time_ms: float
    error_rate: float
    throughput: float
    dependencies_healthy: bool
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Enhanced alert with context and intelligence"""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    component: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    suggested_actions: List[str] = field(default_factory=list)
    related_alerts: List[str] = field(default_factory=list)

class AlertRule:
    """Intelligent alert rule with adaptive thresholds"""
    
    def __init__(self, 
                 name: str,
                 metric_name: str,
                 threshold_func: Callable[[float, Dict], bool],
                 severity: AlertSeverity,
                 cooldown_minutes: int = 15,
                 adaptive: bool = True,
                 context_aware: bool = True):
        self.name = name
        self.metric_name = metric_name
        self.threshold_func = threshold_func
        self.severity = severity
        self.cooldown_minutes = cooldown_minutes
        self.adaptive = adaptive
        self.context_aware = context_aware
        
        # State tracking
        self.last_triggered = None
        self.baseline_calculator = BaselineCalculator()
        self.context_analyzer = ContextAnalyzer()
    
    async def evaluate(self, metric_value: float, context: Dict[str, Any]) -> Optional[Alert]:
        """Evaluate if alert should be triggered"""
        
        # Check cooldown
        if self.last_triggered and datetime.now() - self.last_triggered < timedelta(minutes=self.cooldown_minutes):
            return None
        
        # Calculate adaptive threshold
        threshold_context = context.copy()
        if self.adaptive:
            baseline = await self.baseline_calculator.get_baseline(self.metric_name)
            threshold_context['baseline'] = baseline
            threshold_context['adaptive_multiplier'] = await self._calculate_adaptive_multiplier(context)
        
        # Evaluate threshold
        should_trigger = self.threshold_func(metric_value, threshold_context)
        
        if should_trigger:
            self.last_triggered = datetime.now()
            return await self._create_alert(metric_value, threshold_context)
        
        return None
    
    async def _calculate_adaptive_multiplier(self, context: Dict) -> float:
        """Calculate adaptive threshold multiplier based on context"""
        
        multiplier = 1.0
        
        # Time-based adjustments
        hour = datetime.now().hour
        if 22 <= hour or hour <= 6:  # Night hours
            multiplier *= 1.5  # More lenient at night
        elif 9 <= hour <= 17:  # Business hours
            multiplier *= 0.8  # Stricter during business hours
        
        # Load-based adjustments
        system_load = context.get('system_load', 0.5)
        if system_load > 0.8:
            multiplier *= 1.3  # More lenient under high load
        
        # Historical pattern adjustments
        if context.get('unusual_pattern_detected', False):
            multiplier *= 0.9  # Stricter during unusual patterns
        
        return multiplier
    
    async def _create_alert(self, metric_value: float, context: Dict) -> Alert:
        """Create detailed alert with context and suggestions"""
        
        alert_id = f"{self.name}_{int(time.time())}"
        
        # Generate suggested actions
        suggestions = await self._generate_suggestions(metric_value, context)
        
        # Find related alerts
        related = await self._find_related_alerts(context)
        
        return Alert(
            id=alert_id,
            severity=self.severity,
            title=f"{self.metric_name} threshold exceeded",
            description=f"{self.metric_name} value {metric_value} exceeded threshold in {context.get('component', 'unknown')}",
            component=context.get('component', 'unknown'),
            metric_name=self.metric_name,
            current_value=metric_value,
            threshold_value=context.get('threshold', 0),
            timestamp=datetime.now(),
            context=context,
            suggested_actions=suggestions,
            related_alerts=related
        )
    
    async def _generate_suggestions(self, metric_value: float, context: Dict) -> List[str]:
        """Generate intelligent suggestions based on alert context"""
        
        suggestions = []
        
        if self.metric_name == "response_time_p95":
            suggestions.extend([
                "Check database connection pool usage",
                "Review recent deployments for performance regressions",
                "Consider increasing AI API timeouts",
                "Monitor cache hit rates for degradation"
            ])
        elif self.metric_name == "error_rate":
            suggestions.extend([
                "Check recent error logs for patterns",
                "Verify external API connectivity",
                "Review recent configuration changes",
                "Consider enabling fallback mechanisms"
            ])
        elif self.metric_name == "memory_usage":
            suggestions.extend([
                "Check for memory leaks in recent code changes",
                "Review cache size configurations",
                "Consider garbage collection optimization",
                "Monitor for infinite loops or large data processing"
            ])
        
        return suggestions
    
    async def _find_related_alerts(self, context: Dict) -> List[str]:
        """Find related alerts for correlation analysis"""
        # Implementation would query alert history for correlated alerts
        return []

class BaselineCalculator:
    """Calculates baseline metrics for adaptive thresholds"""
    
    def __init__(self):
        self.baselines = {}
        self.history_window = timedelta(days=7)
    
    async def get_baseline(self, metric_name: str) -> float:
        """Get baseline value for a metric"""
        
        if metric_name not in self.baselines:
            await self._calculate_baseline(metric_name)
        
        baseline_data = self.baselines.get(metric_name, {})
        return baseline_data.get('value', 0.0)
    
    async def _calculate_baseline(self, metric_name: str):
        """Calculate baseline from historical data"""
        
        # Get historical data (implementation would query metrics storage)
        historical_values = await self._get_historical_values(metric_name)
        
        if historical_values:
            # Use median for robust baseline
            baseline_value = statistics.median(historical_values)
            
            self.baselines[metric_name] = {
                'value': baseline_value,
                'calculated_at': datetime.now(),
                'sample_size': len(historical_values)
            }
    
    async def _get_historical_values(self, metric_name: str) -> List[float]:
        """Get historical values for baseline calculation"""
        # Implementation would query metrics database
        return []

class ContextAnalyzer:
    """Analyzes context for intelligent alerting"""
    
    async def analyze_context(self, component: str, metric_name: str) -> Dict[str, Any]:
        """Analyze current context for metric evaluation"""
        
        context = {
            'component': component,
            'metric_name': metric_name,
            'timestamp': datetime.now(),
            'system_load': await self._get_system_load(),
            'time_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday()
        }
        
        # Add component-specific context
        if component == "ai_provider":
            context.update(await self._get_ai_provider_context())
        elif component == "cache":
            context.update(await self._get_cache_context())
        elif component == "database":
            context.update(await self._get_database_context())
        
        return context
    
    async def _get_system_load(self) -> float:
        """Get current system load"""
        return psutil.cpu_percent(interval=1) / 100.0
    
    async def _get_ai_provider_context(self) -> Dict[str, Any]:
        """Get AI provider specific context"""
        return {
            'api_quota_remaining': 0.8,  # Would be retrieved from API
            'recent_error_patterns': [],
            'model_performance_trend': 'stable'
        }
    
    async def _get_cache_context(self) -> Dict[str, Any]:
        """Get cache specific context"""
        return {
            'memory_pressure': False,
            'eviction_rate': 0.1,
            'warming_in_progress': False
        }
    
    async def _get_database_context(self) -> Dict[str, Any]:
        """Get database specific context"""
        return {
            'connection_pool_utilization': 0.6,
            'query_performance_trend': 'stable',
            'replication_lag': 0
        }

class ComponentMonitor:
    """Advanced component-specific monitoring"""
    
    def __init__(self, component_name: str, parent_monitor=None):
        self.component_name = component_name
        self.parent = parent_monitor
        
        # Metrics storage
        self.metrics_buffer = deque(maxlen=1000)
        self.aggregated_metrics = {}
        
        # Performance tracking
        self.operation_timings = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        
        # Health tracking
        self.last_health_check = None
        self.health_status = "unknown"
        
        # Alert management
        self.active_alerts = {}
        self.alert_history = deque(maxlen=100)
        
        # Logger
        self.logger = structlog.get_logger(component=component_name)
    
    async def track_operation(self, operation_name: str, func: Callable, *args, **kwargs):
        """Track operation with comprehensive metrics"""
        
        start_time = time.time()
        operation_id = f"{operation_name}_{int(start_time * 1000)}"
        
        # Add context
        context = {
            'operation_name': operation_name,
            'operation_id': operation_id,
            'component': self.component_name,
            'start_time': start_time
        }
        
        try:
            # Execute operation
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            duration_ms = (time.time() - start_time) * 1000
            await self._record_success(operation_name, duration_ms, context)
            
            return result
            
        except Exception as e:
            # Record error
            duration_ms = (time.time() - start_time) * 1000
            await self._record_error(operation_name, e, duration_ms, context)
            raise
    
    async def record_metric(self, name: str, value: float, 
                          metric_type: MetricType = MetricType.GAUGE,
                          labels: Dict[str, str] = None):
        """Record a custom metric"""
        
        metric = MetricValue(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            labels=labels or {},
            component=self.component_name
        )
        
        self.metrics_buffer.append(metric)
        await self._update_aggregated_metrics(metric)
        
        # Check alert rules
        if self.parent:
            await self.parent.evaluate_alerts(metric, self.component_name)
    
    async def get_health_status(self) -> ComponentHealth:
        """Get comprehensive component health status"""
        
        now = datetime.now()
        
        # Calculate metrics
        recent_errors = sum(1 for m in self.metrics_buffer 
                          if m.metric_type == MetricType.COUNTER and 
                             m.name.endswith('_error') and
                             (now - m.timestamp).seconds < 300)  # Last 5 minutes
        
        recent_operations = sum(1 for m in self.metrics_buffer 
                              if (now - m.timestamp).seconds < 300)
        
        error_rate = recent_errors / max(recent_operations, 1)
        
        # Calculate average response time
        recent_timings = [m.value for m in self.metrics_buffer 
                         if m.name.endswith('_duration_ms') and
                            (now - m.timestamp).seconds < 300]
        
        avg_response_time = statistics.mean(recent_timings) if recent_timings else 0
        
        # Determine status
        if error_rate > 0.1:  # >10% error rate
            status = "unhealthy"
        elif error_rate > 0.05 or avg_response_time > 5000:  # >5% error rate or >5s response
            status = "degraded"
        else:
            status = "healthy"
        
        return ComponentHealth(
            component_name=self.component_name,
            status=status,
            last_check=now,
            response_time_ms=avg_response_time,
            error_rate=error_rate,
            throughput=len(recent_timings) / 5.0,  # Operations per minute
            dependencies_healthy=await self._check_dependencies(),
            custom_metrics=self.aggregated_metrics
        )
    
    async def _record_success(self, operation_name: str, duration_ms: float, context: Dict):
        """Record successful operation"""
        
        self.success_counts[operation_name] += 1
        self.operation_timings[operation_name].append(duration_ms)
        
        # Keep only recent timings
        if len(self.operation_timings[operation_name]) > 100:
            self.operation_timings[operation_name] = self.operation_timings[operation_name][-100:]
        
        # Record metrics
        await self.record_metric(f"{operation_name}_duration_ms", duration_ms, MetricType.TIMER)
        await self.record_metric(f"{operation_name}_success", 1, MetricType.COUNTER)
        
        self.logger.info(
            f"Operation completed successfully",
            operation=operation_name,
            duration_ms=duration_ms,
            **context
        )
    
    async def _record_error(self, operation_name: str, error: Exception, duration_ms: float, context: Dict):
        """Record failed operation"""
        
        self.error_counts[operation_name] += 1
        
        # Record metrics
        await self.record_metric(f"{operation_name}_error", 1, MetricType.COUNTER)
        await self.record_metric(f"{operation_name}_duration_ms", duration_ms, MetricType.TIMER)
        
        self.logger.error(
            f"Operation failed",
            operation=operation_name,
            duration_ms=duration_ms,
            error_type=type(error).__name__,
            error_message=str(error),
            **context,
            exc_info=True
        )
    
    async def _update_aggregated_metrics(self, metric: MetricValue):
        """Update aggregated metrics for dashboard"""
        
        key = f"{metric.name}_{metric.metric_type.value}"
        
        if key not in self.aggregated_metrics:
            self.aggregated_metrics[key] = {
                'count': 0,
                'sum': 0,
                'min': float('inf'),
                'max': float('-inf'),
                'avg': 0,
                'last_value': 0,
                'last_updated': metric.timestamp
            }
        
        agg = self.aggregated_metrics[key]
        agg['count'] += 1
        agg['sum'] += metric.value
        agg['min'] = min(agg['min'], metric.value)
        agg['max'] = max(agg['max'], metric.value)
        agg['avg'] = agg['sum'] / agg['count']
        agg['last_value'] = metric.value
        agg['last_updated'] = metric.timestamp
    
    async def _check_dependencies(self) -> bool:
        """Check if component dependencies are healthy"""
        # Implementation would check dependencies like databases, APIs, etc.
        return True

class EnhancedSystemMonitor:
    """Central enhanced monitoring system with intelligence"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Core components
        self.component_monitors = {}
        self.alert_rules = []
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        
        # Intelligence components
        self.baseline_calculator = BaselineCalculator()
        self.context_analyzer = ContextAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Storage
        self.metrics_storage = MetricsStorage()
        self.redis_client = None
        
        # Dashboard data
        self.dashboard_data = DashboardDataCollector()
        
        # Logger
        self.logger = structlog.get_logger(component="enhanced_system_monitor")
        
        # Initialize default alert rules
        self._setup_default_alert_rules()
    
    async def initialize(self):
        """Initialize the enhanced monitoring system"""
        
        try:
            # Initialize Redis for distributed metrics
            self.redis_client = await aioredis.from_url("redis://localhost:6379")
            await self.redis_client.ping()
            self.logger.info("Redis metrics storage initialized")
        except Exception as e:
            self.logger.warning(f"Redis initialization failed, using local storage: {e}")
        
        # Initialize metrics storage
        await self.metrics_storage.initialize()
        
        # Start background tasks
        asyncio.create_task(self._background_analysis_loop())
        asyncio.create_task(self._dashboard_update_loop())
        asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("Enhanced system monitor initialized successfully")
    
    def register_component(self, component_name: str) -> ComponentMonitor:
        """Register a new component for monitoring"""
        
        if component_name in self.component_monitors:
            return self.component_monitors[component_name]
        
        monitor = ComponentMonitor(component_name, parent=self)
        self.component_monitors[component_name] = monitor
        
        self.logger.info(f"Registered component monitor: {component_name}")
        return monitor
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a custom alert rule"""
        self.alert_rules.append(rule)
        self.logger.info(f"Added alert rule: {rule.name}")
    
    async def evaluate_alerts(self, metric: MetricValue, component: str):
        """Evaluate all alert rules for a metric"""
        
        context = await self.context_analyzer.analyze_context(component, metric.name)
        context.update({
            'metric_value': metric.value,
            'metric_timestamp': metric.timestamp,
            'metric_labels': metric.labels
        })
        
        for rule in self.alert_rules:
            if rule.metric_name == metric.name or rule.metric_name == "all":
                try:
                    alert = await rule.evaluate(metric.value, context)
                    if alert:
                        await self._handle_new_alert(alert)
                except Exception as e:
                    self.logger.error(f"Error evaluating alert rule {rule.name}: {e}")
    
    async def _handle_new_alert(self, alert: Alert):
        """Handle a new alert"""
        
        # Check for duplicate alerts
        if alert.id in self.active_alerts:
            return
        
        # Store alert
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Log alert
        self.logger.warning(
            f"Alert triggered: {alert.title}",
            alert_id=alert.id,
            severity=alert.severity.value,
            component=alert.component,
            metric=alert.metric_name,
            value=alert.current_value,
            threshold=alert.threshold_value
        )
        
        # Send notifications
        await self._send_alert_notifications(alert)
        
        # Store in metrics storage
        await self.metrics_storage.store_alert(alert)
    
    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications to configured channels"""
        
        # Email notifications for critical alerts
        if alert.severity == AlertSeverity.CRITICAL:
            await self._send_email_notification(alert)
        
        # Slack notifications for high+ severity
        if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            await self._send_slack_notification(alert)
        
        # Webhook notifications
        await self._send_webhook_notification(alert)
    
    async def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        # Implementation would send email
        self.logger.info(f"Would send email notification for alert {alert.id}")
    
    async def _send_slack_notification(self, alert: Alert):
        """Send Slack notification"""
        # Implementation would send Slack message
        self.logger.info(f"Would send Slack notification for alert {alert.id}")
    
    async def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification"""
        # Implementation would send webhook
        self.logger.info(f"Would send webhook notification for alert {alert.id}")
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        
        health_data = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "active_alerts": len(self.active_alerts),
            "critical_alerts": len([a for a in self.active_alerts.values() 
                                  if a.severity == AlertSeverity.CRITICAL]),
            "system_metrics": await self._get_system_metrics()
        }
        
        # Get component health
        component_statuses = []
        for name, monitor in self.component_monitors.items():
            component_health = await monitor.get_health_status()
            health_data["components"][name] = asdict(component_health)
            component_statuses.append(component_health.status)
        
        # Determine overall status
        if "unhealthy" in component_statuses:
            health_data["overall_status"] = "unhealthy"
        elif "degraded" in component_statuses:
            health_data["overall_status"] = "degraded"
        elif len(self.active_alerts) > 5:
            health_data["overall_status"] = "degraded"
        
        return health_data
    
    async def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        
        return await self.dashboard_data.collect_dashboard_data(self.component_monitors)
    
    async def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get AI-powered optimization suggestions"""
        
        return await self.performance_analyzer.analyze_and_suggest_optimizations(
            self.component_monitors, self.active_alerts
        )
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network
        network = psutil.net_io_counters()
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3),
            "network_bytes_sent": network.bytes_sent,
            "network_bytes_recv": network.bytes_recv
        }
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules for the system"""
        
        # Response time alerts
        self.add_alert_rule(AlertRule(
            name="high_response_time",
            metric_name="response_time_p95",
            threshold_func=lambda value, ctx: value > ctx.get('baseline', 1000) * ctx.get('adaptive_multiplier', 2),
            severity=AlertSeverity.HIGH,
            adaptive=True
        ))
        
        # Error rate alerts
        self.add_alert_rule(AlertRule(
            name="high_error_rate",
            metric_name="error_rate",
            threshold_func=lambda value, ctx: value > 0.1,  # 10% error rate
            severity=AlertSeverity.CRITICAL,
            cooldown_minutes=5
        ))
        
        # Memory usage alerts
        self.add_alert_rule(AlertRule(
            name="high_memory_usage",
            metric_name="memory_percent",
            threshold_func=lambda value, ctx: value > 85,
            severity=AlertSeverity.HIGH,
            adaptive=True
        ))
        
        # Cache performance alerts
        self.add_alert_rule(AlertRule(
            name="low_cache_hit_rate",
            metric_name="cache_hit_rate",
            threshold_func=lambda value, ctx: value < 0.6,
            severity=AlertSeverity.MEDIUM,
            adaptive=True
        ))
        
        # AI API alerts
        self.add_alert_rule(AlertRule(
            name="ai_api_failures",
            metric_name="ai_api_error_rate",
            threshold_func=lambda value, ctx: value > 0.3,
            severity=AlertSeverity.CRITICAL,
            cooldown_minutes=10
        ))
    
    async def _background_analysis_loop(self):
        """Background loop for continuous analysis"""
        
        while True:
            try:
                # Update baselines
                await self.baseline_calculator._calculate_all_baselines()
                
                # Analyze performance trends
                await self.performance_analyzer.analyze_trends()
                
                # Check for anomalies
                await self._check_for_anomalies()
                
                # Cleanup old data
                await self._cleanup_old_data()
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in background analysis loop: {e}")
                await asyncio.sleep(60)
    
    async def _dashboard_update_loop(self):
        """Background loop for dashboard data updates"""
        
        while True:
            try:
                # Update dashboard data
                await self.dashboard_data.update_cached_data(self.component_monitors)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        
        while True:
            try:
                # Cleanup resolved alerts
                current_time = datetime.now()
                resolved_alerts = []
                
                for alert_id, alert in self.active_alerts.items():
                    if current_time - alert.timestamp > timedelta(hours=24):
                        resolved_alerts.append(alert_id)
                
                for alert_id in resolved_alerts:
                    del self.active_alerts[alert_id]
                
                if resolved_alerts:
                    self.logger.info(f"Cleaned up {len(resolved_alerts)} resolved alerts")
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)
    
    async def _check_for_anomalies(self):
        """Check for system anomalies"""
        
        # Get recent metrics
        recent_metrics = await self.metrics_storage.get_recent_metrics(minutes=60)
        
        # Analyze for anomalies
        anomalies = await self.performance_analyzer.detect_anomalies(recent_metrics)
        
        for anomaly in anomalies:
            self.logger.warning(f"Anomaly detected: {anomaly}")
    
    async def _cleanup_old_data(self):
        """Cleanup old monitoring data"""
        
        # Cleanup old metrics
        cutoff_time = datetime.now() - timedelta(days=7)
        await self.metrics_storage.cleanup_old_metrics(cutoff_time)

class PerformanceAnalyzer:
    """AI-powered performance analysis and optimization"""
    
    def __init__(self):
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.optimization_engine = OptimizationEngine()
    
    async def analyze_trends(self):
        """Analyze performance trends"""
        # Implementation for trend analysis
        pass
    
    async def detect_anomalies(self, metrics: List[MetricValue]) -> List[Dict]:
        """Detect anomalies in metrics"""
        # Implementation for anomaly detection
        return []
    
    async def analyze_and_suggest_optimizations(self, 
                                              component_monitors: Dict[str, ComponentMonitor],
                                              active_alerts: Dict[str, Alert]) -> List[Dict[str, Any]]:
        """Analyze system and suggest optimizations"""
        
        suggestions = []
        
        # Analyze each component
        for name, monitor in component_monitors.items():
            component_suggestions = await self._analyze_component_performance(name, monitor)
            suggestions.extend(component_suggestions)
        
        # Analyze alerts for patterns
        alert_suggestions = await self._analyze_alert_patterns(active_alerts)
        suggestions.extend(alert_suggestions)
        
        return suggestions
    
    async def _analyze_component_performance(self, 
                                           component_name: str, 
                                           monitor: ComponentMonitor) -> List[Dict[str, Any]]:
        """Analyze individual component performance"""
        
        suggestions = []
        health = await monitor.get_health_status()
        
        if health.response_time_ms > 2000:
            suggestions.append({
                "type": "performance",
                "severity": "medium",
                "component": component_name,
                "issue": "High response time",
                "suggestion": "Consider optimizing database queries or adding caching",
                "current_value": health.response_time_ms,
                "target_value": 1000
            })
        
        if health.error_rate > 0.05:
            suggestions.append({
                "type": "reliability",
                "severity": "high",
                "component": component_name,
                "issue": "High error rate",
                "suggestion": "Review error logs and implement additional error handling",
                "current_value": health.error_rate,
                "target_value": 0.01
            })
        
        return suggestions
    
    async def _analyze_alert_patterns(self, active_alerts: Dict[str, Alert]) -> List[Dict[str, Any]]:
        """Analyze alert patterns for insights"""
        
        suggestions = []
        
        # Group alerts by component
        alerts_by_component = defaultdict(list)
        for alert in active_alerts.values():
            alerts_by_component[alert.component].append(alert)
        
        # Look for components with multiple alerts
        for component, alerts in alerts_by_component.items():
            if len(alerts) > 2:
                suggestions.append({
                    "type": "reliability",
                    "severity": "high",
                    "component": component,
                    "issue": f"Multiple active alerts ({len(alerts)})",
                    "suggestion": "Component may need immediate attention or scaling",
                    "alert_count": len(alerts)
                })
        
        return suggestions

class TrendAnalyzer:
    """Analyzes performance trends"""
    pass

class AnomalyDetector:
    """Detects anomalies in system behavior"""
    pass

class OptimizationEngine:
    """Provides optimization recommendations"""
    pass

class MetricsStorage:
    """Handles metrics storage and retrieval"""
    
    async def initialize(self):
        """Initialize metrics storage"""
        pass
    
    async def store_metric(self, metric: MetricValue):
        """Store a metric"""
        pass
    
    async def store_alert(self, alert: Alert):
        """Store an alert"""
        pass
    
    async def get_recent_metrics(self, minutes: int) -> List[MetricValue]:
        """Get recent metrics"""
        return []
    
    async def cleanup_old_metrics(self, cutoff_time: datetime):
        """Cleanup old metrics"""
        pass

class DashboardDataCollector:
    """Collects and caches data for performance dashboard"""
    
    def __init__(self):
        self.cached_data = {}
        self.last_update = None
    
    async def collect_dashboard_data(self, component_monitors: Dict[str, ComponentMonitor]) -> Dict[str, Any]:
        """Collect comprehensive dashboard data"""
        
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "system_overview": await self._get_system_overview(),
            "component_metrics": {},
            "performance_trends": await self._get_performance_trends(),
            "alert_summary": await self._get_alert_summary()
        }
        
        # Collect component-specific data
        for name, monitor in component_monitors.items():
            dashboard_data["component_metrics"][name] = {
                "health": asdict(await monitor.get_health_status()),
                "recent_metrics": monitor.aggregated_metrics,
                "operation_stats": await self._get_operation_stats(monitor)
            }
        
        return dashboard_data
    
    async def update_cached_data(self, component_monitors: Dict[str, ComponentMonitor]):
        """Update cached dashboard data"""
        self.cached_data = await self.collect_dashboard_data(component_monitors)
        self.last_update = datetime.now()
    
    async def _get_system_overview(self) -> Dict[str, Any]:
        """Get system overview metrics"""
        
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        return {
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "uptime_hours": time.time() / 3600,  # Simplified uptime
            "active_connections": 0  # Would be retrieved from actual source
        }
    
    async def _get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trend data"""
        return {
            "response_time_trend": "stable",
            "error_rate_trend": "decreasing",
            "throughput_trend": "increasing"
        }
    
    async def _get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary data"""
        return {
            "total_alerts": 0,
            "critical_alerts": 0,
            "resolved_today": 0,
            "avg_resolution_time_minutes": 0
        }
    
    async def _get_operation_stats(self, monitor: ComponentMonitor) -> Dict[str, Any]:
        """Get operation statistics for a component"""
        
        total_operations = sum(monitor.success_counts.values()) + sum(monitor.error_counts.values())
        total_errors = sum(monitor.error_counts.values())
        
        return {
            "total_operations": total_operations,
            "error_rate": total_errors / max(total_operations, 1),
            "avg_response_time": statistics.mean([
                statistics.mean(timings) for timings in monitor.operation_timings.values()
                if timings
            ]) if monitor.operation_timings else 0
        }

# Create global enhanced monitoring instance
enhanced_system_monitor = EnhancedSystemMonitor()

# Convenience