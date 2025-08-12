import structlog
import time
import asyncio
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from enum import Enum
import aiofiles

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PerformanceMetric:
    operation: str
    duration_ms: float
    success: bool
    ai_provider: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = None
    user_id: Optional[str] = None
    additional_data: Optional[Dict] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class SystemAlert:
    severity: AlertSeverity
    message: str
    operation: str
    error_count: int
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class SystemMonitor:
    """
    Comprehensive system monitoring with structured logging, metrics collection,
    and intelligent alerting for the AI Wardrobe System.
    """
    
    def __init__(self, metrics_file: str = "system_metrics.json"):
        # Initialize structured logging
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger()
        self.metrics: List[PerformanceMetric] = []
        self.alerts: List[SystemAlert] = []
        self.metrics_file = metrics_file
        
        # Alert thresholds
        self.alert_thresholds = {
            "failure_rate_critical": 0.5,      # 50% failure rate
            "failure_rate_high": 0.3,          # 30% failure rate
            "response_time_critical": 10000,   # 10 seconds
            "response_time_high": 5000,        # 5 seconds
            "consecutive_failures": 3           # 3 failures in a row
        }
        
        # Operation criticality levels
        self.critical_operations = {
            'gemini_api_call', 'groq_api_call', 'supabase_query', 
            'notion_webhook', 'weather_api', 'outfit_generation_full',
            'travel_packing_full'
        }
        
        # Recent failures tracking
        self.recent_failures = {}
        
    async def track_operation(self, operation: str, func, *args, **kwargs):
        """
        Track any system operation with comprehensive metrics and error handling.
        
        Args:
            operation: Name of the operation being tracked
            func: Function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            Result of the function execution
            
        Raises:
            Original exception with enhanced logging
        """
        start_time = time.time()
        metric = None
        
        try:
            # Extract user context if available
            user_id = kwargs.get('user_id') or (args[0].get('user_id') if args and isinstance(args[0], dict) else None)
            
            # Log operation start
            self.logger.info(
                f"🚀 Starting operation: {operation}",
                operation=operation,
                user_id=user_id,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys())
            )
            
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Calculate duration
            duration = (time.time() - start_time) * 1000
            
            # Create success metric
            metric = PerformanceMetric(
                operation=operation,
                duration_ms=duration,
                success=True,
                user_id=user_id,
                additional_data={"result_type": type(result).__name__}
            )
            
            # Log success
            self.logger.info(
                f"✅ Operation successful: {operation}",
                operation=operation,
                duration_ms=duration,
                user_id=user_id,
                success=True
            )
            
            # Clear recent failures for this operation
            if operation in self.recent_failures:
                del self.recent_failures[operation]
            
            return result
            
        except Exception as e:
            # Calculate duration
            duration = (time.time() - start_time) * 1000
            
            # Create failure metric
            metric = PerformanceMetric(
                operation=operation,
                duration_ms=duration,
                success=False,
                error_type=type(e).__name__,
                error_message=str(e),
                user_id=user_id,
                additional_data={"args_count": len(args), "kwargs_keys": list(kwargs.keys())}
            )
            
            # Log error with full context
            self.logger.error(
                f"❌ Operation failed: {operation}",
                operation=operation,
                duration_ms=duration,
                error_type=type(e).__name__,
                error_message=str(e),
                user_id=user_id,
                success=False,
                exc_info=True
            )
            
            # Track consecutive failures
            if operation not in self.recent_failures:
                self.recent_failures[operation] = []
            self.recent_failures[operation].append(datetime.now())
            
            # Keep only recent failures (last hour)
            cutoff = datetime.now() - timedelta(hours=1)
            self.recent_failures[operation] = [
                failure_time for failure_time in self.recent_failures[operation]
                if failure_time > cutoff
            ]
            
            # Check if we need to trigger alerts
            await self._check_and_send_alerts(operation, e)
            
            # Re-raise the original exception
            raise e
            
        finally:
            # Always record the metric
            if metric:
                await self._record_metric(metric)
    
    async def _record_metric(self, metric: PerformanceMetric):
        """Record metric and maintain rolling window for memory management"""
        self.metrics.append(metric)
        
        # Maintain rolling window of last 1000 metrics
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]
        
        # Persist metrics to file for analysis
        await self._persist_metrics()
    
    async def _check_and_send_alerts(self, operation: str, error: Exception):
        """Intelligent alerting based on error patterns and operation criticality"""
        
        # Check consecutive failures
        consecutive_failures = len(self.recent_failures.get(operation, []))
        
        # Determine alert severity
        severity = AlertSeverity.LOW
        
        if operation in self.critical_operations:
            if consecutive_failures >= self.alert_thresholds["consecutive_failures"]:
                severity = AlertSeverity.CRITICAL
            elif consecutive_failures >= 2:
                severity = AlertSeverity.HIGH
            else:
                severity = AlertSeverity.MEDIUM
        else:
            if consecutive_failures >= 5:
                severity = AlertSeverity.HIGH
            elif consecutive_failures >= 3:
                severity = AlertSeverity.MEDIUM
        
        # Create and send alert
        alert = SystemAlert(
            severity=severity,
            message=f"Operation {operation} failed: {str(error)}",
            operation=operation,
            error_count=consecutive_failures
        )
        
        await self._send_alert(alert)
    
    async def _send_alert(self, alert: SystemAlert):
        """Send alerts via multiple channels based on severity"""
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        # Log the alert
        self.logger.critical(
            f"🚨 SYSTEM ALERT: {alert.severity.value.upper()}",
            severity=alert.severity.value,
            message=alert.message,
            operation=alert.operation,
            error_count=alert.error_count
        )
        
        # TODO: Add external alerting (Slack, Discord, email, etc.)
        # For now, we'll implement a simple webhook approach
        await self._send_webhook_alert(alert)
    
    async def _send_webhook_alert(self, alert: SystemAlert):
        """Send alert to external webhook if configured"""
        webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        
        if not webhook_url:
            return
        
        try:
            import aiohttp
            
            payload = {
                "text": f"🚨 AI Wardrobe System Alert",
                "attachments": [
                    {
                        "color": self._get_alert_color(alert.severity),
                        "fields": [
                            {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                            {"title": "Operation", "value": alert.operation, "short": True},
                            {"title": "Error Count", "value": str(alert.error_count), "short": True},
                            {"title": "Message", "value": alert.message, "short": False},
                            {"title": "Timestamp", "value": alert.timestamp.isoformat(), "short": True}
                        ]
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, timeout=5) as response:
                    if response.status == 200:
                        self.logger.info("Alert sent successfully to webhook")
                    else:
                        self.logger.warning(f"Failed to send alert to webhook: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Error sending webhook alert: {e}")
    
    def _get_alert_color(self, severity: AlertSeverity) -> str:
        """Get color code for alert based on severity"""
        colors = {
            AlertSeverity.LOW: "#36a64f",      # Green
            AlertSeverity.MEDIUM: "#ff9500",   # Orange
            AlertSeverity.HIGH: "#ff0000",     # Red
            AlertSeverity.CRITICAL: "#8B0000"  # Dark Red
        }
        return colors.get(severity, "#808080")
    
    async def _persist_metrics(self):
        """Persist metrics to file for analysis"""
        try:
            # Convert metrics to serializable format
            metrics_data = [
                {
                    **asdict(metric),
                    'timestamp': metric.timestamp.isoformat()
                }
                for metric in self.metrics[-100:]  # Keep last 100 for file
            ]
            
            async with aiofiles.open(self.metrics_file, 'w') as f:
                await f.write(json.dumps(metrics_data, indent=2))
                
        except Exception as e:
            self.logger.error(f"Failed to persist metrics: {e}")
    
    def get_performance_dashboard(self) -> Dict:
        """Generate real-time performance dashboard data"""
        if not self.metrics:
            return {"status": "no_data", "timestamp": datetime.now().isoformat()}
        
        # Get recent metrics (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        recent_metrics = [m for m in self.metrics if m.timestamp > cutoff]
        
        if not recent_metrics:
            return {"status": "no_recent_data", "timestamp": datetime.now().isoformat()}
        
        return {
            "system_health": self._calculate_health_score(recent_metrics),
            "performance_summary": self._get_performance_summary(recent_metrics),
            "operations_breakdown": self._group_metrics_by_operation(recent_metrics),
            "ai_providers_status": self._analyze_ai_performance(recent_metrics),
            "recent_alerts": [asdict(alert) for alert in self.alerts[-10:]],
            "error_patterns": self._analyze_error_patterns(recent_metrics),
            "timestamp": datetime.now().isoformat(),
            "metrics_count": len(recent_metrics)
        }
    
    def _calculate_health_score(self, metrics: List[PerformanceMetric]) -> Dict:
        """Calculate overall system health score (0-100)"""
        if not metrics:
            return {"score": 100.0, "status": "unknown"}
        
        success_rate = len([m for m in metrics if m.success]) / len(metrics)
        avg_duration = sum(m.duration_ms for m in metrics) / len(metrics)
        
        # Health score calculation
        success_component = success_rate * 70  # 70% weight on success rate
        performance_component = max(0, 100 - (avg_duration / 100)) * 30  # 30% weight on performance
        
        health_score = success_component + performance_component
        health_score = max(0, min(100, health_score))  # Clamp to 0-100
        
        # Determine status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 80:
            status = "good"
        elif health_score >= 60:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "score": round(health_score, 1),
            "status": status,
            "success_rate": round(success_rate * 100, 1),
            "avg_response_time_ms": round(avg_duration, 1)
        }
    
    def _get_performance_summary(self, metrics: List[PerformanceMetric]) -> Dict:
        """Get performance summary statistics"""
        if not metrics:
            return {}
        
        durations = [m.duration_ms for m in metrics]
        successes = [m for m in metrics if m.success]
        failures = [m for m in metrics if not m.success]
        
        return {
            "total_operations": len(metrics),
            "successful_operations": len(successes),
            "failed_operations": len(failures),
            "success_rate_percent": round(len(successes) / len(metrics) * 100, 1),
            "avg_response_time_ms": round(sum(durations) / len(durations), 1),
            "min_response_time_ms": min(durations),
            "max_response_time_ms": max(durations),
            "p95_response_time_ms": round(sorted(durations)[int(len(durations) * 0.95)], 1)
        }
    
    def _group_metrics_by_operation(self, metrics: List[PerformanceMetric]) -> Dict:
        """Group metrics by operation type"""
        operations = {}
        
        for metric in metrics:
            if metric.operation not in operations:
                operations[metric.operation] = {
                    "total": 0,
                    "successes": 0,
                    "failures": 0,
                    "durations": []
                }
            
            op_stats = operations[metric.operation]
            op_stats["total"] += 1
            op_stats["durations"].append(metric.duration_ms)
            
            if metric.success:
                op_stats["successes"] += 1
            else:
                op_stats["failures"] += 1
        
        # Calculate summary stats for each operation
        for op_name, stats in operations.items():
            durations = stats["durations"]
            stats["success_rate"] = round(stats["successes"] / stats["total"] * 100, 1)
            stats["avg_duration_ms"] = round(sum(durations) / len(durations), 1)
            del stats["durations"]  # Remove raw durations from output
        
        return operations
    
    def _analyze_ai_performance(self, metrics: List[PerformanceMetric]) -> Dict:
        """Analyze AI provider performance"""
        ai_metrics = [m for m in metrics if m.ai_provider]
        
        if not ai_metrics:
            return {"status": "no_ai_metrics"}
        
        providers = {}
        for metric in ai_metrics:
            provider = metric.ai_provider
            if provider not in providers:
                providers[provider] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "durations": []
                }
            
            provider_stats = providers[provider]
            provider_stats["total_calls"] += 1
            provider_stats["durations"].append(metric.duration_ms)
            
            if metric.success:
                provider_stats["successful_calls"] += 1
            else:
                provider_stats["failed_calls"] += 1
        
        # Calculate summary for each provider
        for provider, stats in providers.items():
            durations = stats["durations"]
            stats["success_rate"] = round(stats["successful_calls"] / stats["total_calls"] * 100, 1)
            stats["avg_response_time_ms"] = round(sum(durations) / len(durations), 1)
            del stats["durations"]  # Remove raw durations
        
        return providers
    
    def _analyze_error_patterns(self, metrics: List[PerformanceMetric]) -> Dict:
        """Analyze error patterns for insights"""
        failed_metrics = [m for m in metrics if not m.success]
        
        if not failed_metrics:
            return {"status": "no_errors", "total_errors": 0}
        
        error_types = {}
        error_operations = {}
        
        for metric in failed_metrics:
            # Count by error type
            error_type = metric.error_type or "unknown"
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Count by operation
            operation = metric.operation
            error_operations[operation] = error_operations.get(operation, 0) + 1
        
        return {
            "total_errors": len(failed_metrics),
            "error_types": dict(sorted(error_types.items(), key=lambda x: x[1], reverse=True)),
            "error_operations": dict(sorted(error_operations.items(), key=lambda x: x[1], reverse=True)),
            "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        }

# Global system monitor instance
system_monitor = SystemMonitor()