
"""
Monitoring module with lazy loading to prevent circular imports.
This module provides access to monitoring components with graceful fallbacks.
"""

import logging

# Module-level variables for lazy loading
_system_monitor = None

def get_system_monitor():
    """Get system monitor with lazy loading and fallback."""
    global _system_monitor
    if _system_monitor is None:
        try:
            from .system_monitor import system_monitor
            _system_monitor = system_monitor
        except ImportError as e:
            logging.warning(f"System monitor not available: {e}")
            _system_monitor = _create_dummy_monitor()
    return _system_monitor

def _create_dummy_monitor():
    """Create a dummy monitor for graceful degradation."""
    class DummyMonitor:
        async def track_operation(self, name, func, *args, **kwargs):
            """Execute function without monitoring."""
            if hasattr(func, '__call__'):
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            return None
        
        async def get_performance_dashboard(self):
            return {"status": "monitoring_disabled", "message": "System monitor not available"}
        
        def register_component(self, name):
            return self
        
        async def record_metric(self, name, value, **kwargs):
            pass
    
    return DummyMonitor()

# Import asyncio for dummy monitor
import asyncio

# Export the getter functions  
__all__ = [
    'get_system_monitor'
]