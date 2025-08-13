"""
Monitoring module with lazy loading to prevent circular imports.
This module provides access to monitoring components with graceful fallbacks.
"""

import logging
import asyncio

# Module-level variables for lazy loading
_system_monitor = None

def get_system_monitor():
    """Get system monitor with lazy loading and comprehensive fallback."""
    global _system_monitor
    if _system_monitor is None:
        try:
            from .system_monitor import system_monitor
            _system_monitor = system_monitor
            logging.info("✅ System monitor loaded successfully")
        except ImportError as e:
            logging.warning(f"System monitor not available (ImportError): {e}")
            _system_monitor = _create_dummy_monitor()
        except Exception as e:
            logging.error(f"Error loading system monitor: {e}")
            _system_monitor = _create_dummy_monitor()
    return _system_monitor

def _create_dummy_monitor():
    """Create a dummy monitor for graceful degradation."""
    logging.info("Creating dummy monitor for graceful degradation")
    
    class DummyMonitor:
        def __init__(self):
            self._initialized = True
        
        async def track_operation(self, name, func, *args, **kwargs):
            """Execute function without monitoring."""
            try:
                if hasattr(func, '__call__'):
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                return None
            except Exception as e:
                logging.error(f"Error in tracked operation {name}: {e}")
                raise
        
        async def get_performance_dashboard(self):
            return {
                "status": "monitoring_disabled", 
                "message": "System monitor not available",
                "timestamp": "N/A",
                "components": {},
                "performance": {
                    "note": "Monitoring disabled - install required dependencies"
                }
            }
        
        def register_component(self, name):
            """Register a component (no-op in dummy)."""
            return self
        
        async def record_metric(self, name, value, **kwargs):
            """Record a metric (no-op in dummy)."""
            pass
        
        def is_healthy(self):
            return True
    
    return DummyMonitor()

# Export the getter functions  
__all__ = [
    'get_system_monitor'
]