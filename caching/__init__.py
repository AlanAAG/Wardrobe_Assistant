"""
Caching module with lazy loading to prevent circular imports.
This module provides access to caching components with proper error handling.
"""

import logging

# Module-level variables for lazy loading
_advanced_cache = None

def get_advanced_cache():
    """Get advanced cache with lazy loading and comprehensive error handling."""
    global _advanced_cache
    if _advanced_cache is None:
        try:
            # Try to import the advanced cache
            from .advanced_cache import advanced_cache
            _advanced_cache = advanced_cache
            logging.info("✅ Advanced cache loaded successfully")
        except ImportError as e:
            logging.warning(f"Advanced cache not available (ImportError): {e}")
            _advanced_cache = _create_simple_cache()
        except SyntaxError as e:
            logging.error(f"Syntax error in advanced_cache.py: {e}")
            _advanced_cache = _create_simple_cache()
        except Exception as e:
            logging.error(f"Unexpected error loading advanced cache: {e}")
            _advanced_cache = _create_simple_cache()
    return _advanced_cache

def _create_simple_cache():
    """Create a simple in-memory cache as fallback."""
    logging.info("Creating simple fallback cache")
    
    class SimpleCache:
        def __init__(self):
            self._cache = {}
            self._initialized = True
        
        async def get(self, key):
            return self._cache.get(key)
        
        async def set(self, key, value, ttl=None):
            self._cache[key] = value
            return True
        
        async def delete(self, key):
            return self._cache.pop(key, None) is not None
        
        async def clear(self):
            self._cache.clear()
            return True
        
        async def get_stats(self):
            return {
                "type": "simple_fallback",
                "keys": len(self._cache),
                "status": "healthy",
                "cache_type": "memory_only"
            }
        
        def is_healthy(self):
            return hasattr(self, '_initialized') and self._initialized
    
    return SimpleCache()

# Export the getter functions
__all__ = [
    'get_advanced_cache'
]