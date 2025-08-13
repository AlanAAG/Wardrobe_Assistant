
"""
Caching module with lazy loading to prevent circular imports.
This module provides access to caching components with proper error handling.
"""

import logging

# Module-level variables for lazy loading
_advanced_cache = None

def get_advanced_cache():
    """Get advanced cache with lazy loading."""
    global _advanced_cache
    if _advanced_cache is None:
        try:
            from .advanced_cache import advanced_cache
            _advanced_cache = advanced_cache
        except ImportError as e:
            logging.warning(f"Advanced cache not available: {e}")
            # Create a simple fallback cache
            _advanced_cache = _create_simple_cache()
    return _advanced_cache

def _create_simple_cache():
    """Create a simple in-memory cache as fallback."""
    class SimpleCache:
        def __init__(self):
            self._cache = {}
        
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
            return {"keys": len(self._cache), "type": "simple_fallback"}
    
    return SimpleCache()

# Export the getter functions
__all__ = [
    'get_advanced_cache'
]