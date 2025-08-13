import json
import hashlib
from typing import Any, Optional, Dict, List
from datetime import timedelta
import asyncio
import aioredis

# [START monitoring_import]
from monitoring.system_monitor import system_monitor
# [END monitoring_import]

class AdvancedCache:
    def __init__(self):
        self.redis = None
        self.local_fallback = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0
        }
        
        self.ttl_config = {
            'weather': timedelta(hours=1),
            'wardrobe_items': timedelta(hours=6), 
            'outfit_generation': timedelta(minutes=30),
            'travel_packing': timedelta(hours=2),
            'user_preferences': timedelta(days=1),
            'ai_responses': timedelta(hours=12)
        }
    
    async def initialize(self):
        """Initialize Redis connection with fallback"""
        try:
            self.redis = await aioredis.from_url("redis://localhost:6379")
            await self.redis.ping()
            system_monitor.logger.info("Redis cache initialized successfully")
        except Exception as e:
            system_monitor.logger.warning(f"Redis initialization failed, using local fallback: {e}")
            self.redis = None
    
    async def get(self, key: str, cache_type: str = 'default') -> Optional[Any]:
        """Get cached data with intelligent fallback"""
        cache_key = f"{cache_type}:{key}"
        
        try:
            if self.redis:
                data = await self.redis.get(cache_key)
                if data:
                    self.cache_stats['hits'] += 1
                    return json.loads(data)
            
            if cache_key in self.local_fallback:
                entry = self.local_fallback[cache_key]
                if entry['expires'] > asyncio.get_event_loop().time():
                    self.cache_stats['hits'] += 1
                    return entry['data']
                else:
                    del self.local_fallback[cache_key]
            
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            self.cache_stats['errors'] += 1
            system_monitor.logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, data: Any, cache_type: str = 'default'):
        """Set cached data with TTL"""
        cache_key = f"{cache_type}:{key}"
        ttl = self.ttl_config.get(cache_type, timedelta(hours=1))
        
        try:
            serialized = json.dumps(data, default=str)
            
            if self.redis:
                await self.redis.setex(cache_key, int(ttl.total_seconds()), serialized)
            
            self.local_fallback[cache_key] = {
                'data': data,
                'expires': asyncio.get_event_loop().time() + ttl.total_seconds()
            }
            
        except Exception as e:
            system_monitor.logger.error(f"Cache set error: {e}")
    
    def create_cache_key(self, *args, **kwargs) -> str:
        """Create consistent cache keys from parameters"""
        key_data = f"{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_stats(self) -> Dict:
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hit_rate_percent': round(hit_rate, 2),
            'total_requests': total_requests,
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'errors': self.cache_stats['errors'],
            'redis_connected': self.redis is not None
        }

advanced_cache = AdvancedCache()