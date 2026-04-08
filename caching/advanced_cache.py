import os
import time
import json
import logging
from typing import Any, Optional
from datetime import datetime, timedelta
from redis import asyncio as aioredis

logger = logging.getLogger(__name__)

class MemoryCache:
    """Simple in-memory cache with TTL"""
    def __init__(self, ttl_seconds=60):
        self.entries = {}
        self.ttl_seconds = ttl_seconds

    async def get(self, key: str) -> Optional[Any]:
        if key not in self.entries:
            return None
        
        entry = self.entries[key]
        if datetime.now() > entry['expires_at']:
            await self.delete(key)
            return None
            
        return entry['value']

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        expiration = ttl if ttl is not None else self.ttl_seconds
        self.entries[key] = {
            'value': value,
            'expires_at': datetime.now() + timedelta(seconds=expiration)
        }
        return True

    async def delete(self, key: str) -> bool:
        if key in self.entries:
            del self.entries[key]
            return True
        return False

    async def clear(self) -> bool:
        self.entries.clear()
        return True

class RedisCache:
    """Redis cache with TTL"""
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl_seconds=3600):
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.redis_client = None
        self.key_prefix = "aiwardrobe:"
        
    async def initialize(self):
        if self.redis_client is not None:
            return True
            
        try:
            self.redis_client = aioredis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {e}")
            self.redis_client = None
            return False

    async def get(self, key: str) -> Optional[Any]:
        if not await self.initialize():
            return None
            
        try:
            redis_key = f"{self.key_prefix}{key}"
            data = await self.redis_client.get(redis_key)
            if data is None:
                return None
            return json.loads(data)
        except Exception as e:
            logger.warning(f"Redis get failed for {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if not await self.initialize():
            return False
            
        try:
            redis_key = f"{self.key_prefix}{key}"
            expiration = ttl if ttl is not None else self.ttl_seconds
            await self.redis_client.setex(redis_key, expiration, json.dumps(value))
            return True
        except Exception as e:
            logger.warning(f"Redis set failed for {key}: {e}")
            return False
            
    async def delete(self, key: str) -> bool:
        if not await self.initialize():
            return False
            
        try:
            redis_key = f"{self.key_prefix}{key}"
            await self.redis_client.delete(redis_key)
            return True
        except Exception as e:
            logger.warning(f"Redis delete failed for {key}: {e}")
            return False

class AdvancedCache:
    """Manager for multi-level cache: Memory -> Redis"""
    def __init__(self):
        self.memory_cache = MemoryCache(ttl_seconds=60) # 1 min TTL
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_cache = RedisCache(redis_url=redis_url, ttl_seconds=3600) # 1 hr TTL
        
    async def get(self, key: str) -> Optional[Any]:
        # Try memory cache first
        val = await self.memory_cache.get(key)
        if val is not None:
            return val
            
        # Try Redis cache next
        val = await self.redis_cache.get(key)
        if val is not None:
            # Backfill memory cache
            await self.memory_cache.set(key, val)
            return val
            
        return None

    async def set(self, key: str, value: Any, memory_ttl: Optional[int] = None, redis_ttl: Optional[int] = None) -> bool:
        await self.memory_cache.set(key, value, ttl=memory_ttl)
        await self.redis_cache.set(key, value, ttl=redis_ttl)
        return True

    async def delete(self, key: str) -> bool:
        await self.memory_cache.delete(key)
        await self.redis_cache.delete(key)
        return True

    async def get_comprehensive_stats(self) -> dict:
        return {
            "memory_cache_entries": len(self.memory_cache.entries),
            "redis_connected": self.redis_cache.redis_client is not None
        }

# Global instance
advanced_cache = AdvancedCache()