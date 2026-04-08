import os
import logging
from redis import asyncio as aioredis

class WebhookCache:
    """
    A Redis-backed cache to deduplicate recent webhook events.
    This helps prevent processing the same event multiple times in quick succession.
    """
    def __init__(self, ttl_seconds=60):
        self._ttl = ttl_seconds
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        # decode_responses=True ensures we get strings back, not bytes
        self.redis = aioredis.from_url(redis_url, decode_responses=True)
        logging.info(f"WebhookCache initialized with Redis TTL: {self._ttl} seconds. URL: {redis_url}")

    async def is_recently_processed(self, page_id: str) -> bool:
        """
        Checks if a page_id has been processed within the TTL period.
        """
        try:
            cache_key = f"webhook_cache:{page_id}"
            exists = await self.redis.exists(cache_key)
            if exists:
                logging.info(f"Page {page_id} was recently processed. Ignoring.")
                return True
            return False
        except Exception as e:
            logging.error(f"Redis cache check failed: {e}")
            # If Redis fails, we default to False to process the webhook anyway
            return False

    async def add(self, page_id: str):
        """
        Adds a page_id to the cache with SETEX.
        """
        try:
            cache_key = f"webhook_cache:{page_id}"
            await self.redis.setex(cache_key, self._ttl, "1")
            logging.info(f"Page {page_id} added to Redis webhook cache.")
        except Exception as e:
            logging.error(f"Redis cache add failed: {e}")

# Global instance of the webhook cache
webhook_cache = WebhookCache()
