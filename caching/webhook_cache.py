import time
import logging
from collections import OrderedDict

class WebhookCache:
    """
    A simple in-memory cache to deduplicate recent webhook events.
    This helps prevent processing the same event multiple times in quick succession.
    """
    def __init__(self, ttl_seconds=60):
        """
        Initializes the cache.

        Args:
            ttl_seconds (int): Time-to-live for cache entries in seconds.
                               An entry is considered stale after this duration.
        """
        self._cache = OrderedDict()
        self._ttl = ttl_seconds
        logging.info(f"WebhookCache initialized with TTL: {self._ttl} seconds.")

    def is_recently_processed(self, page_id: str) -> bool:
        """
        Checks if a page_id has been processed within the TTL period.

        This method is thread-safe for checking and updating the cache.

        Args:
            page_id (str): The Notion page ID to check.

        Returns:
            bool: True if the page_id is in the cache and not stale, False otherwise.
        """
        if page_id not in self._cache:
            return False

        # Check if the cached entry is stale
        if time.time() - self._cache[page_id] > self._ttl:
            # Entry is stale, remove it
            del self._cache[page_id]
            return False

        # Entry is recent
        logging.info(f"Page {page_id} was recently processed. Ignoring.")
        return True

    def add(self, page_id: str):
        """
        Adds a page_id to the cache with the current timestamp.

        This method is thread-safe for adding entries.

        Args:
            page_id (str): The Notion page ID to add.
        """
        self._cache[page_id] = time.time()
        logging.info(f"Page {page_id} added to webhook cache.")
        self._cleanup()

    def _cleanup(self):
        """
        Removes stale entries from the cache to prevent it from growing indefinitely.
        """
        stale_keys = []
        # Find all stale keys
        for page_id, timestamp in self._cache.items():
            if time.time() - timestamp > self._ttl:
                stale_keys.append(page_id)

        # Remove stale keys
        for key in stale_keys:
            if key in self._cache:
                del self._cache[key]
                logging.debug(f"Removed stale page_id {key} from cache.")

# Global instance of the webhook cache
webhook_cache = WebhookCache()
