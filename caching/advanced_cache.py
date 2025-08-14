import asyncio
import json
import hashlib
import time
import statistics
from typing import Any, Optional, Dict, List, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import pickle
import gzip
import logging

# Advanced multi-layer caching system with intelligence and optimization

class CacheLayer(Enum):
    """Cache layers in order of speed"""
    MEMORY = "memory"        # In-process cache (fastest, ~1ms)
    REDIS = "redis"          # Distributed cache (fast, ~5ms)
    DATABASE = "database"    # Persistent cache (reliable, ~50ms)
    DISK = "disk"           # File-based cache (backup, ~100ms)

class CacheStrategy(Enum):
    """Cache population strategies"""
    LAZY = "lazy"               # Populate on miss
    EAGER = "eager"             # Populate proactively
    SMART = "smart"             # AI-driven population
    SCHEDULED = "scheduled"     # Time-based population

class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"                # Least Recently Used
    LFU = "lfu"                # Least Frequently Used
    TTL = "ttl"                # Time To Live
    SMART = "smart"            # AI-driven eviction

@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int]
    size_bytes: int
    cache_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if not self.ttl_seconds:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds"""
        return (datetime.now() - self.created_at).total_seconds()
    
    def update_access(self):
        """Update access statistics"""
        self.last_accessed = datetime.now()
        self.access_count += 1

@dataclass
class CacheStats:
    """Comprehensive cache statistics"""
    layer_name: str
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    error_count: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0
    avg_access_time_ms: float = 0
    memory_usage_mb: float = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate percentage"""
        total = self.hit_count + self.miss_count
        return (self.hit_count / total * 100) if total > 0 else 0
    
    @property
    def efficiency_score(self) -> float:
        """Calculate cache efficiency score"""
        if self.total_entries == 0:
            return 0
        
        # Weighted score based on hit rate and performance
        hit_weight = self.hit_rate / 100  # 0-1
        perf_weight = max(0, 1 - (self.avg_access_time_ms / 1000))  # 0-1
        size_efficiency = min(1, self.total_entries / max(self.total_size_bytes / 1024, 1))
        
        return (hit_weight * 0.5 + perf_weight * 0.3 + size_efficiency * 0.2) * 100

@dataclass
class CacheConfiguration:
    """Cache layer configuration"""
    max_size_mb: int = 256
    max_entries: int = 10000
    default_ttl_seconds: int = 3600
    eviction_policy: EvictionPolicy = EvictionPolicy.SMART
    compression_enabled: bool = True
    encryption_enabled: bool = False
    replication_enabled: bool = False

class CacheIntelligence:
    """AI-powered cache optimization and management"""
    
    def __init__(self):
        self.access_patterns = defaultdict(list)
        self.performance_history = deque(maxlen=1000)
        self.optimization_suggestions = []
        
    async def analyze_access_patterns(self, entries: Dict[str, CacheEntry]) -> Dict[str, Any]:
        """Analyze cache access patterns for optimization"""
        
        patterns = {
            "hot_keys": [],
            "cold_keys": [],
            "temporal_patterns": {},
            "size_distribution": {},
            "ttl_optimization": {}
        }
        
        if not entries:
            return patterns
        
        # Identify hot and cold keys
        sorted_entries = sorted(entries.values(), key=lambda x: x.access_count, reverse=True)
        total_entries = len(sorted_entries)
        
        # Top 20% are hot keys
        hot_threshold = max(1, total_entries // 5)
        patterns["hot_keys"] = [entry.key for entry in sorted_entries[:hot_threshold]]
        
        # Bottom 20% are cold keys
        cold_threshold = max(1, total_entries // 5)
        patterns["cold_keys"] = [entry.key for entry in sorted_entries[-cold_threshold:]]
        
        # Analyze temporal patterns
        patterns["temporal_patterns"] = await self._analyze_temporal_patterns(entries)
        
        # Size distribution analysis
        patterns["size_distribution"] = await self._analyze_size_distribution(entries)
        
        # TTL optimization suggestions
        patterns["ttl_optimization"] = await self._analyze_ttl_optimization(entries)
        
        return patterns
    
    async def predict_cache_needs(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Predict future cache needs based on historical data"""
        
        predictions = {
            "expected_load": 1.0,
            "hot_keys_prediction": [],
            "capacity_recommendation": {},
            "warming_suggestions": []
        }
        
        if len(historical_data) < 10:
            return predictions
        
        # Analyze load trends
        recent_loads = [data.get("request_count", 0) for data in historical_data[-24:]]
        if recent_loads:
            avg_load = statistics.mean(recent_loads)
            trend_slope = self._calculate_trend_slope(recent_loads)
            predictions["expected_load"] = max(0.1, avg_load * (1 + trend_slope))
        
        # Predict hot keys based on frequency patterns
        key_frequencies = defaultdict(int)
        for data in historical_data[-48:]:  # Last 48 hours
            for key in data.get("accessed_keys", []):
                key_frequencies[key] += 1
        
        # Get top predicted hot keys
        sorted_keys = sorted(key_frequencies.items(), key=lambda x: x[1], reverse=True)
        predictions["hot_keys_prediction"] = [key for key, freq in sorted_keys[:20]]
        
        return predictions
    
    async def optimize_ttl_strategy(self, entries: Dict[str, CacheEntry]) -> Dict[str, int]:
        """Optimize TTL values based on usage patterns"""
        
        ttl_recommendations = {}
        
        for key, entry in entries.items():
            # Calculate optimal TTL based on access patterns
            access_frequency = entry.access_count / max(entry.age_seconds / 3600, 1)  # Per hour
            
            if access_frequency > 10:  # Very frequent access
                optimal_ttl = 7200  # 2 hours
            elif access_frequency > 5:  # Frequent access
                optimal_ttl = 3600  # 1 hour
            elif access_frequency > 1:  # Moderate access
                optimal_ttl = 1800  # 30 minutes
            else:  # Infrequent access
                optimal_ttl = 600   # 10 minutes
            
            # Adjust based on data freshness requirements
            if "real_time" in entry.metadata.get("tags", []):
                optimal_ttl = min(optimal_ttl, 300)  # Max 5 minutes for real-time data
            elif "static" in entry.metadata.get("tags", []):
                optimal_ttl = max(optimal_ttl, 3600)  # Min 1 hour for static data
            
            ttl_recommendations[key] = optimal_ttl
        
        return ttl_recommendations
    
    async def detect_cache_pollution(self, entries: Dict[str, CacheEntry]) -> List[str]:
        """Detect cache entries that are polluting effectiveness"""
        
        pollution_candidates = []
        
        for key, entry in entries.items():
            # Large entries with low access frequency
            if entry.size_bytes > 1024 * 1024 and entry.access_count < 2:  # >1MB, <2 accesses
                pollution_candidates.append(key)
            
            # Very old entries with no recent access
            if entry.age_seconds > 3600 and entry.access_count == 1:  # >1 hour, only 1 access
                pollution_candidates.append(key)
            
            # Entries that consistently miss their TTL
            if entry.is_expired and entry.access_count > 0:
                pollution_candidates.append(key)
        
        return pollution_candidates
    
    async def _analyze_temporal_patterns(self, entries: Dict[str, CacheEntry]) -> Dict[str, Any]:
        """Analyze temporal access patterns"""
        
        patterns = {
            "peak_hours": [],
            "seasonal_patterns": {},
            "access_intervals": {}
        }
        
        # Group accesses by hour
        hourly_access = defaultdict(int)
        for entry in entries.values():
            hour = entry.last_accessed.hour
            hourly_access[hour] += entry.access_count
        
        # Find peak hours (top 25%)
        if hourly_access:
            sorted_hours = sorted(hourly_access.items(), key=lambda x: x[1], reverse=True)
            peak_count = max(1, len(sorted_hours) // 4)
            patterns["peak_hours"] = [hour for hour, count in sorted_hours[:peak_count]]
        
        return patterns
    
    async def _analyze_size_distribution(self, entries: Dict[str, CacheEntry]) -> Dict[str, Any]:
        """Analyze cache entry size distribution"""
        
        sizes = [entry.size_bytes for entry in entries.values()]
        if not sizes:
            return {}
        
        return {
            "mean_size": statistics.mean(sizes),
            "median_size": statistics.median(sizes),
            "max_size": max(sizes),
            "total_size": sum(sizes),
            "size_percentiles": {
                "p90": sorted(sizes)[int(len(sizes) * 0.9)] if sizes else 0,
                "p95": sorted(sizes)[int(len(sizes) * 0.95)] if sizes else 0,
                "p99": sorted(sizes)[int(len(sizes) * 0.99)] if sizes else 0
            }
        }
    
    async def _analyze_ttl_optimization(self, entries: Dict[str, CacheEntry]) -> Dict[str, Any]:
        """Analyze TTL effectiveness and suggest optimizations"""
        
        expired_entries = [e for e in entries.values() if e.is_expired]
        unexpired_entries = [e for e in entries.values() if not e.is_expired]
        
        analysis = {
            "expired_count": len(expired_entries),
            "unexpired_count": len(unexpired_entries),
            "avg_ttl_usage": 0,
            "suggested_adjustments": {}
        }
        
        if unexpired_entries:
            # Calculate average TTL usage percentage
            ttl_usage = []
            for entry in unexpired_entries:
                if entry.ttl_seconds:
                    usage_pct = entry.age_seconds / entry.ttl_seconds
                    ttl_usage.append(usage_pct)
            
            if ttl_usage:
                analysis["avg_ttl_usage"] = statistics.mean(ttl_usage)
        
        return analysis
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope using simple linear regression"""
        
        if len(values) < 2:
            return 0
        
        n = len(values)
        x_values = list(range(n))
        
        # Calculate slope using least squares
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0

class CacheLayer_Base:
    """Base class for cache layer implementations"""
    
    def __init__(self, name: str, config: CacheConfiguration):
        self.name = name
        self.config = config
        self.stats = CacheStats(layer_name=name)
        self.entries = {}
        self.intelligence = CacheIntelligence()
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        raise NotImplementedError
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        raise NotImplementedError
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        raise NotImplementedError
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        raise NotImplementedError
    
    async def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self.stats
    
    def get_all_entries(self) -> Dict[str, CacheEntry]:
        """Get all cache entries for analysis (added method)"""
        return self.entries.copy()

class MemoryCache(CacheLayer_Base):
    """High-performance in-memory cache with intelligent eviction"""
    
    def __init__(self, config: CacheConfiguration):
        super().__init__("memory", config)
        self.entries = {}
        self.access_order = deque()  # For LRU tracking
        self.size_bytes = 0
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        start_time = time.time()
        
        try:
            if key not in self.entries:
                self.stats.miss_count += 1
                return None
            
            entry = self.entries[key]
            
            # Check expiration
            if entry.is_expired:
                await self.delete(key)
                self.stats.miss_count += 1
                return None
            
            # Update access statistics
            entry.update_access()
            self._update_access_order(key)
            
            self.stats.hit_count += 1
            return entry.value
            
        except Exception as e:
            self.stats.error_count += 1
            logging.error(f"Memory cache get error for key {key}: {e}")
            return None
        finally:
            access_time = (time.time() - start_time) * 1000
            self._update_avg_access_time(access_time)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache with intelligent eviction"""
        
        try:
            # Calculate entry size
            serialized_value = pickle.dumps(value)
            if self.config.compression_enabled:
                serialized_value = gzip.compress(serialized_value)
            
            entry_size = len(serialized_value)
            
            # Check if we need to evict entries
            await self._ensure_capacity(entry_size)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl_seconds=ttl or self.config.default_ttl_seconds,
                size_bytes=entry_size,
                cache_type="memory"
            )
            
            # Remove old entry if exists
            if key in self.entries:
                old_entry = self.entries[key]
                self.size_bytes -= old_entry.size_bytes
            
            # Add new entry
            self.entries[key] = entry
            self.size_bytes += entry_size
            self._update_access_order(key)
            
            self.stats.total_entries = len(self.entries)
            self.stats.total_size_bytes = self.size_bytes
            
            return True
            
        except Exception as e:
            self.stats.error_count += 1
            logging.error(f"Memory cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete entry from memory cache"""
        
        try:
            if key in self.entries:
                entry = self.entries[key]
                self.size_bytes -= entry.size_bytes
                del self.entries[key]
                
                # Remove from access order
                try:
                    self.access_order.remove(key)
                except ValueError:
                    pass  # Key not in access order
                
                self.stats.total_entries = len(self.entries)
                self.stats.total_size_bytes = self.size_bytes
                
                return True
            return False
            
        except Exception as e:
            self.stats.error_count += 1
            logging.error(f"Memory cache delete error for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all entries from memory cache"""
        
        try:
            self.entries.clear()
            self.access_order.clear()
            self.size_bytes = 0
            
            self.stats.total_entries = 0
            self.stats.total_size_bytes = 0
            
            return True
            
        except Exception as e:
            self.stats.error_count += 1
            logging.error(f"Memory cache clear error: {e}")
            return False
    
    async def _ensure_capacity(self, new_entry_size: int):
        """Ensure cache has capacity for new entry"""
        
        max_size_bytes = self.config.max_size_mb * 1024 * 1024
        
        # Check size limit
        while (self.size_bytes + new_entry_size > max_size_bytes or 
               len(self.entries) >= self.config.max_entries):
            
            if not self.entries:
                break
            
            # Apply eviction policy
            if self.config.eviction_policy == EvictionPolicy.LRU:
                await self._evict_lru()
            elif self.config.eviction_policy == EvictionPolicy.LFU:
                await self._evict_lfu()
            elif self.config.eviction_policy == EvictionPolicy.TTL:
                await self._evict_expired()
            else:  # SMART eviction
                await self._evict_smart()
    
    async def _evict_lru(self):
        """Evict least recently used entry"""
        
        if self.access_order:
            lru_key = self.access_order[0]
            await self.delete(lru_key)
            self.stats.eviction_count += 1
    
    async def _evict_lfu(self):
        """Evict least frequently used entry"""
        
        if self.entries:
            lfu_key = min(self.entries.keys(), key=lambda k: self.entries[k].access_count)
            await self.delete(lfu_key)
            self.stats.eviction_count += 1
    
    async def _evict_expired(self):
        """Evict expired entries first"""
        
        expired_keys = [k for k, v in self.entries.items() if v.is_expired]
        
        if expired_keys:
            for key in expired_keys[:5]:  # Evict up to 5 expired entries
                await self.delete(key)
                self.stats.eviction_count += 1
        else:
            # Fallback to LRU if no expired entries
            await self._evict_lru()
    
    async def _evict_smart(self):
        """AI-powered smart eviction"""
        
        # First try expired entries
        expired_keys = [k for k, v in self.entries.items() if v.is_expired]
        if expired_keys:
            await self.delete(expired_keys[0])
            self.stats.eviction_count += 1
            return
        
        # Score entries for eviction (lower score = better eviction candidate)
        scored_entries = []
        for key, entry in self.entries.items():
            # Scoring factors
            recency_score = (datetime.now() - entry.last_accessed).total_seconds() / 3600  # Hours
            frequency_score = 1 / max(entry.access_count, 1)
            size_penalty = entry.size_bytes / (1024 * 1024)  # MB
            
            # Combined score (higher = worse candidate)
            score = recency_score + frequency_score + size_penalty
            scored_entries.append((score, key))
        
        if scored_entries:
            # Evict entry with highest score (worst candidate)
            _, worst_key = max(scored_entries)
            await self.delete(worst_key)
            self.stats.eviction_count += 1
    
    def _update_access_order(self, key: str):
        """Update LRU access order"""
        
        # Remove from current position
        try:
            self.access_order.remove(key)
        except ValueError:
            pass  # Key not in order yet
        
        # Add to end (most recent)
        self.access_order.append(key)
    
    def _update_avg_access_time(self, access_time_ms: float):
        """Update average access time"""
        
        # Simple moving average
        if self.stats.avg_access_time_ms == 0:
            self.stats.avg_access_time_ms = access_time_ms
        else:
            self.stats.avg_access_time_ms = (self.stats.avg_access_time_ms * 0.9 + access_time_ms * 0.1)

class RedisCache(CacheLayer_Base):
    """Redis-based distributed cache with clustering support"""
    
    def __init__(self, config: CacheConfiguration, redis_url: str = "redis://localhost:6379"):
        super().__init__("redis", config)
        self.redis_url = redis_url
        self.redis_client = None
        self.key_prefix = "aiwardrobe:"
        self._connection_attempted = False
    
    async def initialize(self):
        """Initialize Redis connection"""
        
        if self._connection_attempted:
            return self.redis_client is not None
        
        self._connection_attempted = True
        
        try:
            # Import the async client from the new redis library
            from redis import asyncio as aioredis
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            logging.info("Redis cache initialized successfully")
            return True
        except ImportError:
            logging.warning("redis-py (with async support) not available - Redis cache disabled")
            return False
        except Exception as e:
            logging.error(f"Redis cache initialization failed: {e}")
            self.redis_client = None
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        
        if not await self.initialize():
            self.stats.error_count += 1
            return None
        
        start_time = time.time()
        
        try:
            redis_key = f"{self.key_prefix}{key}"
            data = await self.redis_client.get(redis_key)
            
            if data is None:
                self.stats.miss_count += 1
                return None
            
            # Deserialize data
            if self.config.compression_enabled:
                data = gzip.decompress(data)
            
            value = pickle.loads(data)
            self.stats.hit_count += 1
            
            return value
            
        except Exception as e:
            self.stats.error_count += 1
            logging.error(f"Redis cache get error for key {key}: {e}")
            return None
        finally:
            access_time = (time.time() - start_time) * 1000
            self._update_avg_access_time(access_time)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        
        if not await self.initialize():
            self.stats.error_count += 1
            return False
        
        try:
            # Serialize data
            data = pickle.dumps(value)
            if self.config.compression_enabled:
                data = gzip.compress(data)
            
            redis_key = f"{self.key_prefix}{key}"
            ttl_seconds = ttl or self.config.default_ttl_seconds
            
            await self.redis_client.setex(redis_key, ttl_seconds, data)
            return True
            
        except Exception as e:
            self.stats.error_count += 1
            logging.error(f"Redis cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        
        if not await self.initialize():
            self.stats.error_count += 1
            return False
        
        try:
            redis_key = f"{self.key_prefix}{key}"
            result = await self.redis_client.delete(redis_key)
            return result > 0
            
        except Exception as e:
            self.stats.error_count += 1
            logging.error(f"Redis cache delete error for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        
        if not await self.initialize():
            self.stats.error_count += 1
            return False
        
        try:
            # Delete all keys with our prefix
            pattern = f"{self.key_prefix}*"
            async for key in self.redis_client.scan_iter(match=pattern):
                await self.redis_client.delete(key)
            
            return True
            
        except Exception as e:
            self.stats.error_count += 1
            logging.error(f"Redis cache clear error: {e}")
            return False
    
    def _update_avg_access_time(self, access_time_ms: float):
        """Update average access time"""
        
        if self.stats.avg_access_time_ms == 0:
            self.stats.avg_access_time_ms = access_time_ms
        else:
            self.stats.avg_access_time_ms = (self.stats.avg_access_time_ms * 0.9 + access_time_ms * 0.1)

class CacheWarmer:
    """Proactive cache warming and maintenance"""
    
    def __init__(self):
        self.warming_tasks = {}
        self.warm_up_interval = 300  # 5 minutes default
        self.last_warm_up = defaultdict(float)
    
    async def warm_up(self, cache_key: str, warm_up_func: Callable, ttl: int = 300):
        """Register a warm-up task for a cache key"""
        self.warming_tasks[cache_key] = (warm_up_func, ttl)
    
    async def run_warm_up(self):
        """Execute warm-up tasks for expired or soon-to-expire entries"""
        current_time = time.time()
        
        for key, (warm_up_func, ttl) in self.warming_tasks.items():
            if current_time - self.last_warm_up[key] > ttl * 0.8:  # Warm up at 80% of TTL
                try:
                    await warm_up_func()
                    self.last_warm_up[key] = current_time
                except Exception as e:
                    logging.error(f"Cache warm-up failed for {key}: {e}")
    
    async def warm_cache_type(self, cache_type: str, keys: Optional[List[str]], cache_manager):
        """Warm cache for specific cache type (fixed method)"""
        try:
            logging.info(f"Warming cache for type: {cache_type}")
            if keys:
                for key in keys:
                    # This would typically fetch data and cache it
                    # Implementation depends on the specific cache type
                    pass
        except Exception as e:
            logging.error(f"Cache warming failed for type {cache_type}: {e}")
    
    async def warm_related_keys(self, keys: List[str], cache_manager):
        """Warm related keys (fixed method)"""
        try:
            for key in keys:
                # Implementation for warming related keys
                pass
        except Exception as e:
            logging.error(f"Related key warming failed: {e}")

class SmartCacheManager:
    """Intelligent multi-layer cache manager with automatic optimization"""
    
    def __init__(self):
        # Cache layers (in order of preference)
        self.layers = {}
        
        # Intelligence components
        self.intelligence = CacheIntelligence()
        self.warmer = CacheWarmer()
        
        # Performance tracking
        self.access_patterns = defaultdict(list)
        self.performance_metrics = deque(maxlen=1000)
        
        # Configuration
        self.default_ttl_map = {
            'weather': 3600,           # 1 hour
            'wardrobe_items': 21600,   # 6 hours
            'ai_responses': 7200,      # 2 hours
            'outfit_generation': 1800, # 30 minutes
            'travel_packing': 3600,    # 1 hour
            'user_preferences': 86400, # 24 hours
            'static_content': 604800   # 1 week
        }
        
        # Monitoring
        self.logger = logging.getLogger(__name__)
        
        # Background task management
        self._background_tasks = []
        self._initialized = False
    
    async def initialize(self):
        """Initialize the cache manager and all layers"""
        
        if self._initialized:
            return
        
        try:
            # Initialize memory cache
            memory_config = CacheConfiguration(
                max_size_mb=256,
                max_entries=5000,
                default_ttl_seconds=3600,
                eviction_policy=EvictionPolicy.SMART,
                compression_enabled=True
            )
            self.layers[CacheLayer.MEMORY] = MemoryCache(memory_config)
            
            # Initialize Redis cache (with graceful fallback)
            redis_config = CacheConfiguration(
                max_size_mb=1024,
                max_entries=50000,
                default_ttl_seconds=7200,
                eviction_policy=EvictionPolicy.LRU,
                compression_enabled=True
            )
            redis_cache = RedisCache(redis_config)
            redis_initialized = await redis_cache.initialize()
            
            if redis_initialized:
                self.layers[CacheLayer.REDIS] = redis_cache
                self.logger.info("Redis cache layer initialized successfully")
            else:
                self.logger.warning("Redis cache layer failed to initialize - using memory-only mode")
            
            # Start background optimization tasks
            self._start_background_tasks()
            
            self._initialized = True
            self.logger.info(f"Smart cache manager initialized with {len(self.layers)} layers")
            
        except Exception as e:
            self.logger.error(f"Cache manager initialization failed: {e}")
            # Ensure at least memory cache is available
            if CacheLayer.MEMORY not in self.layers:
                memory_config = CacheConfiguration()
                self.layers[CacheLayer.MEMORY] = MemoryCache(memory_config)
            self._initialized = True
    
    def _start_background_tasks(self):
        """Start background optimization tasks with error handling"""
        try:
            # Only start tasks if not already running
            if not self._background_tasks:
                optimization_task = asyncio.create_task(self._optimization_loop())
                warming_task = asyncio.create_task(self._warming_loop())
                cleanup_task = asyncio.create_task(self._cleanup_loop())
                
                self._background_tasks = [optimization_task, warming_task, cleanup_task]
                self.logger.info("Background cache optimization tasks started")
        except Exception as e:
            self.logger.error(f"Failed to start background tasks: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown cache manager"""
        try:
            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to finish
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Close Redis connections
            for layer in self.layers.values():
                if hasattr(layer, 'redis_client') and layer.redis_client:
                    await layer.redis_client.close()
            
            self.logger.info("Cache manager shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during cache manager shutdown: {e}")
    
    async def get(self, key: str, cache_type: str = 'default') -> Optional[Any]:
        """Intelligent cache retrieval with automatic promotion"""
        
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Track access pattern
            self._record_access_pattern(key, cache_type)
            
            # Try each layer in order
            for layer_type in [CacheLayer.MEMORY, CacheLayer.REDIS]:
                if layer_type not in self.layers:
                    continue
                
                layer = self.layers[layer_type]
                value = await layer.get(key)
                
                if value is not None:
                    # Record hit and promote to faster layers if needed
                    await self._handle_cache_hit(key, value, layer_type, cache_type)
                    return value
            
            # Cache miss across all layers
            await self._handle_cache_miss(key, cache_type)
            return None
            
        except Exception as e:
            self.logger.error(f"Cache get error for key {key}: {e}")
            return None
        finally:
            access_time = (time.time() - start_time) * 1000
            self._record_performance_metric('get', access_time, key)
    
    async def set(self, key: str, value: Any, cache_type: str = 'default', 
                 ttl: Optional[int] = None) -> bool:
        """Intelligent cache storage with automatic layer optimization"""
        
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        success = False
        
        try:
            # Determine optimal TTL
            optimal_ttl = ttl or self._get_optimal_ttl(cache_type, key)
            
            # Determine which layers to store in based on intelligence
            target_layers = await self._determine_storage_layers(key, value, cache_type)
            
            # Store in target layers
            for layer_type in target_layers:
                if layer_type in self.layers:
                    layer = self.layers[layer_type]
                    layer_success = await layer.set(key, value, optimal_ttl)
                    if layer_success:
                        success = True
            
            return success
            
        except Exception as e:
            self.logger.error(f"Cache set error for key {key}: {e}")
            return False
        finally:
            access_time = (time.time() - start_time) * 1000
            self._record_performance_metric('set', access_time, key)
    
    async def delete(self, key: str) -> bool:
        """Delete from all cache layers"""
        
        success = False
        
        try:
            # Delete from all layers
            for layer in self.layers.values():
                layer_success = await layer.delete(key)
                if layer_success:
                    success = True
            
            return success
            
        except Exception as e:
            self.logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern"""
        
        invalidated_count = 0
        
        try:
            # For each layer that supports pattern matching
            for layer in self.layers.values():
                if hasattr(layer, 'invalidate_pattern'):
                    count = await layer.invalidate_pattern(pattern)
                    invalidated_count += count
            
            self.logger.info(f"Invalidated {invalidated_count} keys matching pattern: {pattern}")
            return invalidated_count
            
        except Exception as e:
            self.logger.error(f"Pattern invalidation error for pattern {pattern}: {e}")
            return 0
    
    async def warm_cache(self, cache_type: str, keys: List[str] = None):
        """Proactively warm cache with predicted data"""
        
        await self.warmer.warm_cache_type(cache_type, keys, self)
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics across all layers"""
        
        stats = {
            "timestamp": datetime.now().isoformat(),
            "layers": {},
            "overall_performance": {
                "total_layers": len(self.layers),
                "avg_access_time_ms": 0,
                "cache_efficiency_score": 0
            },
            "intelligence_insights": await self._get_intelligence_insights()
        }
        
        # Collect layer stats
        total_hit_rate = 0
        total_efficiency = 0
        
        for layer_type, layer in self.layers.items():
            layer_stats = await layer.get_stats()
            stats["layers"][layer_type.value] = {
                "hit_rate": layer_stats.hit_rate,
                "efficiency_score": layer_stats.efficiency_score,
                "total_entries": layer_stats.total_entries,
                "total_size_mb": layer_stats.total_size_bytes / (1024 * 1024),
                "avg_access_time_ms": layer_stats.avg_access_time_ms,
                "hit_count": layer_stats.hit_count,
                "miss_count": layer_stats.miss_count,
                "error_count": layer_stats.error_count
            }
            
            total_hit_rate += layer_stats.hit_rate
            total_efficiency += layer_stats.efficiency_score
        
        # Calculate overall performance
        if self.layers:
            stats["overall_performance"]["avg_hit_rate"] = total_hit_rate / len(self.layers)
            stats["overall_performance"]["avg_efficiency_score"] = total_efficiency / len(self.layers)
        
        return stats
    
    async def _handle_cache_hit(self, key: str, value: Any, hit_layer: CacheLayer, cache_type: str):
        """Handle cache hit with intelligent promotion"""
        
        # If hit in Redis but not Memory, consider promoting to memory
        if hit_layer == CacheLayer.REDIS and CacheLayer.MEMORY in self.layers:
            should_promote = await self._should_promote_to_memory(key, cache_type)
            if should_promote:
                memory_cache = self.layers[CacheLayer.MEMORY]
                await memory_cache.set(key, value, self._get_optimal_ttl(cache_type, key))
    
    async def _handle_cache_miss(self, key: str, cache_type: str):
        """Handle cache miss with intelligent warming"""
        
        # Record miss for analysis
        self.access_patterns[key].append({
            'timestamp': datetime.now(),
            'type': 'miss',
            'cache_type': cache_type
        })
        
        # Consider warming related keys
        related_keys = await self._get_related_keys(key, cache_type)
        if related_keys:
            asyncio.create_task(self.warmer.warm_related_keys(related_keys, self))
    
    async def _should_promote_to_memory(self, key: str, cache_type: str) -> bool:
        """Determine if a key should be promoted to memory cache"""
        
        # Check access frequency
        recent_accesses = [
            access for access in self.access_patterns[key]
            if datetime.now() - access['timestamp'] < timedelta(hours=1)
        ]
        
        # Promote if accessed more than 3 times in the last hour
        if len(recent_accesses) > 3:
            return True
        
        # Promote hot cache types
        hot_cache_types = ['weather', 'user_preferences', 'wardrobe_items']
        if cache_type in hot_cache_types:
            return True
        
        return False
    
    async def _determine_storage_layers(self, key: str, value: Any, cache_type: str) -> List[CacheLayer]:
        """Determine optimal storage layers for a key-value pair"""
        
        layers = []
        
        # Always try to store in memory for hot data
        if await self._is_hot_data(key, cache_type):
            layers.append(CacheLayer.MEMORY)
        
        # Always store in Redis for persistence
        if CacheLayer.REDIS in self.layers:
            layers.append(CacheLayer.REDIS)
        
        # Default to memory if no other decision
        if not layers and CacheLayer.MEMORY in self.layers:
            layers.append(CacheLayer.MEMORY)
        
        return layers
    
    async def _is_hot_data(self, key: str, cache_type: str) -> bool:
        """Determine if data should be considered 'hot' and cached aggressively"""
        
        # Hot cache types
        hot_cache_types = ['weather', 'user_preferences', 'wardrobe_items']
        if cache_type in hot_cache_types:
            return True
        
        # Check access frequency
        recent_accesses = len([
            access for access in self.access_patterns[key]
            if datetime.now() - access['timestamp'] < timedelta(minutes=30)
        ])
        
        return recent_accesses > 2
    
    def _get_optimal_ttl(self, cache_type: str, key: str) -> int:
        """Get optimal TTL for a cache type and key"""
        
        base_ttl = self.default_ttl_map.get(cache_type, 3600)
        
        # Adjust based on access patterns
        if key in self.access_patterns:
            access_frequency = len([
                access for access in self.access_patterns[key]
                if datetime.now() - access['timestamp'] < timedelta(hours=24)
            ])
            
            # More frequent access = longer TTL
            if access_frequency > 10:
                base_ttl *= 2
            elif access_frequency < 2:
                base_ttl //= 2
        
        return max(300, min(base_ttl, 86400))  # Between 5 minutes and 24 hours
    
    async def _get_related_keys(self, key: str, cache_type: str) -> List[str]:
        """Get keys related to the given key for warming"""
        
        related_keys = []
        
        # Simple pattern-based relationships
        if cache_type == 'wardrobe_items':
            # Related wardrobe items might share category or aesthetic
            related_keys.extend([
                f"wardrobe_category_{key.split('_')[-1]}",
                f"wardrobe_aesthetic_{key.split('_')[-1]}"
            ])
        elif cache_type == 'weather':
            # Related weather for nearby times
            related_keys.extend([
                f"weather_forecast_{key.split('_')[-1]}",
                f"weather_extended_{key.split('_')[-1]}"
            ])
        
        return related_keys
    
    def _record_access_pattern(self, key: str, cache_type: str):
        """Record access pattern for analysis"""
        
        self.access_patterns[key].append({
            'timestamp': datetime.now(),
            'type': 'access',
            'cache_type': cache_type
        })
        
        # Keep only recent access patterns
        cutoff = datetime.now() - timedelta(hours=24)
        self.access_patterns[key] = [
            access for access in self.access_patterns[key]
            if access['timestamp'] > cutoff
        ]
    
    def _record_performance_metric(self, operation: str, duration_ms: float, key: str):
        """Record performance metric"""
        
        self.performance_metrics.append({
            'operation': operation,
            'duration_ms': duration_ms,
            'key': key,
            'timestamp': datetime.now()
        })
    
    async def _get_intelligence_insights(self) -> Dict[str, Any]:
        """Get insights from cache intelligence analysis"""
        
        insights = {
            "hot_keys": [],
            "optimization_suggestions": [],
            "predicted_misses": [],
            "efficiency_recommendations": []
        }
        
        try:
            # Analyze access patterns across all layers
            all_entries = {}
            
            # Collect entries from all layers
            for layer in self.layers.values():
                if hasattr(layer, 'get_all_entries'):
                    layer_entries = layer.get_all_entries()
                    all_entries.update(layer_entries)
            
            if all_entries:
                # Analyze patterns
                patterns = await self.intelligence.analyze_access_patterns(all_entries)
                insights.update(patterns)
            
        except Exception as e:
            self.logger.error(f"Error getting intelligence insights: {e}")
        
        return insights
    
    async def _optimization_loop(self):
        """Background optimization loop"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Collect all entries for analysis
                all_entries = {}
                for layer in self.layers.values():
                    if hasattr(layer, 'get_all_entries'):
                        layer_entries = layer.get_all_entries()
                        all_entries.update(layer_entries)
                
                if all_entries:
                    # Analyze and optimize
                    patterns = await self.intelligence.analyze_access_patterns(all_entries)
                    pollution = await self.intelligence.detect_cache_pollution(all_entries)
                    
                    # Clean up polluted entries
                    for key in pollution[:10]:  # Limit cleanup to prevent performance impact
                        await self.delete(key)
                    
                    if pollution:
                        self.logger.info(f"Cleaned up {min(len(pollution), 10)} polluted cache entries")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _warming_loop(self):
        """Background cache warming loop"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self.warmer.run_warm_up()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in warming loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Cleanup old access patterns
                cutoff = datetime.now() - timedelta(hours=24)
                cleaned_keys = []
                
                for key, patterns in list(self.access_patterns.items()):
                    filtered_patterns = [p for p in patterns if p['timestamp'] > cutoff]
                    if filtered_patterns:
                        self.access_patterns[key] = filtered_patterns
                    else:
                        cleaned_keys.append(key)
                
                # Remove empty pattern entries
                for key in cleaned_keys:
                    del self.access_patterns[key]
                
                if cleaned_keys:
                    self.logger.debug(f"Cleaned up {len(cleaned_keys)} old access pattern entries")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)  # Wait before retrying


# Create global advanced cache instance
advanced_cache = SmartCacheManager()

# Convenience functions for easy integration
async def get_cached(key: str, cache_type: str = 'default') -> Optional[Any]:
    """Convenience function to get from cache"""
    return await advanced_cache.get(key, cache_type)

async def set_cached(key: str, value: Any, cache_type: str = 'default', ttl: Optional[int] = None) -> bool:
    """Convenience function to set in cache"""
    return await advanced_cache.set(key, value, cache_type, ttl)

async def delete_cached(key: str) -> bool:
    """Convenience function to delete from cache"""
    return await advanced_cache.delete(key)

async def get_cache_stats() -> Dict[str, Any]:
    """Convenience function to get cache statistics"""
    return await advanced_cache.get_comprehensive_stats()

async def initialize_cache():
    """Initialize the advanced cache system"""
    await advanced_cache.initialize()

async def shutdown_cache():
    """Shutdown the advanced cache system"""
    await advanced_cache.shutdown()

# Export key classes and functions
__all__ = [
    'CacheLayer',
    'CacheStrategy', 
    'EvictionPolicy',
    'CacheEntry',
    'CacheStats',
    'CacheConfiguration',
    'CacheIntelligence',
    'MemoryCache',
    'RedisCache',
    'CacheWarmer',
    'SmartCacheManager',
    'advanced_cache',
    'get_cached',
    'set_cached',
    'delete_cached',
    'get_cache_stats',
    'initialize_cache',
    'shutdown_cache'
]