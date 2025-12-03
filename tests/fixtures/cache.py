"""
Test Data Caching System
Intelligent caching for test data to improve test execution performance
Memory-based and disk-based caching with TTL and size limits
"""

import json
import pickle
import time
import threading
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict, Set, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
import weakref

from .base import AbstractDataCache, DataScope

class MemoryCache(AbstractDataCache):
    """Memory-based cache with TTL and LRU eviction"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data with TTL check"""
        with self._lock:
            if key not in self._cache:
                return None
            
            value, expires_at = self._cache[key]
            
            # Check if expired
            if expires_at < time.time():
                del self._cache[key]
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cached data with TTL"""
        try:
            with self._lock:
                ttl = ttl or self.default_ttl
                expires_at = time.time() + ttl
                
                self._cache[key] = (value, expires_at)
                self._cache.move_to_end(key)
                
                # Evict if over size limit
                while len(self._cache) > self.max_size:
                    self._cache.popitem(last=False)
                
                return True
        except Exception:
            return False
    
    def delete(self, key: str) -> bool:
        """Delete cached data"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> bool:
        """Clear all cached data"""
        try:
            with self._lock:
                self._cache.clear()
            return True
        except Exception:
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        return self.get(key) is not None
    
    def cleanup_expired(self) -> int:
        """Remove expired entries, return count removed"""
        count = 0
        current_time = time.time()
        
        with self._lock:
            expired_keys = []
            for key, (_, expires_at) in self._cache.items():
                if expires_at < current_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
                count += 1
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            current_time = time.time()
            expired_count = sum(
                1 for _, expires_at in self._cache.values() 
                if expires_at < current_time
            )
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'expired_count': expired_count,
                'hit_ratio': getattr(self, '_hit_ratio', 0.0)
            }

class DiskCache(AbstractDataCache):
    """Disk-based cache with file storage"""
    
    def __init__(self, cache_dir: Path, max_size_mb: int = 100, default_ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self._lock = threading.RLock()
        self._index_file = self.cache_dir / ".cache_index.json"
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create index
        self._index = self._load_index()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data from disk"""
        with self._lock:
            key_hash = self._hash_key(key)
            
            if key_hash not in self._index:
                return None
            
            entry = self._index[key_hash]
            
            # Check if expired
            if entry['expires_at'] < time.time():
                self._remove_entry(key_hash)
                return None
            
            # Load data from file
            cache_file = self.cache_dir / f"{key_hash}.cache"
            if not cache_file.exists():
                self._remove_entry(key_hash)
                return None
            
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Update access time
                entry['accessed_at'] = time.time()
                self._save_index()
                
                return data
            except Exception:
                self._remove_entry(key_hash)
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cached data to disk"""
        try:
            with self._lock:
                key_hash = self._hash_key(key)
                ttl = ttl or self.default_ttl
                
                # Serialize data
                cache_file = self.cache_dir / f"{key_hash}.cache"
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f)
                
                # Update index
                file_size = cache_file.stat().st_size
                current_time = time.time()
                
                self._index[key_hash] = {
                    'key': key,
                    'size': file_size,
                    'created_at': current_time,
                    'accessed_at': current_time,
                    'expires_at': current_time + ttl
                }
                
                self._save_index()
                self._cleanup_if_needed()
                
                return True
        except Exception:
            return False
    
    def delete(self, key: str) -> bool:
        """Delete cached data from disk"""
        with self._lock:
            key_hash = self._hash_key(key)
            return self._remove_entry(key_hash)
    
    def clear(self) -> bool:
        """Clear all cached data"""
        try:
            with self._lock:
                # Remove all cache files
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
                
                # Clear index
                self._index.clear()
                self._save_index()
                
                return True
        except Exception:
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        return self.get(key) is not None
    
    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        count = 0
        current_time = time.time()
        
        with self._lock:
            expired_keys = []
            for key_hash, entry in self._index.items():
                if entry['expires_at'] < current_time:
                    expired_keys.append(key_hash)
            
            for key_hash in expired_keys:
                if self._remove_entry(key_hash):
                    count += 1
            
            self._save_index()
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_size = sum(entry['size'] for entry in self._index.values())
            current_time = time.time()
            expired_count = sum(
                1 for entry in self._index.values() 
                if entry['expires_at'] < current_time
            )
            
            return {
                'entries': len(self._index),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'expired_count': expired_count
            }
    
    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key"""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def _load_index(self) -> Dict[str, Any]:
        """Load cache index from disk"""
        try:
            if self._index_file.exists():
                with open(self._index_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def _save_index(self) -> bool:
        """Save cache index to disk"""
        try:
            with open(self._index_file, 'w') as f:
                json.dump(self._index, f)
            return True
        except Exception:
            return False
    
    def _remove_entry(self, key_hash: str) -> bool:
        """Remove cache entry and file"""
        try:
            if key_hash in self._index:
                cache_file = self.cache_dir / f"{key_hash}.cache"
                if cache_file.exists():
                    cache_file.unlink()
                del self._index[key_hash]
                return True
        except Exception:
            pass
        return False
    
    def _cleanup_if_needed(self) -> None:
        """Cleanup cache if over size limit"""
        total_size = sum(entry['size'] for entry in self._index.values())
        
        if total_size <= self.max_size_bytes:
            return
        
        # Sort by access time (LRU)
        entries_by_access = sorted(
            self._index.items(),
            key=lambda x: x[1]['accessed_at']
        )
        
        # Remove oldest entries until under limit
        for key_hash, entry in entries_by_access:
            if total_size <= self.max_size_bytes:
                break
            
            if self._remove_entry(key_hash):
                total_size -= entry['size']

class HybridCache(AbstractDataCache):
    """Hybrid cache combining memory and disk storage"""
    
    def __init__(self, cache_dir: Path, memory_max_size: int = 100, 
                 disk_max_size_mb: int = 100, default_ttl: int = 3600):
        self.memory_cache = MemoryCache(memory_max_size, default_ttl)
        self.disk_cache = DiskCache(cache_dir, disk_max_size_mb, default_ttl)
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data, check memory first then disk"""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.set(key, value)
        
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cached data in both memory and disk"""
        with self._lock:
            memory_success = self.memory_cache.set(key, value, ttl)
            disk_success = self.disk_cache.set(key, value, ttl)
            return memory_success or disk_success
    
    def delete(self, key: str) -> bool:
        """Delete cached data from both caches"""
        with self._lock:
            memory_success = self.memory_cache.delete(key)
            disk_success = self.disk_cache.delete(key)
            return memory_success or disk_success
    
    def clear(self) -> bool:
        """Clear all cached data"""
        with self._lock:
            memory_success = self.memory_cache.clear()
            disk_success = self.disk_cache.clear()
            return memory_success and disk_success
    
    def exists(self, key: str) -> bool:
        """Check if key exists in either cache"""
        return self.memory_cache.exists(key) or self.disk_cache.exists(key)
    
    def cleanup_expired(self) -> int:
        """Remove expired entries from both caches"""
        memory_count = self.memory_cache.cleanup_expired()
        disk_count = self.disk_cache.cleanup_expired()
        return memory_count + disk_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from both caches"""
        return {
            'memory': self.memory_cache.get_stats(),
            'disk': self.disk_cache.get_stats()
        }

class ScopedCache:
    """Cache with automatic cleanup based on data scope"""
    
    def __init__(self, cache: AbstractDataCache):
        self.cache = cache
        self._scope_keys: Dict[DataScope, Set[str]] = {
            scope: set() for scope in DataScope
        }
        self._lock = threading.RLock()
    
    def set(self, key: str, value: Any, scope: DataScope, ttl: Optional[int] = None) -> bool:
        """Set cached data with scope tracking"""
        with self._lock:
            success = self.cache.set(key, value, ttl)
            if success:
                self._scope_keys[scope].add(key)
            return success
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data"""
        return self.cache.get(key)
    
    def delete(self, key: str) -> bool:
        """Delete cached data and remove from scope tracking"""
        with self._lock:
            success = self.cache.delete(key)
            if success:
                for scope_set in self._scope_keys.values():
                    scope_set.discard(key)
            return success
    
    def cleanup_scope(self, scope: DataScope) -> int:
        """Cleanup all data for a specific scope"""
        count = 0
        with self._lock:
            keys_to_remove = list(self._scope_keys[scope])
            for key in keys_to_remove:
                if self.cache.delete(key):
                    count += 1
            self._scope_keys[scope].clear()
        return count
    
    def cleanup_temporary(self) -> int:
        """Cleanup all temporary data"""
        return self.cleanup_scope(DataScope.TEMPORARY)
    
    def cleanup_test_function(self) -> int:
        """Cleanup all test function scoped data"""
        return self.cleanup_scope(DataScope.TEST_FUNCTION)
    
    def cleanup_test_module(self) -> int:
        """Cleanup all test module scoped data"""
        return self.cleanup_scope(DataScope.TEST_MODULE)

class TestDataCacheManager:
    """Manager for test data caching with automatic lifecycle management"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create hybrid cache
        self._hybrid_cache = HybridCache(self.cache_dir)
        self._scoped_cache = ScopedCache(self._hybrid_cache)
        
        # Cleanup thread
        self._cleanup_thread = None
        self._cleanup_interval = 300  # 5 minutes
        self._shutdown = threading.Event()
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        # Register cleanup on exit
        weakref.finalize(self, self._cleanup_all)
    
    def get_cache(self) -> ScopedCache:
        """Get the scoped cache instance"""
        return self._scoped_cache
    
    def cache_test_data(self, key: str, value: Any, scope: DataScope, 
                       ttl: Optional[int] = None) -> bool:
        """Cache test data with scope"""
        return self._scoped_cache.set(key, value, scope, ttl)
    
    def get_test_data(self, key: str) -> Optional[Any]:
        """Get cached test data"""
        return self._scoped_cache.get(key)
    
    def cleanup_scope(self, scope: DataScope) -> int:
        """Cleanup data by scope"""
        return self._scoped_cache.cleanup_scope(scope)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self._hybrid_cache.get_stats()
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread"""
        def cleanup_worker():
            while not self._shutdown.wait(self._cleanup_interval):
                try:
                    # Cleanup expired entries
                    self._hybrid_cache.cleanup_expired()
                    
                    # Cleanup temporary data older than 1 hour
                    self._scoped_cache.cleanup_temporary()
                    
                except Exception as e:
                    print(f"Cache cleanup error: {e}")
        
        self._cleanup_thread = threading.Thread(
            target=cleanup_worker, 
            daemon=True, 
            name="TestDataCacheCleanup"
        )
        self._cleanup_thread.start()
    
    def _cleanup_all(self) -> None:
        """Cleanup all cache data"""
        self._shutdown.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
        self._hybrid_cache.clear()

# Global cache manager instance
_cache_manager: Optional[TestDataCacheManager] = None

def get_cache_manager(cache_dir: Optional[Path] = None) -> TestDataCacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        if cache_dir is None:
            cache_dir = Path(__file__).parent / "test_cache"
        _cache_manager = TestDataCacheManager(cache_dir)
    return _cache_manager

def initialize_cache_manager(cache_dir: Path) -> TestDataCacheManager:
    """Initialize global cache manager with specific directory"""
    global _cache_manager
    _cache_manager = TestDataCacheManager(cache_dir)
    return _cache_manager
EOF < /dev/null