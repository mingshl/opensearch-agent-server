"""LRU Cache Implementation for Agent Management.

Provides thread-safe LRU cache for caching Strands agents per thread ID.
This prevents unbounded memory growth in long-running servers by automatically
evicting least recently used agents when the cache reaches its maximum size.

**Key Components:**
- `LRUAgentCache` - Thread-safe LRU cache for agent instances

**Features:**
- Thread-safe operations using asyncio.Lock
- Automatic LRU eviction when cache is full
- Configurable cache size
- Efficient O(1) operations for get/put

**Usage Example:**
```python
from server.agent_cache import LRUAgentCache
from strands import Agent as StrandsAgentCore

cache = LRUAgentCache(max_size=100)

# Cache an agent
await cache.put("thread-123", agent_instance)

# Retrieve cached agent
agent = await cache.get("thread-123")

# Remove agent
await cache.remove("thread-123")
```
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict

from strands import Agent as StrandsAgentCore

__all__ = ["LRUAgentCache"]


class LRUAgentCache:
    """LRU cache for thread agents with size limit.

    Implements a least-recently-used cache that automatically evicts
    the least recently used entries when the cache reaches its maximum size.
    This prevents unbounded memory growth in long-running servers.

    Thread-safe: All cache operations are protected by an internal lock,
    making the cache safe for concurrent access from multiple coroutines.

    Attributes:
        max_size: Maximum number of agents to cache (default: 100)
        cache: OrderedDict maintaining insertion/access order
        _lock: Internal asyncio.Lock for thread-safe operations

    """

    def __init__(self, max_size: int = 100) -> None:
        """Initialize LRU cache.

        Args:
            max_size: Maximum number of agents to cache. Defaults to 100.
                     Set to 0 or negative to disable caching.

        """
        self.max_size = max_size if max_size > 0 else 0
        # OrderedDict maintains insertion order - useful for LRU eviction
        # The type hint `OrderedDict[str, StrandsAgentCore]` means:
        # - Keys are strings (thread IDs)
        # - Values are StrandsAgentCore instances
        self.cache: OrderedDict[str, StrandsAgentCore] = OrderedDict()
        # asyncio.Lock provides thread-safe access in async code
        # Use `async with self._lock:` to acquire/release the lock
        self._lock = asyncio.Lock()  # Internal lock for thread-safe operations

    async def get(self, key: str) -> StrandsAgentCore | None:
        """Get agent from cache, moving it to end (most recently used).

        Thread-safe: Protected by internal lock.

        Args:
            key: Thread ID

        Returns:
            Cached agent if found, None otherwise

        """
        # `async with` is Python's async context manager syntax
        # It ensures the lock is acquired before entering the block
        # and released when exiting (even if an exception occurs)
        async with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                # OrderedDict.move_to_end() moves the key-value pair to the end
                # This makes it the "most recently used" item
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    async def put(self, key: str, value: StrandsAgentCore) -> str | None:
        """Add agent to cache, evicting LRU entry if cache is full.

        Thread-safe: Protected by internal lock.

        Args:
            key: Thread ID
            value: Agent instance to cache

        Returns:
            Evicted thread_id if an entry was evicted, None otherwise

        """
        async with self._lock:
            if self.max_size == 0:
                # Caching disabled
                return None

            evicted_key = None

            if key in self.cache:
                # Update existing entry
                self.cache.move_to_end(key)
                self.cache[key] = value
            else:
                # Add new entry
                self.cache[key] = value

                # Evict LRU entry if cache is full
                if len(self.cache) > self.max_size:
                    evicted_key = next(iter(self.cache))
                    del self.cache[evicted_key]

            return evicted_key

    async def remove(self, key: str) -> bool:
        """Remove agent from cache.

        Thread-safe: Protected by internal lock.

        Args:
            key: Thread ID

        Returns:
            True if entry was removed, False if not found

        """
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    async def clear(self) -> int:
        """Clear all entries from cache.

        Thread-safe: Protected by internal lock.

        Returns:
            Number of entries cleared

        """
        async with self._lock:
            count = len(self.cache)
            self.cache.clear()
            return count

    async def size(self) -> int:
        """Get the number of cached entries.

        Thread-safe: Protected by internal lock.

        Returns:
            Number of entries currently in the cache

        """
        async with self._lock:
            return len(self.cache)

    async def contains(self, key: str) -> bool:
        """Check if key exists in cache.

        Thread-safe: Protected by internal lock.

        Args:
            key: Thread ID to check

        Returns:
            True if key exists in cache, False otherwise

        """
        async with self._lock:
            return key in self.cache
