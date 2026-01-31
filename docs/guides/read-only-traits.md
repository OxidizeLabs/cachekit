# Read-Only Cache Traits

This guide explains how to use read-only traits for safe, side-effect-free cache inspection.

## Overview

CacheKit provides a comprehensive set of read-only traits that allow you to inspect cache state without triggering evictions, updating access patterns, or requiring mutable access. This is useful for:

- **Monitoring & Observability**: Check cache stats without affecting behavior
- **Concurrent Readers**: Multiple threads can inspect cache state with read locks only
- **Testing & Debugging**: Examine cache state without side effects
- **API Design**: Clearly signal when functions don't modify cache state

## Trait Hierarchy

```text
ReadOnlyCache<K, V>              ← Base read-only trait
    ├── CoreCache<K, V>          ← Adds insert/get (extends ReadOnlyCache)
    │   ├── FifoCacheTrait       ← FIFO-specific operations
    │   └── MutableCache         ← Adds remove operation
    │       ├── LruCacheTrait
    │       ├── LfuCacheTrait
    │       └── LrukCacheTrait
    │
    └── Policy-Specific Read-Only Traits
        ├── ReadOnlyFifoCache    ← FIFO inspection
        ├── ReadOnlyLruCache     ← LRU inspection
        ├── ReadOnlyLfuCache     ← LFU inspection
        └── ReadOnlyLruKCache    ← LRU-K inspection
```

## Base Read-Only Trait

The `ReadOnlyCache<K, V>` trait provides fundamental inspection operations:

```rust
use cachekit::traits::ReadOnlyCache;
use cachekit::policy::lru_k::LrukCache;

fn print_cache_stats<C: ReadOnlyCache<u64, String>>(cache: &C) {
    println!("Cache size: {}/{}", cache.len(), cache.capacity());
    println!("Is empty: {}", cache.is_empty());
    println!("Has key 42: {}", cache.contains(&42));
}

let mut cache = LrukCache::new(100);
cache.insert(42, "answer".to_string());

// Works with immutable reference - no side effects!
print_cache_stats(&cache);
```

### Methods

| Method | Description | Side Effects |
|--------|-------------|--------------|
| `contains(&K) -> bool` | Check if key exists | None |
| `len() -> usize` | Get number of entries | None |
| `is_empty() -> bool` | Check if cache is empty | None |
| `capacity() -> usize` | Get maximum capacity | None |

## Policy-Specific Read-Only Traits

Each cache policy has a corresponding read-only trait for inspecting policy-specific metadata.

### FIFO: `ReadOnlyFifoCache`

Inspect insertion order without modifying the queue:

```rust
use cachekit::traits::{ReadOnlyFifoCache, CoreCache};
use cachekit::policy::fifo::FifoCache;

fn analyze_fifo_state<C: ReadOnlyFifoCache<u64, String>>(cache: &C) {
    // Peek at oldest entry without removing it
    if let Some((key, value)) = cache.peek_oldest() {
        println!("Oldest: key={}, value={}", key, value);

        // Get its position in the queue
        if let Some(rank) = cache.age_rank(key) {
            println!("Age rank: {} (0 = oldest)", rank);
        }
    }
}

let mut cache = FifoCache::new(10);
cache.insert(1, "first".to_string());
cache.insert(2, "second".to_string());

analyze_fifo_state(&cache);  // No side effects!
```

**Methods:**
- `peek_oldest() -> Option<(&K, &V)>` - View oldest entry
- `age_rank(&K) -> Option<usize>` - Get position in queue (0 = oldest)

### LRU: `ReadOnlyLruCache`

Inspect recency order without affecting LRU list:

```rust
use std::sync::Arc;
use cachekit::traits::{ReadOnlyLruCache, CoreCache};
use cachekit::policy::lru::LruCore;

fn analyze_lru_state<C: ReadOnlyLruCache<u64, String>>(cache: &C) {
    // Peek at least recently used entry
    if let Some((key, value)) = cache.peek_lru() {
        println!("LRU: key={}, value={}", key, value);

        // Get recency rank (0 = most recent)
        if let Some(rank) = cache.recency_rank(key) {
            println!("Recency rank: {}", rank);
        }
    }
}

let mut cache: LruCore<u64, &str> = LruCore::new(10);
cache.insert(1, Arc::new("first"));
cache.insert(2, Arc::new("second"));
cache.get(&1);  // Make key 1 MRU

analyze_lru_state(&cache);  // Doesn't change LRU order!
```

**Methods:**
- `peek_lru() -> Option<(&K, &V)>` - View LRU entry
- `recency_rank(&K) -> Option<usize>` - Get position (0 = MRU, higher = older)

### LFU: `ReadOnlyLfuCache`

Inspect frequency information without updating counters:

```rust
use std::sync::Arc;
use cachekit::traits::{ReadOnlyLfuCache, CoreCache};
use cachekit::policy::lfu::LfuCache;

fn analyze_lfu_state<C: ReadOnlyLfuCache<u64, String>>(cache: &C) {
    // Peek at least frequently used entry
    if let Some((key, value)) = cache.peek_lfu() {
        println!("LFU: key={}, value={}", key, value);

        // Get access frequency
        if let Some(freq) = cache.frequency(key) {
            println!("Frequency: {} accesses", freq);
        }
    }
}

let mut cache: LfuCache<u64, &str> = LfuCache::new(10);
cache.insert(1, Arc::new("value"));
cache.get(&1);  // Frequency = 2 (insert + get)
cache.get(&1);  // Frequency = 3

analyze_lfu_state(&cache);  // Doesn't increment frequency!
```

**Methods:**
- `peek_lfu() -> Option<(&K, &V)>` - View LFU entry
- `frequency(&K) -> Option<u64>` - Get access count

### LRU-K: `ReadOnlyLruKCache`

Inspect K-distance and access history without recording accesses:

```rust
use cachekit::traits::{ReadOnlyLruKCache, CoreCache};
use cachekit::policy::lru_k::LrukCache;

fn analyze_lru_k_state<C: ReadOnlyLruKCache<u64, String>>(cache: &C) {
    println!("K value: {}", cache.k_value());

    if let Some((key, value)) = cache.peek_lru_k() {
        println!("LRU-K: key={}, value={}", key, value);

        // Get access statistics
        if let Some(count) = cache.access_count(key) {
            println!("Access count: {}", count);
        }

        if let Some(dist) = cache.k_distance(key) {
            println!("K-distance: {}", dist);
        }

        if let Some(history) = cache.access_history(key) {
            println!("Access history (recent first): {:?}", history);
        }

        if let Some(rank) = cache.k_distance_rank(key) {
            println!("K-distance rank: {} (0 = evict first)", rank);
        }
    }
}

let mut cache = LrukCache::with_k(10, 2);
cache.insert(1, "value".to_string());
cache.get(&1);  // Second access

analyze_lru_k_state(&cache);  // No access recorded!
```

**Methods:**
- `peek_lru_k() -> Option<(&K, &V)>` - View LRU-K entry
- `k_value() -> usize` - Get the K parameter
- `access_count(&K) -> Option<usize>` - Get number of accesses
- `k_distance(&K) -> Option<u64>` - Get K-th access timestamp
- `access_history(&K) -> Option<Vec<u64>>` - Get full history (recent first)
- `k_distance_rank(&K) -> Option<usize>` - Get eviction rank

## Use Cases

### 1. Monitoring & Observability

```rust
use cachekit::traits::{ReadOnlyCache, ReadOnlyLruCache};

fn cache_health_check<C>(cache: &C) -> CacheHealth
where
    C: ReadOnlyCache<u64, Vec<u8>> + ReadOnlyLruCache<u64, Vec<u8>>,
{
    let utilization = cache.len() as f64 / cache.capacity() as f64;
    let has_lru_victim = cache.peek_lru().is_some();

    CacheHealth {
        size: cache.len(),
        utilization: (utilization * 100.0) as u8,
        can_evict: has_lru_victim,
    }
}

struct CacheHealth {
    size: usize,
    utilization: u8,  // 0-100%
    can_evict: bool,
}
```

### 2. Testing Without Side Effects

```rust
#[cfg(test)]
mod tests {
    use cachekit::traits::{ReadOnlyLruCache, CoreCache};
    use cachekit::policy::lru::LruCore;
    use std::sync::Arc;

    #[test]
    fn test_lru_order_preserved() {
        let mut cache: LruCore<u64, &str> = LruCore::new(3);
        cache.insert(1, Arc::new("a"));
        cache.insert(2, Arc::new("b"));
        cache.insert(3, Arc::new("c"));

        // Inspect without affecting order
        assert_eq!(cache.peek_lru().map(|(k, _)| *k), Some(1));
        assert_eq!(cache.recency_rank(&1), Some(2));  // Oldest
        assert_eq!(cache.recency_rank(&3), Some(0));  // Newest

        // Verify peek didn't change order
        assert_eq!(cache.peek_lru().map(|(k, _)| *k), Some(1));
    }
}
```

### 3. Concurrent Readers

```rust
use std::sync::Arc;
use parking_lot::RwLock;
use cachekit::traits::{ReadOnlyCache, ReadOnlyLruCache};
use cachekit::policy::lru::LruCore;

// Shared cache with many readers
let cache = Arc::new(RwLock::new(LruCore::<u64, Arc<String>>::new(1000)));

// Spawn multiple reader threads
let handles: Vec<_> = (0..10).map(|i| {
    let cache = cache.clone();
    std::thread::spawn(move || {
        // Only need read lock for inspection
        let guard = cache.read();

        println!("Thread {}: size={}, has_42={}",
            i, guard.len(), guard.contains(&42));

        if let Some((key, _)) = guard.peek_lru() {
            println!("Thread {}: LRU key={}", i, key);
        }
    })
}).collect();

for handle in handles {
    handle.join().unwrap();
}
```

### 4. Clear API Intent

```rust
use cachekit::traits::{ReadOnlyCache, CoreCache};

// Function signature clearly shows it won't modify cache
fn compute_hit_rate<C: ReadOnlyCache<u64, Vec<u8>>>(
    cache: &C,
    hits: u64,
    total: u64,
) -> f64 {
    // Can inspect but not modify
    let size = cache.len();
    println!("Cache size: {}", size);

    hits as f64 / total as f64
}

// This function CAN modify cache
fn warm_cache<C: CoreCache<u64, Vec<u8>>>(
    cache: &mut C,
    data: &[(u64, Vec<u8>)],
) {
    for (k, v) in data {
        cache.insert(*k, v.clone());
    }
}
```

## Comparison: Read-Only vs Mutable Operations

| Operation | Read-Only Trait | Mutable Operation | Side Effects |
|-----------|----------------|-------------------|--------------|
| Check existence | `contains(&K)` | `get(&mut K)` | None vs Updates access |
| View LRU | `peek_lru()` | `get(&mut K)` | None vs Moves to MRU |
| View frequency | `frequency(&K)` | `get(&mut K)` | None vs Increments count |
| View oldest | `peek_oldest()` | `pop_oldest()` | None vs Removes entry |

## Best Practices

1. **Use `&self` for Inspection**: Read-only traits work with immutable references
   ```rust
   fn inspect<C: ReadOnlyCache<K, V>>(cache: &C) { }  // ✓ Good
   fn inspect<C: CoreCache<K, V>>(cache: &mut C) { }  // ✗ Overkill
   ```

2. **Combine Traits for Rich Inspection**: Use multiple trait bounds
   ```rust
   fn analyze<C>(cache: &C)
   where
       C: ReadOnlyCache<u64, String> + ReadOnlyLruCache<u64, String>,
   {
       // Can use both base and policy-specific inspection
   }
   ```

3. **Prefer Read-Only for Metrics**: Collecting metrics shouldn't affect behavior
   ```rust
   struct MetricsCollector;

   impl MetricsCollector {
       fn collect<C: ReadOnlyCache<K, V>>(&self, cache: &C) -> Metrics {
           Metrics {
               size: cache.len(),
               capacity: cache.capacity(),
               utilization: cache.len() as f64 / cache.capacity() as f64,
           }
       }
   }
   ```

4. **Document Side Effects**: Make it clear when functions modify cache state
   ```rust
   /// Inspects cache without side effects.
   fn peek<C: ReadOnlyCache<K, V>>(cache: &C) { }

   /// Updates LRU order as a side effect.
   fn access<C: CoreCache<K, V>>(cache: &mut C, key: &K) {
       cache.get(key);
   }
   ```

## Performance Considerations

- **Read-only operations are typically faster**: No list manipulation or counter updates
- **Better concurrency**: Multiple readers can use shared `&self` references
- **No allocations**: Inspection doesn't trigger evictions that might allocate

## See Also

- [Trait Hierarchy](../api-surface.md) - Full trait documentation
- [Choosing a Policy](choosing-a-policy.md) - Policy comparison guide
- [Concurrency Patterns](../../examples/) - Thread-safe cache usage
