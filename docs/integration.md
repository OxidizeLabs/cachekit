# Integration Guide

This guide covers how to integrate CacheKit into your application.

## CacheBuilder API

The `CacheBuilder` provides a unified, ergonomic API for creating caches with any eviction policy. It hides internal implementation details (like `Arc<V>` wrapping) and provides a consistent interface.

### Basic Usage

```rust
use cachekit::builder::{CacheBuilder, CachePolicy};

// Create a cache with the builder
let mut cache = CacheBuilder::new(1000).build::<u64, String>(CachePolicy::Lru);

// Standard operations
cache.insert(1, "value".to_string());
let value = cache.get(&1);
cache.clear();
```

### Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CacheBuilder                                   │
│                                                                             │
│   CacheBuilder::new(capacity)                                               │
│         │                                                                   │
│         ▼                                                                   │
│   .build::<K, V>(policy)                                                    │
│         │                                                                   │
│         ├─── CachePolicy::Fifo ────► FifoCache<K, V>                       │
│         ├─── CachePolicy::Lru ─────► LruCore<K, V>                         │
│         ├─── CachePolicy::LruK ────► LrukCache<K, V>                       │
│         ├─── CachePolicy::Lfu ─────► LfuCache<K, V>                        │
│         ├─── CachePolicy::HeapLfu ─► HeapLfuCache<K, V>                    │
│         └─── CachePolicy::TwoQ ────► TwoQCore<K, V>                        │
│                                                                             │
│         ▼                                                                   │
│   Cache<K, V>  (unified wrapper)                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  .insert(key, value)  → Option<V>                                   │   │
│   │  .get(&key)           → Option<&V>                                  │   │
│   │  .contains(&key)      → bool                                        │   │
│   │  .len() / .is_empty() → usize / bool                                │   │
│   │  .capacity()          → usize                                       │   │
│   │  .clear()                                                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Type Constraints

The `Cache<K, V>` wrapper requires:

| Type | Bounds | Reason |
|------|--------|--------|
| `K` | `Copy + Eq + Hash + Ord` | Key indexing, heap ordering (HeapLFU) |
| `V` | `Clone + Debug` | Value extraction, debug formatting |

### Policy Selection

| Policy | Use Case | Trade-offs |
|--------|----------|------------|
| `Fifo` | Simple, predictable eviction | No recency/frequency tracking |
| `Lru` | Temporal locality | Vulnerable to scans |
| `LruK { k }` | Scan resistance | Extra memory for history |
| `Lfu` | Stable hot spots | Slow to adapt to changes |
| `HeapLfu` | Large caches, frequent evictions | O(log n) eviction |
| `TwoQ { probation_frac }` | Mixed workloads | Two-queue overhead |

## Direct Policy Access

For advanced use cases requiring policy-specific operations (e.g., `touch()`, `frequency()`, `k_distance()`), use the underlying cache implementations directly.

### Trait Hierarchy

```text
                     CoreCache<K, V>
                           │
           ┌───────────────┴───────────────┐
           │                               │
  FifoCacheTrait<K, V>            MutableCache<K, V>
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
           LruCacheTrait<K, V>    LfuCacheTrait<K, V>    LrukCacheTrait<K, V>
```

### Example: LRU-specific Operations

```rust
use std::sync::Arc;
use cachekit::policy::lru::LruCore;
use cachekit::traits::{CoreCache, LruCacheTrait};

let mut cache: LruCore<u64, &str> = LruCore::new(100);
cache.insert(1, Arc::new("first"));
cache.insert(2, Arc::new("second"));

// Touch to mark as recently used
cache.touch(&1);

// Peek at LRU entry
if let Some((key, _)) = cache.peek_lru() {
    println!("LRU key: {}", key);
}

// Pop LRU entry
let evicted = cache.pop_lru();
```

### Example: LFU-specific Operations

```rust
use std::sync::Arc;
use cachekit::policy::lfu::LfuCache;
use cachekit::traits::{CoreCache, LfuCacheTrait};

let mut cache: LfuCache<u64, &str> = LfuCache::new(100);
cache.insert(1, Arc::new("value"));

// Check frequency
println!("Frequency: {:?}", cache.frequency(&1));

// Boost frequency without accessing value
cache.increment_frequency(&1);

// Reset frequency (demote hot entry)
cache.reset_frequency(&1);
```

### Example: LRU-K-specific Operations

```rust
use cachekit::policy::lru_k::LrukCache;
use cachekit::traits::{CoreCache, LrukCacheTrait};

// Create LRU-2 cache
let mut cache = LrukCache::with_k(100, 2);
cache.insert(1, "value");

// Check access count and K-distance
println!("Access count: {:?}", cache.access_count(&1));
println!("K-distance: {:?}", cache.k_distance(&1));

// Touch to record access
cache.touch(&1);
```

## Thread Safety

Individual cache implementations are **NOT** thread-safe by default. For concurrent access:

### Option 1: External Synchronization

```rust
use std::sync::{Arc, RwLock};
use cachekit::policy::lru_k::LrukCache;
use cachekit::traits::CoreCache;

let cache = Arc::new(RwLock::new(LrukCache::<u64, String>::new(100)));

// Read access
{
    let guard = cache.read().unwrap();
    let _ = guard.contains(&1);
}

// Write access
{
    let mut guard = cache.write().unwrap();
    guard.insert(1, "value".to_string());
}
```

### Option 2: Builder with External Lock

```rust
use std::sync::{Arc, Mutex};
use cachekit::builder::{CacheBuilder, CachePolicy};

let cache = Arc::new(Mutex::new(
    CacheBuilder::new(100).build::<u64, String>(CachePolicy::Lru)
));

// Use from multiple threads
let cache_clone = cache.clone();
std::thread::spawn(move || {
    let mut guard = cache_clone.lock().unwrap();
    guard.insert(1, "value".to_string());
});
```

## Metrics Integration

When the `metrics` feature is enabled, caches expose metrics via `metrics_snapshot()`:

```rust
// Requires: cachekit = { ..., features = ["metrics"] }
use cachekit::policy::lru_k::LrukCache;

let mut cache = LrukCache::<u64, String>::new(100);
cache.insert(1, "value".to_string());
cache.get(&1);

#[cfg(feature = "metrics")]
{
    let metrics = cache.metrics_snapshot();
    println!("Hits: {}", metrics.hits());
    println!("Misses: {}", metrics.misses());
    println!("Hit rate: {:.2}%", metrics.hit_rate() * 100.0);
}
```
