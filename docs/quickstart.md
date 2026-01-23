# Quickstart

Get up and running with CacheKit in minutes.

## Install

Add CacheKit to your `Cargo.toml`:

```toml
[dependencies]
cachekit = "0.2.0-alpha"
```

## Build Your First Cache

Use `CacheBuilder` with an eviction policy:

```rust
use cachekit::builder::{CacheBuilder, CachePolicy};

fn main() {
    // Create an LRU cache with capacity 100
    let mut cache = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Lru);

    // Insert values
    cache.insert(1, "alpha".to_string());
    cache.insert(2, "beta".to_string());

    // Read values (updates recency)
    assert_eq!(cache.get(&1), Some(&"alpha".to_string()));

    // Check presence and size
    assert!(cache.contains(&2));
    assert_eq!(cache.len(), 2);
}
```

## Pick a Policy

Start with one of these:

- `CachePolicy::Lru` for low latency and strong temporal locality.
- `CachePolicy::S3Fifo { small_ratio: 0.1, ghost_ratio: 0.9 }` for scan-heavy workloads.
- `CachePolicy::Lfu { bucket_hint: None }` for frequency-driven workloads.
- `CachePolicy::TwoQ { probation_frac: 0.25 }` for mixed workloads with occasional scans.

See the full policy guidance in `docs/policies/README.md`.

## Thread Safety

Cache implementations are not thread-safe by default. Wrap with a lock for shared access, or enable the `concurrency` feature for built-in concurrent wrappers:

```rust
use std::sync::{Arc, Mutex};
use cachekit::builder::{CacheBuilder, CachePolicy};

let cache = Arc::new(Mutex::new(
    CacheBuilder::new(100).build::<u64, String>(CachePolicy::Lru)
));

let cache_clone = cache.clone();
std::thread::spawn(move || {
    let mut guard = cache_clone.lock().unwrap();
    guard.insert(1, "value".to_string());
});
```

Enable the feature:

```toml
[dependencies]
cachekit = { version = "0.2.0-alpha", features = ["concurrency"] }
```

Tiny example with a built-in concurrent wrapper:

```rust
use cachekit::ds::ConcurrentClockRing;

let cache = ConcurrentClockRing::new(2);
cache.insert("a", 1);
cache.insert("b", 2);

assert_eq!(cache.get(&"a"), Some(1));
```

## Metrics (Optional)

Enable the `metrics` feature to collect basic hit/miss data:

```toml
[dependencies]
cachekit = { version = "0.2.0-alpha", features = ["metrics"] }
```

```rust
use cachekit::policy::lru::LruCore;
use cachekit::traits::CoreCache;

let mut cache = LruCore::<u64, String>::new(100);
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

## Direct Policy Access

Use a policy directly when you need policy-specific operations:

```rust
use std::sync::Arc;
use cachekit::policy::lru::LruCore;
use cachekit::traits::{CoreCache, LruCacheTrait};

let mut cache: LruCore<u64, &str> = LruCore::new(100);
cache.insert(1, Arc::new("value"));

if let Some((key, _)) = cache.peek_lru() {
    println!("LRU key: {}", key);
}
```

## Next Steps

- [Integration details](integration.md)
- [Policy deep dives](policies/README.md)
- [Benchmarks](benchmarks.md)
