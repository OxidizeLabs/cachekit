# HashMap store

This store module is implemented in `cachekit::store::hashmap` and provides `HashMap`-backed stores with entry-count capacity enforcement.

## Architecture
- Keys are stored directly in a `HashMap<K, Arc<V>>`.
- Capacity is enforced by entry count (`len() <= capacity()`), not by bytes.

## Key Components
- `HashMapStore<K, V>`: single-threaded store.
- `ConcurrentHashMapStore<K, V>`: `RwLock`-protected store for multi-threaded use.
- `ShardedHashMapStore<K, V>`: per-shard locks to reduce contention.

## Core Operations
- `try_insert`: insert/update by key; fails with `StoreFull` when at capacity.
- `get`: fetch by key (updates hit/miss counters).
- `remove`, `clear`, `contains`, `len`.

## Performance Trade-offs
- O(1) average lookup/insert/remove.
- Stores `Arc<V>`; clones are cheap on reads, but inserts still allocate the `Arc`.
- Sharding reduces lock contention but adds an extra hashing step to pick the shard.

## When to Use
- You want the simplest general-purpose store keyed by owned keys.
- Capacity by entry count is sufficient.
- You want a straightforward concurrent store (global lock) or a sharded one.

## Example Usage
```rust
use std::sync::Arc;

use cachekit::store::hashmap::HashMapStore;
use cachekit::store::traits::StoreMut;

let mut store: HashMapStore<u64, String> = HashMapStore::new(2);
store.try_insert(1, Arc::new("a".to_string())).unwrap();
assert!(store.contains(&1));
```

## Type Constraints
- `K: Eq + Hash` for key lookup.

## Thread Safety
- `HashMapStore` is single-threaded.
- `ConcurrentHashMapStore` and `ShardedHashMapStore` are `Send + Sync`.

## Implementation Notes
- Capacity is checked before insertion; eviction is driven by the policy layer.
- Metrics are stored using atomics for compatibility with concurrent variants.
