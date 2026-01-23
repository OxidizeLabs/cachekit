# Weight store

This store module is implemented in `cachekit::store::weight` and provides a weight-aware store that enforces both an entry-count limit and a total weight limit (typically “bytes”). For an overview of all store types, see [Stores](README.md).

## Architecture
- Stores `Arc<V>` values in a `HashMap<K, WeightEntry<V>>`.
- Tracks a running `total_weight` to enforce a weight capacity.
- Uses a caller-provided weight function `F: Fn(&V) -> usize`.

## Capacity Semantics
- Two limits are enforced: entry count (`capacity()`) and total weight (`capacity_weight()`).
- `try_insert` returns `Err(StoreFull)` when inserting a new key would exceed the entry limit, or when inserting/updating would exceed the weight limit.
- Updates recompute weight and may fail if the updated value is “too large” for the configured weight capacity.
- The store does not evict on its own; eviction is driven by the policy layer (which should call `remove` and then `record_eviction` to keep eviction metrics accurate).

## Key Components
- `WeightStore<K, V, F>`: single-threaded weight-aware store.
- `ConcurrentWeightStore<K, V, F>`: `RwLock`-protected store for multi-threaded use.

## Core Operations
- `try_insert`: insert/update by key while enforcing entry and weight limits.
- `get`: fetch by key (updates hit/miss counters).
- `remove`: delete by key and adjust `total_weight`.
- `total_weight`, `capacity_weight`, `clear`.

## Performance Trade-offs
- Inserts/updates compute weight; the weight function is on the hot path.
- Reads are cheap (weight is stored per entry, not recomputed on get).
- Keeps “size accounting” separate from the eviction policy (policy decides what to evict).

## When to Use
- Values vary widely in size and you want size-based capacity.
- You want observability into how “full” the store is in bytes/weight.

## Example Usage
```rust
use std::sync::Arc;

use cachekit::store::traits::StoreMut;
use cachekit::store::weight::WeightStore;

let mut store = WeightStore::with_capacity(10, 64, |v: &String| v.len());
store.try_insert("k1", Arc::new("value".to_string())).unwrap();
assert!(store.total_weight() <= store.capacity_weight());
```

## Example: Concurrent usage
```rust
use std::sync::Arc;

use cachekit::store::traits::ConcurrentStore;
use cachekit::store::weight::ConcurrentWeightStore;

let store = ConcurrentWeightStore::with_capacity(10, 64, |v: &String| v.len());
store.try_insert("k1", Arc::new("value".to_string())).unwrap();
assert!(store.total_weight() <= store.capacity_weight());
```

## Type Constraints
- `K: Eq + Hash` for key lookup.
- `F: Fn(&V) -> usize` to compute weight.

## Thread Safety
- `WeightStore` is single-threaded.
- `ConcurrentWeightStore` is `Send + Sync` via `RwLock`.

## Implementation Notes
- Updates recompute weight and adjust `total_weight` (can fail with `StoreFull`).
- Entry capacity and weight capacity are both enforced.
- In `ConcurrentWeightStore`, `get` takes a write lock because it updates metrics; this can increase contention in read-heavy workloads.
- Weight is an accounting mechanism, not a guarantee of actual memory usage; the accuracy depends on your chosen weight function.

## Weight Function Guidelines
- Keep it fast (it runs on every insert/update).
- Keep it deterministic and stable for a given value (avoid time/randomness/global state).
- Prefer “monotonic with size” (larger values should not report smaller weights).
- For `ConcurrentWeightStore`, the weight function must be `Send + Sync` (capture-only thread-safe state).
