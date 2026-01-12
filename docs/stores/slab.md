# Slab store

This store module is implemented in `cachekit::store::slab` and provides a slab-backed store with stable `EntryId` indirection (useful for policy metadata structures that want stable handles).

## Architecture
- Values are stored in a slab (`Vec<Option<Entry<...>>>`) with a free-list for slot reuse.
- A `HashMap<K, EntryId>` maps keys to slab slots.
- `EntryId` is a compact handle into the slab.

## Key Components
- `EntryId`: stable handle to a slot (until that slot is freed).
- `SlabStore<K, V, M>`: core store with a configurable `ValueModel`.
- `SharedSlabStore<K, V>`: returns `Arc<V>` values (cheap clone on reads).
- `OwnedSlabStore<K, V>`: stores `V` and returns `&V` on reads.
- `ConcurrentSlabStore<K, V>`: `RwLock`-protected `SharedSlabStore`.

## Core Operations
- `try_insert`: insert/update by key, reusing free slots when possible.
- `entry_id`: get the `EntryId` for an existing key.
- `get_by_id`, `key_by_id`: stable handle lookups.
- `remove`, `clear`.

## Performance Trade-offs
- `EntryId` avoids storing large keys in policy data structures.
- One extra indirection (key → id → entry) vs direct map lookup.
- Slot reuse can reduce allocation churn in eviction-heavy workloads.

## When to Use
- A policy needs stable IDs to maintain O(1) metadata updates (e.g. lists/arenas).
- You want to store a value once and pass around compact handles.
- You expect heavy churn and want reuse-friendly allocation behavior.

## Example Usage
```rust
use std::sync::Arc;

use cachekit::store::slab::{SharedSlabStore, SlabStore};
use cachekit::store::traits::StoreMut;

let mut store: SharedSlabStore<u64, String> = SlabStore::new(4);
store.try_insert(1, Arc::new("a".to_string())).unwrap();
let id = store.entry_id(&1).unwrap();
assert_eq!(store.get_by_id(id).as_deref().map(String::as_str), Some("a"));
```

## Type Constraints
- `K: Eq + Hash` for key lookup.
- `M: ValueModel<V>` controls how values are stored and returned.

## Thread Safety
- `SlabStore` is single-threaded.
- `ConcurrentSlabStore` is `Send + Sync` via `RwLock`.

## Implementation Notes
- `EntryId` values become invalid after removal of that entry (IDs are reused).
- `OwnedSlabStore` is useful when you want zero `Arc` overhead on reads (borrowed output).
