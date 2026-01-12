# Handle store

This store module is implemented in `cachekit::store::handle` and provides a store keyed by compact handles (IDs) instead of full keys. It’s intended to be used alongside a `KeyInterner` (or any other handle allocator) so that policy metadata never has to clone large keys.

## Architecture
- Stores values in a `HashMap<H, Arc<V>>`, where `H` is a compact handle type.
- Policies operate on handles; an interner maps keys ↔ handles outside the store.

## Key Components
- `HandleStore<H, V>`: single-threaded handle-backed store.
- `ConcurrentHandleStore<H, V>`: `RwLock`-protected store for multi-threaded use.
- `KeyInterner`: a common way to obtain stable handles for keys (in `cachekit::ds`).

## Core Operations
- `try_insert`: insert/update by handle.
- `get`: fetch by handle (updates hit/miss counters).
- `remove`, `clear`.

## Performance Trade-offs
- Avoids cloning/storing large keys inside policy data structures.
- Requires an extra mapping layer (key → handle) managed by the caller.
- Stores `Arc<V>` values for cheap cloning on reads.

## When to Use
- You already have stable handles for keys (interning, IDs, indices).
- Keys are large/expensive to clone and you want to keep policies “handle-only”.

## Example Usage
```rust
use std::sync::Arc;

use cachekit::ds::KeyInterner;
use cachekit::store::handle::HandleStore;
use cachekit::store::traits::StoreMut;

let mut interner = KeyInterner::new();
let handle = interner.intern("alpha".to_string());

let mut store: HandleStore<_, String> = HandleStore::new(2);
store.try_insert(handle, Arc::new("value".to_string())).unwrap();
```

## Type Constraints
- `H: Copy + Eq + Hash` for handle lookup.

## Thread Safety
- `HandleStore` is single-threaded.
- `ConcurrentHandleStore` is `Send + Sync` via `RwLock`.

## Implementation Notes
- Handles must remain stable for as long as the entry is expected to be retrievable.
- Key lifecycle (interning and cleanup) is managed by the caller, not the store.
