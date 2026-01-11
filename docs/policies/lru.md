# LRU (Least Recently Used)

## Goal
Evict the entry that has not been accessed for the longest time.

## Core Data Structures
Canonical O(1) LRU:
- `HashMap<K, NodeHandle>` mapping keys to nodes
- Intrusive doubly-linked list storing recency order (head = MRU, tail = LRU)

In `cachekit`, `src/policy/lru.rs` uses:
- `HashMap<K, NonNull<Node<K>>>` for O(1) node access
- an intrusive list (manual `prev`/`next` pointers) to support O(1) move-to-front
- a separate store mapping `K -> Arc<V>` (value ownership)
- an optional concurrent wrapper (`parking_lot::RwLock`) for multi-threaded use

## Operations

### `get(key)`
- Lookup key; if present: move its node to the head (MRU) and return value.

### `insert(key, value)`
- If `key` exists: update stored value and move node to head.
- Else:
  - If at capacity: evict tail node, remove its key from the index and store.
  - Insert new node at head and add to index + store.

### `pop_lru()`
- Remove tail node (LRU) and delete from index + store.

## Complexity & Overhead
- Lookup/update/evict: O(1)
- Memory: per-entry list pointers + hash entry + key copy (intrusive list usually stores keys, not values)

## Concurrency Notes
Strict global LRU mutates shared metadata on every hit; typical approaches:
- Single lock around the entire structure (simple, but can contend)
- Sharding (key->shard) for scalability
- Approximate LRU (Clock) when write-amplification or contention is too high

`cachekit` provides a concurrent wrapper in the LRU module; be mindful that:
- every hit does a write (reordering), so read-heavy workloads can still contend.

## Safety / Invariants (Rust)
If using intrusive pointers (like `NonNull<Node<_>>`):
- Nodes must not move in memory while linked (allocate in a stable arena/Box and manage lifetime).
- Every `prev/next` pointer update must preserve list invariants; always update both sides.
- On removal/eviction, ensure you unlink before freeing and remove from the hashmap index.

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
