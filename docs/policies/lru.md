# LRU (Least Recently Used)

**Feature:** `policy-lru`

## Goal
Evict the entry that has not been accessed for the longest time.

## Core Data Structures

`src/policy/lru.rs` uses a pool-based design:

- `SlotArena<Node<K, V>>` — contiguous `Vec` with a free list for O(1) slot reuse.
  Each `Node` stores `prev`/`next` as `Option<SlotId>` indices, plus the key and `Arc<V>`.
- `FxHashMap<K, SlotId>` — O(1) key-to-slot lookup.
- `head`/`tail: Option<SlotId>` — doubly-linked recency order (head = MRU, tail = LRU).
- `ConcurrentLruCache` — thread-safe wrapper via `parking_lot::RwLock`.

At steady state (cache full), every insert evicts the tail node (returning its
slot to the free list) then inserts the new node (reusing a free slot). No heap
allocations occur after the initial warm-up phase.

## Operations

### `get(key)`
- Lookup key in map; if present: detach node, attach at head (MRU), return `&Arc<V>`.

### `insert(key, value)`
- If `key` exists: replace value in-place, detach + attach at head.
- Else:
  - If at capacity: `pop_tail()` removes the LRU node from the arena (slot returned to free list), remove its key from the map.
  - `arena.insert(Node{...})` reuses a free slot or appends.
  - Insert `SlotId` into the map, attach at head.

### `pop_lru()`
- Remove tail node from arena + map, return `(K, Arc<V>)`.

### `peek(key)` / `peek_lru()`
- Read-only lookup; no list reordering.

## Complexity & Overhead
- Lookup / update / evict: O(1)
- Memory per entry: `SlotId` prev/next (2 × 12 bytes) + key + `Arc<V>` + `Option` discriminant in arena slot
- Steady-state allocations: **zero** (free-list recycling)

## Concurrency Notes
Strict global LRU mutates shared metadata on every hit; `cachekit` provides:
- `ConcurrentLruCache` — single `RwLock` around `LruCore`.
- `get()` requires a **write** lock (moves node to head).
- `peek()` requires only a **read** lock (no reordering).
- Read-heavy workloads can still contend on `get()`; prefer `peek()` when recency updates are not needed.

## Safety / Invariants (Rust)
- **No `unsafe`**: all list manipulation uses `SlotId` indices into the arena.
- **ABA prevention**: `SlotId` includes a generation counter; stale handles return `None`.
- **Automatic cleanup**: `SlotArena` drops all live nodes when `LruCore` is dropped.
- `Send`/`Sync` auto-derive from the safe field types; no manual `unsafe impl`.

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
