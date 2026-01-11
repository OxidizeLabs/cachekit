# FIFO (First-In, First-Out)

## Goal
Evict the oldest inserted resident entry when the cache is full.

## Core Data Structures
Typical O(1) FIFO implementation:
- `HashMap<K, V>` for key lookup and storage
- `VecDeque<K>` (or ring buffer) to track insertion order

In `cachekit`, FIFO is implemented in `src/policy/fifo/` with:
- a key/value map plus an insertion-order queue
- **lazy stale entry handling** (queue entries can become stale if the key is removed/updated)

## Operations

### `insert(key, value)`
- If `key` already exists: update the stored value, **do not** change insertion order.
- If cache is at capacity: evict from the front of the queue until a *live* key is found.
- Insert the new key/value and push the key to the back of the queue.

### `get(key)`
- Lookup only; FIFO does not reorder on access.

### `pop_oldest()`
- Pop from queue front until you find a key still present in the map; remove it from the map and return it.

## Complexity & Overhead
- Lookup: O(1)
- Insert: O(1) amortized; eviction may skip stale queue entries
- Memory: `HashMap` + queue of keys

## Edge Cases / Implementation Notes
- **Stale keys**: if you support `remove(key)` (or “update” that replaces storage), you must reconcile the queue.
  - Eager cleanup requires O(n) removal from the middle of a `VecDeque`.
  - Lazy cleanup keeps hot path O(1) and pays cleanup during eviction.
- **Duplicate keys in queue**: if your update path pushes keys again, you must handle duplicates on eviction; the repo implementation avoids changing order on update.

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
