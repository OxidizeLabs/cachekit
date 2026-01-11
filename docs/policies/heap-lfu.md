# Heap LFU (Priority Queue LFU)

## Goal
Implement LFU using a heap to choose the minimum frequency victim, trading O(1) for simpler structure and O(log n) operations.

## Core Data Structures
Typical heap LFU:
- `HashMap<K, V>` (or store) for values
- `HashMap<K, u64>` authoritative frequencies
- `BinaryHeap<Reverse<(u64, K)>>` min-heap of (frequency, key)

In `cachekit`, this is implemented in `src/policy/heap_lfu.rs` using **lazy stale entries**:
- Every access pushes a new `(freq, key)` into the heap.
- On eviction, heap entries are popped until one matches the authoritative frequency map.

## Operations

### `get(key)`
- Increment `frequencies[key]`
- Push `(new_freq, key)` into the heap (older heap entries become stale)

### `insert(key, value)`
- Insert/update value.
- Initialize or increment frequency.
- Push updated `(freq, key)` to heap.
- If at capacity: repeatedly pop heap until you find a non-stale victim; remove it from store + frequency map.

## Complexity & Overhead
- `get`: O(log n) (heap push)
- eviction: O(log n) amortized, but may pop multiple stale entries
- memory: heap can grow beyond live entries unless you periodically rebuild

## Implementation Notes
- To bound heap growth, periodically rebuild the heap from the authoritative frequency map when heap size exceeds a threshold multiple of live entries (this is the approach documented in the module).

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
