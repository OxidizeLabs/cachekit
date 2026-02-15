# NRU (Not Recently Used)

**Feature:** `policy-nru`

## Goal
Cheap eviction using a coarse "recently used" signal rather than a full ordering. Approximate LRU with minimal overhead and simpler implementation than Clock.

## Core Idea
Each entry has a reference bit `R`. On eviction:
1. Scan for first entry with `R = 0` (unreferenced)
2. If all entries are referenced, clear all reference bits and evict first entry

Unlike Clock which uses a continuously advancing hand, NRU scans linearly from the beginning when looking for victims.

- On access (`get`): set `ref = 1`
- On update (`insert` existing): set `ref = 1`
- On new insert: set `ref = 0` (new entries start unreferenced, cold start)
- On eviction: find first unreferenced entry, or clear all bits and evict first

## Core Data Structures

Implementation in [`NruCache`](../../src/policy/nru.rs):
- `map: FxHashMap<K, Entry<V>>` for O(1) key → entry lookup
  - Each `Entry` contains: `index`, `value`, `referenced` (bool)
- `keys: Vec<K>` dense array of keys for eviction scanning
- No hand pointer (unlike Clock)

## Operations

### `get(key)`
- Look up entry via hashmap
- Set `referenced = true`
- Return value reference
- Cost: O(1)

### `insert(key, value)`
- If key exists: update value, set `referenced = true`, return old value
- If at capacity: run eviction (linear scan)
- Insert new entry with `referenced = true`
- Cost: O(1) best case, O(n) worst case (eviction scan)

### `evict()`
Phase 1 - Find unreferenced:
- Scan keys vec for first entry with `referenced = false`
- If found: evict that entry (swap-remove)
- Cost: O(k) where k is position of first unreferenced entry

Phase 2 - All referenced:
- If no unreferenced entry found, clear all reference bits
- Evict first entry (index 0)
- Cost: O(n) to clear all bits

### `contains(key)`
- Hash lookup only
- Does not affect reference bit
- Cost: O(1)

### `remove(key)`
- Remove from map and swap-remove from keys vec
- Update index of swapped key
- Cost: O(1)

## Complexity & Overhead

| Operation  | Time   | Notes                                     |
|------------|--------|-------------------------------------------|
| `get`      | O(1)   | Hash lookup + bit set                     |
| `insert`   | O(n)*  | *Worst case if all entries referenced     |
| `contains` | O(1)   | Hash lookup only                          |
| `remove`   | O(1)   | Hash lookup + swap-remove                 |

- Metadata per entry: key + value + index + 1 bool (referenced)
- Memory layout: HashMap + Vec (keys stored twice)
- Simpler implementation than Clock (no hand management)

## Trade-offs vs Other Policies

| Aspect         | NRU                           | Clock                     | True LRU                  |
|----------------|-------------------------------|---------------------------|---------------------------|
| Access cost    | O(1) bit set                  | O(1) bit set              | O(1) list move            |
| Eviction cost  | O(n) worst case               | O(1) amortized            | O(1)                      |
| Granularity    | Binary (used/not used)        | Binary with hand sweep    | Full order                |
| Implementation | Simple (just scan)            | Moderate (hand + ring)    | Complex (linked list)     |
| Overhead/entry | ~17 bytes (index + ref)       | ~1 byte (ref bit)         | ~16 bytes (2 pointers)    |

## When to Use

**Use NRU when:**
- You need simple, coarse eviction tracking
- Memory for full LRU list is too expensive
- You can tolerate O(n) eviction in worst case
- Access patterns have temporal locality
- You want simpler implementation than Clock

**Avoid NRU when:**
- You need O(1) eviction guarantees (use Clock with hand, or LRU)
- You need fine-grained recency tracking (use LRU)
- You need scan resistance (use S3-FIFO, LRU-K)
- You need frequency tracking (use LFU)
- Eviction performance is critical (use Clock or LRU)

## Comparison with Clock

NRU vs Clock similarities:
- Both use single reference bit per entry
- Both provide coarse recency approximation
- Both set reference bit on access

NRU vs Clock differences:
- NRU scans linearly from start on eviction
- Clock uses hand pointer for continuous sweep
- NRU simpler to implement (no hand management)
- Clock has better amortized eviction cost
- NRU may have better cache locality for small caches (fewer data structures)

## Thread Safety

- `NruCache`: Implements `Send` and `Sync` when K and V are Send/Sync
- Not internally synchronized; wrap in `Mutex` for concurrent access
- See tests for `Arc<Mutex<NruCache>>` examples

## Implementation Notes

- New inserts start with `referenced = false` (cold start, only accessed items get referenced)
- Uses swap-remove technique for O(1) removal from keys vec
- Maintains index in Entry for fast swap-remove updates
- Zero capacity is clamped to 1
- Clears all reference bits only when all entries are referenced (lazy clearing)

## Performance Context

Expected behavior on workload with temporal locality:
- Hit rate: Between Clock and Random
- Access latency: Similar to Clock (just bit set)
- Eviction latency: Slower than Clock due to linear scan
- Memory overhead: Similar to Clock

Best for: Small-to-medium caches where O(n) scan is acceptable and simplicity is valued over eviction performance.

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
- NRU is a simplified variant of Clock, trading eviction performance for implementation simplicity
