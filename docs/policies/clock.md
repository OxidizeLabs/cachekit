# Second-Chance / Clock

**Feature:** `policy-clock`

## Goal
Approximate LRU with lower overhead by avoiding strict recency ordering and linked list manipulation.

## Core Idea
Maintain entries in a circular buffer ("clock"). Each entry has a reference bit.
- On access (`get`): set `ref = 1`
- On update (`insert` existing): set `ref = 1`
- On new insert: set `ref = 0` (new entries start unreferenced)
- On eviction: advance the hand:
  - if `ref = 0`, evict it
  - if `ref = 1`, set `ref = 0` (second chance) and continue

## Core Data Structures

Implementation uses [`ClockRing`](../policy-ds/clock-ring.md):
- `index: FxHashMap<K, usize>` for O(1) key → slot lookup
- `slots: Vec<Option<Entry<K, V>>>` circular buffer
  - Each entry has: `key`, `value`, `referenced` (bool)
  - Entry layout is `#[repr(C)]` with `referenced` first for cache-line optimization
  - `None` indicates empty slot
- `hand: usize` clock pointer into the ring

## Operations

### `get(key)`
- Look up slot via hashmap
- Set `referenced = true`
- Return value reference
- Cost: O(1)

### `insert(key, value)`
- If key exists: update value, set `referenced = true`, return old value
- If at capacity: run eviction (sweep with clock hand)
- Find slot using hand position (O(1) amortized, no linear scan)
- Insert with `referenced = false`
- Cost: O(1) amortized

### `evict()`
- Sweep from current hand position:
  - If entry is referenced: clear bit (second chance), advance hand
  - If entry is unreferenced: evict, return slot
- Cost: O(n) worst case, O(1) amortized

### `contains(key)`
- Hash lookup only
- Does not affect reference bit
- Cost: O(1)

### `remove(key)`
- Remove from index and clear slot
- Cost: O(1)

### Additional Operations (via `as_ring()`)
- `peek(key)`: Get value without setting reference bit
- `touch(key)`: Set reference bit without retrieving value
- `peek_victim()`: Preview next eviction candidate
- `pop_victim()`: Manually evict next candidate

## Complexity & Overhead

| Operation  | Time   | Notes                                     |
|------------|--------|-------------------------------------------|
| `get`      | O(1)   | Hash lookup + bit set                     |
| `insert`   | O(1)*  | *Amortized; eviction may sweep            |
| `contains` | O(1)   | Hash lookup only                          |
| `remove`   | O(1)   | Hash lookup + clear slot                  |

- Metadata per entry: key + value + 1 bool (referenced)
- Memory layout: contiguous Vec (cache-friendly)
- Entry layout optimized with `#[repr(C)]` for cache locality

## Trade-offs vs True LRU

| Aspect         | Clock                         | True LRU                  |
|----------------|-------------------------------|---------------------------|
| Access cost    | O(1) bit set                  | O(1) list move            |
| Memory layout  | Contiguous (cache-friendly)   | Scattered nodes           |
| Eviction       | Approximate LRU               | Exact LRU                 |
| Overhead/entry | ~1 byte (ref bit)             | ~16 bytes (2 pointers)    |

## Thread Safety

- `ClockCache`: Implements `Send` and `Sync` when K and V are Send/Sync
- Not internally synchronized; wrap in `Mutex` for concurrent access
- See [`ConcurrentClockRing`](../policy-ds/clock-ring.md) for a thread-safe alternative

## Implementation Notes
- New inserts start unreferenced; only accesses and updates set the reference bit
- Uses hand-based slot finding for O(1) amortized empty slot discovery (no linear scan)
- Zero capacity is clamped to 1
- Built on the [`ClockRing`](../policy-ds/clock-ring.md) data structure

## References
- Wikipedia: https://en.wikipedia.org/wiki/Page_replacement_algorithm#Clock
