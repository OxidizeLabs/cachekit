# Second-Chance / Clock

## Goal
Approximate LRU with lower overhead by avoiding strict recency ordering.

## Core Idea
Maintain entries in a circular list (“clock”). Each entry has a reference bit.
- On access: set `ref = 1`
- On eviction: advance the hand:
  - if `ref = 0`, evict it
  - if `ref = 1`, set `ref = 0` and continue

## Core Data Structures
Practical O(1) Clock (amortized):
- `Vec<EntryMeta>` (dense array of slots) with `{ key, ref_bit, occupied, ... }`
- `HashMap<K, slot_index>` for O(1) lookup
- `hand: usize` pointer into the ring

## Operations

### `get(key)`
- Find slot via hashmap; set `ref_bit = 1`.

### `insert(key, value)`
- If present: update value, set `ref_bit = 1`.
- Else:
  - If free slot exists: occupy it, set `ref_bit = 1`.
  - If full: run eviction loop advancing `hand` until a `ref_bit == 0` victim is found.

## Complexity & Overhead
- Lookup: O(1)
- Insert: O(1) amortized; worst-case can scan many entries if all are “referenced”
- Metadata: 1 bit (or byte) per entry + hash index

## Implementation Notes
- Store values in a separate dense array or hashmap; keep metadata hot and compact.
- For concurrency, Clock is often easier than strict LRU because you don’t reorder a global list on every hit.

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
