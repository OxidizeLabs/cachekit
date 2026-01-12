# NRU (Not Recently Used)

## Goal
Cheap eviction using a coarse “recently used” signal rather than a full ordering.

## Core Idea
Each entry has a reference bit `R`. Periodically (each epoch), clear all `R` bits.
On eviction, pick any entry with `R = 0` (not used this epoch).

## Core Data Structures
Practical NRU:
- `HashMap<K, Entry>` for key/value
- `Vec<RBit>` or `bitset` indexed by slot id
- Epoch control:
  - global “epoch tick” that triggers clearing
  - or per-entry epoch stamping (`last_touched_epoch`)

## Operations

### `get(key)`
- Mark as referenced:
  - set `R = 1`, or
  - set `last_touched_epoch = current_epoch`

### `evict()`
- If using a bitset: scan for any `R=0` victim (can be O(n) unless you maintain a candidate list).
- If using epoch stamping: choose an entry with `last_touched_epoch != current_epoch`.

## Complexity & Overhead
- NRU is cheap but often needs scanning unless you maintain additional structure.
- Good when you already have a natural epoch (e.g., timer tick) and can accept approximation.

## Implementation Notes
- If scanning is unacceptable, combine with a `Clock` hand or maintain a queue of candidates.

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
