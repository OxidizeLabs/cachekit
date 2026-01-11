# LRU-K

## Goal
Improve scan resistance by evicting based on the K-th most recent access time, rather than last access time.

## Core Idea
Track the last `K` access timestamps for each key.
- If an entry has fewer than `K` accesses, treat it as “cold/probation”.
- Evict cold entries first (they haven’t demonstrated reuse).
- Otherwise, evict the entry with the oldest K-th most recent access (largest backward K-distance).

In `cachekit`, LRU-K is implemented in `src/policy/lru_k.rs` with segmented tracking for `< K` and `>= K` entries.

## Core Data Structures
Common LRU-K:
- `HashMap<K, EntryMeta>` where `EntryMeta` holds:
  - ring buffer / `VecDeque<u64>` of timestamps (size K)
  - membership in cold/hot segments (optional optimization)
- A store for values (often `HashMap<K, V>` or separate arena)

For efficient victim selection, many implementations maintain:
- Separate candidate structures for cold vs hot
- LRU ordering among cold entries (by first access) and among hot entries (by K-distance)

## Operations

### `touch/get(key)`
- Push current timestamp into the key’s history (keep only last K).
- If history length reaches K: promote to hot segment.
- Update segment ordering based on chosen tie-break rules.

### `pop_lru_k()`
Eviction priority:
1. any entry with `< K` accesses (cold): evict oldest by first access time
2. otherwise: evict the entry with the oldest K-distance (K-th most recent access is oldest)

## Complexity & Overhead
- More metadata than LRU: needs K timestamps per entry (or compressed form).
- Hot-path updates are heavier than LRU; keep operations O(1) by using small fixed K and fixed-size buffers.

## References
- O’Neil, O’Neil, Weikum (1993): “The LRU-K page replacement algorithm for database disk buffering”.
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
