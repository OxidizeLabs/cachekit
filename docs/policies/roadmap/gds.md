# GreedyDual-Size (GDS)

## Goal
Improve byte-level cache efficiency by preferring entries with better value-per-byte
when item sizes and miss costs vary.

## Core Idea
Each resident entry carries a priority `H`; eviction removes the smallest `H`.

A common GDS-style score:
- `H = (cost / size) + L`

Where:
- `cost`: estimated miss penalty (latency, backend pressure, or monetary cost)
- `size`: object size in bytes
- `L`: global inflation term set to the `H` of the most recently evicted entry

The inflation term prevents old low-value entries from sticking forever.

## Core Data Structures (Typical)
- Hash index `K -> EntryMeta`
- Min-heap keyed by `H` for victim selection
- Global inflation scalar `L`

## Complexity & Overhead
- Heap updates are usually O(log n) on insert/update/evict
- More metadata than LRU/FIFO due to per-entry score and object size/cost

## Notes For CacheKit
- Useful when optimizing byte hit rate, not just request hit rate.
- Integrates naturally if entry size is already tracked in storage metadata.
- Keep score updates explicit and avoid per-access allocation.

## References
- Cao, Irani (1997): “GreedyDual-Size: An algorithm for web caching”.
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
