# GreedyDual-Size / GreedyDual-Size-Frequency (GDS/GDSF)

## Goal
Make eviction consider **object size** and/or **miss cost** (common in web/CDN caches), not just request count.

## Core Idea
Assign each entry a key `H` (priority). Evict the smallest `H`.

Common scoring shape (varies by variant):
- `H = (cost * freq) / size + L`
Where:
- `cost`: retrieval cost (latency, backend load, $ cost)
- `freq`: access frequency (for GDSF-like variants)
- `size`: bytes
- `L`: an “inflation” value updated to the `H` of the last evicted item (prevents tiny objects from dominating forever)

## Core Data Structures
Typical implementation:
- `HashMap<K, Entry>` for lookup
- min-heap or ordered map keyed by `H` for victim selection

Implementation choices:
- Exact heap: O(log n) per access/update
- Approximate: bucket priorities or periodic recomputation

## When It Matters
Use when:
- object sizes vary a lot and you care about byte hit rate
- misses have heterogeneous cost

## References
- Cao, Irani (1997): “GreedyDual-Size: An algorithm for web caching”.
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
