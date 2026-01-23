# Policy Data Structures

These documents describe reusable data structures that show up across cache eviction policies (LRU/LRU-K/LFU/ARC/Clock/etc).

**Index**

- [IntrusiveList](intrusive-list.md) — O(1) recency lists (LRU/SLRU/2Q/ARC)
- [SlabArena](intrusive-list.md) — stable handles + free list (avoid pointer chasing)
- [ClockRing](clock-ring.md) — Clock/second-chance ring + hand
- [GhostList](ghost-list.md) — history (“ghost”) lists for adaptive policies (ARC/CAR)
- [FixedHistory](fixed-history.md) — fixed-size access history per key (LRU-K style)
- [FrequencyBuckets](frequency-buckets.md) — LFU bucket lists + `min_freq` tracking
- [LazyHeap](lazy-heap.md) — heap + authoritative map with stale entries (heap LFU, OPT simulators)
