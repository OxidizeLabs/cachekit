# Policy Data Structures

These documents describe reusable data structures that show up across cache eviction policies (LRU/LRU-K/LFU/ARC/Clock/etc).

**Index**

- `intrusive-list.md` — O(1) recency lists (LRU/SLRU/2Q/ARC)
- `slot-arena.md` — stable handles + free list (avoid pointer chasing)
- `clock-ring.md` — Clock/second-chance ring + hand
- `ghost-list.md` — history (“ghost”) lists for adaptive policies (ARC/CAR)
- `fixed-history.md` — fixed-size access history per key (LRU-K style)
- `frequency-buckets.md` — LFU bucket lists + `min_freq` tracking
- `lazy-heap.md` — heap + authoritative map with stale entries (heap LFU, OPT simulators)
