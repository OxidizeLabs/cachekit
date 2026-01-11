# Policy Implementation Notes

These documents describe how to implement common cache eviction policies, with a bias toward practical, production implementations (O(1) hot-paths, bounded metadata, and clear invariants).

**In this repo** (implemented policies):
- FIFO: `src/policy/fifo/`
- LRU: `src/policy/lru.rs`
- LRU-K: `src/policy/lru_k.rs`
- LFU: `src/policy/lfu.rs`
- Heap LFU: `src/policy/heap_lfu.rs`

**Index**

- `opt.md` — Belady OPT / MIN (for trace evaluation)
- `random.md` — Random eviction
- `fifo.md` — FIFO
- `lifo.md` — LIFO / FILO
- `lru.md` — LRU
- `mru.md` — MRU
- `clock.md` — Second-chance / Clock
- `nru.md` — NRU
- `lfu.md` — LFU (bucketed)
- `mfu.md` — MFU
- `heap-lfu.md` — LFU via heap (lazy stale entries)
- `lfu-aging.md` — Aging / decayed LFU patterns
- `lru-k.md` — LRU-K
- `2q.md` — 2Q
- `slru.md` — Segmented LRU (SLRU)
- `arc.md` — ARC
- `car.md` — CAR (Clock with Adaptive Replacement)
- `lirs.md` — LIRS
- `clock-pro.md` — CLOCK-Pro
- `gdsf.md` — GreedyDual-Size/Frequency style (size/cost-aware)
- `ttl.md` — TTL (expiration; not an eviction policy)
