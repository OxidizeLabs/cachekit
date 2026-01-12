# Policy Implementation Notes

These documents describe how to implement common cache eviction policies, with a bias toward practical, production implementations (O(1) hot-paths, bounded metadata, and clear invariants).

**In this repo** (implemented policies):
- FIFO: `src/policy/fifo/`
- LRU: `src/policy/lru.rs`
- LRU-K: `src/policy/lru_k.rs`
- LFU: `src/policy/lfu.rs`
- Heap LFU: `src/policy/heap_lfu.rs`

**Index (Implemented)**

- `fifo.md` — FIFO
- `lru.md` — LRU
- `lru-k.md` — LRU-K
- `lfu.md` — LFU (bucketed)
- `heap-lfu.md` — LFU via heap (lazy stale entries)

**Roadmap (Not Implemented Yet)**

- `roadmap/opt.md` — Belady OPT / MIN (for trace evaluation)
- `roadmap/random.md` — Random eviction
- `roadmap/lifo.md` — LIFO / FILO
- `roadmap/mru.md` — MRU
- `roadmap/clock.md` — Second-chance / Clock
- `roadmap/nru.md` — NRU
- `roadmap/mfu.md` — MFU
- `roadmap/lfu-aging.md` — Aging / decayed LFU patterns
- `roadmap/2q.md` — 2Q
- `roadmap/slru.md` — Segmented LRU (SLRU)
- `roadmap/arc.md` — ARC
- `roadmap/car.md` — CAR (Clock with Adaptive Replacement)
- `roadmap/lirs.md` — LIRS
- `roadmap/clock-pro.md` — CLOCK-Pro
- `roadmap/gdsf.md` — GreedyDual-Size/Frequency style (size/cost-aware)
- `roadmap/ttl.md` — TTL (expiration; not an eviction policy)
