# Policy Implementation Notes

These documents describe how to implement common cache eviction policies, with a bias toward practical, production implementations (O(1) hot-paths, bounded metadata, and clear invariants).

**In this repo** (implemented policies):
- FIFO: `src/policy/fifo.rs`
- LRU: `src/policy/lru.rs`
- LRU-K: `src/policy/lru_k.rs`
- LFU: `src/policy/lfu.rs`
- Heap LFU: `src/policy/heap_lfu.rs`
- 2Q: `src/policy/two_q.rs`
- S3-FIFO: `src/policy/s3_fifo.rs`
- Second-chance / Clock: `src/policy/clock.rs`
- CLOCK-Pro: `src/policy/clock_pro.rs`

**Roadmap (Not Implemented Yet)**

- `roadmap/opt.md` — Belady OPT / MIN (for trace evaluation)
- `roadmap/random.md` — Random eviction
- `roadmap/lifo.md` — LIFO / FILO
- `roadmap/mru.md` — MRU
- `roadmap/nru.md` — NRU
- `roadmap/mfu.md` — MFU
- `roadmap/lfu-aging.md` — Aging / decayed LFU patterns
- `roadmap/slru.md` — Segmented LRU (SLRU)
- `roadmap/arc.md` — ARC
- `roadmap/car.md` — CAR (Clock with Adaptive Replacement)
- `roadmap/lirs.md` — LIRS
- `roadmap/gdsf.md` — GreedyDual-Size/Frequency style (size/cost-aware)
- `roadmap/ttl.md` — TTL (expiration; not an eviction policy)
