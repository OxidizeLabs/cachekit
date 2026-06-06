# Policy spec matrix

**Canonical index** for the spec-first harness. Other docs ([static-analysis.md](../static-analysis.md), [abstract_models README](../../../tests/abstract_models/README.md)) link here rather than duplicating rows.

| Policy | Spec maturity | Tier | Harness mode(s) | Spec doc | Exact model | Reference model | Bounded module | Op strategy | Traits asserted | TLA+ |
|--------|---------------|------|-----------------|----------|-------------|-----------------|----------------|-------------|-----------------|------|
| FIFO | reference, tla | exact | DualRun, CrossModel | [fifo.md](fifo.md) | `exact/fifo.rs` (`FifoModel`) | `reference/fifo.rs` (`NaiveFifoModel`) | — | `standard_op_list` | VictimInspectable, EvictingCache | [Fifo.tla](Fifo.tla) |
| LRU | reference, tla | exact | DualRun, CrossModel | [lru.md](lru.md) | `exact/lru.rs` (`LruOccupancyModel`) | `reference/lru.rs` (`NaiveLruModel`) | — | `standard_op_list` | VictimInspectable, RecencyTracking, EvictingCache | [Lru.tla](Lru.tla) |
| Fast-LRU | reference | exact | DualRun, CrossModel | [fast-lru.md](fast-lru.md) | `exact/lru.rs` (`LruOccupancyModel`) | `reference/lru.rs` (`NaiveLruModel`) | — | `op_strategy_with_get_mut` | VictimInspectable, RecencyTracking, EvictingCache | — |
| LIFO | reference | exact | DualRun, CrossModel | [lifo.md](lifo.md) | `exact/lifo.rs` (`LifoModel`) | `reference/lifo.rs` (`NaiveLifoModel`) | — | `standard_op_list` | VictimInspectable, EvictingCache | — |
| MRU | reference | exact | DualRun, CrossModel | [mru.md](mru.md) | `exact/mru.rs` (`MruModel`) | `reference/mru.rs` (`NaiveMruModel`) | — | `standard_op_list` | EvictingCache | — |
| LFU | reference | exact | DualRun, CrossModel | [lfu.md](lfu.md) | `exact/lfu.rs` (`LfuModel`) | `reference/lfu.rs` (`NaiveLfuModel`) | — | `standard_op_list` | VictimInspectable, FrequencyTracking, EvictingCache | — |
| Heap-LFU | reference | exact | DualRun, CrossModel | [heap-lfu.md](heap-lfu.md) | `exact/heap_lfu.rs` (`HeapLfuModel`) | `reference/heap_lfu.rs` (`NaiveHeapLfuModel`) | — | `standard_op_list` | residency (`Ord` tie-break) | — |
| MFU | reference | exact | DualRun, CrossModel | [mfu.md](mfu.md) | `exact/mfu.rs` (`MfuModel`) | `reference/mfu.rs` (`NaiveMfuModel`) | — | `standard_op_list_mfu_safe` | residency (seq tie-break) | — |
| LRU-K | reference | exact | DualRun, CrossModel | [lru-k.md](lru-k.md) | `exact/lru_k.rs` (`LruKModel`) | `reference/lru_k.rs` (`NaiveLruKModel`) | — | `standard_op_list` | HistoryTracking, EvictingCache | — |
| Clock | stub | mirror | DualRun | [clock.md](clock.md) | `exact/clock.rs` (`ClockModel`) | — | — | `standard_op_list` | EvictingCache | — |
| 2Q | stub | mirror | DualRun | [two-q.md](two-q.md) | `exact/two_q.rs` (`TwoQModel`) | — | — | `standard_op_list_no_evict` | residency | — |
| SLRU | stub | mirror | DualRun | [slru.md](slru.md) | `exact/slru.rs` (`SlruModel`) | — | — | `standard_op_list_no_evict` | residency | — |
| NRU | stub | mirror | DualRun | [nru.md](nru.md) | `exact/nru.rs` (`NruModel`) | — | — | `short_op_list_no_evict` | residency | — |
| ARC | stub | bounded | InvariantOnly | [arc.md](arc.md) | — | — | `bounded/arc.rs` | `standard_op_list` | `debug_validate_invariants` | — |
| CAR | stub | bounded | InvariantOnly | [car.md](car.md) | — | — | `bounded/car.rs` | `standard_op_list` | `debug_validate_invariants` | — |
| Clock-PRO | stub | bounded | InvariantOnly | [clock-pro.md](clock-pro.md) | — | — | `bounded/clock_pro.rs` | `standard_op_list` | `debug_validate_invariants` | — |
| S3-FIFO | stub | bounded | InvariantOnly | [s3-fifo.md](s3-fifo.md) | — | — | `bounded/s3_fifo.rs` | `op_strategy_with_get_mut` | `check_invariants`, residency | — |
| TTL | stub | composed | DualRun | [ttl.md](ttl.md) | `exact/lru.rs` (`LruOccupancyModel`) | — | — | TTL integration traces | `LruOccupancyModel` + deadlines | — |
| Random | — | — | — | — | — | — | — | — | — | — |

**Status:** Random policy is deferred (no semantic harness).

## Harness modes

| Mode | Oracle |
|------|--------|
| **DualRun** | `exact/` `PolicyModel` vs implementation |
| **CrossModel** | `reference/` vs `exact/` (all exact-tier policies with reference models) |
| **InvariantOnly** | `len <= capacity` + `debug_validate_invariants` / `check_invariants` |

## Spec maturity

| Level | Meaning |
|-------|---------|
| `stub` | Transcribed from executable oracle; independent reference pending |
| `reference` | Independent naive `reference/` model exists |
| `tla` | Machine-readable TLA+ module + TLC runbook |

## Related

- [template.md](template.md) — spec skeleton for new policies
- [tla-guide.md](tla-guide.md) — adding TLA+ specs
- [README.md](README.md) — pipeline and checklist
