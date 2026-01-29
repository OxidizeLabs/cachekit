# Cache Replacement Policies

This document summarizes common cache replacement (eviction) policies, their tradeoffs, and when to use (or avoid) each. It’s written as a practical companion to [Design overview](../design/design.md).

Implementation notes live in the [per-policy docs](./) and the [policy data structures](../policy-ds/README.md).

Terminology used below:
- **Admission**: whether an item is allowed into cache at all (some “policies” combine admission + eviction).
- **Eviction**: which resident item to remove when making space.
- **Scan pollution**: one-time accesses (e.g., large scans) pushing out genuinely hot items.
- **Metadata cost**: extra per-entry memory and CPU needed to maintain the policy.

## How This Doc Is Organized

- Quick guidance and a simple decision flow
- Policy catalog (with implemented vs roadmap separation)
- Practical tradeoffs and rules of thumb

## How To Choose (Quick Guidance)

These recommendations mirror the latest benchmark guide in [Benchmarks](../benchmarks/latest/index.md).
Results depend on workload shape, cache size, and access distribution.

Pick based on workload first:
- **Strong temporal locality + low latency**: `LRU` or `Clock` (fastest in benchmarks).
- **One-hit-wonder dominated / scan-heavy**: `S3-FIFO` or `Heap-LFU`; `LRU-K` or `2Q` for mixed scans + reuses.
- **Frequency matters more than recency** (hot keys repeat with long gaps): `LFU` or `Heap-LFU`; `LRU-K` for multi-access signals.
- **Need low overhead & simple**: `LRU` or `Clock` (fast ops, minimal metadata); `FIFO`/`Random` when simplicity trumps hit rate.
- **Need adaptive across shifting workloads**: `S3-FIFO` or `2Q`.

If you can only implement one “general purpose” policy for mixed workloads, `S3-FIFO` or `LRU` are the strongest defaults in current benchmarks, with `2Q` as a good alternative when scans are common.

## Policy Catalog (Summaries)

### Implemented Policies (CacheKit)

| Policy | Summary | Doc |
|--------|---------|-----|
| LRU | Strong default for temporal locality | [LRU doc](lru.md) |
| MRU | Evicts most recent (niche: cyclic patterns) | [MRU doc](mru.md) |
| SLRU | Segmented LRU with probation/protected | [SLRU doc](slru.md) |
| LFU | Frequency-driven, stable hot sets | [LFU doc](lfu.md) |
| Heap-LFU | LFU with heap eviction | [Heap-LFU doc](heap-lfu.md) |
| LRU-K | Scan-resistant recency | [LRU-K doc](lru-k.md) |
| 2Q | Probation + protected queues | [2Q doc](2q.md) |
| FIFO | Simple insertion-order | [FIFO doc](fifo.md) |
| Clock | Approximate LRU | [Clock doc](clock.md) |
| Clock-PRO | Scan-resistant Clock variant | [Clock-PRO doc](clock-pro.md) |
| S3-FIFO | Scan-resistant FIFO | [S3-FIFO doc](s3-fifo.md) |

### Roadmap Policies (Planned)

See [Policy roadmap](roadmap/README.md) for upcoming policies (ARC, CAR, LIRS, etc.).

### Implemented Policy Summaries (Short)

- **LRU**: Strong default for temporal locality; fast; scan-vulnerable.
- **MRU**: Opposite of LRU; evicts most recent; only for specific cyclic patterns.
- **SLRU**: Segmented LRU; simple scan resistance via probation/protected segments.
- **Clock**: LRU-like with lower overhead; good for low-latency paths.
- **S3-FIFO**: Scan-resistant FIFO with ghost history; strong general-purpose choice.
- **LFU / Heap-LFU**: Frequency-driven; stable hot sets; slower to adapt.
- **LRU-K**: Good scan resistance; more metadata per entry.
- **2Q**: Simple scan resistance; requires queue sizing.
- **FIFO**: Predictable insertion order; weak under strong locality.
- **Clock-PRO**: Scan-resistant Clock variant; more complexity.

For broader policy taxonomy (OPT, ARC, CAR, LIRS, Random, etc.), use the
[Policy roadmap](roadmap/README.md) and reference material below.

## Practical Tradeoffs (What Changes In Real Systems)

- **Scan resistance**: `LRU`/`Clock` are vulnerable; `S3-FIFO`, `Heap-LFU`, `LRU-K`, and `2Q` handle scans better.
- **Metadata & CPU**: `Random`/`FIFO` < `Clock` < `LRU` < `2Q`/`SLRU` < `LRU-K`/`ARC`/`LIRS`.
- **Concurrency**: strict global `LRU` lists can contend; `Clock` and sharded designs often scale better.
- **Adaptivity**: `LFU` needs decay to adapt; `ARC`-family adapts via history; static partitions (`2Q`/`SLRU`) need tuning.
- **Predictability**: simpler policies are easier to reason about under tail-latency constraints; complex policies can have more edge cases.

## When To Use / Not Use (Rules Of Thumb)

- Use `LRU` when you have **temporal locality** and need **low latency**; it is consistently fast in benchmarks.
- Prefer `Clock` when you want **LRU-like** behavior with **lower overhead** and similar latency.
- For **scan-heavy** workloads, avoid plain `LRU`; prefer `S3-FIFO` or `Heap-LFU`, with `2Q` or `LRU-K` as alternatives.
- Use `LFU`/`Heap-LFU` when **frequency is predictive** and the hot set is stable; expect slower adaptation to shifts.
- For **shifting patterns**, `S3-FIFO` or `2Q` adapts better in benchmarks than frequency-only policies.
- Use cost/size-aware policies (GDS/GDSF) when optimizing **byte hit rate** or **miss cost**, not just request count.

## Quick Decision Flow

- Need lowest latency? Start with `LRU` or `Clock`.
- Scan-heavy workloads? Prefer `S3-FIFO` or `Heap-LFU`.
- Frequency matters? Use `LFU`/`Heap-LFU` or `LRU-K`.
- Shifting patterns? Try `S3-FIFO` or `2Q`.

## See Also

- [Choosing a policy](../guides/choosing-a-policy.md)
- [Benchmarks overview](../benchmarks/overview.md)
- [Benchmark workloads](../benchmarks/workloads.md)

## Reference Material

- Wikipedia: Cache replacement policies: https://en.wikipedia.org/wiki/Cache_replacement_policies
- LRU-K: “The LRU-K page replacement algorithm for database disk buffering” (O’Neil, O’Neil, Weikum), 1993.
- 2Q: “2Q: A Low Overhead High Performance Buffer Management Replacement Algorithm” (Johnson, Shasha), 1994.
- ARC: “ARC: A Self-Tuning, Low Overhead Replacement Cache” (Megiddo, Modha), 2003.
- LIRS: “LIRS: An Efficient Low Inter-reference Recency Set Replacement Policy to Improve Buffer Cache Performance” (Jiang, Zhang), 2002.
- OPT (Belady): “A study of replacement algorithms for a virtual-storage computer” (Belady), 1966.
- GDS/GDSF: “GreedyDual-Size: An algorithm for web caching” (Cao, Irani), 1997.
