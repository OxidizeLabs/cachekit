# Cache Replacement Policies

This document summarizes common cache replacement (eviction) policies, their tradeoffs, and when to use (or avoid) each. It’s written as a practical companion to `docs/design.md`.

Implementation notes live in the per-policy docs under `docs/policies/` and the data-structure docs under `docs/policy-ds/`.

Terminology used below:
- **Admission**: whether an item is allowed into cache at all (some “policies” combine admission + eviction).
- **Eviction**: which resident item to remove when making space.
- **Scan pollution**: one-time accesses (e.g., large scans) pushing out genuinely hot items.
- **Metadata cost**: extra per-entry memory and CPU needed to maintain the policy.

## How To Choose (Quick Guidance)

These recommendations mirror the latest benchmark guide in `docs/benchmarks/latest/index.md`.

Pick based on workload first:
- **Strong temporal locality + low latency**: `LRU` or `Clock` (fastest in benchmarks).
- **One-hit-wonder dominated / scan-heavy**: `S3-FIFO` or `Heap-LFU`; `LRU-K` or `2Q` for mixed scans + reuses.
- **Frequency matters more than recency** (hot keys repeat with long gaps): `LFU` or `Heap-LFU`; `LRU-K` for multi-access signals.
- **Need low overhead & simple**: `LRU` or `Clock` (fast ops, minimal metadata); `FIFO`/`Random` when simplicity trumps hit rate.
- **Need adaptive across shifting workloads**: `S3-FIFO` or `2Q`.

If you can only implement one “general purpose” policy for mixed workloads, `S3-FIFO` or `LRU` are the strongest defaults in current benchmarks, with `2Q` as a good alternative when scans are common.

## Policy Catalog (Summaries)

### OPT / MIN (Belady’s Optimal)

**Idea**: evict the item whose *next* use is farthest in the future.

- **Pros**: best possible hit rate for a known future; gold-standard for evaluation.
- **Cons**: requires knowing the future; not implementable in real systems (except offline traces/simulators).
- **Use when**: benchmarking and comparing other policies on recorded traces.
- **Avoid when**: building a real cache.

### Random

**Idea**: evict a uniformly random resident item.

- **Pros**: trivial; very low overhead; surprisingly decent under some adversarial patterns.
- **Cons**: ignores locality; unstable hit rate; can evict hot items.
- **Use when**: you need the simplest possible eviction with constant overhead.
- **Avoid when**: locality exists and you can afford minimal tracking.

### FIFO (First-In, First-Out)

**Idea**: evict the oldest inserted item.

- **Pros**: simple; O(1); predictable; low metadata.
- **Cons**: ignores reuse; can be very poor when early inserts stay hot.
- **Use when**: insert order correlates with staleness (e.g., streaming-ish workloads), or you want predictability.
- **Avoid when**: strong temporal locality; “old but hot” keys are common.

### LIFO / FILO (Last-In, First-Out)

**Idea**: evict the most recently inserted item.

- **Pros**: can work for some cyclic/scan-like patterns where newest items are least likely to be reused.
- **Cons**: counterproductive under temporal locality; uncommon in general-purpose caches.
- **Use when**: you have evidence newest items are least reusable.
- **Avoid when**: typical request caches with recency locality.

### LRU (Least Recently Used)

**Idea**: evict the item not accessed for the longest time.

- **Pros**: strong default for temporal locality; intuitive; stable.
- **Cons**: vulnerable to scan pollution; maintaining exact LRU can be costly under high concurrency.
- **Use when**: workloads have strong recency locality; you can tolerate metadata and updates on every access.
- **Avoid when**: large sequential scans are common; cache is highly contended and strict ordering is too expensive.

### MRU (Most Recently Used)

**Idea**: evict the most recently accessed item.

- **Pros**: can outperform LRU for some “looping scan” patterns where the just-touched item won’t be reused soon.
- **Cons**: performs poorly for typical temporal locality.
- **Use when**: known cyclic access where the most-recently-used item is least likely to be reused next.
- **Avoid when**: you’re unsure; MRU is rarely a safe default.

### Second-Chance / Clock

**Idea**: approximate LRU using a circular list and a referenced bit; give items a “second chance”.

- **Pros**: O(1) amortized; lower overhead than strict LRU; good concurrency properties in practice.
- **Cons**: approximation quality depends on implementation; still suffers from scan pollution in many forms.
- **Use when**: you want LRU-like behavior with cheaper metadata and fewer writes.
- **Avoid when**: you specifically need scan resistance or frequency awareness.

### NRU (Not Recently Used)

**Idea**: evict an item whose “referenced” bit is not set; bits are periodically cleared (epochs).

- **Pros**: very low overhead; works well when you can batch/reset reference bits cheaply.
- **Cons**: coarse recency signal; behavior depends heavily on epoch length.
- **Use when**: you already have hardware/software reference bits or can cheaply track “touched this epoch”.
- **Avoid when**: you need tight recency ordering.

### LFU (Least Frequently Used)

**Idea**: evict the item with the smallest access count.

- **Pros**: strong when popularity is stable and skewed; resists scan pollution better than LRU.
- **Cons**: “cache pollution by history” (once-hot items stick around); needs **aging/decay** to adapt; counters add overhead.
- **Use when**: hot items remain hot for long periods; frequency is the primary predictor.
- **Avoid when**: the hot set shifts quickly; you can’t implement decay/aging safely.

### MFU (Most Frequently Used)

**Idea**: evict the most frequently used item.

- **Pros**: can work in specific “burst then never again” patterns (items that were heavily used are now “done”).
- **Cons**: usually the opposite of what you want; not a general-purpose choice.
- **Use when**: you have evidence “most frequent so far” implies “least likely to be reused now”.
- **Avoid when**: almost always.

### Aging / Decayed LFU (LFU with time decay)

**Idea**: combine frequency with time so old counts lose influence (e.g., periodic halving, exponential decay).

- **Pros**: avoids LFU’s “stale hot” problem; adapts to changing popularity.
- **Cons**: more complexity; decay schedule can be tricky; still more metadata than LRU/Clock.
- **Use when**: you want frequency but with adaptivity to phase changes.
- **Avoid when**: extremely latency-sensitive hot paths where counter maintenance dominates.

### LRU-K

**Idea**: evict based on the K-th most recent access time (e.g., `K=2` tracks the 2nd most recent touch).

- **Pros**: filters one-time accesses; much more scan-resistant than LRU.
- **Cons**: more metadata per entry; more expensive updates; needs careful implementation to stay O(1) in practice.
- **Use when**: mixed point-lookups + scans; DB buffer pools; workloads with many one-hit-wonders.
- **Avoid when**: you need the simplest possible policy or can’t afford per-entry history.

### 2Q

**Idea**: use two queues: a short “probation” FIFO for new items and a main LRU for items that are accessed again.

- **Pros**: simple scan resistance; cheaper than LRU-K; widely used pattern.
- **Cons**: requires tuning queue sizes; still mainly recency-based once admitted to main queue.
- **Use when**: you want an easy scan-resistant upgrade over LRU.
- **Avoid when**: you can’t tolerate tuning knobs or workload changes dramatically.

### SLRU (Segmented LRU)

**Idea**: split LRU into segments (e.g., probationary + protected); promotion requires reuse.

- **Pros**: reduces scan pollution; simple; common in practice.
- **Cons**: needs segment sizing; not as adaptive as ARC-style approaches.
- **Use when**: you want low-complexity scan resistance with LRU semantics.
- **Avoid when**: workload shifts require continual retuning.

### ARC (Adaptive Replacement Cache)

**Idea**: adaptively balances recency vs frequency using two LRU lists plus “ghost” history lists to tune itself.

- **Pros**: strong across many workloads; self-tuning between scan resistance and frequency-ish behavior.
- **Cons**: more complex; more metadata (including ghost entries); harder to implement lock-efficiently.
- **Use when**: you need robust performance across shifting patterns and can afford complexity.
- **Avoid when**: memory overhead must be minimal or implementation complexity is a hard constraint.

### CAR (Clock with Adaptive Replacement)

**Idea**: ARC-like adaptivity but with Clock structures to reduce overhead.

- **Pros**: retains ARC’s adaptivity with lower overhead in some implementations.
- **Cons**: still complex; behavior depends on details.
- **Use when**: you want ARC-like behavior but prefer Clock-style mechanics.
- **Avoid when**: you need simplicity or have no room for ghost/history metadata.

### LIRS (Low Inter-reference Recency Set)

**Idea**: use inter-reference recency (distance between repeated touches) to classify and protect frequently reused items.

- **Pros**: excellent scan resistance in many workloads; strong theoretical grounding.
- **Cons**: implementation complexity; metadata overhead; harder to explain/debug than LRU variants.
- **Use when**: you can invest in a high-quality scan-resistant policy for DB-like workloads.
- **Avoid when**: you need a small, simple policy surface.

### CLOCK-Pro

**Idea**: Clock-based policy that differentiates hot/cold pages and tracks recent history to handle scans better than Clock.

- **Pros**: good scan resistance with Clock mechanics; practical for OS/DB buffer caches.
- **Cons**: more complex than Clock; tuning/implementation details matter.
- **Use when**: you want better-than-Clock scan handling without full ARC machinery.
- **Avoid when**: you want the simplest possible eviction logic.

### Size/Cost-Aware Policies (GDS / GDSF family)

**Idea**: evict based on a “value” score that accounts for retrieval cost and/or object size (common in web caches).

- **Pros**: optimizes for byte hit rate or cost-weighted hit rate; better than LRU when object sizes vary widely.
- **Cons**: more bookkeeping; needs cost/size signals; can be less intuitive.
- **Use when**: objects have large size variance; misses have heterogeneous cost (e.g., network fetch cost).
- **Avoid when**: costs are uniform and you only care about request hit rate.

### TTL / Time-Based Expiration (Not a Replacement Policy)

**Idea**: entries expire after a time-to-live, regardless of recency/frequency.

- **Pros**: bounds staleness; essential for correctness in many domains.
- **Cons**: does not optimize hit rate by itself; still needs an eviction policy when full.
- **Use when**: correctness requires freshness bounds (configs, tokens, CDN-like caching).
- **Avoid when**: you treat TTL as a substitute for eviction optimization.

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

## Reference Material

- Wikipedia: Cache replacement policies: https://en.wikipedia.org/wiki/Cache_replacement_policies
- LRU-K: “The LRU-K page replacement algorithm for database disk buffering” (O’Neil, O’Neil, Weikum), 1993.
- 2Q: “2Q: A Low Overhead High Performance Buffer Management Replacement Algorithm” (Johnson, Shasha), 1994.
- ARC: “ARC: A Self-Tuning, Low Overhead Replacement Cache” (Megiddo, Modha), 2003.
- LIRS: “LIRS: An Efficient Low Inter-reference Recency Set Replacement Policy to Improve Buffer Cache Performance” (Jiang, Zhang), 2002.
- OPT (Belady): “A study of replacement algorithms for a virtual-storage computer” (Belady), 1966.
- GDS/GDSF: “GreedyDual-Size: An algorithm for web caching” (Cao, Irani), 1997.
