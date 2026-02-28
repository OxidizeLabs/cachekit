# Hyperbolic Caching

## Goal
Combine recency and frequency in a compact, practical scoring model with good
hit-rate behavior under skewed and shifting workloads.

## Core Idea
Each resident entry gets a score that decays with age and increases with
frequency. A common formulation is proportional to:
- `score = freq / age`

Where:
- `freq`: access count or weighted access signal
- `age`: time or logical ticks since insertion/update

Eviction removes the smallest score, so stale/low-value entries fall out first.

## Core Data Structures (Typical)
- Hash index `K -> EntryMeta`
- Per-entry counters: `freq`, `last_touch` (or logical timestamp)
- Victim selector:
  - exact ordered structure by score, or
  - approximate buckets/periodic rescoring for lower overhead

## Complexity & Overhead
- Exact score ordering is often O(log n) per score-changing update
- Approximate variants trade precision for lower CPU cost
- Metadata is moderate: one frequency counter + age/timestamp per entry

## Notes For CacheKit
- Good candidate when plain LRU underperforms and full ML-style policies are too heavy.
- Prefer monotonic logical clocks and integer math on hot paths.
- Benchmark against `LRU`, `Heap-LFU`, `S3-FIFO`, and `TinyLFU/W-TinyLFU`.

## References
- Blankstein et al.: “Hyperbolic Caching: Flexible Caching for Web Applications”.
- Wikipedia (taxonomy context): https://en.wikipedia.org/wiki/Cache_replacement_policies
