# AdaptSize

## Goal
Size-aware admission policy that maximizes object hit rate (OHR) rather than byte hit rate (BHR).

## Core Idea
Combine size-aware eviction (e.g., GDSF) with an admission policy that learns
which object sizes should be admitted:
- Track exponentially weighted moving average (EWMA) of hit rates per size class
- Admit object of size `s` with probability based on learned benefit
- Adapts to workload: if small objects have better hit rates, prefer admitting them

Typical implementation:
- Partition sizes into classes (e.g., log-scale buckets)
- Maintain per-class admission probability
- Update probabilities via EWMA on hits/misses

## Core Data Structures
- Base cache with size-aware eviction (e.g., GDSF)
- Admission table: `size_class -> admission_probability`
- Per-class EWMA state for hit rate estimation

## Complexity & Overhead
- O(1) admission decision (lookup + RNG)
- Extra memory: small fixed-size table for size classes
- Adapts over time via EWMA updates

## When To Use
- Variable object sizes (CDN, object cache, HTTP proxy)
- Want to optimize object hit rate, not byte hit rate
- Complement to GDSF or other size-aware eviction
- Workloads where size correlates with reuse (common: small objects are hotter)

## Implementation Notes For CacheKit
- Pair with GDSF eviction for full size-aware stack
- Choose size class granularity (e.g., powers of 2)
- Tune EWMA decay rate for adaptation speed
- Optional: combine with frequency-based admission (TinyLFU-style)

## References
- Berger et al. (2017): "AdaptSize: Orchestrating the Hot Object Memory Cache in a Content Delivery Network", NSDI 2017.
- Paper: https://www.usenix.org/system/files/conference/nsdi17/nsdi17-berger.pdf
