# LeCaR (Learning Cache Replacement)

## Goal
Adapt eviction behavior online by learning when recency-focused or
frequency-focused decisions work better for the current workload.

## Core Idea
LeCaR treats eviction as a lightweight online-learning problem:
- Keep two experts (commonly LRU-like and LFU-like signals).
- Maintain weights for each expert based on recent regret/reward.
- On eviction pressure, pick victims according to the weighted mixture.

Under shifting workloads, weights move toward the better-performing strategy.

## Core Data Structures (Typical)
- Hash index `K -> EntryMeta`
- Recency structure (LRU list or clock-style metadata)
- Frequency counters (or compact approximate counters)
- Learning state (expert weights, step size, recent feedback)

## Complexity & Overhead
- Extra per-operation arithmetic for weight updates
- More metadata than single-policy designs
- Performance depends on stable, well-tuned feedback signals

## Notes For CacheKit
- Best viewed as a research/benchmark policy before production defaulting.
- Keep learning state compact and avoid floating-point work in hot loops if possible.
- Evaluate under non-stationary workloads with hotspot shifts.

## References
- Vietri et al. (2018): “Driving cache replacement with ML-based LeCaR”.
- Wikipedia (taxonomy context): https://en.wikipedia.org/wiki/Cache_replacement_policies
