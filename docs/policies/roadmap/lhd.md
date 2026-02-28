# LHD (Learning Hit Density)

## Goal
Maximize value per byte by using a learned estimate of hit probability over
remaining lifetime, especially when object sizes vary substantially.

## Core Idea
LHD estimates each object's future hit density (roughly expected future hits per
byte over a horizon) from observed reuse behavior. Eviction prefers entries with
the lowest estimated hit density.

Compared to simple size-aware scoring, LHD adapts estimates from workload data
instead of relying on fixed formulas.

## Core Data Structures (Typical)
- Hash index `K -> EntryMeta`
- Size-aware entry metadata (`size`, age/timestamp, access stats)
- Lightweight learned model/state (bucketized statistics tables)
- Victim selector keyed by estimated hit density

## Complexity & Overhead
- Higher implementation complexity than GDS/GDSF
- Additional memory for model tables and per-entry predictors
- Runtime cost depends on estimator complexity (can be O(1) with table lookups)

## Notes For CacheKit
- High-value candidate for byte hit rate optimization and heterogeneous object sizes.
- Start with coarse bucketed estimators to keep hot paths predictable.
- Benchmark against `GDS`, `GDSF`, and `Hyperbolic` on mixed-size traces.

## References
- Beckmann et al. (2018): “LHD: Improving Cache Hit Rate by Maximizing Hit Density”.
- USENIX OSDI 2018 paper and follow-up implementations.
