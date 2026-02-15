# LFRU / LRFU (Least Recently/Frequently Used)

## Goal
Unified recency-frequency policy with tunable interpolation parameter.

## Core Idea
Assign each entry a combined recency-frequency (CRF) value:
- `CRF(t) = Σ (1/2)^(λ·(t - tᵢ))` over all past access times `tᵢ`
- Evict entry with smallest CRF

Tuning parameter λ:
- λ=0 → pure LRU (only last access time matters)
- λ→∞ → pure LFU (access count matters, timing doesn't)
- λ=1 (typical) → balanced recency + frequency

Efficient implementation: exponentially decay CRF at each access rather than
summing over full history.

## Core Data Structures
- Hash index `K -> Entry`
- Per-entry: `crf: f64`, `last_update: u64`
- Min-heap or lazy-update priority structure keyed by CRF

## Complexity & Overhead
- O(1) get (hash lookup + decay + update)
- O(log n) eviction if using exact heap
- Can approximate with bucketed priorities for O(1) amortized
- Extra metadata: CRF value + timestamp per entry

## When To Use
- Mixed workloads where both recency and frequency are predictive
- Want single policy that adapts via λ tuning
- Willing to accept float arithmetic or fixed-point approximation

## Implementation Notes For CacheKit
- Pre-choose λ or make it configurable
- Consider fixed-point arithmetic to avoid float instability
- Lazy heap updates to reduce eviction overhead
- Optional: dynamic λ adjustment based on hit rate feedback

## References
- Lee et al. (2001): "LRFU: A Spectrum of Policies that Subsumes the Least Recently Used and Least Frequently Used Policies", IEEE Transactions on Computers.
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
