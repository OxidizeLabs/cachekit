# TinyLFU / W-TinyLFU (Admission + Eviction Design)

## Goal
Improve hit rate under skewed and one-hit-wonder workloads by using
frequency-based admission instead of admitting every miss.

## Core Idea
TinyLFU is primarily an **admission policy**:
- Keep an approximate recent frequency sketch (usually Count-Min Sketch).
- On a miss, compare candidate frequency against a sampled victim.
- Admit only if the candidate appears "hotter" than the victim.

W-TinyLFU combines:
- **Window cache** (small recency-focused region, typically LRU)
- **Main cache** (segmented/protected region)
- **TinyLFU admission gate** between window and main

This keeps recent bursts responsive while avoiding long-tail pollution.

## Core Data Structures (Typical)
- Hash index `K -> Entry`
- Window segment (e.g., LRU list or ring)
- Main segment (e.g., SLRU-style probation/protected)
- Frequency sketch (Count-Min Sketch)
- Optional reset/aging counter for sketch decay

## Complexity & Overhead
- Access/update in sketch: O(1) with small constant factors
- Admission decision: O(1)
- Extra memory for sketch and segmented metadata
- Approximate counts can produce false positives, but usually good tradeoff

## Notes For CacheKit
- Fits best as **policy composition**: storage + segmented eviction + admission.
- Keep hot-path updates allocation-free (pre-sized sketch, fixed segments).
- Make admission optional/configurable for apples-to-apples benchmarks.
- Benchmark against `S3-FIFO`, `ARC/CAR`, and `Heap-LFU` on:
  - Zipfian
  - scan + point lookup mixes
  - shifting hotspots

## References
- Einziger et al. (2017): "TinyLFU: A Highly Efficient Cache Admission Policy".
- Caffeine design notes (W-TinyLFU implementation details).
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
