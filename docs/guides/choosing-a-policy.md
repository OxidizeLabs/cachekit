# Choosing a Policy

This guide summarizes practical trade-offs and mirrors the benchmark-driven guidance
in the [latest benchmark guide](../benchmarks/latest/index.md).

**Feature flags:** Each policy is gated behind a feature flag (e.g. `policy-lru`, `policy-s3-fifo`). Enable only the policies you need for smaller builds. See [Compatibility and Features](compatibility-and-features.md).

## Quick Picks

- **General purpose, skewed workloads**: `LRU` or `S3-FIFO`
- **Scan-heavy workloads**: `S3-FIFO` or `Heap-LFU`
- **Low latency required**: `LRU` or `Clock`
- **Memory constrained**: `LRU` or `Clock`
- **Frequency-aware**: `LFU`, `Heap-LFU`, or `LRU-K`
- **Shifting patterns**: `S3-FIFO` or `2Q`
- **Mixed one-hit + frequent**: `2Q` or `S3-FIFO`

## Policy Summaries

- **LRU**: Great default for temporal locality; fast; scan-vulnerable.
- **Clock**: LRU-like with lower overhead; similar latency to LRU.
- **S3-FIFO**: Strong scan resistance with low overhead; solid default for mixed workloads.
- **LFU / Heap-LFU**: Frequency-driven; stable hot sets; slower to adapt.
- **LRU-K**: Strong scan resistance; more metadata per entry.
- **2Q**: Simple scan resistance; needs queue sizing.

## Deep Dives

- [Implemented policies](../policies/README.md)
- [Policy roadmap](../policies/roadmap/README.md)
