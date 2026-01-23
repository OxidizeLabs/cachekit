# CacheKit Docs

CacheKit is a Rust library for building high-performance caches with pluggable eviction policies and cache policies and supporting data structures.

Key features:
- Multiple eviction policies (LRU, LFU, LRU-K, 2Q, Clock, S3-FIFO)
- Tiered cache building blocks with a unified builder API
- Optional metrics and benchmark tooling

## Getting Started

- [Quickstart](quickstart.md) — Install and build your first cache
- [Integration guide](integration.md) — CacheBuilder API, policy selection, thread safety
- [Design overview](design.md) — Architectural decisions and performance principles
- [API surface](api-surface.md) — Module map and entrypoints

## Policies

- [Policy overview](policies/README.md) — Implemented policies
- [Policy roadmap](policies/roadmap/README.md) — Future policies
- [Choosing a policy](choosing-a-policy.md) — Practical selection guide
- [Glossary](glossary.md) — Shared terminology

## Internals

- [Stores](stores/README.md) — Storage backends
- [Policy data structures](policy-ds/README.md) — Implementation details
- [Benchmarks](benchmarks.md) — Performance benchmarks
- [Workloads](workloads.md) — Synthetic workload generators for benchmarking

## Benchmarks

- [Benchmark quickstart](benchmarks/QUICKSTART.md) — View and run benchmarks
- [Benchmark docs](benchmarks/README.md) — Reports, artifacts, and publishing

## Release and Maintenance

- [Release checklist](release-checklist.md)
- [Releasing CacheKit](releasing.md)
- [CI/CD release cycle](ci-cd-release-cycle.md)
- [CD/CI continuous fuzzing](fuzzing-cicd.md)
- [Documentation style guide](style-guide.md)
- [Compatibility and features](compatibility-and-features.md)
- [FAQ and gotchas](faq.md)

## Testing and Fuzzing

- [Testing strategy](testing.md)
- [Adding fuzz targets](adding-fuzz-targets.md)
