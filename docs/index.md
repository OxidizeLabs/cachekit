# CacheKit Docs

CacheKit is a Rust library for building high-performance caches with pluggable eviction policies and supporting data structures.

Key features:
- Multiple eviction policies (LRU, LFU, LRU-K, 2Q, Clock, S3-FIFO)
- Composable cache building blocks with a unified builder API
- Optional metrics and benchmark tooling

## Getting Started

- [Quickstart](getting-started/quickstart.md) — Install and build your first cache
- [Integration guide](getting-started/integration.md) — CacheBuilder API, policy selection, thread safety
- [Design overview](design/design.md) — Architectural decisions and performance principles
- [API surface](guides/api-surface.md) — Module map and entrypoints

## Policies

- [Policy overview](policies/README.md) — Implemented policies
- [Policy roadmap](policies/roadmap/README.md) — Future policies
- [Choosing a policy](guides/choosing-a-policy.md) — Practical selection guide
- [Glossary](guides/glossary.md) — Shared terminology

## Internals

- [Stores](stores/README.md) — Storage backends
- [Policy data structures](policy-ds/README.md) — Implementation details

## Benchmarks

- [Benchmark quickstart](benchmarks/QUICKSTART.md) — View and run benchmarks
- [Benchmark docs](benchmarks/README.md) — Reports, artifacts, and publishing
- [Benchmarks](benchmarks/overview.md) — Performance benchmarks
- [Workloads](benchmarks/workloads.md) — Synthetic workload generators for benchmarking

## Release and Maintenance

- [Release checklist](release/release-checklist.md)
- [Releasing CacheKit](release/releasing.md)
- [CI/CD release cycle](release/ci-cd-release-cycle.md)
- [CD/CI continuous fuzzing](testing/fuzzing-cicd.md)
- [Documentation style guide](design/style-guide.md)
- [Compatibility and features](guides/compatibility-and-features.md)
- [FAQ and gotchas](guides/faq.md)

## Testing and Fuzzing

- [Testing strategy](testing/testing.md)
- [Adding fuzz targets](testing/adding-fuzz-targets.md)
