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
- [Cache trait hierarchy](design/trait-hierarchy.md) — Kernel trait, capability traits, read/mutate split
- [Concurrency](design/concurrency.md) — `Concurrent*` wrappers, lock discipline, sharded primitives
- [Builder and runtime dispatch](design/builder-and-dyn-dispatch.md) — `CachePolicy`, `DynCache`, enum dispatch
- [Weighted eviction](design/weighted-eviction.md) — `WeightStore`, dual limits, GDS/GDSF pre-staging
- [Metrics](design/metrics.md) — Recorder / snapshot / exporter split, Prometheus integration
- [Error model](design/error-model.md) — Panic vs `Result` discipline, four error types
- [Benchmarking design](design/benchmarking.md) — Benchmark layers, policy registry, JSON artifacts
- [Hashing and key identity](design/hashing.md) — Hasher choices, key interning, shard routing
- [Sharding](design/sharding.md) — Sharded primitives, routing, capacity semantics
- [Storage layer](design/storage.md) — Store trait family, concrete stores, `StoreMetrics` baseline
- [Serialization](design/serialization.md) — `serde` surface and cache-state persistence boundaries
- [Non-goals](design/non-goals.md) — Explicit boundaries and out-of-scope features
- [TTL design](design/ttl.md) — Worked example of every principle in one feature
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
- [Testing catalog](testing/catalog.md) — test types, current coverage, and gaps
- [Policy semantic testing](testing/static-analysis.md)
- [Adding fuzz targets](testing/adding-fuzz-targets.md)
