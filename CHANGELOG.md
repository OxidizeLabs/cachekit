# Changelog

All notable changes to cachekit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet

### Changed
- Nothing yet

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Nothing yet

### Security
- Nothing yet

## [0.1.0-alpha] - 2026-01-13

### Added
- Documentation index in `README.md`.
- Example programs for LRU, LRU-K, LFU, and Heap LFU in `examples/`.
- Handle-based store (`HandleStore`, `ConcurrentHandleStore`) for zero-copy keys.
- Concurrent slab store (`ConcurrentSlabStore`) with EntryId indirection.
- Weight-aware store (`WeightStore`, `ConcurrentWeightStore`) for size-based limits.
- HashMap stores now support custom hashers (`BuildHasher`).
- Workload-style hit-rate benchmarks and grouped policy/micro-op benches.
- Documentation style guide (`docs/style-guide.md`) and expanded module docs.

### Changed
- Example comments now include expected output and brief explanations.
- Benchmarks grouped by end-to-end, policy, micro-ops, and workloads.
- LFU performance thresholds adjusted for debug/noisy environments.

### Fixed
- Clippy warnings across benches and store modules.
