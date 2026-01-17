# Changelog

All notable changes to cachekit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Unified cache builder (`CacheBuilder`) with support for all eviction policies (FIFO, LRU, LRU-K, LFU, HeapLFU, 2Q).
- New 2Q eviction policy (`TwoQCore`) with configurable probation/protected queue ratios.
- `ConcurrentStoreRead` trait for read-only concurrent store operations.
- `StoreFactory` and `ConcurrentStoreFactory` traits for creating store instances.

### Changed
- Refactored store traits to separate single-threaded and concurrent ownership models:
  - `StoreCore`/`StoreMut` now use direct value ownership (`&V`, `V`) for zero-overhead single-threaded access.
  - `ConcurrentStore` uses `Arc<V>` for safe shared ownership across threads.
- `HashMapStore` and `SlabStore` now store values directly (not `Arc`-wrapped) for single-threaded use.
- `HandleStore` and `WeightStore` no longer implement generic store traits; they expose specialized `Arc<V>` APIs directly.
- All cache policies updated to work with the new trait structure.

### Removed
- `StoreEvictionHook` struct (eviction recording moved to direct method calls).
- `#[must_use]` attributes on `try_insert` methods (redundant with `Result`).

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
