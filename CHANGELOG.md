# Changelog

All notable changes to cachekit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-02-18

### Added
- CAR (Clock with Adaptive Replacement) cache policy (`CarCache`) combining ARC-like adaptivity with Clock mechanics for enhanced concurrency and reduced overhead.
- Policy feature flags for modular builds:
  - `policy-fifo`, `policy-lru`, `policy-fast-lru`, `policy-lru-k`, `policy-lfu`, `policy-heap-lfu`
  - `policy-two-q`, `policy-s3-fifo`, `policy-arc`, `policy-car`, `policy-lifo`, `policy-mfu`
  - `policy-mru`, `policy-random`, `policy-slru`, `policy-clock`, `policy-clock-pro`, `policy-nru`
  - `policy-all` feature for enabling all policies at once
- Default features now include: `policy-s3-fifo`, `policy-lru`, `policy-fast-lru`, `policy-lru-k`, `policy-clock`
- CAR policy integration in `CacheBuilder` for convenient cache creation
- TinyLFU/W-TinyLFU added to roadmap for future implementation

### Documentation
- Complete CAR policy documentation (`docs/policies/car.md`) with architecture, operations, and performance trade-offs
- Enhanced README with comprehensive Table of Contents and Quick Start section
- Updated policy documentation to include feature flag requirements for each policy
- Improved integration guide with feature flag usage examples
- Added TinyLFU/W-TinyLFU to policy roadmap
- New compatibility and features guide (`docs/guides/compatibility-and-features.md`)

### Changed
- Default feature set now includes specific policies instead of all policies, enabling smaller builds
- Benchmark support now uses `policy-all` feature for comprehensive testing
- Policy module organization updated to support conditional compilation
- Documentation consistently references feature flags for policy enablement

### Benefits
- **Modular Builds**: Enable only the policies you need for smaller binary sizes
- **Adaptive Eviction**: CAR policy provides ARC-like adaptivity with better concurrency
- **Flexible Configuration**: Choose between comprehensive defaults or minimal custom builds

## [0.3.0] - 2026-02-11

### Added
- Read-only trait support for side-effect-free cache inspection:
  - `ReadOnlyCache<K, V>` base trait for immutable inspection operations
  - `ReadOnlyFifoCache<K, V>` for FIFO-specific inspection (peek_oldest, age_rank)
  - `ReadOnlyLruCache<K, V>` for LRU-specific inspection (peek_lru, recency_rank)
  - `ReadOnlyLfuCache<K, V>` for LFU-specific inspection (peek_lfu, frequency)
  - `ReadOnlyLruKCache<K, V>` for LRU-K-specific inspection (peek_lru_k, k_distance, access_history)
- Documentation guide for read-only traits (`docs/guides/read-only-traits.md`)
- Read-only trait exports in prelude for convenient access

### Changed
- `CoreCache` now extends `ReadOnlyCache` to inherit immutable inspection methods
- Updated trait hierarchy to separate read-only operations from write operations
- Enhanced architecture diagrams and trait documentation to show read-only pattern

### Benefits
- **No Side Effects**: Inspection operations don't trigger evictions or update access patterns
- **Better Concurrency**: Multiple readers can use shared `&self` references with read locks
- **Clear API Intent**: Function signatures signal whether cache state will be modified
- **Testing Support**: Examine cache state without affecting test outcomes

## [0.2.0] - 2026-01-31

### Added
- MFU (Most Frequently Used) cache policy (`MfuCache`) with frequency-based eviction that keeps most frequently accessed items.
- LIFO (Last In, First Out) cache policy (`LifoCache`) with stack-based eviction.
- Random eviction cache policy (`RandomCache`) with XorShift64 RNG for unpredictable eviction patterns.
- MRU (Most Recently Used) cache policy (`MruCache`) with recency-based eviction of most recently accessed items.
- Segmented LRU (SLRU) cache policy (`SlruCache`) with probationary and protected segments for scan resistance.
- Clock cache replacement policy (`ClockCache`) implementing the Clock (Second-Chance) algorithm with O(1) access operations.
- Clock-PRO cache policy (`ClockProCache`) implementing the advanced Clock algorithm with hot/cold/test page classification.
- NRU (Not Recently Used) cache policy (`NruCache`) with simple reference bit tracking.
- Example programs: `basic_mfu.rs`, `basic_lifo.rs`, `basic_random.rs`, `basic_mru.rs`, `basic_slru.rs`, `basic_clock.rs`, `basic_clock_pro.rs`, `basic_nru.rs`.
- Invariant validation for cache policies to ensure internal consistency during operations.
- Regression test for FIFO behavior in `FrequencyBuckets`.

### Documentation
- Complete documentation for MFU policy (`docs/policies/mfu.md`) with architecture and usage patterns.
- Complete documentation for LIFO policy (`docs/policies/lifo.md`) with stack semantics and use cases.
- Complete documentation for Random eviction policy (`docs/policies/random.md`) with RNG strategy.
- Complete documentation for MRU policy (`docs/policies/mru.md`) with recency-based eviction.
- Complete documentation for SLRU policy (`docs/policies/slru.md`) with segment management and promotion strategies.
- Complete documentation for Clock policy (`docs/policies/clock.md`) with second-chance algorithm details.
- Complete documentation for Clock-PRO policy (`docs/policies/clock-pro.md`) with hot/cold/test page management.
- Complete documentation for NRU policy (`docs/policies/nru.md`) with reference bit tracking.
- Updated examples in cache policy documentation to use integer keys and values for consistency.

### Changed
- Random eviction policy now uses XorShift64 for RNG state management, improving performance and predictability.
- Updated dependencies and enhanced benchmarking structure for better performance analysis.

### Removed
- Legacy performance tests for LFU, LRU-K, and LRU policies (consolidated into newer test suite).

## [0.2.0-alpha] - 2026-01-19

### Added
- S3-FIFO cache policy (`S3FifoCache`) implementing the scan-resistant FIFO variant with small/main/ghost queues.
- Clock-PRO cache policy (`ClockProCache`) implementing the advanced Clock algorithm with hot/cold/test page classification.
- Clock cache replacement policy (`ClockCache`) implementing the Clock (Second-Chance) algorithm with O(1) access operations.
- `FastLru` policy for single-threaded performance with cache-line optimized layout and direct value storage.
- S3-FIFO benchmarks (`benches/s3_fifo.rs`) with workload generators.
- Comparison benchmarks (`benches/comparison.rs`) for evaluating cache policies against external libraries (moka, quick_cache).
- Human-readable benchmark reports (`benches/reports.rs`) for cache policy comparison tables without criterion overhead.
- New workload generators for benchmarking: `ScrambledZipfian`, `Latest`, `ShiftingHotspot`, `Exponential`.
- `rand` and `rand_distr` dependencies for workload simulation.
- Unified cache builder (`CacheBuilder`) with support for all eviction policies (FIFO, LRU, LRU-K, LFU, HeapLFU, 2Q).
- New 2Q eviction policy (`TwoQCore`) with configurable probation/protected queue ratios.
- Example programs: `basic_s3_fifo.rs` for S3-FIFO cache policy, `basic_two_q.rs` for 2Q cache policy, `basic_builder.rs` for CacheBuilder API usage.
- DHAT heap profiling binary (`dhat_profile`) for memory analysis.
- Integration guide documentation (`docs/integration.md`).
- Documentation tests added to CI workflow.
- `ConcurrentStoreRead` trait for read-only concurrent store operations.
- `StoreFactory` and `ConcurrentStoreFactory` traits for creating store instances.

### Documentation
- S3-FIFO cache policy documentation (`docs/policies/s3-fifo.md`) with architecture, queue management, and ghost filtering.
- Clock policy documentation moved from roadmap to main policies (`docs/policies/clock.md`).
- Benchmark README enhanced with detailed performance insights, hit rate comparisons, and policy selection guide.
- 2Q cache policy documentation (`docs/policies/2q.md`) with goals, data structures, operations, and complexity analysis.
- README enhanced with Quick Start section and examples for all eviction policies.
- Complete documentation for `src/policy/lru_k.rs`:
  - Architecture diagram showing cache layout and K-distance calculation.
  - Scan resistance explanation with before/after diagrams.
  - Docstrings with examples for `LrukCache`, `new()`, `with_k()`, and all trait methods.
  - Private method docstrings with complexity notes.
- Complete documentation for `src/policy/lfu.rs`:
  - Architecture diagram showing frequency buckets and eviction flow.
  - LFU vs LRU comparison diagram.
  - Docstrings with examples for `LfuCache`, `LFUHandleCache`, and all public methods.
  - Batch operation examples (`insert_batch`, `remove_batch`, `touch_batch`).
  - Metrics snapshot documentation.
- Complete documentation for `src/policy/heap_lfu.rs`:
  - Architecture diagram showing min-heap structure and stale entry handling.
  - Standard LFU vs Heap LFU performance comparison.
  - Docstrings with examples for `HeapLfuCache` and all public methods.
  - Private method docstrings explaining lazy deletion and heap rebuild strategy.
- Documentation enhancements for data structures: ClockRing, FrequencyBuckets, GhostList, KeyInterner, IntrusiveList, LazyMinHeap, SlotArena, FixedHistory, ShardSelector.
- Documentation enhancements for store implementations: HandleStore, SlabStore, HashMapStore, WeightStore.
- Documentation enhancements for cache policies: LRU, TwoQ.
- Documentation enhancements for cache traits and CacheBuilder.

### Changed
- FIFO policy simplified to single implementation (removed separate metrics variant).
- `dhat_profile` moved from `src/bin/` to `examples/` directory.
- Bumped `lru` dependency from 0.12.5 to 0.16.3.
- Switched to `rustc_hash::FxHashMap` for improved hashing performance across data structures.
- Raw pointer linked lists in `LruCore` and `LrukCache` for improved cache locality and reduced indirection.
- Enhanced `TwoQCore` implementation with helper methods (`detach_from_probation`, `attach_to_protected`) and comprehensive tests.
- Refactored store traits to separate single-threaded and concurrent ownership models:
  - `StoreCore`/`StoreMut` now use direct value ownership (`&V`, `V`) for zero-overhead single-threaded access.
  - `ConcurrentStore` uses `Arc<V>` for safe shared ownership across threads.
- `HashMapStore` and `SlabStore` now store values directly (not `Arc`-wrapped) for single-threaded use.
- `HandleStore` and `WeightStore` no longer implement generic store traits; they expose specialized `Arc<V>` APIs directly.
- All cache policies updated to work with the new trait structure.

### Removed
- `manager` module (placeholder `cache_manager` removed).
- `StoreEvictionHook` struct (eviction recording moved to direct method calls).
- `#[must_use]` attributes on `try_insert` methods (redundant with `Result`).

### Fixed
- Removed flaky conflict rate assertion from LFU high-contention stress test for Windows compatibility.

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
