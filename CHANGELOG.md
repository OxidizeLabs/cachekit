# Changelog

All notable changes to cachekit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- S3-FIFO cache policy (`S3FifoCache`) implementing the scan-resistant FIFO variant with small/main/ghost queues.
- Clock-PRO cache policy (`ClockProCache`) implementing the advanced Clock algorithm with hot/cold/test page classification.
- Clock cache replacement policy (`ClockCache`) implementing the Clock (Second-Chance) algorithm with O(1) access operations.
- `FastLru` policy for single-threaded performance with cache-line optimized layout and direct value storage.
- S3-FIFO benchmarks (`benches/s3_fifo.rs`) with workload generators.
- Comparison benchmarks (`benches/comparison.rs`) for evaluating cache policies against external libraries (moka, quick_cache).
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
