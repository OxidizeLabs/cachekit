# Changelog

All notable changes to cachekit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking
- `ClockRing::with_hasher` / `ClockRing::try_with_hasher` and the `ConcurrentClockRing` equivalents now require a third argument: a `KeysAreTrusted` marker. This forces every caller that opts out of the default DoS-resistant `RandomState` hasher to acknowledge the trade-off at the call site, making hash-collision-DoS exposure reviewable via `grep KeysAreTrusted`. Callers using `ClockRing::new` / `try_new` are unaffected. Migration: add `, KeysAreTrusted::new()` to existing `with_hasher` / `try_with_hasher` calls.

### Added
- `ClockRing::KeysAreTrusted` — ZST acknowledgement type required by hasher-configurable constructors. Constructible via `KeysAreTrusted::new()` or `KeysAreTrusted::default()`; see its type-level docs for guidance on when a non-randomized hasher is appropriate.
- Re-export `KeysAreTrusted` from `cachekit::ds`.
- `#[track_caller]` on `ClockRing::with_hasher` so `MAX_CAPACITY` panics blame the call site rather than the constructor body.
- `ClockRing::try_clone` — fallible clone that mirrors `try_new` / `try_with_hasher` by routing every backing allocation through `try_reserve_exact` / `try_reserve`, surfacing allocator failure as `ClockRingError::AllocationFailed` instead of aborting the process. Prefer this over `Clone::clone` when cloning with attacker-influenced capacity.
- `LazyMinHeap::MAX_CAPACITY` constant and `LazyMinHeapError` enum (`CapacityTooLarge` / `AllocationFailed`), re-exported from `cachekit::ds` as `LAZY_HEAP_MAX_CAPACITY` / `LazyMinHeapError`. Mirrors the `ClockRing` pattern.
- `LazyMinHeap::try_with_capacity` / `LazyMinHeap::try_reserve` / `LazyMinHeap::try_rebuild` — fallible allocation variants that surface allocator failure as `LazyMinHeapError::AllocationFailed` instead of aborting the process. Use these when the heap population size is attacker-influenced.
- `LazyMinHeap::with_auto_rebuild` / `LazyMinHeap::set_auto_rebuild` / `LazyMinHeap::auto_rebuild_factor` — opt-in automatic rebuild cadence that runs `maybe_rebuild(factor)` after every `update`, capping the stale-entry heap growth exploited by the update-storm DoS described below.
- `LazyMinHeap::renumber_seqs` — public helper that renumbers every live entry with fresh sequential `seq` values. Called automatically on `update` at the `u64::MAX` boundary; exposed for long-running processes that want to force the renumber explicitly.

### Fixed
- **LazyMinHeap unbounded stale-entry growth** — `update` pushes a new heap entry on every call and never removes the old one, so an attacker that can drive `update` on a small key set faster than the caller drains via `pop_best` / `rebuild` could grow the backing `BinaryHeap` to arbitrary size while `len()` stayed tiny, exhausting memory. `LazyMinHeap::with_auto_rebuild` / `set_auto_rebuild` now let callers bound `heap_len` to roughly `len() * factor` by running `maybe_rebuild` after each `update`. The DoS exposure and its mitigation are documented in a new `## Security Considerations` module section. Regression test `auto_rebuild_bounds_stale_heap_growth`.
- **LazyMinHeap constructor abort on oversized capacity** — `LazyMinHeap::with_capacity` / `LazyMinHeap::reserve` / `LazyMinHeap::from_iter` forwarded their argument straight into `HashMap` / `BinaryHeap` allocation, so a capacity close to `usize::MAX` (or an adversarial `size_hint` of the same size) aborted the process inside the allocator. `with_capacity` / `reserve` now reject capacities above `MAX_CAPACITY` (panic with a `#[track_caller]` message), `try_with_capacity` / `try_reserve` surface both bounds and allocator failure as `LazyMinHeapError`, and `from_iter` clamps to the fallible path and silently falls back to zero-capacity when the reported `size_hint` cannot be honored. Regression tests `try_with_capacity_rejects_oversized_capacity`, `with_capacity_panics_on_oversized_capacity`, `try_reserve_rejects_oversized_total`, `from_iter_clamps_size_hint_to_max_capacity`.
- **LazyMinHeap sequence-number wraparound** — `pop_best` skips stale entries by checking `current.score == entry.score && current.seq == entry.seq`. The counter used `wrapping_add`, so after ~2⁶⁴ `update` calls a wrap could let a stale heap entry satisfy the equality check and be popped as if live (and break FIFO tie-breaking on the way there). The counter is now guarded by a direct `u64::MAX` check and renumbers every live entry via `renumber_seqs` before overflow; FIFO tie-breaking order is preserved across the renumber by sorting live entries on their existing `seq`. Regression tests `renumber_seqs_preserves_fifo_for_equal_scores` and `update_at_seq_saturation_renumbers_without_stale_match`.
- **LazyMinHeap `approx_bytes` overflow** — `capacity() * size_of::<_>()` could overflow `usize` for pathologically large capacities, panicking in debug and wrapping silently in release. `approx_bytes` now uses `saturating_mul` / `saturating_add`. Regression test `approx_bytes_does_not_overflow`.
- **LazyMinHeap `Debug` leaks keys and scores** — the derived `Debug` recursed through every `(key, score)` pair and every stale heap entry, turning `tracing::debug!`, `dbg!`, and panic-unwind backtraces into a disclosure channel for caches keyed on session tokens / API keys / auth headers. `Debug` is now hand-written, reports only `len`, `heap_len`, `seq`, and `auto_rebuild_factor`, and is implementable for *any* `LazyMinHeap<K, S>` regardless of whether `K` / `S` implement `Debug`. Regression test `debug_impl_does_not_leak_keys_or_scores`.

- **ClockRing `insert_swap` silent value drop** — when the index mapped a key to a slot that had been emptied (a broken invariant reachable only via a malformed `Hash`/`Eq`/`Clone` impl on `K`), the caller's value was silently discarded. The repair path now places the value in the empty slot, refreshes the index entry, and recomputes `len` from slot occupancy; a `debug_assert!` surfaces the root-cause corruption in debug/test/fuzz builds.
- **ClockRing `insert_swap` silent value drop on out-of-bounds index corruption** — the sibling of the repair path above also silently discarded the caller's value when the index mapped a key to a slot index that was *out of bounds* for `slots` (reachable only via state corruption, never from safe use of this module). For `V` owning a file descriptor, lock, `Arc<_>`, or any resource with observable `Drop`, this leaked the resource across a single corruption event without surfacing any error. `insert_swap` now hands the value back via the `replaced` return position and removes the stale index entry, preserving the no-silent-value-loss contract; `ClockRing::insert` drops it at the caller boundary (matching the update path) and `ConcurrentClockRing::insert` drops it after releasing the write lock. A `debug_assert!` surfaces the root-cause corruption in debug/test/fuzz builds. Regression test `insert_swap_returns_value_when_index_points_out_of_bounds`.
- **ClockRing `step` helper now total for `cap == 0`** — previously protected only by a `debug_assert!` (stripped in release), so a future caller that forgot the capacity-zero guard would hit a release-mode division-by-zero panic. `step(_, 0)` now returns `0`.
- **ClockRing `insert_swap` eviction-path exception safety** — a panicking `K::Clone` (most plausibly OOM in a heap-allocating `Clone` impl) reached during full-ring eviction previously left the ring with an empty slot while `len == capacity`, turning the next `insert` into an `unreachable!("occupied slot missing under full ring")` panic. For `ConcurrentClockRing` this converted a single transient failure into a persistent-panic DoS for every caller sharing the ring. The fix clones the new key *before* any destructive state change, so a panic there leaves invariants intact. Regression tests `insert_swap_eviction_panic_in_clone_preserves_invariants` and `insert_swap_eviction_panic_in_clone_does_not_leak_evictee` guard both the invariant and the index integrity.
- **ClockRing `insert_swap` / `pop_victim` eviction-path exception safety vs panicking `Hash` / `Eq`** — both sites previously ran `self.slots[idx].take()` *before* `self.index.remove(&evictee.key)`. A panic in a user-supplied `Hash` or `Eq` impl during the subsequent `HashMap::remove` would then leave the ring with an empty slot while `len == capacity` (for `insert_swap`) or `len` un-decremented (for `pop_victim`), again tripping `unreachable!("occupied slot missing under full ring")` on the next `insert` — a persistent-panic DoS amplified across every thread sharing a `ConcurrentClockRing` handle. The fix reads the evictee key *by borrow* and removes it from the index **before** emptying the slot, so a panic in the hasher leaves slot / `len` / `referenced` untouched. Regression tests `insert_swap_eviction_panic_in_hash_preserves_invariants` and `pop_victim_panic_in_hash_preserves_invariants` install a `Hash` impl with a caller-controlled panic budget and verify ring invariants and follow-up-insert recovery.

### Changed
- `ClockRing::clone` is now a hand-written impl (replacing the `#[derive(Clone)]`) so the abort-on-OOM failure mode inherited from `Vec`/`HashMap` is documented on the impl itself and callers are directed to `try_clone` when recovery matters. Behavior and trait bounds (`K: Clone, V: Clone, S: Clone`) are unchanged.
- `ClockRing` / `ConcurrentClockRing` `Debug` impls are now hand-written (replacing the derived `Debug`) and redact all stored keys and values, exposing only `len`, `capacity`, `hand`, and metrics counters when the `metrics` feature is enabled. The derived `Debug` recursed through every `Entry<K, V>` and `HashMap<K, usize, S>` entry, turning `tracing::debug!`, `dbg!`, and panic-unwind backtraces into a secret-exposure channel for caches keyed on session tokens / API keys / auth headers or holding sensitive values. Callers that need full contents can iterate via `ClockRing::iter` / `into_inner().iter()` and print entries they've vetted. No trait bounds were added: `Debug` is now implementable for *any* `ClockRing<K, V, S>` regardless of whether `K` / `V` / `S` implement `Debug`. Regression tests `debug_impl_does_not_leak_keys_or_values` and `concurrent_debug_impl_does_not_leak_keys_or_values`.

### Documentation
- New `## Memory Budgeting` module-level section documenting that `approx_bytes` counts `size_of::<V>()` only and does not follow heap pointers — workloads with variable-sized values should enforce a byte budget at the call site.
- New `## Timing Side Channels` module-level section documenting that `HashMap`-backed lookup timing reveals key presence, and recommending `ClockRing` not be used as the backing store for caches whose key set must remain confidential against a co-located attacker.
- `Extend<(K, V)> for ClockRing` impl now carries a memory-budget caveat pointing callers at `pop_victim`-based byte accounting when values are attacker-influenced; evicted entries are silently dropped by `Extend` and the aggregate memory use during a bulk insert is bounded only by `capacity × max(size_of(V_i))` across the iterator.

## [0.7.0] - 2026-04-09

### Breaking
- Replace `ReadOnlyCache`, `CoreCache`, `MutableCache`, `FifoCacheTrait`, `LruCacheTrait`, `LfuCacheTrait`, `LrukCacheTrait` with a single unified `Cache<K, V>` trait.
- Add five optional capability traits: `EvictingCache`, `VictimInspectable`, `RecencyTracking`, `FrequencyTracking`, `HistoryTracking`.
- Policy-specific methods (`pop_oldest`, `pop_lru`, `pop_lfu`, `pop_lru_k`, `age_rank`, etc.) are now inherent methods, not trait methods.
- Rename `builder::Cache` to `builder::DynCache` to avoid collision with the new `traits::Cache` trait.
- Remove `CacheTierManager` and `CacheTier` traits (no existing implementations).
- All 18 policies now implement `Cache<K, V>` with universal `peek` and `remove` support.

## [0.6.0] - 2026-03-31

### Breaking
- Rename `LFUHandleCache` to `LfuHandleCache` for Rust naming consistency.
- Rename `ARCCore` to `ArcCore` for Rust naming consistency.

### Added
- `Clone` / `Default` and iterators where applicable across FIFO, LRU, LRU-K, LIFO, Clock, MRU, NRU, Heap LFU, S3 FIFO, Random, and related types.
- Expanded error-handling documentation with conversion helpers and examples.

### Fixed
- CAR policy correctness and related fixes (#64).
- Store, metrics, and data-structure API guideline alignments (#54, #55, #56, #57).
- FIFO policy module documentation (#52).

### Changed
- MFU policy simplification and documentation refresh.
- SLRU capacity handling and documentation updates.
- Random core API and documentation improvements.
- Dependency and CI workflow maintenance (including GitHub Pages action updates).

## [0.5.0] - 2026-03-02

### Fixed
- **ConcurrentSlabStore TOCTOU race conditions** — Separate `RwLock`s for index, entries, and free list allowed data corruption on concurrent update-after-remove, capacity overshoot under parallel inserts, and inconsistent reads during `clear()`. Consolidated into a single `SlabInner` behind one `RwLock` for full mutation atomicity.
- **ARC ghost-list directory leak** — Case 4 (complete miss) did not prune ghost lists B1/B2, violating the paper's invariant that T1+T2+B1+B2 ≤ 2×capacity. Fixed to match the original ARC eviction logic.
- **Capacity-0 coercion in Clock, ClockPro, NRU** — Constructors silently coerced `capacity=0` to 1 via `.max(1)`, inconsistent with other policies. Now honors zero capacity and rejects inserts gracefully.
- **MFU insert return value at capacity 0** — `MfuCore::insert` returned `Some(value)` for rejected inserts instead of `None`.
- **MRU / SLRU / TwoQ `CoreCache::insert` inconsistency** — Trait impl duplicated update logic that diverged from the inherent method. Unified by delegating the trait impl to the inherent `insert`, which now returns `Option<V>`.
- **ConcurrentWeightStore missing metrics** — `try_insert` and `remove` did not update insert/update/remove counters.

### Added
- `GhostList::evict_lru()` for popping the least-recently-used ghost entry.
- `ClockRing` iterators: `Iter`, `IterMut`, `IntoIter`, `Keys`, `Values`, `ValuesMut`.
- Integration test suite `tests/slab_concurrency.rs` for ConcurrentSlabStore race conditions and atomicity.
- Integration test suite `tests/policy_invariants.rs` for cross-policy capacity-0 semantics.
- Per-policy regression tests for capacity-0 behavior, insert return values, ARC ghost-list bounds, and ConcurrentWeightStore metrics.
- Metrics tracking for `ClockRing`, `ArcCache`, `CarCache`, and `ClockProCache`.
- `cargo-deny` policy configuration (`deny.toml`) and CI checks for licenses/advisories.

### Changed
- `ClockRing` module documentation updated with rustdoc intra-doc links and expanded operations table.
- `LruCore` internals refactored to a pool-based, allocation-free design for improved hot-path predictability.
- `SlotId`/`SlotArena` internals improved for stability and performance, including iterator refinements and corrected `approx_bytes` accounting.
- `ClockRing::insert` slot management optimized to reduce unnecessary evictions.
- `WeightStore` weight accounting updated to checked arithmetic with stronger invariant enforcement.
- CI/release automation updated for benchmark reporting and crate publishing workflow improvements.
- Dependency refreshes (including `chrono` patch update and lockfile refreshes).

### Documentation
- Policy roadmap expanded and refreshed (including SIEVE, LHD, LeCaR, and Hyperbolic entries).
- Additional architecture/API-guideline documentation polish across core data structures and policy docs.

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
  - Docstrings with examples for `LfuCache`, `LfuHandleCache`, and all public methods.
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
