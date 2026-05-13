# Design Overview

This document collects the design principles that shape `cachekit`. Each
section pairs a principle with the concrete artifact in the source tree
that realizes it, so the prose stays grounded in the code rather than
floating as advice.

For a worked example that applies every principle below to one feature,
see the [TTL design doc](ttl.md). For interface conventions, the
[Rust API Guidelines checklist](https://rust-lang.github.io/api-guidelines/checklist.html)
is the companion reference; module-level documentation follows the
[doc style guide](style-guide.md).

## 1. Workload First, Policy Second

Cache policy only matters relative to workload.

Identify access patterns:
- Hot-set traffic: skewed keys, low churn on the hot set, high churn at the tail.
- Scan-heavy traffic: large working sets, weak temporal locality.
- Mixed traffic: bursts of hot data over large cold sets.

Measure:
- Reuse distance / stack distance.
- Read/write ratio.
- Temporal vs spatial locality.

Choose policies accordingly:
- `LRU` / `Clock`: good for temporal locality, vulnerable to scans.
- `LRU-K` / `2Q` / `SLRU`: better at filtering one-off accesses.
- `ARC` / `CAR`: adaptive recency/frequency balance without manual tuning.
- `S3-FIFO` / `Heap-LFU`: strong general-purpose defaults under scans.

All of the above ship today; see [`docs/policies/`](../policies/README.md)
for the implemented catalog and [`docs/policies/roadmap/`](../policies/roadmap/README.md)
for planned policies (LIRS, TinyLFU, SIEVE, GDS/GDSF, etc.).

When picking a policy or tuning a cache, design for the workload you
expect — not the average of all workloads.

## 2. Memory Layout Matters More Than Algorithms

In a cache, memory layout often dominates policy.

Prefer:
- Contiguous storage (`Vec`, slabs, arenas).
- Index-based indirection over pointer chasing.

Avoid:
- Excessive `Box`, `Arc`, linked lists with heap-allocated nodes.
- `HashMap` lookups in hot paths if avoidable.

Techniques:
- Store metadata (recency, freq, flags) in tightly packed structs.
- Separate hot metadata from cold payloads.
- Use slab allocators for fixed-size entries.

cachekit realizes this through reusable building blocks under
[`src/ds/`](../../src/ds): [`SlotArena`](../../src/ds/slot_arena.rs)
hands out stable `Handle`s backed by a `Vec`, [`IntrusiveList`](../../src/ds/intrusive_list.rs)
threads recency lists through those slots without per-node allocation,
and [`ClockRing`](../../src/ds/clock_ring.rs) keeps Clock-style state in
a single contiguous array. See [`docs/policy-ds/`](../policy-ds/README.md)
for the full primitive catalog.

Cache misses caused by your own data structure are as bad as upstream misses.

## 3. Concurrency Strategy Is Core Design, Not a Wrapper

Locking strategy shapes everything.

Options:
- Global lock: simple, often fast enough for small cores, dies under high contention.
- Sharded caches: hash key → shard, each shard independently locked.
- Lock-free or mostly-lock-free: hard in Rust, only worth it if contention dominates.

cachekit ships the first option today via the `concurrency` feature:
`Concurrent*` wrappers (e.g. `ConcurrentLruCache`, `ConcurrentSlotArena`,
`ConcurrentClockRing`) place a `parking_lot::RwLock` around the
single-threaded core. The wrappers deliberately do **not** implement
`Cache<K, V>` directly when that would force returning `&V` across a
lock boundary — they expose `Option<Arc<V>>` style APIs instead. See
[`src/policy/lru.rs`](../../src/policy/lru.rs),
[`src/ds/slot_arena.rs`](../../src/ds/slot_arena.rs), and
[`src/ds/clock_ring.rs`](../../src/ds/clock_ring.rs).

Rust-specific notes:
- For `RwLock`, prefer `parking_lot` for fairness control and lower
  uncontended overhead. For `Mutex`, the futex-based `std::sync::Mutex`
  on Rust 1.85+ is competitive on Linux/macOS; `parking_lot::Mutex`
  still wins on raw uncontended speed and offers nicer guard ergonomics.
- Avoid `Arc<Mutex<…>>` in hot paths.

Future directions worth exploring but **not currently implemented**:
sharded caches (hash key → shard, per-shard lock), per-thread caches with
periodic merge, and RCU-style read paths for read-heavy workloads.

## 4. Avoid Per-Operation Allocation

Allocations kill throughput.

Pre-allocate:
- Entry pools — see [`SlotArena`](../../src/ds/slot_arena.rs) and the
  free-list discipline in [`src/store/slab.rs`](../../src/store/slab.rs).
- Node arrays — intrusive lists thread through arena slots rather than
  allocating per-node (see [`src/ds/intrusive_list.rs`](../../src/ds/intrusive_list.rs)).

Reuse:
- Free lists (slab-backed).
- Slabs sized once at construction time via `CacheBuilder::new(capacity)`.

Use:
- `Vec` with explicit capacity management.
- `rustc-hash` (via the `rustc-hash` dep) for cheap key hashing in
  hot-path lookups.

Avoid:
- Creating new `Arc`, `String`, `Vec` per lookup.
- Hidden clones of `K` on the eviction path.

If `malloc` shows up in your flamegraph, your cache is already slow.

## 5. Eviction Must Be Predictable and Cheap

Eviction is the critical slow path.

O(1) eviction is the goal.

Avoid unbounded tree walks or scans in eviction paths.

Maintain:
- Direct indices / `Handle`s to eviction candidates (see
  [`src/store/handle.rs`](../../src/store/handle.rs) and the
  [`Cache`](../../src/store/traits.rs) trait).
- Eviction lists or clock hands (intrusive list head, `ClockRing` hand).
- Lazy heaps where amortized O(log n) is acceptable
  ([`LazyMinHeap`](../../src/ds/lazy_heap.rs); used by Heap-LFU and TTL).

Be careful with:
- Background eviction threads (synchronization overhead).
- Lazy cleanup that grows unbounded; bound it with rebuild thresholds
  (e.g. `LazyMinHeap::with_auto_rebuild`).

Eviction cost must be comparable to lookup cost, not orders of magnitude higher.

## 6. Metrics Are Not Optional

You cannot tune what you do not measure.

Track at least:
- Hit / miss rate.
- Eviction count and reason (capacity vs. expiration).
- Insert/update rate.

cachekit exposes these through [`StoreMetrics`](../../src/store/traits.rs)
and per-policy metric structs (e.g. `LruMetrics`), gated behind the
`metrics` feature so non-instrumented builds pay nothing. The
`expirations` counter on `Expiring<C>` follows the same pattern (see
[`src/policy/expiring.rs`](../../src/policy/expiring.rs)).

Roadmap counters:
- Scan pollution rate.
- Lock contention or wait time.

Expose:
- Lightweight counters in the hot path.
- Optional detailed metrics behind feature flags.

Metrics should guide design decisions, not justify them afterward.

## 7. Separate Policy From Storage

Design in layers:
- Storage layer: how entries live in memory, allocation, layout,
  indexing — [`src/store/`](../../src/store).
- Policy layer: LRU, FIFO, LFU, LRU-K, 2Q, ARC, CAR, Clock, Clock-PRO,
  S3-FIFO, … — manipulates metadata and ordering only
  ([`src/policy/`](../../src/policy)).
- Capability layer: opt-in extension traits ([`RecencyTracking`](../../src/traits.rs),
  `FrequencyTracking`, `HistoryTracking`, `ExpiringCache`) that policies
  implement when the underlying signal exists. This is how `Expiring<C>`
  composes over any policy without touching policy code.
- Integration layer: ties application objects, payloads, or IDs into
  cache entries via [`CacheBuilder`](../../src/builder.rs) and the
  `DynCache` runtime dispatcher.

Related docs:
- [Policy overview](../policies/README.md)
- [Policy roadmap](../policies/roadmap/README.md)
- [Policy data structures](../policy-ds/README.md)
- [Read-only traits](../guides/read-only-traits.md)

This makes:
- Benchmarking easier.
- Policy experimentation cheap.
- Reasoning about performance clearer.

## 8. Beware of "Nice" Rust APIs in Hot Paths

Ergonomics often cost performance.

Avoid in critical loops:
- Heavy generics causing code bloat across many monomorphizations.
- Trait objects for hot dispatch.
- Closures capturing state.
- Iterator chains where a plain `for` loop would do.

Prefer:
- Explicit loops.
- Concrete types and monomorphized fast paths.
- Enum dispatch over `Box<dyn Trait>` when polymorphism is needed at the
  edges — this is exactly the trade `DynCache` makes (see §13).

You can wrap fast internals in nice APIs at the edges.

## 9. Scans Are the Enemy of Caches

In scan-heavy workloads:

Large sequential reads destroy LRU-style caches.

Solutions:
- Scan-resistant policies: `LRU-K`, `2Q`, `SLRU`, `ARC`, `CAR`,
  `Clock-PRO`, `S3-FIFO`, `Heap-LFU` — all implemented today.
- Explicit "scan mode" hints from the caller or workload layer.
- Bypass cache for known one-shot reads.

If you ignore scans, your cache will look great in microbenchmarks and
terrible in production.

## 10. Benchmark Like a System, Not a Library

Do not rely on uniform-random key benchmarks.

Use:
- Zipfian distributions.
- Mixed read/write workloads.
- Scan + point lookup mixtures.
- Time-varying hot sets.

Measure:
- Throughput.
- Tail latency.
- Memory overhead.
- Eviction cost.

cachekit's benchmark harness covers these dimensions; see
[`docs/benchmarks/workloads.md`](../benchmarks/workloads.md) and the
runners under [`benches/`](../../benches).

A cache that is 5 % faster on uniform-random keys but 50 % worse under
scans is a bad cache.

## 11. Rust Hot-Path Hazards Beyond Allocation

`Arc` is expensive in hot paths; minimize it and lift `Arc::clone` out
of inner loops.

The borrow checker can push you toward indirection — fight it with:
- Index-based access (`Handle`s, slot indices) instead of `&mut` chains.
- Interior mutability only where unavoidable; prefer `Cell<T>` over
  `RefCell<T>` when `T: Copy`, and atomics when the value lives behind
  a shared reference.

Beware of:
- Hidden clones, particularly of keys on the eviction path.
- Trait object dispatch on read/insert.
- Over-generic designs whose monomorphization cost dwarfs their benefit.

Rust can match C on hot paths, but only when systems-level discipline
survives contact with the type system.

## 12. Design for Failure Modes

Ask:
- What happens under memory pressure?
- What happens when eviction cannot keep up?
- What happens under pathological access patterns?

Add:
- Backpressure or rejection when full.
- Bypass modes.
- Emergency eviction strategies.

A cache that collapses under stress is worse than no cache.

## 13. Compile-Time and Runtime Composition

cachekit's externally visible surface is shaped by two composition
mechanisms that together let users pay only for what they use.

**Per-policy feature flags.** Every policy is behind a Cargo feature
(`policy-lru`, `policy-s3-fifo`, …), with `policy-all` for "everything"
and a small default of `policy-s3-fifo`, `policy-lru`, `policy-fast-lru`,
`policy-lru-k`, `policy-clock`. Optional capabilities are gated the
same way: `metrics`, `concurrency`, `serde`, and `ttl`. Downstream
crates can disable defaults and select the minimum surface they need;
see [`Cargo.toml`](../../Cargo.toml).

**Capability traits + runtime dispatch.** Extension traits
([`RecencyTracking`](../../src/traits.rs), `FrequencyTracking`,
`HistoryTracking`, `ExpiringCache`) keep optional behavior off the
core `Cache<K, V>` trait. For ergonomic builder construction without
forcing trait objects on the user, [`CacheBuilder`](../../src/builder.rs)
returns a [`DynCache<K, V>`](../../src/builder.rs) that dispatches via
an internal enum match rather than `Box<dyn Cache>`. When TTL is
enabled, the builder returns a sibling `DynExpiringCache<K, V>` that
threads the expiry check around each variant's `Cache` call — a worked
example of capability composition. See [`docs/design/ttl.md`](ttl.md)
for the full design and [`src/policy/expiring.rs`](../../src/policy/expiring.rs)
for the decorator itself.

## Bottom Line

High-performance caches are not about clever algorithms — they are about:
- Memory layout.
- Allocation discipline.
- Contention control.
- Eviction predictability.
- Workload realism.

In Rust, your main enemy is not safety — it is abstraction overhead and
accidental allocation. Design from the metal upward, then wrap it in
something pleasant to use.

## See Also

Design docs:
- [Concurrency](concurrency.md) — `Concurrent*` wrappers, `RwLock`
  discipline, sharded primitives, `ConcurrentCache` marker
- [Cache trait hierarchy](trait-hierarchy.md) — `Cache<K, V>` kernel,
  capability traits, read/mutate split, object safety
- [Builder and runtime dispatch](builder-and-dyn-dispatch.md) —
  `CachePolicy`, `DynCache`, enum-vs-`Box<dyn>` trade-off, adding new
  policies
- [Weighted eviction](weighted-eviction.md) — `WeightStore` dual
  limits, weight function contract, GDS/GDSF pre-staging
- [Metrics](metrics.md) — recorder / snapshot / exporter split,
  `MetricsCell`, Prometheus exporter, feature gating
- [Error model](error-model.md) — panic vs `Result` discipline,
  four error types, debug-only invariant checks
- [Benchmarking](benchmarking.md) — benchmark layers, monomorphic policy
  registry, JSON artifact schema, reproducibility rules
- [Hashing and key identity](hashing.md) — hasher choices, `KeyInterner`,
  `ShardSelector`, HashDoS trade-offs
- [Sharding](sharding.md) — current sharded primitives, routing,
  capacity semantics, roadmap for sharded caches
- [Serialization](serialization.md) — current `serde` surface, cache-state
  persistence boundaries, TTL and hash-seed rules
- [Non-goals](non-goals.md) — explicit boundaries for what cachekit does
  not try to be
- [TTL](ttl.md) — applied example of every principle above
- [Doc style guide](style-guide.md)

Reference docs:
- [Policy overview](../policies/README.md) and [roadmap](../policies/roadmap/README.md)
- [Policy data structures](../policy-ds/README.md)
- [Stores](../stores/README.md)
- [Read-only traits](../guides/read-only-traits.md)
- [Choosing a policy](../guides/choosing-a-policy.md)
- [Benchmarks overview](../benchmarks/overview.md) and [workloads](../benchmarks/workloads.md)
- [Rust API Guidelines checklist](https://rust-lang.github.io/api-guidelines/checklist.html)
