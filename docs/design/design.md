# Design Overview

> Status: top-level design overview for cachekit. Indexes the
> architectural layers and the substantive design decisions, with the
> original numbered principles preserved as Appendix A. Every
> subsystem (concurrency, storage, metrics, TTL, …) has its own
> companion design doc; this doc names what they collectively decide.

For a worked example that applies every concern below to a single
feature, see the [TTL design doc](ttl.md). For interface conventions,
the [Rust API Guidelines checklist](https://rust-lang.github.io/api-guidelines/checklist.html)
is the companion reference; both module-level rustdoc and the design
docs themselves follow the [doc style guide](style-guide.md).

## Architecture at a glance

```text
+------------------------------------------------------------------+
|  Integration: CacheBuilder, DynCache, DynExpiringCache           |
+------------------------------------------------------------------+
|  Capability traits (opt-in):                                     |
|   RecencyTracking, FrequencyTracking, HistoryTracking,           |
|   ExpiringCache, EvictingCache, VictimInspectable                |
+------------------------------------------------------------------+
|  Policy kernel: Cache<K, V>                                      |
|   18 implemented policies (LRU, LFU, S3-FIFO, ARC, CAR, ...)     |
+------------------------------------------------------------------+
|  Storage: StoreCore / StoreMut (+ concurrent peers)              |
|   HashMapStore, SlabStore, HandleStore, WeightStore (sibling)    |
+------------------------------------------------------------------+
        ^                 ^                 ^                ^
   +----+----+      +-----+-----+     +-----+-----+    +----+-----+
   | Metrics |      |    TTL    |     |Concurrency|    | Hashing  |
   |(2-layer)|      | Expiring  |     |  RwLock + |    |    +     |
   |         |      | decorator |     |  sharding |    | Sharding |
   +---------+      +-----------+     +-----------+    +----------+
             (cross-cutting concerns, mostly feature-gated)
```

Read bottom-up: storage owns layout and ownership; policies layer
eviction order on top of that; capability traits expose optional
signals when the policy has them; the integration layer turns
runtime configuration into one concrete cache type. Cross-cutting
concerns (metrics, TTL, concurrency, hashing, sharding) are
orthogonal to the layer stack — each has its own design doc and
attaches to the layer where its choices live.

## Layered map

### Storage layer

Stores own keys and values. They expose `StoreCore`/`StoreMut` for
sequential access and `ConcurrentStoreRead`/`ConcurrentStore` for
concurrent access, signal capacity refusal with `StoreFull`, and
ship one always-on counter struct (`StoreMetrics`).

Four concrete stores ship: `HashMapStore` (default), `SlabStore`
(arena with stable `EntryId` handles), `HandleStore`
(interner-keyed), and `WeightStore` (byte-aware sibling that
deliberately diverges from `StoreMut`'s contract — see D14). See
[Storage layer](storage.md).

### Policy layer

Policies decide eviction order. Every policy implements the
object-safe `Cache<K, V>` kernel; 18 ship today, organised in the
catalog at [`docs/policies/`](../policies/README.md). The kernel
deliberately separates `peek` (`&self`, side-effect-free) from `get`
(`&mut self`, policy-updating) so concurrent wrappers can take a
read lock on `peek` paths (see D3). See
[Cache trait hierarchy](trait-hierarchy.md).

### Capability layer

Capabilities are opt-in extension traits that expose signals which
*some but not all* policies own: `RecencyTracking`,
`FrequencyTracking`, `HistoryTracking`, `ExpiringCache`,
`EvictingCache`, `VictimInspectable`. Each extends `Cache<K, V>` —
generic code requires the capability bound rather than feature-flag-
gating methods on the kernel (D2). See
[Cache trait hierarchy §"Layer 2 — Capability traits"](trait-hierarchy.md#layer-2--capability-traits).

### Integration layer

`CacheBuilder` turns a `CachePolicy` enum + capacity + optional
defaults (TTL, hasher) into a `DynCache<K, V>` (or
`DynExpiringCache<K, V>` when TTL is configured). `DynCache`
dispatches through an internal enum match rather than
`Box<dyn Cache>` — devirtualising the hot path while keeping the
user-facing type uniform (D9). See
[Builder and runtime dispatch](builder-and-dyn-dispatch.md).

### Cross-cutting concerns

- [Concurrency](concurrency.md) — `Concurrent*` wrappers, `RwLock`
  discipline, sharded primitives, `ConcurrentCache` marker.
- [Metrics](metrics.md) — recorder / snapshot / exporter split,
  Prometheus exporter, `MetricsCell` soundness contract.
- [TTL](ttl.md) — `Expiring<C>` decorator, `ExpirationIndex`,
  `Clock` abstraction, decorator-vs-embedded trade.
- [Hashing](hashing.md) — `RandomState` at public boundaries,
  `FxHash` internally, `ShardSelector` for routing.
- [Sharding](sharding.md) — sharded primitives at the
  data-structure / store layer; sharded cache policies are roadmap.
- [Error model](error-model.md) — three-tier panic / `Result` /
  invariant discipline, four error types.
- [Benchmarking](benchmarking.md) — benchmark layers, monomorphic
  policy registry, JSON artifact schema, reproducibility rules.
- [Serialization](serialization.md) — `serde` scope (snapshots
  only), reasons cache state is not serialisable today.
- [Non-goals](non-goals.md) — explicit boundaries on what cachekit
  does *not* try to be.

## Design decisions

Numbered ADR-style entries. Each entry documents one substantive
choice that shaped the public or internal surface; the canonical
companion doc carries the full detail.

### D1. Policy / storage separation

**Context.** Combining "what to evict" and "how entries are laid
out in memory" into one type couples eviction strategy to storage
and prevents policy experimentation.

**Decision.** Two layers with explicit interfaces: stores own keys
and values and expose `StoreCore`/`StoreMut`; policies own eviction
metadata and consume the store. Capacity refusal is signalled by
`StoreFull`; the policy decides who to evict and retries.

**Alternatives considered.**
- One trait that combines lookup and eviction. Rejected: policy
  experimentation requires substituting one half without the other.
- Store-driven eviction with policy callbacks. Rejected: forces a
  callback indirection on every insert.

**Consequences.**
- A policy runs over `HashMapStore`, `SlabStore`, or a custom store
  without policy changes.
- The store layer is its own design space (see
  [storage.md](storage.md)); `WeightStore` is the precedent for
  divergence (D14).
- Tests can drive the store independently of any policy.

**See also.** Appendix §7; [storage.md](storage.md),
[trait-hierarchy.md](trait-hierarchy.md).

### D2. Object-safe `Cache<K, V>` kernel + opt-in capability traits

**Context.** Some policies expose signals (recency rank, frequency
count, K-distance history) that others don't. Putting all of them
on the kernel forces meaningless defaults and breaks object safety.

**Decision.** `Cache<K, V>` carries only what every policy must do
(`contains` / `len` / `capacity` / `peek` / `get` / `insert` /
`remove` / `clear`). Optional signals live in extension traits
(`RecencyTracking`, `FrequencyTracking`, `HistoryTracking`,
`ExpiringCache`, `EvictingCache`, `VictimInspectable`) which
policies implement only when the signal exists in their metadata.

**Alternatives considered.**
- Feature-gated methods on the kernel. Rejected: the gating signal
  is policy identity, not build configuration.
- `Option`-returning defaults. Rejected: silently returning `None`
  on policies that don't support a method is a footgun.

**Consequences.**
- Generic code like `fn warm<C: FrequencyTracking>(...)`
  type-checks only on policies that genuinely track frequency.
- Trait surface grows by one trait per signal, but only the
  policies that own the signal pay anything.
- The kernel stays object-safe, so `Box<dyn Cache<K, V>>` works
  for test harnesses and registries (even though shipped runtime
  dispatch uses an enum — see D9).

**See also.** Appendix §7, §13;
[trait-hierarchy.md](trait-hierarchy.md).

### D3. `peek` (side-effect-free) vs `get` (policy-updating) split

**Context.** A read that updates LRU recency or LFU frequency
cannot be served from a shared read lock — it mutates policy state.
A single read method forces every lookup through a write lock.

**Decision.** Two distinct read methods on `Cache<K, V>`:
- `peek(&self, &K) -> Option<&V>` — honest read, no policy mutation.
- `get(&mut self, &K) -> Option<&V>` — recorded read; LRU moves the
  entry to MRU, Clock sets the reference bit, etc.

**Alternatives considered.**
- One `get(&self, …)` with interior mutability. Rejected: forces
  the concurrent wrapper to serialise every read through one lock
  shape regardless of intent.

**Consequences.**
- Concurrent wrappers take `RwLock::read` on `peek` / `contains` /
  `len` / `capacity`; `RwLock::write` on `get` / `insert` /
  `remove` / `clear`. Read-heavy workloads scale across cores.
- Callers must know which one they want. `get` on a read-heavy
  workload silently kills scalability; documented at every call
  site.

**See also.** Appendix §3;
[trait-hierarchy.md §"peek vs get"](trait-hierarchy.md#peek-vs-get--the-readmutate-split),
[concurrency.md](concurrency.md).

### D4. Sequential `&V` + concurrent owned values (two trait families)

**Context.** Borrowed references cannot outlive the lock guard
they were extracted from. A concurrent cache cannot expose
`Cache::get`'s `Option<&V>` without holding the lock across the
borrow, which serialises readers.

**Decision.** Two parallel trait families. Sequential
(`Cache<K, V>`, `StoreCore`, `StoreMut`) returns `&V` / owned `V`.
Concurrent stores (`ConcurrentStoreRead`, `ConcurrentStore`) return
`Arc<V>` and take `&self`. Concurrent cache wrappers also take
`&self`, but their concrete APIs may return `Arc<V>` (LRU) or cloned
`V` (FIFO, S3-FIFO), depending on the underlying policy storage.
Concurrent cache wrappers do **not** implement `Cache<K, V>`; they
expose their own concrete API.

**Alternatives considered.**
- Unified `Arc<V>`-only traits. Rejected: forces an `Arc<V>`
  round-trip on every sequential lookup.
- Hand out lock guards. Rejected: ergonomically terrible and forces
  callers to manage the lock.

**Consequences.**
- Concurrent store hits pay one atomic refcount bump. Concurrent
  cache wrappers that return cloned `V` instead pay the value's
  clone cost.
- Generic code that wants "any thread-safe cache" bounds on
  `ConcurrentCache + Send + Sync`, not `Cache<K, V>`.
- `DynCache` (sequential) and the `Concurrent*` wrappers
  (concurrent) are separate dispatch paths.

**See also.** Appendix §3;
[concurrency.md](concurrency.md),
[storage.md §"Layer 2"](storage.md#layer-2--concurrentstoreread--concurrentstore).

### D5. `parking_lot::RwLock` for concurrent wrappers

**Context.** The choice of lock primitive shapes uncontended cost,
fairness, and writer-starvation behaviour.

**Decision.** Every `Concurrent*` wrapper uses
`parking_lot::RwLock`. Gated behind the `concurrency` Cargo feature.

**Alternatives considered.**
- `std::sync::RwLock`. Rejected for now: writer-starvation hazard
  on some platforms; revisit if `parking_lot` becomes a build
  burden.
- `Mutex` (any variant). Rejected: serialises readers that only
  consult `peek` / `contains` / `len`, defeating D3's split.
- Lock-free / RCU. Rejected as current design; covered in
  [non-goals.md §"Not Lock-Free"](non-goals.md#not-lock-free).

**Consequences.**
- `peek` paths scale linearly with cores on read-heavy workloads.
- `get` paths serialise through the write lock — fundamental, not
  fixable in the wrapper.
- `parking_lot` doesn't poison on panic; relevant only under
  `panic = "unwind"` (the crate's release default is `abort`).

**See also.** Appendix §3;
[concurrency.md §"Lock primitive choice"](concurrency.md#lock-primitive-choice).

### D6. Slab / arena / intrusive layout over pointer chasing

**Context.** A cache implemented over `Box`-allocated nodes with
linked-list pointers pays cache misses on every traversal step,
fights the borrow checker on every mutation, and fragments memory.

**Decision.** Build policies over reusable primitives in
[`src/ds/`](../../src/ds): `SlotArena` for stable `Handle`-indexed
entries, `IntrusiveList` for recency lists threading through arena
slots, `ClockRing` for contiguous reference-bit storage,
`FrequencyBuckets` for LFU.

**Alternatives considered.**
- `Box<Node>` linked lists. Rejected: every traversal step is a
  cache miss; `Box` allocations dominate hot paths.
- `Rc` / `Arc<Node>`. Rejected: refcount overhead in the hot path.

**Consequences.**
- Every policy that needs ordering threads its metadata through
  one of the arena / intrusive primitives. Adding a new policy
  with novel metadata typically means adding a new primitive to
  `src/ds/` rather than rolling its own.
- Stable handles enable O(1) eviction (D8).

**See also.** Appendix §2;
[`src/ds/`](../../src/ds),
[`docs/policy-ds/`](../policy-ds/README.md).

### D7. No per-operation allocation in hot paths

**Context.** Allocation on `get` / `insert` dominates flamegraphs
and makes tail latency unpredictable.

**Decision.** Pre-size pools, slabs, and intrusive lists at
construction time. Policy and store hot paths avoid `Box::new` and
`Vec::push`-that-grows; allocation is acceptable on `new` /
`with_capacity` paths and caller-driven `clear` / replace paths.
The integration layer has narrow exceptions where a policy stores
shared values internally: `DynCache` wraps inserted values in
`Arc<V>` for the LRU, LFU, and Heap-LFU variants.

**Alternatives considered.**
- Per-thread arenas. Roadmap; the single-arena story suffices for
  the current policies and avoids cross-thread arena coordination.
- "Allow allocation, document the cost." Rejected: defeats the
  performance contract.

**Consequences.**
- Insert into a full store returns `StoreFull` rather than
  triggering a `Vec` grow; cache policies handle that by evicting
  according to policy and retrying the store insert.
- Benchmarks must pre-allocate values; see
  [benchmarking.md §"Value Construction Discipline"](benchmarking.md#value-construction-discipline).
- The `WeightStore` weight-function contract requires
  `Fn(&V) -> usize` to be O(1); a traversal-based weight function
  silently regresses insert latency.

**See also.** Appendix §4;
[`src/ds/slot_arena.rs`](../../src/ds/slot_arena.rs),
[`src/store/slab.rs`](../../src/store/slab.rs).

### D8. Predictable eviction via direct handles / indices

**Context.** Eviction is the slow path on a full cache. A policy
that scans to find a victim trades constant-time inserts for
linear-time evictions, which dominates tail latency.

**Decision.** Policies targeting interactive hot paths maintain
direct indices or `Handle`s (intrusive list head / tail, `ClockRing`
hand, lazy-heap root) to the eviction candidate. Eviction cost should
be comparable to lookup cost, not orders of magnitude higher. `NruCache`
is the documented exception: it is intentionally simple and can scan
O(n) on eviction; callers that need O(1) eviction should use `Clock`,
`LRU`, or another direct-victim policy.

**Alternatives considered.**
- Scan-on-demand. Rejected for any policy targeting interactive
  workloads.
- Lazy O(log n) priority queues are acceptable when amortised cost
  dominates — `LazyMinHeap` (Heap-LFU, TTL) is the precedent.

**Consequences.**
- `EvictingCache::evict_one` is `#[must_use]`. Most shipped policies
  are O(1) or O(log n) amortised; `NruCache` is O(n) worst case by
  design.
- Lazy structures must bound staleness:
  `LazyMinHeap::with_auto_rebuild` prevents unbounded tombstone
  growth.

**See also.** Appendix §5;
[`src/store/handle.rs`](../../src/store/handle.rs),
[`src/ds/lazy_heap.rs`](../../src/ds/lazy_heap.rs),
[`src/traits.rs`](../../src/traits.rs) (`EvictingCache`).

### D9. Enum dispatch (`DynCache`) over `Box<dyn Cache>` for runtime selection

**Context.** Users want to pick a policy at runtime from
configuration without writing one call site per policy.
`Cache<K, V>` is object-safe, but `Box<dyn Cache<K, V>>` pays a
vtable indirection on every method call.

**Decision.** `CacheBuilder` returns `DynCache<K, V>`, a wrapper
around a closed `CacheInner<K, V>` enum with one variant per
builder-wired policy. Method calls dispatch through `match`, which
the optimiser devirtualises when the variant is invariant across an
inner loop. CAR is implemented as `CarCore` but is not yet exposed
through `CachePolicy` / `DynCache`.

**Alternatives considered.**
- `Box<dyn Cache<K, V>>`. Rejected for the runtime dispatcher;
  still available to users who want open polymorphism.
- One concrete `LruCache`-like default. Rejected: configurability
  is a goal.

**Consequences.**
- Adding a new builder-wired policy is a six-arm edit (see
  [builder-and-dyn-dispatch.md §"Adding a new policy"](builder-and-dyn-dispatch.md#adding-a-new-policy)).
- `DynCache` requires the *union* of every variant's trait bounds
  (`K: Copy + Eq + Hash + Ord`, `V: Clone + Debug`).
- A small `Arc<V>` round-trip is paid on the LRU / LFU / Heap-LFU
  variants that store `Arc<V>` internally.

**See also.** Appendix §8, §13;
[builder-and-dyn-dispatch.md](builder-and-dyn-dispatch.md).

### D10. Per-policy feature flags + curated default set

**Context.** Users who only need one or two policies shouldn't pay
compile time and binary size for the other 16.

**Decision.** Every policy is behind its own `policy-*` Cargo
feature. `policy-all` enables every policy. The default feature
set is a curated subset (`policy-s3-fifo`, `policy-lru`,
`policy-fast-lru`, `policy-lru-k`, `policy-clock`) chosen to cover
the most-recommended workloads. Optional capabilities (`metrics`,
`concurrency`, `serde`, `ttl`) are gated the same way.

**Alternatives considered.**
- One mega-feature. Rejected: penalises minimum-surface users.
- No defaults, force every consumer to enumerate. Rejected:
  ergonomic regression for the common case.

**Consequences.**
- A user with `default-features = false, features = ["policy-lru"]`
  gets one inner variant and no other policy code.
- One sharp edge: `policy-fast-lru` is in the default set, and
  `FastLru` is `!Send + !Sync` (uses `NonNull<Node>`). With the
  default features, `DynCache<K, V>` is also `!Send + !Sync`.
  Builds that need a sendable `DynCache` should disable
  `policy-fast-lru` or use a [`Concurrent*` wrapper](concurrency.md)
  directly.

**See also.** Appendix §13;
[`Cargo.toml`](../../Cargo.toml),
[builder-and-dyn-dispatch.md §"`Send + Sync` is conditional"](builder-and-dyn-dispatch.md#send--sync-is-conditional).

### D11. Two-layer metrics

**Context.** Universal counters (hit rate, eviction count) are
part of the store trait surface and provide the observability
baseline every implementation can report. Per-policy detailed
metrics (Clock hand advances, ARC ghost hits, LRU recency-rank
reads) are expensive to plumb and unnecessary for many consumers.

**Decision.** Two parallel metrics surfaces:
- `StoreMetrics` (seven counters: `hits`, `misses`, `inserts`,
  `updates`, `removes`, `evictions`, `expirations`), always on,
  in [`src/store/traits.rs`](../../src/store/traits.rs).
- Feature-gated policy-layer hierarchy (per-policy `*Metrics`
  recorders, `*Snapshot` value types, Prometheus exporter) behind
  the `metrics` Cargo feature.

**Alternatives considered.**
- All metrics behind one feature flag. Rejected: the store-layer
  counters are part of the universal trait contract.
- No counters at all. Rejected: cannot tune what you do not
  measure.

**Consequences.**
- The `metrics` feature opts into per-policy detail and the
  exporter; turning it off does not strip the store-layer
  baseline.
- `MetricsCell` (the `&self` interior-mutability counter) carries
  a narrow soundness contract: increments must happen under
  *exclusive* synchronisation, not under a shared `RwLock::read`.

**See also.** Appendix §6; [metrics.md](metrics.md).

### D12. Three-tier error model

**Context.** Different failures need different responses. A
programmer who passes `k = 0` to LRU-K should learn at the call
site, not handle a `Result`. A user-supplied config file with
`small_ratio = 2.0` should be routed through a fallible constructor
such as `S3FifoCache::try_with_ratios`. The current
`CacheBuilder::build` path treats `CachePolicy` values as already
validated program input and panics on invalid parameters; a future
`try_build` would be the tier-2 surface for external configuration.
An internal invariant violation should be diagnosable but not fatal
in test runs.

**Decision.** Three tiers:
1. **Programming error** → panic (`assert!`, `panic!`,
   `debug_assert!`).
2. **User-supplied input / expected resource failure** →
   `Result<_, ErrorType>`, using narrow error types such as
   `ConfigError`, `StoreFull`, `LazyMinHeapError`, or
   `std::collections::TryReserveError` passthrough.
3. **Invariant violation** → `check_invariants` methods returning
   `Result<(), InvariantError>` under `debug_assertions` / `test`.

**Alternatives considered.**
- One `CachekitError` enum. Rejected: each surface has different
  recovery semantics, and downstream code rarely wants the union.
- Panic on every failure. Rejected for tier-2 surfaces that take
  user-supplied input.

**Consequences.**
- Public constructors that accept user-tunable knobs should come in
  pairs where that surface is exposed to external configuration:
  panicking `new` / `build` and fallible `try_*` variants.
- `panic = "abort"` is the release default; programming errors
  terminate the process rather than unwinding.

**See also.** Appendix §12; [error-model.md](error-model.md).

### D13. TTL as decorator (`Expiring<C>`) rather than per-policy embedded

**Context.** TTL is orthogonal to eviction policy. Embedding
`expires_at` into every policy's node would add 8 bytes per entry
to every cache regardless of whether the user wants TTL.

**Decision.** Phase 1 ships TTL as a generic decorator
`Expiring<C, K, V, T>` that wraps any `C: Cache<K, V>` with a
shared `ExpirationIndex` (lazy min-heap of deadlines). The builder
returns `DynExpiringCache<K, V>` when `.with_default_ttl(...)` is
configured; otherwise it returns the plain `DynCache<K, V>`.
Phase 2 would profile and selectively embed `expires_at` into hot
policies (LRU, S3-FIFO) only if benchmarks justify it.

**Alternatives considered.**
- Per-policy embedded `expires_at`. Deferred: per-entry overhead
  is unwanted in non-TTL builds.
- Storage-level TTL. Rejected: policies still hold stale metadata
  for now-expired keys.

**Consequences.**
- Zero churn on the 18 existing policies.
- Decorator pays one extra hash probe per read; benchmark-gated
  for the Phase 2 embedded variants.
- `DynExpiringCache` is structurally distinct from `DynCache` so
  `Expiring<Expiring<…>>` is unrepresentable — type-level
  prevention of the "two clocks, two indexes" footgun.

**See also.** [ttl.md](ttl.md).

### D14. `WeightStore` as sibling of `StoreCore` / `StoreMut`, not subtype

**Context.** Byte-budgeted caches (images, blobs, variable-size
values) need a second budget alongside entry count. Updates can
legitimately fail when a larger replacement value exceeds the
budget, which `StoreMut::try_insert`'s "updates always succeed"
contract cannot express.

**Decision.** `WeightStore<K, V, F>` is a sibling of the store
trait family, not a subtype. It returns `Arc<V>` (even
single-threaded) and enforces a dual entry-count + weight cap.
`ConcurrentWeightStore` *does* implement `ConcurrentStoreRead` /
`ConcurrentStore` because those traits already use `Arc<V>`
returns.

**Alternatives considered.**
- Fold weight into `StoreMut`. Rejected: forces every store to
  carry a weight slot and a weight function.
- A separate `WeightCache` policy. Roadmap (GDS / GDSF); today
  `WeightStore` is the substrate and no policy in the tree
  consumes it.

**Consequences.**
- Code generic over `StoreMut` cannot accept a `WeightStore`
  without adaptation. Documented sharp edge.
- `WeightStore` pre-stages GDS / GDSF by providing the
  per-entry-size half of cost / size eviction.

**See also.** [weighted-eviction.md](weighted-eviction.md).

### D15. Mixed hasher defaults (`RandomState` at public boundaries, `FxHash` internally)

**Context.** A single hasher choice forces a single trade-off:
HashDoS-resistance on public APIs (slower) or speed on internal
hot paths (vulnerable when keys are user-controlled).

**Decision.** Public stores (`HashMapStore`, `ClockRing`) default
to `RandomState`. Internal policy maps and `WeightStore` use
`FxHashMap`. Shard routing (`ShardSelector`) uses keyed
SipHash-1-3. Each public constructor that takes a non-randomised
hasher requires explicit caller opt-in (`with_hasher`,
`KeysAreTrusted`).

**Alternatives considered.**
- `RandomState` everywhere. Rejected: leaves performance on the
  table for internal trusted-key paths.
- `FxHash` everywhere. Rejected: HashDoS exposure on public APIs.

**Consequences.**
- `WeightStore`'s `FxHashMap` is a documented sharp edge — its
  target use case (variable-size values keyed by request paths)
  often has user-derived keys.
- Sharded routing resists shard-pinning attacks because the
  routing hash is keyed.

**See also.** [hashing.md](hashing.md).

## Summary

The decisions above cluster around five concerns that consistently
dominate benchmark and review feedback. The themes are not novel;
the value in stating them here is naming where each was paid for
in concrete code.

- **Memory layout** — contiguous storage, stable handles, intrusive
  lists, contiguous metadata rings (D6). Layout is chosen before
  any policy decision.
- **Allocation discipline** — pre-sized pools, no `Box::new` or
  grow-on-demand `Vec` operations in policy / store hot paths, and
  `StoreFull` over silent store growth (D7).
- **Contention control** — `peek` vs `get` separation (D3) and
  `parking_lot::RwLock` discipline (D5) so read-heavy workloads
  scale; sharded primitives where the data structure permits.
- **Eviction predictability** — O(1) or amortised O(log n) victims
  via direct handles or lazy heaps for hot-path policies (D8);
  documented exceptions like NRU are explicit, and lazy cleanup is
  bounded.
- **Workload realism** — scan-resistant policies in the catalog
  (Appendix §9), Zipfian / scan / mixed benchmarks (Appendix §10),
  and policy / storage separation (D1) so the harness can swap one
  half without disturbing the other.

The trade against ergonomic Rust idioms — enum dispatch over
`Box<dyn Cache>` (D9), arena handles over `Box<Node>` (D6), and owned
values at concurrent boundaries (D4) — is deliberate. The decisions
above name where that cost was paid and where it was refused.

## Appendix A: Principles

The original 13 principles that shape cachekit. The decisions
above are the implementation of these principles as concrete
choices; this appendix preserves the principle statements and
their stable section numbers so sibling design docs can cite
`design.md §N`.

### 1. Workload First, Policy Second

Cache policy only matters relative to workload. Identify access
patterns (hot-set, scan-heavy, mixed), measure reuse distance and
read/write ratio, and choose accordingly: `LRU` / `Clock` for
temporal locality, `LRU-K` / `2Q` / `SLRU` for scan filtering,
`ARC` / `CAR` for adaptive recency / frequency balance, `S3-FIFO` /
`Heap-LFU` for strong general-purpose defaults under scans. All 18
policies ship as concrete types (CAR is the one not yet wired into
`DynCache` — see the
[CAR builder gap](builder-and-dyn-dispatch.md#car-builder-gap)).
See [`docs/policies/`](../policies/README.md) for the implemented
catalog and [`docs/policies/roadmap/`](../policies/roadmap/README.md)
for planned policies (LIRS, TinyLFU, SIEVE, GDS / GDSF, …). Design
for the workload you expect — not the average of all workloads.

*Realised by:* the policy catalog itself; see also
[choosing-a-policy.md](../guides/choosing-a-policy.md).

### 2. Memory Layout Matters More Than Algorithms

Memory layout often dominates policy. Prefer contiguous storage
(`Vec`, slabs, arenas) and index-based indirection over pointer
chasing; avoid excessive `Box`, `Arc`, and linked lists with
heap-allocated nodes; store metadata in tightly packed structs;
separate hot metadata from cold payloads. Cachekit realises this
through reusable building blocks under
[`src/ds/`](../../src/ds): [`SlotArena`](../../src/ds/slot_arena.rs)
hands out stable `Handle`s, [`IntrusiveList`](../../src/ds/intrusive_list.rs)
threads recency lists through arena slots without per-node
allocation, and [`ClockRing`](../../src/ds/clock_ring.rs) keeps
Clock-style state in a single contiguous array. See
[`docs/policy-ds/`](../policy-ds/README.md) for the full primitive
catalog. Cache misses caused by your own data structure are as bad
as upstream misses.

*Realised by:* D6 (slab / arena / intrusive layout).

### 3. Concurrency Strategy Is Core Design, Not a Wrapper

Locking strategy shapes everything. Options: global lock (simple,
dies under contention), sharded caches (per-shard lock), and
lock-free (hard in Rust, only worth it if contention dominates).
Cachekit ships the global-lock option as `Concurrent*` wrappers
(`parking_lot::RwLock` around the single-threaded core, gated by
the `concurrency` feature) and partial sharding at the
data-structure / store layer (`ShardedHashMapStore`,
`ShardedSlotArena`, `ShardedFrequencyBuckets`). A generic
`Sharded<C: Cache<K, V>>` wrapping any policy is not yet shipped;
RCU-style read paths and per-thread caches with periodic merge are
roadmap. Avoid `Arc<Mutex<…>>` in hot paths.

*Realised by:* D4 (`&V` vs `Arc<V>` trait families), D5
(`parking_lot::RwLock`).

### 4. Avoid Per-Operation Allocation

Allocations kill throughput. Pre-allocate entry pools (see
[`SlotArena`](../../src/ds/slot_arena.rs) and the free-list
discipline in [`src/store/slab.rs`](../../src/store/slab.rs)) and
node arrays (intrusive lists thread through arena slots rather
than allocating per-node). Reuse free lists; size slabs once at
construction. Use `Vec` with explicit capacity and `rustc-hash`'s
`FxHashMap` for cheap key hashing in hot-path lookups (see
[hashing.md](hashing.md) for the threat model). Avoid creating
new `Arc` / `String` / `Vec` per lookup and hidden clones of `K`
on the eviction path. If `malloc` shows up in your flamegraph,
your cache is already slow.

*Realised by:* D7 (no per-operation allocation).

### 5. Eviction Must Be Predictable and Cheap

Eviction is the critical slow path; O(1) is the goal for hot-path
policies. Maintain direct indices / `Handle`s to eviction candidates
([`src/store/handle.rs`](../../src/store/handle.rs); the
[`EvictingCache`](../../src/traits.rs) capability trait carries
`evict_one`), eviction lists or clock hands (intrusive list head,
`ClockRing` hand), and lazy heaps where amortised O(log n) is
acceptable ([`LazyMinHeap`](../../src/ds/lazy_heap.rs); used by
Heap-LFU and TTL). `NruCache` deliberately trades that guarantee for
simplicity and has O(n) worst-case eviction. Be careful with
background eviction threads and unbounded lazy cleanup; bound it with
rebuild thresholds (e.g. `LazyMinHeap::with_auto_rebuild`). For
latency-sensitive caches, eviction cost must be comparable to lookup
cost, not orders of magnitude higher.

*Realised by:* D8 (O(1) eviction via direct handles).

### 6. Metrics Are Not Optional

You cannot tune what you do not measure. Track at least hit /
miss rate, eviction count and reason (capacity vs expiration), and
insert / update rate. Cachekit exposes these in two layers: an
unconditional store-layer baseline
([`StoreMetrics`](../../src/store/traits.rs), seven counters that
ship in every build) and a feature-gated policy-layer hierarchy
(per-policy `*Metrics` recorders, `*Snapshot` types, Prometheus
exporter) behind the `metrics` Cargo feature. Lightweight counters
on the hot path, detailed metrics behind a feature flag — see
[Metrics design](metrics.md) for the recorder / snapshot /
exporter split. Metrics should guide design decisions, not justify
them afterward.

*Realised by:* D11 (two-layer metrics).

### 7. Separate Policy From Storage

Design in layers: storage layer (how entries live in memory,
allocation, layout, indexing — [`src/store/`](../../src/store);
design rationale in [storage.md](storage.md)); policy layer (LRU,
FIFO, LFU, LRU-K, 2Q, ARC, CAR, Clock, Clock-PRO, S3-FIFO, … —
manipulates metadata and ordering only,
[`src/policy/`](../../src/policy)); capability layer (opt-in
extension traits — `RecencyTracking`, `FrequencyTracking`,
`HistoryTracking`, `ExpiringCache` — implemented when the signal
exists, which is how `Expiring<C>` composes over any policy
without touching policy code); integration layer (ties application
objects to cache entries via [`CacheBuilder`](../../src/builder.rs)
and the `DynCache` runtime dispatcher).

*Realised by:* D1 (policy / storage separation), D2 (object-safe
kernel + capability traits), D9 (`DynCache` enum dispatch).

### 8. Beware of "Nice" Rust APIs in Hot Paths

Ergonomics often cost performance. Avoid in critical loops: heavy
generics causing code bloat across many monomorphisations, trait
objects for hot dispatch, closures capturing state, and iterator
chains where a plain `for` loop would do. Prefer explicit loops,
concrete types and monomorphised fast paths, and enum dispatch
over `Box<dyn Trait>` when polymorphism is needed at the edges —
this is exactly the trade `DynCache` makes (D9). You can wrap
fast internals in nice APIs at the edges.

*Realised by:* D9 (enum dispatch over `Box<dyn Cache>`).

### 9. Scans Are the Enemy of Caches

Large sequential reads destroy LRU-style caches. Solutions:
scan-resistant policies (`LRU-K`, `2Q`, `SLRU`, `ARC`, `CAR`,
`Clock-PRO`, `S3-FIFO`, `Heap-LFU` — all implemented today),
explicit "scan mode" hints from the caller or workload layer, and
bypass cache for known one-shot reads. If you ignore scans, your
cache will look great in microbenchmarks and terrible in
production.

*Realised by:* the scan-resistant policy catalog itself;
benchmarked under the workloads in
[benchmarks/workloads.md](../benchmarks/workloads.md).

### 10. Benchmark Like a System, Not a Library

Do not rely on uniform-random key benchmarks. Use Zipfian
distributions, mixed read / write workloads, scan + point lookup
mixtures, and time-varying hot sets. Measure throughput, tail
latency, memory overhead, and eviction cost. Cachekit's benchmark
harness covers these dimensions — see
[benchmarking.md](benchmarking.md) for the design (benchmark
layers, monomorphic policy registry, JSON artifact schema),
[`docs/benchmarks/workloads.md`](../benchmarks/workloads.md) for
the workload catalog, and the runners under
[`benches/`](../../benches). A cache that is 5 % faster on
uniform-random keys but 50 % worse under scans is a bad cache.

*Realised by:* the benchmark harness; design captured in
[benchmarking.md](benchmarking.md).

### 11. Rust Hot-Path Hazards Beyond Allocation

`Arc` is expensive in hot paths; minimise it and lift `Arc::clone`
out of inner loops. Fight borrow-checker-induced indirection with
index-based access (`Handle`s, slot indices) instead of `&mut`
chains; use interior mutability only where unavoidable, preferring
`Cell<T>` over `RefCell<T>` when `T: Copy` and atomics when the
value lives behind a shared reference. Beware hidden clones
(particularly of keys on the eviction path), trait-object dispatch
on read / insert (a specific instance of §8), and over-generic
designs whose monomorphisation cost dwarfs their benefit. Rust can
match C on hot paths, but only when systems-level discipline
survives contact with the type system.

*Realised by:* D6 (slab / arena / intrusive), D9 (enum dispatch).

### 12. Design for Failure Modes

Ask: what happens under memory pressure, when eviction cannot keep
up, and under pathological access patterns? Add backpressure or
rejection when full (`StoreFull`), bypass modes, and emergency
eviction strategies. Cross-thread back-pressure semantics across a
layered cache stack remain a roadmap topic; today the answer is
`StoreFull` propagation up to the caller. A cache that collapses
under stress is worse than no cache.

*Realised by:* D12 (three-tier error model); see also
[error-model.md](error-model.md).

### 13. Compile-Time and Runtime Composition

Cachekit's externally visible surface is shaped by two composition
mechanisms that together let users pay only for what they use.
**Per-policy feature flags**: every policy is behind a Cargo
feature (`policy-lru`, `policy-s3-fifo`, …), with `policy-all` for
"everything" and a small default of `policy-s3-fifo`, `policy-lru`,
`policy-fast-lru`, `policy-lru-k`, `policy-clock`; optional
capabilities are gated the same way (`metrics`, `concurrency`,
`serde`, `ttl`). One sharp edge worth naming inline: the default
feature set includes `policy-fast-lru`, which is `!Send + !Sync`,
so default-feature `DynCache` is also `!Send + !Sync`. **Capability
traits + runtime dispatch**: extension traits keep optional
behaviour off the core `Cache<K, V>` trait; the builder returns a
[`DynCache<K, V>`](../../src/builder.rs) that dispatches via an
internal enum match rather than `Box<dyn Cache>`. When the builder
is configured with `.with_default_ttl(...)`, it returns a sibling
`DynExpiringCache<K, V>` that threads the expiry check around each
variant's `Cache` call — a worked example of capability composition.

*Realised by:* D2 (capability traits), D9 (enum dispatch), D10
(per-policy feature flags), D13 (TTL as decorator).

## See Also

Design docs:
- [Concurrency](concurrency.md) — `Concurrent*` wrappers, `RwLock`
  discipline, sharded primitives, `ConcurrentCache` marker
- [Cache trait hierarchy](trait-hierarchy.md) — `Cache<K, V>` kernel,
  capability traits, read / mutate split, object safety
- [Builder and runtime dispatch](builder-and-dyn-dispatch.md) —
  `CachePolicy`, `DynCache`, enum-vs-`Box<dyn>` trade-off, adding
  new policies
- [Weighted eviction](weighted-eviction.md) — `WeightStore` dual
  limits, weight-function contract, GDS / GDSF pre-staging
- [Metrics](metrics.md) — recorder / snapshot / exporter split,
  `MetricsCell`, Prometheus exporter, feature gating
- [Error model](error-model.md) — panic vs `Result` discipline,
  four error types, debug-only invariant checks
- [Benchmarking](benchmarking.md) — benchmark layers, monomorphic
  policy registry, JSON artifact schema, reproducibility rules
- [Hashing and key identity](hashing.md) — hasher choices,
  `KeyInterner`, `ShardSelector`, HashDoS trade-offs
- [Sharding](sharding.md) — current sharded primitives, routing,
  capacity semantics, roadmap for sharded caches
- [Storage layer](storage.md) — store trait family
  (`StoreCore` / `StoreMut` and their concurrent peers), concrete
  stores, `StoreMetrics` baseline, `WeightStore`'s divergence
- [Serialization](serialization.md) — current `serde` surface,
  cache-state persistence boundaries, TTL and hash-seed rules
- [Non-goals](non-goals.md) — explicit boundaries for what cachekit
  does not try to be
- [TTL](ttl.md) — applied example of every layer above
- [Doc style guide](style-guide.md)

Reference docs:
- [Policy overview](../policies/README.md) and
  [roadmap](../policies/roadmap/README.md)
- [Policy data structures](../policy-ds/README.md)
- [Stores](../stores/README.md)
- [Read-only traits](../guides/read-only-traits.md)
- [Choosing a policy](../guides/choosing-a-policy.md)
- [Benchmarks overview](../benchmarks/overview.md) and
  [workloads](../benchmarks/workloads.md)
- [Rust API Guidelines checklist](https://rust-lang.github.io/api-guidelines/checklist.html)
