# Storage Layer

> Status: design rationale for the store trait family in
> [`src/store/traits.rs`](../../src/store/traits.rs) and the concrete
> stores under [`src/store/`](../../src/store). Companion to
> [`design.md`](design.md) §7 (policy/storage separation),
> [`trait-hierarchy.md`](trait-hierarchy.md) (the parallel cache-trait
> family), [`concurrency.md`](concurrency.md), and
> [`weighted-eviction.md`](weighted-eviction.md).

cachekit splits caches into two layers: a *policy* that decides what
to evict and a *store* that owns the keys and values. This doc
covers the store side. It explains why the store traits look the way
they do, what each shipped concrete store is for, how the
sequential/concurrent split mirrors the cache-trait family, and why
[`WeightStore`](../../src/store/weight.rs) deliberately sits outside
the rest of the family.

## Goals

The store layer is shaped around four things:

1. **Decouple ownership from policy.** A policy that doesn't know how
   entries are laid out in memory can be swapped without rewriting
   storage code, and a store can be swapped without rewriting policy
   code. This is the policy/storage separation rule in
   [`design.md`](design.md) §7.
2. **Make capacity refusal explicit, not implicit.** When a store is
   full, it returns [`StoreFull`](../../src/store/traits.rs) rather
   than silently evicting. The caller (policy or user) decides what to
   evict. This is the core of the layering — without it, every store
   would have to ship an eviction strategy.
3. **Mirror the sequential / concurrent split that already exists
   for caches.** The sequential traits return owned `V` and borrowed
   `&V`; the concurrent traits return `Arc<V>`. The reasoning is the
   same as for the cache trait family in
   [`trait-hierarchy.md`](trait-hierarchy.md) — references cannot
   safely outlive lock guards.
4. **Keep the always-on observability minimal but useful.**
   [`StoreMetrics`](../../src/store/traits.rs) ships unconditionally
   with seven counters: `hits`, `misses`, `inserts`, `updates`,
   `removes`, `evictions`, and `expirations`. None of them are
   feature-gated; the richer per-policy metrics hierarchy is, but
   the store-layer baseline is not. `expirations` stays at `0` on
   stores that do not own a TTL surface — the TTL count for an
   `Expiring<…>` decorator is exposed separately via
   [`Expiring::expirations()`](../../src/policy/expiring.rs).

## Map of the hierarchy

```text
Single-threaded (direct ownership)        Concurrent (shared ownership)
────────────────────────────────          ───────────────────────────────

  ┌────────────────────┐                    ┌──────────────────────────┐
  │     StoreCore      │                    │   ConcurrentStoreRead    │
  │  get(&K) -> &V     │                    │  get(&K) -> Arc<V>       │
  │  contains, len,    │                    │  contains, len,          │
  │  capacity, metrics │                    │  capacity, metrics       │
  └─────────┬──────────┘                    └────────────┬─────────────┘
            │ extends                                    │ extends
            ▼                                            ▼
  ┌────────────────────┐                    ┌──────────────────────────┐
  │      StoreMut      │                    │     ConcurrentStore      │
  │  try_insert,       │                    │  try_insert (Arc<V>),    │
  │  remove, clear     │                    │  remove, clear           │
  └────────────────────┘                    └──────────────────────────┘

  ┌────────────────────┐                    ┌──────────────────────────┐
  │   StoreFactory     │                    │ ConcurrentStoreFactory   │
  │ type Store: ...    │                    │ type Store: ...          │
  │ new(capacity)      │                    │ new(capacity)            │
  └────────────────────┘                    └──────────────────────────┘

  StoreMetrics (unconditional, 7 counters)
  StoreFull   (zero-sized error type)
```

Every concrete store implements exactly one column. The
single-threaded stores use direct ownership; the concurrent stores
use `Arc<V>` because borrowed references cannot outlive
`RwLockReadGuard`. See
[`concurrency.md`](concurrency.md#why-concurrent-does-not-implement-cachek-v)
for the equivalent argument at the cache-trait level.

## Layer 1 — `StoreCore` / `StoreMut`

The sequential surface. Three design choices worth naming:

### `&V` return position

[`StoreCore::get`](../../src/store/traits.rs) returns `Option<&V>`.
The borrow is tied to `&self`, so callers can read without cloning.
This is the right shape for a sequential trait because the alternative
(`V` by value) forces `V: Clone` everywhere or hands out `Arc<V>` on
every call.

The concurrent counterpart cannot do this — see
[Layer 2](#layer-2--concurrentstoreread--concurrentstore) below.

### `try_insert` returns `Result<Option<V>, StoreFull>`

Three independent failure modes hide in this one signature:

| Outcome | Return value | Meaning |
|---------|--------------|---------|
| New key fits | `Ok(None)` | Inserted; no previous value |
| Existing key updated | `Ok(Some(old))` | Replaced; old value handed back |
| Store full and key is new | `Err(StoreFull)` | Caller must evict and retry |

Updates to an existing key **always succeed** — they cannot push the
store past capacity by entry count, and the previous value is handed
back as `Ok(Some(old))`. Capacity refusal is a `StoreFull` *only*
when the key is new and inserting it would exceed the entry count.

[`WeightStore`](../../src/store/weight.rs) extends this to a
dual-limit model where updates *can* fail (a larger value replacing a
smaller one can exceed the weight budget). See
[`weighted-eviction.md`](weighted-eviction.md#dual-limit-model) for
the full table.

### No automatic eviction

The store never evicts on its own. Returning `StoreFull` is the
signal to the caller that *they* must decide who to remove. This is
the layering rule from [`design.md`](design.md) §7 made concrete:

- A policy layered over `StoreMut` evicts its chosen victim, then
  retries `try_insert`.
- A user using a `StoreMut` directly evicts a key they pick (random,
  oldest-by-some-criterion, etc.), then retries.

A store that evicted on its own would lock a single eviction
strategy into every consumer, and would prevent layering a *better*
eviction policy on top.

## Layer 2 — `ConcurrentStoreRead` / `ConcurrentStore`

The concurrent surface mirrors the sequential one with two
substitutions:

- Returns are `Arc<V>` rather than `&V` / `V`. References cannot
  outlive the lock guard they were extracted from; `Arc<V>` carries
  ownership safely past lock release.
- All methods take `&self`. Internal synchronization (almost always
  `parking_lot::RwLock`) is the store's responsibility, not the
  caller's. The wrapper does **not** require `&mut self` for
  mutation.

Implementors must be `Send + Sync`. The trait bound is on the trait
declaration itself
(`pub trait ConcurrentStoreRead<K, V>: Send + Sync`), so any code
generic over `ConcurrentStoreRead` automatically requires thread
safety from the implementor.

This is the same shape `Concurrent*` cache wrappers use — see
[`concurrency.md`](concurrency.md#the-dominant-pattern-sequential-core-concurrent-wrapper)
for the parallel reasoning.

## Layer 3 — Factories

```rust,ignore
pub trait StoreFactory<K, V> {
    type Store: StoreMut<K, V>;
    fn new(capacity: usize) -> Self::Store;
}

pub trait ConcurrentStoreFactory<K, V> {
    type Store: ConcurrentStore<K, V>;
    fn new(capacity: usize) -> Self::Store;
}
```

Factory traits exist so generic code can construct a store without
naming the concrete type. They mirror `CacheFactory` in
[`trait-hierarchy.md`](trait-hierarchy.md#cachefactory-and-cacheconfig).
In practice most code constructs stores directly; the factories are
used by test harnesses and benchmark runners that want to
parameterise across store implementations.

## `StoreMetrics`: the always-on baseline

```rust,ignore
#[non_exhaustive]
pub struct StoreMetrics {
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
    pub updates: u64,
    pub removes: u64,
    pub evictions: u64,
    pub expirations: u64,
}
```

Two things distinguish this from the policy-layer metrics in
[`metrics.md`](metrics.md):

- **It ships in every build.** No `#[cfg(feature = "metrics")]`
  gate. The seven counters here are universal enough to be a
  baseline contract every store satisfies.
- **It is read-only at the trait surface.** `StoreCore::metrics()`
  returns a snapshot `StoreMetrics` by value. How a store records
  the increments (plain `u64`, `AtomicU64`, `MetricsCell`,
  `StoreCounters` in
  [`src/store/weight.rs`](../../src/store/weight.rs)) is an
  implementation detail. Concurrent stores typically use
  `AtomicU64`; single-threaded stores use plain `u64` or `Cell<u64>`.

`StoreMetrics` is `#[non_exhaustive]`, so adding a new universal
counter is a minor version bump. The `expirations` field landed
this way (added when the TTL surface needed time-driven removals
distinguished from capacity-driven evictions).

For per-policy detail (recency rank reads, LFU bucket promotions,
S3-FIFO ghost hits) see the policy-layer metrics behind the
`metrics` feature.

## `StoreFull` error semantics

[`StoreFull`](../../src/store/traits.rs) is a zero-sized type that
carries no data. The caller already knows what they tried to insert;
attaching the key/value to the error would force `K: Clone` /
`V: Clone` on the error path for no information gain.

The error is co-located with the trait that returns it
(`src/store/traits.rs`) rather than in `src/error.rs`. The reasoning
matches the broader error model
([`error-model.md`](error-model.md#why-four-error-types-not-one)):
each error type lives near the surface that produces it.

## Concrete stores

Four concrete store types ship today, plus their concurrent
counterparts. Each picks a different point in the memory-layout
space.

| Store | Backing | Key shape | Threading | When to use |
|---|---|---|---|---|
| [`HashMapStore`](../../src/store/hashmap.rs) | `HashMap<K, V, S>` | `K: Eq + Hash` | sequential | Default; any cache where the key drives layout |
| [`ConcurrentHashMapStore`](../../src/store/hashmap.rs) | `RwLock<HashMap<…>>` | same | concurrent | Default concurrent shape |
| [`ShardedHashMapStore`](../../src/store/hashmap.rs) | N `RwLock<HashMap<…>>` shards | same | concurrent, contention-aware | When one `RwLock` is the bottleneck |
| [`SlabStore`](../../src/store/slab.rs) | slab arena with `EntryId` handles | `K: Eq + Hash` | sequential | Policies that need stable `EntryId`s for intrusive metadata |
| [`ConcurrentSlabStore`](../../src/store/slab.rs) | `RwLock<SlabStore>` | same | concurrent | Concurrent slab access |
| [`HandleStore`](../../src/store/handle.rs) | `HashMap<H, Arc<V>>` | opaque handle `H` | sequential | When keys are pre-interned and only the handle is in the hot path |
| [`ConcurrentHandleStore`](../../src/store/handle.rs) | `RwLock<HashMap<H, Arc<V>>>` | same | concurrent | Concurrent variant of the above |
| [`WeightStore`](../../src/store/weight.rs) | `FxHashMap` + per-entry weight | `K: Eq + Hash` | sequential | Variable-size values; byte-budgeted caches |
| [`ConcurrentWeightStore`](../../src/store/weight.rs) | `RwLock<WeightStore>` | same | concurrent | Concurrent variant of the above |

### `HashMapStore`

The default public store. Uses `std::collections::hash_map::RandomState`
by default for HashDoS-resistant hashing on the public surface; users
who control the key source can opt into a faster hasher via
`with_hasher`. See [`hashing.md`](hashing.md) for the threat model.

This is the right choice when keys are typed
(`String`, `u64`, `(TenantId, ResourceId)`, …) and the policy does
not need stable per-entry handles. Most caches built through
`CacheBuilder` end up here either directly or indirectly.

### `SlabStore`

Backs stores in a slab arena. Each entry has a stable `EntryId`
handle that survives mutations to other entries. This is essential
for policies that thread intrusive metadata through entries — LRU's
recency list, S3-FIFO's small/main queues, NRU's reference bit
ring — because pointer chasing without stable indirection makes the
borrow checker rejection-prone and pointer chasing hostile to the
cache hierarchy.

Use `SlabStore` directly when building a policy that wants slot
handles. Most users reach it indirectly through the policy types
that consume it.

### `HandleStore`

A specialised shape: keys are stored elsewhere (typically a
[`KeyInterner`](../../src/ds/interner.rs)) and the store maps
`Handle -> Arc<V>`. The motivation is to avoid cloning large keys
on every operation when many policies (LFU bucket maps,
frequency-bucket arrays, ARC ghost lists) need a compact key proxy
anyway.

`HandleStore` returns `Arc<V>` even in the single-threaded variant.
This is the same divergence
[`WeightStore`](#weightstores-deliberate-divergence) takes, and for
the same reason: the values targeted by this shape (interned blobs,
deduplicated payloads) benefit from cheap shared ownership.

### `WeightStore`'s deliberate divergence

`WeightStore` does **not** implement `StoreCore` / `StoreMut`. It is
a sibling of the trait family, not a subtype. The reasons live in
[`weighted-eviction.md`](weighted-eviction.md) but worth recapping
here:

- It returns `Arc<V>` (not `&V`) even in the single-threaded
  variant. This is necessary for the concurrent variant and the
  single-threaded variant inherits the same shape so users can swap
  between them by changing one type.
- Its `try_insert` enforces a *dual* limit (entry count and weight
  budget). Updates can fail when the weight delta would exceed
  budget. `StoreMut::try_insert`'s contract is "updates always
  succeed," which `WeightStore` cannot honour.
- It takes an `F: Fn(&V) -> usize` weight function. Carrying that
  third type parameter would propagate through every layer of
  `StoreMut`-generic code unnecessarily.

The concurrent variant *does* implement `ConcurrentStoreRead` /
`ConcurrentStore` because those return `Arc<V>` and accept `Arc<V>`
on insert. The asymmetry is awkward but honest — the concurrent
trait family already has the shape `WeightStore` needs.

## Sharded stores

`ShardedHashMapStore<K, V, S>` is the only sharded store that ships
today. It owns N independent `RwLock<HashMap<…>>` shards, each
addressed by hashing the key through a
[`ShardSelector`](../../src/ds/shard.rs).

| Property | Single concurrent | Sharded |
|---|---|---|
| Lock acquisition | One global `RwLock` per op | One shard `RwLock` per op |
| Hot key contention | Yes — all readers/writers compete | Only readers/writers on the same shard |
| Capacity model | Single global cap | Per-shard caps that sum to global cap |
| Eviction quality | Global victim picking | Per-shard victim picking |
| Implementation complexity | Low | Medium |

See [`sharding.md`](sharding.md) for the full discussion. Note that
the sharded primitive lives at the *data-structure* / *store* layer;
a sharded *cache policy* (e.g. `ShardedLruCache`) is roadmap.

## Why not a single unified `Store` trait?

`StoreCore` could in principle subsume `StoreMut` (just make all
methods `&mut self`-or-`&self`). It doesn't, for the same reason
`Cache<K, V>` separates `peek` from `get`: a read-only surface lets
concurrent wrappers acquire only the read lock.

`StoreCore` + `StoreMut` could in principle merge with
`ConcurrentStoreRead` + `ConcurrentStore` via an `Arc<V>`-returning
universal variant. That collapses the sequential `&V` fast path into
an unnecessary `Arc<V>` round-trip, which is exactly what the
sequential `Cache::get -> Option<&V>` shape is trying to avoid.

Two parallel families is the cost of letting both shapes pay only
for what they use.

## Adding a new store

Checklist for landing a new store implementation:

1. **Pick the layer.** Sequential (`StoreCore` / `StoreMut`) or
   concurrent (`ConcurrentStoreRead` / `ConcurrentStore`). Usually
   both, with the concurrent variant wrapping the sequential one in
   `RwLock`.
2. **Implement the read trait first.** `get`, `contains`, `len`,
   `capacity`, `metrics`. Override `metrics()` to expose your
   counters rather than the default-zero implementation.
3. **Implement the mut trait.** `try_insert`, `remove`, `clear`.
   `try_insert` must return `Err(StoreFull)` for new keys at
   capacity; updates to existing keys must not fail (unless the
   store has additional invariants like `WeightStore`'s weight
   budget — document the divergence at the module level).
4. **Add a `StoreFactory` impl** if the store has a stable
   `new(capacity)` constructor and is likely to be parameterised
   over in generic code.
5. **Implement `Send + Sync`** for the concurrent variant. The
   sequential variant typically is not `Sync` (because it holds
   `Cell<u64>` for `MetricsCell` counters or `RefCell` for any
   interior state).
6. **Document the threat model.** Which hasher does the store
   default to? Is it HashDoS-resistant? Are there public surfaces
   that expose internal counters that could leak entry-size
   information? Match the
   [`hashing.md`](hashing.md) discipline.
7. **Add `docs/stores/<name>.md`** following the
   [doc style guide](style-guide.md#design-doc-style). Link the new
   doc from [`docs/stores/README.md`](../stores/README.md).
8. **Write proptest or fuzz coverage** for invariants:
   `len == sum(entries)`, metric counters monotonic,
   `try_insert(k, v)` followed by `remove(k)` round-trips. See
   [`docs/testing/testing.md`](../testing/testing.md) for the
   conventions.

## When not to add a new store

The store layer is small on purpose. Before adding a new store,
check:

- **Is the difference a *policy* difference or a *layout*
  difference?** Different eviction strategies belong above the
  store, not at the store layer.
- **Is the shape already covered by a hasher swap or sharding?**
  `HashMapStore::with_hasher(FxBuildHasher)` and
  `ShardedHashMapStore` cover most of the obvious knobs.
- **Does it justify its own trait-family divergence?**
  `WeightStore` is the precedent for diverging — variable weights
  forced a dual-limit model that `StoreMut` cannot express. New
  stores that fit `StoreMut`'s contract should implement it rather
  than introduce a sibling.

## Failure modes worth naming

- **`StoreFull` from `try_insert` on a `WeightStore`-style store
  with weight budget remaining.** Caller should consult both
  `len()` and (for weight-aware stores) `total_weight()` to know
  which budget bit. The error type is the same; the resolution
  differs.
- **Panic during a user-supplied callback in
  `ConcurrentWeightStore::try_insert`.** The weight function runs
  inside the write lock. Under `panic = "unwind"` the lock is
  released (parking_lot doesn't poison), but the inner state is
  whatever the panicking weight function left it in. Under the
  crate's release-default `panic = "abort"`, the process exits
  before any observer can see partial state. See
  [`error-model.md`](error-model.md#operational-contract-panic-profile).
- **Hash collisions on `FxHashMap`-backed stores under adversarial
  keys.** `WeightStore` and policy-internal maps are the targets;
  see [`hashing.md`](hashing.md#fxhash-hot-internal-default) for
  the trade-off and the user-facing escape hatches.

## See also

- [Design overview](design.md) — §7 frames policy/storage
  separation at the principles level
- [Cache trait hierarchy](trait-hierarchy.md) — parallel trait
  family at the policy layer; the `&V` vs `Arc<V>` reasoning is
  shared
- [Concurrency](concurrency.md) — `Concurrent*` wrapper pattern
  applied at the store layer
- [Weighted eviction](weighted-eviction.md) — `WeightStore`'s
  dual-limit model and deliberate divergence from `StoreMut`
- [Hashing and key identity](hashing.md) — store-level hasher
  defaults and overrides
- [Sharding](sharding.md) — `ShardedHashMapStore` and the roadmap
  for sharded cache policies
- [Metrics](metrics.md) — relationship between the always-on
  `StoreMetrics` baseline and the feature-gated policy-layer
  metrics
- [Error model](error-model.md) — `StoreFull` semantics, panic
  behaviour during user-supplied callbacks
- [Stores reference](../stores/README.md) — runtime-behaviour
  documentation for each concrete store
- [`src/store/traits.rs`](../../src/store/traits.rs) — canonical
  trait definitions
