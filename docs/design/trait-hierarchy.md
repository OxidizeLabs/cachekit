# Cache Trait Hierarchy

> Status: design rationale for the trait surface in
> [`src/traits.rs`](../../src/traits.rs). Companion to the cross-cutting
> principles in [`docs/design/design.md`](design.md) §7 and the concurrency
> rationale in [`docs/design/concurrency.md`](concurrency.md).

cachekit exposes its policies through a small, layered trait hierarchy.
One kernel trait (`Cache<K, V>`) covers what every policy must do;
optional capability traits expose signals that some policies have and
others don't. This document explains why the surface is shaped this
way, what each trait promises, and how to add new capabilities without
breaking the kernel.

## Goals

The trait surface optimizes for four things, roughly in order:

1. **Code written against the kernel survives a policy swap.** Users
   writing `fn warm<C: Cache<K, V>>(c: &mut C, …)` can pick any of
   the 18 implemented concrete policies without changing call sites.
2. **Optional behaviour is visible only when present.** A policy that
   doesn't track frequency should not have a `frequency()` method that
   returns garbage or panics. Capability traits exist so this remains
   true.
3. **The kernel stays object-safe.** `Box<dyn Cache<K, V>>` is needed
   for runtime dispatch (the `DynCache` enum is the chosen alternative,
   but object safety keeps the door open and keeps the trait usable in
   trait objects elsewhere).
4. **The read/mutate split is explicit.** `peek` and `contains` are
   side-effect-free `&self` methods; `get` is `&mut self` because it
   updates policy state. This drops out of point 3 but is worth naming
   on its own because it shapes the concurrent surface
   ([`docs/design/concurrency.md`](concurrency.md)).

## Map of the hierarchy

```text
                            ┌───────────────────────┐
                            │      Cache<K, V>      │   object-safe kernel
                            │  contains, len,       │
                            │  capacity, peek, get, │
                            │  insert, remove,      │
                            │  clear, is_empty      │
                            └───────────┬───────────┘
                                        │ extends
            ┌───────────────┬───────────┼───────────┬──────────────────┐
            ▼               ▼           ▼           ▼                  ▼
    EvictingCache   VictimInspect   RecencyTrack   FrequencyTrack   HistoryTrack
    evict_one()     peek_victim()   touch,         frequency()      access_count,
                                    recency_rank                    k_distance,
                                                                    access_history,
                                                                    k_value

   ConcurrentCache  CacheFactory + CacheConfig    AsyncCacheFuture  (utility traits,
   (unsafe marker)  (constructor abstraction)     (Phase 2)         not extensions)
```

All capability traits in the upper row extend `Cache<K, V>`. They
compose by being implemented additively — `LrukCache` implements
`Cache`, `RecencyTracking`, `FrequencyTracking`, **and**
`HistoryTracking` because it tracks all three signals.

## Layer 1 — `Cache<K, V>`

The kernel trait. Every policy implements it. The full signature lives
in [`src/traits.rs`](../../src/traits.rs); the design decisions worth
naming are:

```rust
pub trait Cache<K, V> {
    fn contains(&self, key: &K) -> bool;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn capacity(&self) -> usize;

    fn peek(&self, key: &K) -> Option<&V>;
    fn get(&mut self, key: &K) -> Option<&V>;
    fn insert(&mut self, key: K, value: V) -> Option<V>;
    fn remove(&mut self, key: &K) -> Option<V>;
    fn clear(&mut self);
}
```

### Object safety

The signature deliberately avoids every feature that would break
object safety:

- No generic methods.
- No `Self` in return position (except by reference, which is allowed).
- No `where Self: Sized` bounds.
- No `impl Trait` returns.

This costs ergonomics — batch operations like `insert_many`,
`get_or_insert_with(closure)`, and `extend(iter)` stay as inherent
methods on each policy rather than landing on `Cache<K, V>` itself.
That trade is intentional: keeping the trait object-safe means
`DynCache<K, V>` is *able* to dispatch through it (even though the
shipped `DynCache` is an enum dispatcher rather than a trait object —
see [`design.md`](design.md) §13). It also keeps `Box<dyn Cache<K, V>>`
available for users writing test harnesses, factories, or registries
that need true type erasure.

### `peek` vs `get` — the read/mutate split

This is the most consequential design decision in the kernel:

- **`peek(&self, …) -> Option<&V>`** does not update recency, frequency,
  reference bits, segment placement, or any policy state. It is the
  honest read.
- **`get(&mut self, …) -> Option<&V>`** is the policy-tracked read.
  An LRU `get` moves the entry to MRU; an LFU `get` bumps the
  frequency counter; a Clock `get` sets the reference bit.

Three things fall out of the split:

1. **`peek` is usable behind a read lock.** Concurrent wrappers
   ([`docs/design/concurrency.md`](concurrency.md)) implement their
   `peek` with `RwLock::read`, allowing multiple readers to proceed
   in parallel. `get` requires `RwLock::write` because it mutates.
2. **`peek` is testable as a pure function.** Hit-rate measurements,
   invariant assertions, and debug prints can use `peek` without
   perturbing the policy.
3. **`len` / `contains` / `capacity` are also `&self`.** They live
   alongside `peek` in the read-locked surface of concurrent wrappers,
   for the same reason.

`contains` is its own method — not `peek(key).is_some()` — because
some policies (S3-FIFO, ARC, CAR with ghost lists) can answer
"is this key resident?" cheaper than they can return a value reference.

### `&V` return positions

Returning `&V` rather than `V`-by-value or `Arc<V>` is the right
choice **for the sequential trait**. Callers who need ownership can
clone; callers who don't pay nothing. The cost shows up in concurrent
wrappers, which cannot return `&V` across a lock boundary — that's
why `Concurrent*` types deviate from `Cache<K, V>` (covered in detail
in [`concurrency.md`](concurrency.md)).

### Default methods

Only `is_empty` has a default. Adding more defaults — even ones that
seem obviously implementable in terms of other methods — would push
performance regressions onto policies that have cheaper specialised
implementations. The hashmap-backed `contains` is faster than the
default `peek(…).is_some()` because it skips fetching the value, and
that difference matters on hot lookup paths.

## Layer 2 — Capability traits

Each capability trait extends `Cache<K, V>` and exposes a signal that
**some but not all** policies have. The rule is:

> Implement the capability trait only when the policy genuinely
> exposes that signal. Do not stub out the methods with sentinel
> returns.

### `EvictingCache<K, V>: Cache<K, V>`

```rust
fn evict_one(&mut self) -> Option<(K, V)>;
```

Forces a single eviction by policy. Returns the evicted entry or
`None` if the cache is empty. Useful for benchmarks ("evict 1 % of
the cache and measure"), background cleanup, and capacity-on-demand
patterns. Implemented by FIFO, LIFO, LRU, FastLRU, Heap-LFU, S3-FIFO,
Clock, Clock-PRO, LRU-K, MFU, MRU, plus the LFU variants.

Policies that **do not** implement it: ARC, CAR, NRU, Random, SLRU,
2Q. The rustdoc on `EvictingCache` lists this set explicitly. The
reason is policy-specific:

- ARC / CAR evict via adaptive choice across two queues; "evict one
  by policy" is ambiguous without an insertion that drives the
  adaptation.
- NRU sweeps reference bits; an isolated `evict_one` may scan the
  whole cache.
- Random has no order; users who want random eviction should call
  `remove(random_key)` themselves.
- SLRU / 2Q's victim depends on which segment is over-quota,
  which only happens under capacity pressure.

The trait is `#[must_use]` on its return because dropping the evicted
entry on the floor is rarely what callers want.

### `VictimInspectable<K, V>: Cache<K, V>`

```rust
fn peek_victim(&self) -> Option<(&K, &V)>;
```

Read-only access to the entry that would be evicted next. Only
implemented by policies whose victim is cheap and stable to identify
without mutating state — FIFO, LIFO, LRU, FastLRU. Clock-family
policies don't implement it because identifying the victim requires
advancing the hand (a mutation). LFU-family policies don't implement
it because the heap top can be a stale entry that hasn't been popped
yet ([`LazyMinHeap`](../../src/ds/lazy_heap.rs)).

The signature is deliberately `&self`-only. Anything that would force
`&mut self` (lazy heap rebuild, clock-hand advance, ARC adaptation)
disqualifies the policy from implementing it.

### `RecencyTracking<K, V>: Cache<K, V>`

```rust
fn touch(&mut self, key: &K) -> bool;
fn recency_rank(&self, key: &K) -> Option<usize>;
```

For policies that order entries by access recency: LRU, FastLRU,
LRU-K. `touch` is `get` without the value lookup — useful when you
want to refresh recency for a key whose value you already have.
`recency_rank` returns 0 for the MRU entry and `len() - 1` for the
LRU. Both are stable across `peek`/`contains`/`len` calls but invalidate
on any `&mut` call.

### `FrequencyTracking<K, V>: Cache<K, V>`

```rust
fn frequency(&self, key: &K) -> Option<u64>;
```

For policies that track access frequency: LFU, Heap-LFU, MFU, LRU-K.
The `u64` return is intentional even though some policies use smaller
counters internally (LFU uses small saturating counters under
[`FrequencyBuckets`](../../src/ds/frequency_buckets.rs)) — exposing
`u64` keeps the trait stable across counter-width changes.

### `HistoryTracking<K, V>: Cache<K, V>`

```rust
fn access_count(&self, key: &K) -> Option<usize>;
fn k_distance(&self, key: &K) -> Option<u64>;
fn access_history(&self, key: &K) -> Option<Vec<u64>>;
fn k_value(&self) -> usize;
```

LRU-K style access-history inspection. Currently implemented only by
`LrukCache`. The `access_history` return is a `Vec<u64>` because the
history is bounded by K and callers typically inspect it as a unit;
exposing the underlying [`FixedHistory`](../../src/ds/fixed_history.rs)
would couple consumers to an internal type.

`k_value()` is on the trait rather than as a constructor argument
witness because LRU-K's K is policy-configured and consumers writing
generic code over `HistoryTracking` need to read it without knowing
the concrete type.

## Why capability traits, not feature flags?

cachekit could expose recency / frequency / history through methods
on `Cache<K, V>` itself, gated by Cargo features. It doesn't, for
three reasons:

- **Compile-time gating doesn't match the actual gating signal.**
  Whether a method is meaningful depends on the **policy**, not on
  the **build**. A `policy-all` build still has policies that can't
  answer `frequency()`.
- **Method-level defaults that return `None` are a footgun.** Code
  that calls `cache.frequency(&k)` on an LRU cache would silently
  return `None` and pass through review.
- **Trait bounds carry information.** `fn warm<C: FrequencyTracking>()`
  documents at the type-system level that the function only makes
  sense for frequency-tracking caches.

The trade is one extra `use` statement at call sites — `use
cachekit::traits::{Cache, RecencyTracking};` — which is a small price
for the correctness gain.

## Utility traits

Three traits live alongside the hierarchy but are not extensions of
`Cache<K, V>`.

### `unsafe trait ConcurrentCache: Send + Sync`

Marker trait, no methods. Implementing it asserts that the type
handles internal synchronization safely. Covered in detail in
[`concurrency.md`](concurrency.md#concurrentcache-marker-trait-not-capability-trait).

### `CacheFactory<K, V>` and `CacheConfig`

```rust
pub trait CacheFactory<K, V> {
    type Cache: Cache<K, V>;
    fn new(capacity: usize) -> Self::Cache;
    fn with_config(config: CacheConfig) -> Self::Cache;
}
```

Constructor abstraction for generic code that needs to build caches
without naming the concrete type. `CacheConfig` is a `#[non_exhaustive]`
struct with builder-style `with_*` methods, mirroring the wider
`CacheBuilder` shape in [`src/builder.rs`](../../src/builder.rs).

In practice most code constructs caches directly (`LruCache::new(…)`)
or through `CacheBuilder`. `CacheFactory` mostly exists for test
harnesses and benchmark runners that want to parameterise across
policies; the trait's `Cache` associated type makes that ergonomic.

### `AsyncCacheFuture<K, V>: Send + Sync`

Phase 2 placeholder. The methods (`supports_async_get`,
`supports_async_insert`) default to `false` and no policy overrides
them. The trait exists so that async-native policies can be added in
the future without breaking the existing surface.

## Read/mutate split rationale (recapitulated)

Worth stating once more in one place: the methods on `Cache<K, V>`
split cleanly into two groups:

| `&self` (read-locked-safe) | `&mut self` (write-locked) |
|----------------------------|----------------------------|
| `contains`, `len`, `is_empty`, `capacity` | `get`, `insert`, `remove`, `clear` |
| `peek`                     |                            |
| (capability) `peek_victim`, `recency_rank`, `frequency`, `access_count`, `k_distance`, `access_history`, `k_value` | (capability) `evict_one`, `touch` |

This is the contract the concurrent wrappers rely on. Adding a new
`Cache` method that mutates state through `&self` (interior mutability)
would break the lock-granularity story; adding one that takes `&mut
self` but doesn't logically mutate would prevent the read-lock fast
path in `Concurrent*` wrappers.

## Object safety vs. ergonomic methods

Some operations naturally belong on `Cache<K, V>` but would break
object safety. They live as inherent methods on each policy instead:

- `extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I)`
- `get_or_insert_with<F: FnOnce() -> V>(&mut self, key: K, f: F) -> &V`
- `insert_many(&mut self, items: impl IntoIterator<Item = (K, V)>)`
  with buffer reuse

The rule: anything taking a generic closure, generic iterator, or
returning `impl Trait` is an inherent method, not a trait method.
The trait stays object-safe; the policy types stay ergonomic.

## Adding a new capability trait

Checklist for new capability traits:

1. **The signal must exist in the implementing policy's metadata.**
   No defaults that return `None`/`0`/`false` for "doesn't apply."
2. **Bound on `Cache<K, V>`.** Capability traits compose with the
   kernel; they don't replace it.
3. **Object safety is optional for capability traits** but
   recommended. Trait objects of capability traits show up rarely;
   ergonomic generic methods are fine.
4. **Name follows the noun-of-the-signal pattern.** `RecencyTracking`,
   `FrequencyTracking`, `HistoryTracking`. New ones should follow
   suit: `WeightTracking`, `CostTracking`, `AdmissionTracking`.
5. **Re-export from `prelude`.** Capability traits live in the same
   `use cachekit::prelude::*;` namespace as the kernel.
6. **Document the implementing-policy set.** The rustdoc on
   `EvictingCache` lists policies that opt out; new traits should
   do the same for the smaller set that opts in.

## Future capability traits

Sketched in priority order:

- **`ExpiringCache<K, V>: Cache<K, V>`** — TTL surface, per
  [`docs/design/ttl.md`](ttl.md) §4(a). Signature:

  ```rust
  fn insert_with_ttl(&mut self, key: K, value: V, ttl: Duration) -> Option<V>;
  fn ttl_status(&self, key: &K) -> TtlStatus;
  fn set_ttl(&mut self, key: &K, ttl: Duration) -> bool;
  fn purge_expired(&mut self) -> usize;
  ```

  Implemented by the `Expiring<C>` decorator over any `Cache<K, V>`.

- **`WeightTracking<K, V>: Cache<K, V>`** — surface for weight-aware
  caches built on [`WeightStore`](../../src/store/weight.rs). Likely
  signature:

  ```rust
  fn weight(&self, key: &K) -> Option<usize>;
  fn total_weight(&self) -> usize;
  fn weight_capacity(&self) -> usize;
  ```

  Needed before GDS/GDSF (roadmap policies) can be expressed
  generically.

- **`AdmissionTracking<K, V>: Cache<K, V>`** — exposes ghost-list /
  admission-history state for ARC, CAR, S3-FIFO, Clock-PRO,
  TinyLFU. Specifically: was this key ever resident, and if so when
  did it leave? Useful for adaptive workloads where the caller
  wants to know whether a miss is a one-hit-wonder or a returning
  member of the working set.

The trait is intentionally not added until a second policy implements
it. The `RecencyTracking` / `FrequencyTracking` / `HistoryTracking`
naming established the convention; adding `WeightTracking` only when
GDS lands keeps the surface honest.

## See also

- [Design overview](design.md) — §7 frames the layering at the
  principles level, §13 covers `DynCache` runtime dispatch
- [Concurrency](concurrency.md) — read/mutate split + `ConcurrentCache`
- [TTL design](ttl.md) — applied example: `ExpiringCache` as a new
  capability trait
- [Read-only traits](../guides/read-only-traits.md) — user-facing
  guidance on the `peek` / `get` split
- [`src/traits.rs`](../../src/traits.rs) — the canonical definitions
- [Storage layer](storage.md) — parallel trait family at the store
  layer (sequential + concurrent), with the same `&V` vs. `Arc<V>`
  split reasoning
- [`src/store/traits.rs`](../../src/store/traits.rs) — canonical
  store-trait definitions
