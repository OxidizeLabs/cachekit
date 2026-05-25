# TTL / Time-Based Expiration — Design Notes

> Status: **Phase 1 implemented and shipped behind the `ttl` feature flag.**
> Phase 2 (per-policy embedded `expires_at`, `ConcurrentExpiring`, timer-wheel
> swap-in, `serde`) is deferred. Companion to the implementation tracker at
> [`docs/policies/roadmap/ttl.md`](../policies/roadmap/ttl.md).

TTL is **not** a replacement policy; it is an expiration rule that coexists
with an eviction policy. This document captures the rationale behind how TTL
is wired into `cachekit` and preserves the project's invariants:

- policy ↔ storage separation (see [`src/store/traits.rs`](../../src/store/traits.rs))
- allocation-free hot paths
- O(1) eviction with index/handle indirection
- explicit, opt-in concurrency and metrics

## Current State

Phase 1 is fully landed behind the `ttl` feature flag
([`Cargo.toml`](../../Cargo.toml)):

- [`src/time.rs`](../../src/time.rs) — `Clock` trait, `StdClock`, `MockClock`
  (ms ticks; `&self`; `Send + Sync`; saturating arithmetic).
- [`src/ds/expiration_index.rs`](../../src/ds/expiration_index.rs) —
  `ExpirationIndex<K>` wrapping `LazyMinHeap<K, u64>` with auto-rebuild and
  TTL-specialised operations (`set_deadline`, `next_deadline`, `pop_expired`,
  `drain_expired`).
- [`src/ds/lazy_heap.rs`](../../src/ds/lazy_heap.rs) — added
  `LazyMinHeap::peek_best` so `ExpirationIndex` reads stale-tombstoned
  roots in place without touching heap internals.
- [`src/traits.rs`](../../src/traits.rs) — `Tick`, `TtlStatus`, and the
  `ExpiringCache<K, V>` capability trait alongside `RecencyTracking` etc.
- [`src/policy/expiring.rs`](../../src/policy/expiring.rs) —
  `Expiring<C, K, V, T = StdClock>` decorator with the ordering invariant
  and logical-read/physical-purge split.
- [`src/builder.rs`](../../src/builder.rs) — `impl Cache<K, V>` for
  `DynCache<K, V>`, `CacheBuilder::with_default_ttl(Duration)` returning
  `ExpiringBuilder`, and `DynExpiringCache<K, V>` (private-constructor
  wrapper around `Expiring<DynCache<K, V>, _, _, StdClock>`).
- [`src/prelude.rs`](../../src/prelude.rs) — re-exports `Clock`,
  `StdClock`, `MockClock`, `Tick`, `TtlStatus`, `ExpiringCache`,
  `DynExpiringCache`, and `ExpiringBuilder` under the `ttl` feature.
- [`tests/ttl_integration_test.rs`](../../tests/ttl_integration_test.rs) —
  integration tests plus proptest invariants over expiry order.
- [`benches/ttl_overhead.rs`](../../benches/ttl_overhead.rs) — Zipfian /
  scan / mixed workloads comparing `LruCore<u64, u64>` against
  `Expiring<LruCore<u64, u64>>` with a 60-second default TTL.

Phase 2 has **not** landed:

- Per-policy embedded `expires_at: u64` in `LruCore::Node` /
  `S3FifoCache::Node` (gated on bench results from Phase 1).
- `ConcurrentExpiring<C>` returning owned/`Arc<V>` values.
- Timer-wheel swap-in for `ExpirationIndex`.
- `serde` support for `Tick` / `TtlStatus`.

`CacheInner` still wires only 17 of the 18 policy modules under
`src/policy/`; `policy::car` is gated by `policy-car` but absent from
`DynCache`. TTL-via-builder for CAR remains gated on closing that gap.

---

## 1. Design Patterns

Five viable patterns, ordered roughly by invasiveness.

### a) Decorator / wrapper cache (chosen)

A new struct `Expiring<C, K, V, T>` owns an inner `C: Cache<K, V>` plus a
per-key expiry index, intercepting `get` / `peek` / `insert` / `remove` to
consult the index. This is what shipped — see
[`src/policy/expiring.rs`](../../src/policy/expiring.rs).

```rust
pub struct Expiring<C, K, V, T = StdClock> {
    inner: C,
    index: ExpirationIndex<K>,
    clock: T,
    default_ttl_ticks: Option<u64>,
    #[cfg(feature = "metrics")]
    expirations: u64,
    _value: PhantomData<fn() -> V>,
}

impl<C, K, V, T> Cache<K, V> for Expiring<C, K, V, T>
where
    C: Cache<K, V>,
    K: Eq + Hash + Clone,
    T: Clock,
{ /* … */ }
```

`K` must appear as a type parameter on the wrapper itself because the index
is keyed by `K`; threading it only through the `Cache<K, V>` impl is not
enough. `V` is recorded as `PhantomData<fn() -> V>` so the inner cache's
value type is fixed at construction without dragging `V` into auto-trait
bounds.

- **Pros:** zero churn on the 18 existing policies; opt-in; composes with
  `DynCache`; matches the policy/storage separation rule in `.cursorrules`.
- **Cons:** when the index is hash-keyed (e.g., `LazyMinHeap<K, u64>`),
  reads pay one extra hash probe; when the index is intrusive over a
  shared `SlotArena` (option E in §3) reads pay only a pointer compare.
  Two sources of truth that must stay consistent; cannot piggy-back on
  intrusive list slots inside `LruCore` / `S3FifoCache` without crossing
  the wrapper boundary.
- **Ordering invariant:** the wrapper must always remove from the inner
  cache *before* removing from the expiration index. A panic in the inner
  removal leaves the index pointing at a key the cache still holds (next
  `pop_expired` will be a no-op or remove the now-stale entry), which is
  recoverable. The reverse order leaves the index missing a key the cache
  still holds, which silently loses the deadline. Document and test this
  ordering — it is the single non-obvious correctness rule of the wrapper.
- **Important semantic constraint:** today's [`Cache`](../../src/traits.rs)
  trait has `peek`, `contains`, and `len` as `&self` methods. A decorator
  cannot physically remove expired entries from those methods unless it adds
  interior mutability. Shipped behaviour: `peek` and `contains` are
  *logical* reads — expired entries are invisible to them, while physical
  cleanup happens on `get`, `insert`, `remove`, `clear`, or explicit
  `purge_expired`. **Decision (shipped):** `Cache::len` returns physical
  occupancy. `Expiring::live_len(&self) -> usize` is exposed as an
  inherent method that walks the expiration index once (O(n), no
  allocation, `&self` — no internal mutation needed because
  `ExpirationIndex::iter` is borrow-only). The distinction is documented
  on both `Cache::len` and `Expiring::live_len`.
- **Mutation semantics:** expired-but-resident entries should behave as
  logically missing. `get` / `remove` should purge and return `None`;
  `insert` / `insert_with_ttl` should purge the stale value before inserting
  and return `None` rather than exposing a value that was already expired.
  The only time insertion/removal returns the previous value is when the prior
  entry was still live at the operation's `now`.

### b) Capability trait + per-policy implementation

Add an extension trait alongside `RecencyTracking` / `FrequencyTracking`:

```rust
pub trait ExpiringCache<K, V>: Cache<K, V> {
    fn insert_with_ttl(&mut self, key: K, value: V, ttl: Duration) -> Option<V>;
    fn ttl_status(&self, key: &K) -> TtlStatus;
    fn set_ttl(&mut self, key: &K, ttl: Duration) -> bool;
    fn purge_expired(&mut self) -> usize;
}

pub enum TtlStatus {
    Missing,
    Immortal,
    Expired,
    Live { remaining: Duration, deadline: Tick },
}
```

See §4(a) for the `Tick` newtype rationale. Each policy embeds an
`expires_at: u64` field in its node struct (`Node<K, V>` in
[`src/policy/lru.rs`](../../src/policy/lru.rs),
[`src/policy/s3_fifo.rs`](../../src/policy/s3_fifo.rs), etc.) and shares an
`ExpirationIndex` data structure. The embedded `u64` is the in-memory
representation; the public `TtlStatus::Live` surface still hands callers a
`Tick`, not a raw `u64`.

- **Pros:** optimal layout — expiration co-located with the existing
  slot/node; `purge_expired` can interact with policy ordering (e.g., update
  the LRU list in place).
- **Cons:** 18 policies × N integration points; touches every
  `Cache::insert` / `get`; harder to gate cleanly behind a feature flag.

### c) Storage-level TTL

Move expiration into a new `ExpiringStore<S>` that wraps a
`StoreCore` / `StoreMut`. Policies stay TTL-unaware; the store returns `None`
(and increments `evictions`) on expired reads.

- **Pros:** pure to the policy ↔ storage separation; works for *every*
  policy without code change; natural fit for `HashMapStore`.
- **Cons:** policies that hold their own metadata for a now-expired key
  (intrusive list links, frequency buckets, ghost entries) end up with
  dangling metadata until the policy notices on its next eviction. Needs an
  "evict by key" callback on the policy.

### d) Observer / callback (entry-aware policy)

Combine (b) and (c) by giving policies an `on_evict(key)` hook the store can
invoke when it lazily detects expiration.

- **Pros:** keeps both sides consistent.
- **Cons:** requires a new policy trait and re-plumbing; adds an indirection.

### e) Trait-object mixin (not recommended)

A `Box<dyn ExpirationIndex>` injected into any policy. Conflicts with the
.cursorrules guidance to "minimize Arc usage in hot paths" and "avoid heavy
Rust ergonomics in hot loops (trait objects, …)".

### Recommendation (shipped as Phase 1)

(a) was shipped as the `ttl` feature, and the wrapper deliberately offers
*logical* expiration over the current `Cache` trait rather than a
zero-overhead embedded TTL.

For builder integration, the shipped approach is a **hybrid of options
(1) and (2)** from §4(c):

- `DynCache<K, V>` got an `impl Cache<K, V>` so `Expiring<DynCache, …>`
  type-checks (option 1's mechanism — single delegation, no per-policy
  match-arm duplication outside the existing `DynCache` boilerplate).
- A separate public type `DynExpiringCache<K, V>` wraps
  `Expiring<DynCache<K, V>, K, V, StdClock>` with the inner field
  **private**; its only constructor is `ExpiringBuilder::build`
  (option 2's surface — no public way to feed a `DynExpiringCache` back
  into an `Expiring`).

`DynExpiringCache` does **not** implement `Cache<K, V>`. Because
`Expiring<C>` requires `C: Cache<K, V>`,
`Expiring<DynExpiringCache>` is structurally unrepresentable, which
restores the "one TTL layer" guarantee that pure option (1) couldn't
give without a runtime check. The cost is one extra public type plus
a thin mirror of `DynCache`'s inherent methods.

Phase 2: where profiling justifies it, embed `expires_at` into specific
policies (b) — LRU, FastLRU and S3-FIFO are the high-value targets. The
embed must be opt-in per-node so non-TTL users do not pay 8 bytes per
entry (see §6, step 7).

---

## 2. Which Policies Can Use It

TTL is orthogonal to eviction, but the *interaction* differs by policy.

| Policy | Embedded TTL fit | Interaction notes |
|---|---|---|
| FIFO, LIFO | Trivial — single VecDeque/Vec; one extra `u64` per entry | Eviction order independent; expire-first then evict-by-policy |
| LRU, FastLRU | Excellent — `Node<K,V>` already has `prev/next/key/value`, add `u64` | Expired LRU node short-circuits `pop_lru` |
| S3-FIFO | Excellent — slot arena nodes; expiry can drop from Small/Main without changing admission history | Ghost list should track *capacity evictions*, not TTL expiry |
| LRU-K, SLRU, 2Q | Natural — multi-segment node already exists; expiry can drop probationary entries cheaply | Promote-and-expire interaction matters: do not promote already-expired |
| Clock, Clock-PRO, NRU | Natural — clock entry already has a reference bit; add `u64`; sweep can expire on the way past | Clock-PRO's adaptive logic must treat expiry-evictions separately from cold-list evictions |
| LFU, Heap-LFU, MFU | Works, but TTL-evictions distort frequency stats. Either decay frequency on expire or accept skew | Co-locate with the existing `LazyMinHeap` if desired |
| ARC, CAR | Same as LFU — adaptive parameter `p` should not move on a TTL eviction | Track TTL-evictions separately to avoid mis-tuning. CAR is not currently a `DynCache` variant; TTL-via-builder for CAR is gated on closing that gap first. |
| Random | Trivial; no ordering to corrupt | — |

The least useful combinations are MFU and pure ARC where TTL competes with
the policy's signal. Document the warning rather than disabling.

---

## 3. Data Structures & Algorithms

The codebase already owns most of the building blocks: `SlotArena`,
`LazyMinHeap`, `IntrusiveList`, `GhostList`, `ClockRing`. Concrete options
for the expiration index follow.

### A) Lazy min-heap of `(expires_at, key)` (shipped)

`ds::LazyMinHeap<K, S>` already existed at
[`src/ds/lazy_heap.rs`](../../src/ds/lazy_heap.rs) and explicitly listed TTL
in its use cases. Insertion is O(log n); `pop_best` is amortized O(log n);
`update` is O(log n) with `maybe_rebuild` to bound staleness.

The shipped `ExpirationIndex` wrapper lives at
[`src/ds/expiration_index.rs`](../../src/ds/expiration_index.rs):

```rust
pub type Deadline = u64;

pub struct ExpirationIndex<K> { /* LazyMinHeap<K, Deadline> */ }

impl<K: Eq + Hash + Clone> ExpirationIndex<K> {
    pub fn new() -> Self;
    pub fn with_capacity(capacity: usize) -> Self;

    pub fn set_deadline(&mut self, key: K, expires_at: Deadline) -> Option<Deadline>;
    pub fn deadline_of<Q>(&self, key: &Q) -> Option<Deadline>
    where K: Borrow<Q>, Q: Hash + Eq + ?Sized;
    pub fn remove<Q>(&mut self, key: &Q) -> Option<Deadline>
    where K: Borrow<Q>, Q: Hash + Eq + ?Sized;

    pub fn next_deadline(&mut self) -> Option<(&K, Deadline)>;
    pub fn pop_expired(&mut self, now: Deadline) -> Option<(K, Deadline)>;
    pub fn drain_expired(&mut self, now: Deadline)
        -> impl Iterator<Item = (K, Deadline)> + '_;

    pub fn iter(&self) -> Iter<'_, K>;
    pub fn set_auto_rebuild(&mut self, factor: Option<usize>) -> &mut Self;
}
```

Auto-rebuild defaults to factor 2 — stale heap entries are bounded at
`2 * live_len()` — and callers that mutate every entry many times per
epoch can tighten this via `set_auto_rebuild`.

`LazyMinHeap` had destructive `pop_best` but no non-destructive
live-minimum peek. Phase 1 added a
`peek_best(&mut self) -> Option<(&K, &S)>` primitive to `LazyMinHeap`
that drains stale-tombstoned roots in place (mutating because lazy
deletion may need to advance past tombstones, immutable observation
otherwise). `ExpirationIndex::next_deadline` is a thin wrapper around
it, so the wrapper does not couple to the heap's internal staleness
representation. The primitive is reusable outside TTL.

- **Pros:** smallest delta — reuse an existing primitive, single allocation
  pool, no clock-tick budget.
- **Cons:** `purge_expired(now)` is O(k log n) for k newly-expired entries;
  per-insert log n is heavier than LRU's O(1); keys are cloned into the heap
  and backing `HashMap`; stale heap entries must be bounded with
  `with_auto_rebuild` / `maybe_rebuild`.

### B) Hashed timer wheel (single wheel)

N slots, each a `Vec<K>` (or `IntrusiveList<SlotId>`); insert at
`slot = (expires_at / tick) % N`; on advance, drain the current slot.

- **Pros:** O(1) insert; O(1) per-tick expire; no per-insert log.
- **Cons:** bounded TTL range = `N * tick`; long TTLs need overflow lists;
  one-shot timers only.

### C) Hierarchical timer wheel (Linux-style)

Multiple wheels (e.g., 256 ms, 16 s, 1024 s, …) cascading on overflow.

- **Pros:** O(1) amortized for arbitrary TTL ranges; widely deployed (Netty,
  the older Tokio `time` module).
- **Cons:** most complex; cascading is bookkeeping-heavy and increases code
  size; overkill for a cache library.

### D) Sorted index over `expires_at`

A `BTreeMap<u64, SmallVec<[K; 4]>>` or `IntrusiveList` per bucket time.

- **Pros:** easy `purge_expired(now)` via `range(..=now)`; predictable.
- **Cons:** `BTreeMap` is allocation-heavy and branchy; violates the
  "favor memory layout efficiency" rule.

### E) Intrusive expiry list per slot/segment

For `LruCore`-style policies, a *second* doubly-linked list through the same
`SlotArena`, ordered by insertion-time TTL. If TTL is uniform per-cache (a
single global `default_ttl`), this becomes O(1): a simple FIFO over
expiration order, no priority queue needed.

- **Pros:** zero heap allocation; O(1) for the common "all entries share the
  same TTL" case; perfect fit for `SlotArena` + `IntrusiveList`.
- **Cons:** does not help when TTL is per-entry-variable.

### F) Generation / epoch counter (invalidation, not TTL)

Tag every entry with the cache's `epoch`; bump epoch to invalidate
everything; lazy purge on access.

- **Pros:** O(1) full invalidation, useful primitive for "burst flush".
- **Cons:** not real TTL — closer to a `clear`-on-mismatch.

### Time source

Do not hardcode `std::time::Instant`. Introduce a small `Clock` trait so
tests/benches can use a mock and so users on `no_std`-adjacent targets can
plug their own:

```rust
pub trait Clock {
    /// Monotonic ticks. Recommended unit: milliseconds.
    fn now(&self) -> u64;
}
```

- `StdClock(Instant)` is the default.
- `MockClock(AtomicU64)` is essential for deterministic tests, `proptest`
  strategies, and fuzz seeds.
- The trait deliberately takes `&self` (not `&mut self`). This keeps
  `Cache::peek`/`Cache::contains` capable of consulting the clock through
  shared references, and it keeps the clock free to live behind the same
  read lock as the inner cache. The cost is that any clock with mutable
  state (notably `MockClock`) must use interior mutability — `AtomicU64`
  is the recommended choice because it is also `Send + Sync` and so
  composes with the `Concurrent*` wrappers without further work.
- Storing `u64` ticks (not `Instant`) shrinks the hot path to a single
  integer compare (one branch in any sane codegen), avoids the
  `Option`-returning `Instant::checked_duration_since` round-trip, and
  keeps the deadline cheaply comparable, serializable, and 8-byte aligned.

### Algorithm cheat sheet

- **On insert:** convert `ttl` to ticks, compute
  `expires_at = clock.now().checked_add(ttl_ticks)`, and index the deadline.
  On overflow, **saturate to `u64::MAX`** with documented "effectively
  never expires" semantics. Saturation (rather than returning an error)
  is chosen because `insert_with_ttl` returns `Option<V>` with no error
  channel; changing the return type to `Result<Option<V>, _>` for a
  failure mode that only triggers on TTLs of ≥500 million years is a poor
  trade. Document this in the rustdoc so callers passing
  `Duration::MAX` get expected behavior.
- **On zero TTL:** do not store a new entry. For an existing key,
  `insert_with_ttl(key, value, Duration::ZERO)` should remove the existing
  entry and return the previous value only if that entry is still live. If it
  is already expired, purge it and return `None`. Per-entry TTL always
  wins over `default_ttl`, including `Duration::ZERO` — i.e. a caller
  explicitly opting into immediate expiry must not be silently upgraded
  to the default TTL.
- **On read:** if `entry.expires_at <= clock.now()` → policy-remove + count
  as miss. The comparison is `<=` (not `<`) because the deadline is the
  first tick at which the entry is no longer live — equality means "this
  tick has begun and the entry is already past it." With millisecond
  ticks and `Duration::ZERO` defined as immediate expiry, `<=` is the
  only choice consistent with both: a one-tick TTL inserted at tick `t`
  expires at exactly tick `t + 1`. `peek` / `contains` may report the
  key as absent without removing it when only `&self` is available.
- **On remove:** an expired resident entry is purged and returns `None`;
  callers should not observe stale values through mutation APIs.
- **On insert/update:** check the prior entry's deadline before returning the
  replaced value. Replacing a live entry returns `Some(old_value)`; replacing
  an expired resident entry purges it first and returns `None`.
- **Periodic (or on insert when full):**
  while `ExpirationIndex::next_deadline()` returns a deadline `<= now`,
  call `pop_expired(now)` and remove that key from the wrapped cache.
  This is exactly what `Expiring::purge_expired` does today.
- **Eviction precedence:** "evict expired first, then policy victim" — the
  rule already documented in
  [`docs/policies/roadmap/ttl.md`](../policies/roadmap/ttl.md).

---

## 4. Unified API & Integration

`cachekit` already has a strong pattern for optional capabilities:
extension traits in [`src/traits.rs`](../../src/traits.rs) and a `DynCache`
enum in [`src/builder.rs`](../../src/builder.rs). TTL fits the same shape.

### a) New capability trait alongside the others

```rust
pub trait ExpiringCache<K, V>: Cache<K, V> {
    fn insert_with_ttl(&mut self, key: K, value: V, ttl: Duration) -> Option<V>;
    fn ttl_status(&self, key: &K) -> TtlStatus;
    fn set_ttl(&mut self, key: &K, ttl: Duration) -> bool;
    fn purge_expired(&mut self) -> usize;
}

/// Monotonic deadline expressed in the cache's tick unit.
///
/// Wrapping the raw `u64` keeps the tick representation out of the
/// public API surface; users compare it via `Tick`'s methods or convert
/// to/from `Duration` via the cache.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct Tick(u64);

pub enum TtlStatus {
    Missing,
    Immortal,
    Expired,
    Live { remaining: Duration, deadline: Tick },
}
```

This sits beside `RecencyTracking` / `FrequencyTracking` /
`HistoryTracking` and stays object-safe. Prefer `ttl_status` over separate
`expires_at` / `ttl_remaining` methods so users can distinguish missing keys,
non-expiring keys, expired-but-not-yet-purged keys, and live TTL entries.

Exposing `Tick` rather than a bare `u64` matters because the tick unit
(ms vs. ns vs. wall-clock seconds) is a private implementation choice.
The newtype lets the unit move without breaking downstream callers, and
gives a natural place to hang `Tick::saturating_add(Duration)`,
`Tick::as_duration_since(&self, Clock)`, and `serde` glue.

**Precedence rule for `default_ttl` and per-entry TTL:** per-entry TTL
always wins. Calling `insert_with_ttl(k, v, Duration::ZERO)` on a cache
with `default_ttl(Some(60s))` configured must produce immediate expiry,
not a 60-second TTL. Plain `insert(k, v)` uses the default; specifying
any TTL — including zero — overrides it. State this explicitly in both
the trait docs and the builder docs because the "default + explicit
override" pattern is ambiguous in user code.

### b) `Clock` abstraction for testability

A `Clock` parameter on `Expiring<C, K, V, T>` (default `StdClock`) and on
any policy that embeds expiry. Mirrors how `RandomCore` keeps `rng_state`
rather than calling `rand::thread_rng()` directly. The shipped
[`Clock`](../../src/time.rs) trait is object-safe and requires
`Send + Sync + Debug`, with `StdClock(Instant)` and `MockClock(AtomicU64)`
covering production and test cases respectively.

### c) Builder integration (shipped)

The shipped surface:

```rust
use cachekit::builder::{CacheBuilder, CachePolicy};
use std::time::Duration;

let mut cache = CacheBuilder::new(1000)
    .with_default_ttl(Duration::from_secs(60))
    .build::<u64, String>(CachePolicy::Lru);

cache.insert(1, "v".to_string());                              // uses default TTL
cache.insert_with_ttl(2, "fast".into(), Duration::from_secs(5)); // overrides
```

`CacheBuilder::with_default_ttl(Duration)` returns an `ExpiringBuilder`
whose `build::<K, V>(policy)` returns a `DynExpiringCache<K, V>` rather
than a `DynCache<K, V>`. The two paths considered were:

1. Add `impl Cache<K, V> for DynCache<K, V>` and wrap the result in
   `Expiring<DynCache<K, V>, …>` directly.
2. Introduce a separate `DynExpiringCache<K, V>` returned by a
   TTL-specific builder path, avoiding a recursive enum at the cost of
   another public type.

The shipped code uses the **mechanism of (1) plus the surface of (2)**.
`DynCache<K, V>` now implements `Cache<K, V>` (see
[`builder.rs`](../../src/builder.rs)), so `Expiring<DynCache, …>` is
internally type-checkable. `DynExpiringCache<K, V>` wraps that
`Expiring<DynCache<K, V>, K, V, StdClock>` with the inner field
**private**, and the `Cache` trait is **not** implemented on the public
wrapper — therefore `Expiring<DynExpiringCache>` is structurally
unrepresentable. The "one new variant, not 18" goal holds: the new type
mirrors `DynCache`'s inherent methods (one delegation per method, not
per policy) and adds the TTL surface.

### d) Feature gating (shipped)

A `ttl` feature lives in [`Cargo.toml`](../../Cargo.toml). TTL depends
only on `std::time` and the existing `LazyMinHeap` — no new runtime
dependency. The following modules are gated behind
`#[cfg(feature = "ttl")]`:

- `src/time.rs` (`Clock`, `StdClock`, `MockClock`, `duration_to_ticks`)
- `src/ds/expiration_index.rs` (`ExpirationIndex`, `Deadline`, iters)
- `src/policy/expiring.rs` (`Expiring`)
- `src/builder.rs::ttl_support` (`ExpiringBuilder`, `DynExpiringCache`)
- `Tick` / `TtlStatus` / `ExpiringCache` items in `src/traits.rs`

When `ttl` is off the `ds` module does not grow a time abstraction and
`DynCache` reverts to its original surface.

Metrics integration is a single counter — `Expiring::expirations: u64`
behind `#[cfg(feature = "metrics")]`, with an accessor that returns 0
when the feature is off so call sites compile unconditionally. Per-policy
metrics structures (`LruMetrics`, `StoreMetrics`) were intentionally
left alone — the decorator-owned counter is enough for Phase 1 because
every TTL purge funnels through the wrapper.

### e) Concurrent variants (Phase 2)

The existing `Concurrent*` wrappers (`ConcurrentLruCache` in
[`src/policy/lru.rs`](../../src/policy/lru.rs), `ConcurrentSlotArena` in
[`src/ds/slot_arena.rs`](../../src/ds/slot_arena.rs),
`ConcurrentClockRing` in [`src/ds/clock_ring.rs`](../../src/ds/clock_ring.rs))
wrap their single-threaded core in `parking_lot::RwLock`.
`ConcurrentExpiring<C>` is **not** shipped yet; when it lands it must
follow two non-negotiable rules:

1. **Return owned/`Arc<V>`, not `&V`.** The `Cache::get(&mut self) -> Option<&V>`
   signature cannot be implemented safely on `ConcurrentExpiring<C>`
   without holding the write lock across the borrow, which serializes
   readers and defeats the point of `RwLock`. `ConcurrentExpiring<C>`
   should expose `fn get(&self, key: &K) -> Option<Arc<V>>` (and the
   sibling mutators), and should **not** implement `Cache<K, V>`. It is a
   concrete type with its own API, mirroring how `ConcurrentLruCache`
   already deviates from `Cache<K, V>`.
2. **Atomic expiry-and-removal.** The expiry check, policy removal, and
   index removal must be one atomic write-locked operation. Splitting
   them allows a concurrent `set_ttl` or `insert` to race with a stale
   expiry decision and produce a "you remove an entry I just renewed"
   outcome. A read-locked fast path (check `expires_at <= now` under a
   read lock, escalate to write lock for the actual removal) is safe so
   long as the write-locked path re-checks the deadline before acting.

Today, callers needing thread-safety wrap a `DynExpiringCache` in
`Arc<RwLock<…>>` exactly as they would for any other `Cache`.

### f) `DynCache` touchpoint (shipped)

`DynCache<K, V>` gained an `impl Cache<K, V>` so that
`Expiring<DynCache, K, V, StdClock>` type-checks — see
[`src/builder.rs`](../../src/builder.rs). All other policy structs
already implemented `Cache<K, V>`; this fills the last gap so the
decorator works through the runtime-selected enum without a per-policy
match arm.

`DynExpiringCache<K, V>` (also in `builder.rs`) lives next to `DynCache`
and mirrors `DynCache`'s inherent methods plus the TTL surface
(`insert_with_ttl`, `ttl_status`, `set_ttl`, `purge_expired`,
`live_len`, `default_ttl`, `expirations`). Its `Debug` impl reports
`default_ttl`, `len`, and `capacity` via `finish_non_exhaustive` —
no keys or deadlines leak.

### g) `prelude.rs` (shipped)

[`src/prelude.rs`](../../src/prelude.rs) re-exports `Clock`, `StdClock`,
`MockClock`, `Tick`, `TtlStatus`, `ExpiringCache`, `DynExpiringCache`,
and `ExpiringBuilder` under `#[cfg(feature = "ttl")]`. `Expiring` itself
is not re-exported — callers reach it through
`cachekit::policy::expiring::Expiring` when they need the raw decorator,
but most code should consume the builder-vended `DynExpiringCache`.

---

## 5. Trade-offs Side-by-Side

### Pattern trade-offs

| Pattern | Hot-path read cost | Hot-path insert cost | Code churn | Fits .cursorrules | When to pick |
|---|---|---|---|---|---|
| (a1) `Expiring<C>` + heap index (A) | +1 hash probe, +1 cmp | O(log n) heap insert + key clone | Low, plus `DynExpiringCache` plumbing | Mostly — preserves separation, not allocation-free | First TTL iteration; benchmark the overhead |
| (a2) `Expiring<C>` + intrusive list (E) | +1 cmp (no hash probe) | O(1) list push | Low–medium (intrusive list reuse) | Yes for uniform TTL | When TTL is uniform across all entries |
| (b) Per-policy embedded | +1 cmp | +1 cmp | High (18 policies × edits) | Yes — best layout | Long-term, after profiling shows the decorator overhead matters |
| (c) Storage-level | +1 cmp inside store | +1 cmp inside store | Medium (new store + callbacks) | Mostly | If also adding size-aware eviction or weight-aware stores |
| (d) Observer/callback | medium | medium | High | Mixed (adds vtable-ish hop) | Multi-tenant caches with heterogeneous policies |
| (e) Trait-object mixin | high (vcall) | high (vcall) | Low | No — violates trait-object/Arc rule | Don't |

### Index data-structure trade-offs

| Index | Insert | Expire batch | Memory / entry | Variable TTL? | Fits `ds` module |
|---|---|---|---|---|---|
| Lazy min-heap (A) | O(log n) + key clone | O(k log n) amortized | HashMap entry + heap entry + sequence; stale entries bounded by rebuild | Yes | Already exists (`LazyMinHeap`) |
| Single timer wheel (B) | O(1) | O(slot size) per tick | ~8 B + slot vec | Bounded | New ds module (`TimerWheel`) |
| Hierarchical wheel (C) | O(1) am. | O(1) am. with cascade | ~8 B + multi-wheel | Yes | New, large addition |
| BTreeMap (D) | O(log n) | O(log n) range drain | ~32 B + alloc | Yes | Doesn't fit (allocations) |
| Intrusive expiry list (E) | O(1) | O(k) | 2 × usize per entry | No (uniform TTL) | Reuses `SlotArena` + `IntrusiveList` |
| Epoch tag (F) | O(1) | O(1) (lazy) | 8 B per entry | N/A | Trivial |

### Eviction-policy interaction trade-offs

- **TTL-eviction-first** is universally good (cheap miss, frees space without
  using the policy's signal), but it changes hit/miss accounting. Either
  expose `evictions_ttl` and `evictions_capacity` separately in metrics, or
  benchmarks will be misread.
- For frequency-sensitive policies (LFU, MFU, ARC, CAR), TTL-evictions
  should **not** update frequency or the adaptive parameter `p`. This is
  the single biggest correctness footgun.
- For LRU-K and SLRU, an expired probationary entry is fine to drop
  silently; an expired protected/hot entry should still call the demotion
  path so segment counters stay consistent.
- For S3-FIFO, an expired entry **should not** seed the ghost list. The ghost
  list should represent keys rejected by capacity pressure, not keys whose
  freshness window elapsed. TTL expiry should remove resident state without
  teaching the admission policy that the key deserved to survive.

### Time source trade-offs

- `Instant` → `u64` ticks: lossy but predictable; `Duration::as_millis()`
  cast to `u64` covers ~585 million-year cache lifetimes (u64::MAX ms ≈
  5.85 × 10⁸ years).
- Calling `Instant::now()` on every `get` is ~15–25 ns on modern Linux
  (vDSO `CLOCK_MONOTONIC`), ~30–60 ns on macOS, and noticeably more on
  Windows. The overhead is small but measurable against an LRU `get` of
  ~30 ns. Consider amortizing via a coarse-clock thread (read once per
  ms) when latency matters; benchmark before optimizing.
- `MockClock(AtomicU64)` is essential for deterministic tests and for
  `proptest` / fuzz strategies. `AtomicU64` (rather than `Cell<u64>` or
  bare `u64`) is required because `Clock::now(&self)` is `&self`-only
  and `MockClock` must remain `Send + Sync` to compose with concurrent
  cache wrappers.

### API trade-offs

- **Single `default_ttl` vs. per-entry `insert_with_ttl`:** support both.
  The default is the 90% case (CDN, API caches); per-entry is needed for
  negative caching ("not found" entries with shorter TTL).
- **Return value of `get` on expired key:** `None` is the right call.
  Returning `Some(&V)` with a side-channel "stale" flag is over-engineering
  for a cache library.
- **Sliding vs. absolute TTL:** pick *absolute* by default (`expires_at` set
  on insert) and add `touch_extends_ttl: bool` as an opt-in; sliding TTL
  silently corrupts time-based bounds.
- **Status reporting:** use a status enum rather than `Option<Duration>` so
  users can distinguish missing, immortal, expired, and live entries.
- **Serialization:** monotonic ticks are not portable across process restarts.
  If `serde` support is added, serialize TTL entries as relative remaining
  durations captured at serialization time, not raw `Instant`-derived ticks.

---

## 6. Phased Roadmap

### Phase 1 — landed

All six items below are merged behind the `ttl` feature flag. The
shipped sequence preserves policy/storage separation and keeps TTL
opt-in, but the decorator does not preserve every hot-path invariant:
inserts pay heap maintenance, the index clones keys, and expired entries
may remain physically resident until a mutable operation purges them.
The Phase 2 benchmark gate (step 7) is therefore part of the design,
not optional cleanup.

1. `src/policy/expiring.rs` — `Expiring<C, K, V, T = StdClock>`
   decorator. `peek` / `contains` are logical reads; `Cache::len`
   reports physical occupancy; `Expiring::live_len(&self)` walks the
   index once for the live count (see §1(a) Decision). The shipped
   signature is `&self`, not `&mut self`, because
   `ExpirationIndex::iter` is borrow-only.
2. `src/ds/expiration_index.rs` — `ExpirationIndex<K>` backed by
   `LazyMinHeap<K, u64>` with auto-rebuild enabled (factor 2 by
   default) to bound stale heap growth. The door is open to swap in a
   timer wheel later. Added a `peek_best` primitive to `LazyMinHeap`
   itself (see §3.A) so `ExpirationIndex::next_deadline` /
   `pop_expired` / `drain_expired` do not couple to heap internals.
3. `src/time.rs` — `Clock` trait + `StdClock` / `MockClock`. `Clock`
   takes `&self`, is object-safe, and requires `Send + Sync + Debug`.
4. `src/traits.rs` — `Tick`, `TtlStatus`, and the `ExpiringCache<K, V>`
   capability trait (object-safe; sits alongside `RecencyTracking`,
   `FrequencyTracking`, `HistoryTracking`).
5. `CacheBuilder::with_default_ttl(Duration)` returns an
   `ExpiringBuilder` whose `build` produces a `DynExpiringCache<K, V>`
   (separate public type with a private inner field; cannot be wrapped
   in another `Expiring`). To make this work without per-policy
   plumbing, `DynCache<K, V>` gained an `impl Cache<K, V>` — see §1
   Recommendation and §4(c).
6. Feature flag `ttl`; counter `Expiring::expirations` (gated on
   `metrics`, accessor returns 0 unconditionally so call sites
   compile); doctests on every public item;
   [`tests/ttl_integration_test.rs`](../../tests/ttl_integration_test.rs)
   exercises the decorator and the builder path under proptest;
   [`benches/ttl_overhead.rs`](../../benches/ttl_overhead.rs) compares
   plain LRU vs. `Expiring<LRU>` under Zipfian / scan / mixed
   workloads.

### Phase 2 — deferred

7. **Embedded `expires_at`.** Profile Phase 1 with the existing
   `ttl_overhead` group and, if the extra hash hit shows up in
   flamegraphs, embed `expires_at: u64` into `LruCore::Node` and
   `S3FifoCache::Node` (the two highest-traffic policies in the
   existing benches at [`benches/`](../../benches)) — but **opt-in per
   node**, not unconditionally. Two viable shapes:
   - A const generic `Node<K, V, const TTL: bool>` so non-TTL caches
     monomorphize to the slimmer layout.
   - A separate type `LruWithTtl<K, V>` (and `S3FifoWithTtl<K, V>`)
     that wraps the slot arena with a parallel `Vec<u64>` keyed by
     slot handle.

   Embedding `expires_at` unconditionally would add 8 bytes per node to
   every LRU and S3-FIFO instance — a 10–25% memory regression for the
   common case of fixed-size value caches — and would regress the very
   benchmarks Phase 1 uses as a gate. The `.cursorrules` "keep metadata
   tight" rule applies here.
8. **`ConcurrentExpiring<C>`** following §4(e) (owned/`Arc<V>` return,
   atomic expiry-and-removal).
9. **Timer-wheel swap-in** as an alternative `ExpirationIndex` backend
   when TTL is uniform and high-throughput (see §3.B/§3.C).
10. **`serde`** support for `Tick` / `TtlStatus`, with the relative-
    duration serialization rule from §5 (API trade-offs).
11. **CAR in `DynCache`.** TTL-via-builder for CAR is blocked until
    `policy::car` becomes a `CacheInner` variant.

---

## 7. Open Questions

Still open:

- Should `purge_expired` only be exposed publicly (as today), or also
  fire on a background thread or on insert-when-full? Phase 1 ships the
  pull-only API and lets the caller drive cadence; configurable
  background sweepers may follow once we have data on what's actually
  needed.
- How should serialization (under `serde` feature) handle `expires_at` —
  the current §5 recommendation is relative remaining duration, but
  restoring long-lived caches may need wall-clock deadlines. Open
  until a serialization API is proposed.
- Is there demand for *negative* TTL (entries that become valid only
  after a delay)? Probably no, but worth confirming before locking the
  API.
- Should `purge_expired` continue to return only a `usize` count, or
  also offer a variant that yields the evicted `(K, V)` pairs? Phase 1
  trait method returns `usize`; users who need the values can build on
  the lower-level `ExpirationIndex::drain_expired` iterator paired with
  manual cache removals.

Resolved during the Phase 1 design pass (kept here for posterity):

- `Cache::len` reports physical occupancy. `Expiring::live_len(&self)`
  walks the index once to give the exact live count; `&self` suffices
  because `ExpirationIndex::iter` is borrow-only — see §1(a).
- Builder integration uses a separate `DynExpiringCache<K, V>` with a
  private inner field plus an `impl Cache<K, V> for DynCache<K, V>`,
  combining option (1)'s plumbing with option (2)'s surface — see §1
  Recommendation and §4(c). `Expiring<DynExpiringCache>` is
  structurally unrepresentable.
- The `Clock` trait lives at `src/time.rs` (top-level), not inside
  `ds`. Revisit if `no_std` support becomes a constraint.
- `Tick` is exposed as a newtype rather than a bare `u64` — keeps the
  tick unit (ms today) private to the `Clock` implementation.

---

## References

### Shipped source

- [`src/time.rs`](../../src/time.rs) — `Clock`, `StdClock`, `MockClock`
- [`src/ds/expiration_index.rs`](../../src/ds/expiration_index.rs) —
  `ExpirationIndex`
- [`src/ds/lazy_heap.rs`](../../src/ds/lazy_heap.rs) — `LazyMinHeap`
  (with the Phase 1 `peek_best` addition)
- [`src/traits.rs`](../../src/traits.rs) — `Tick`, `TtlStatus`,
  `ExpiringCache`
- [`src/policy/expiring.rs`](../../src/policy/expiring.rs) — `Expiring`
  decorator
- [`src/builder.rs`](../../src/builder.rs) — `impl Cache for DynCache`,
  `ExpiringBuilder`, `DynExpiringCache`
- [`src/prelude.rs`](../../src/prelude.rs) — re-exports
- [`tests/ttl_integration_test.rs`](../../tests/ttl_integration_test.rs)
- [`benches/ttl_overhead.rs`](../../benches/ttl_overhead.rs)

### Supporting docs

- [`docs/policies/roadmap/ttl.md`](../policies/roadmap/ttl.md) —
  implementation tracker and Quick Start
- [`docs/policy-ds/lazy-heap.md`](../policy-ds/lazy-heap.md) — lazy heap
  primitive used as the index

### External

- [Wikipedia: Cache replacement policies](https://en.wikipedia.org/wiki/Cache_replacement_policies)
