# TTL / Time-Based Expiration — Design Exploration

> Status: design exploration. Companion to the high-level stub at
> [`docs/policies/roadmap/ttl.md`](../policies/roadmap/ttl.md).

TTL is **not** a replacement policy; it is an expiration rule that coexists
with an eviction policy. This document explores how TTL can be introduced into
`cachekit` while preserving the project's invariants:

- policy ↔ storage separation (see [`src/store/traits.rs`](../../src/store/traits.rs))
- allocation-free hot paths
- O(1) eviction with index/handle indirection
- explicit, opt-in concurrency and metrics

## Current State

- No TTL exists in source today (`rg ttl|expir|Instant` finds only docs and
  benchmark labels).
- A high-level stub already exists at
  [`docs/policies/roadmap/ttl.md`](../policies/roadmap/ttl.md).
- The `ds::LazyMinHeap` primitive at [`src/ds/lazy_heap.rs`](../../src/ds/lazy_heap.rs)
  explicitly lists "TTL expiry heaps" as a use case.
- The capability-trait pattern at [`src/traits.rs`](../../src/traits.rs)
  (`RecencyTracking`, `FrequencyTracking`, `HistoryTracking`) gives a clean
  injection point for an `ExpiringCache` trait.
- The runtime-policy enum at [`src/builder.rs`](../../src/builder.rs)
  (`DynCache` / `CacheInner`) makes a TTL wrapper composable in one variant
  rather than 18 per-policy edits. Note: `CacheInner` currently wires
  17 of the 18 policy modules under `src/policy/`; `policy::car` is not
  yet a variant. Closing that gap is a prerequisite for "TTL works for
  every policy via `DynCache`".

---

## 1. Design Patterns

Five viable patterns, ordered roughly by invasiveness.

### a) Decorator / wrapper cache

A new struct `Expiring<C>` owns an inner `C: Cache<K, V>` plus a per-key
expiry index, intercepting `get` / `peek` / `insert` / `remove` to consult
the index.

```rust
pub struct Expiring<C, K, T = StdClock> {
    inner: C,
    index: ExpirationIndex<K>,
    clock: T,
    default_ttl: Option<Duration>,
}

impl<C, K, V, T> Cache<K, V> for Expiring<C, K, T>
where
    C: Cache<K, V>,
    K: Eq + Hash + Clone,
    T: Clock,
{ /* … */ }
```

`K` must appear as a type parameter on the wrapper itself because the index
is keyed by `K`; threading it only through the `Cache<K, V>` impl is not
enough.

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
  interior mutability. The first slice should define them as *logical* reads:
  expired entries are invisible to `peek` / `contains`, while physical cleanup
  happens on `get`, `insert`, `remove`, `clear`, or explicit
  `purge_expired`. **Decision:** `Cache::len` returns physical occupancy
  — it is cheaper, matches the underlying cache trait, and is the only
  thing implementable through `&self`. Surprise after time advances is
  mitigated by exposing `Expiring::live_len(&mut self) -> usize` as an
  inherent method on the wrapper, which can amortize an internal sweep.
  Document the distinction in both rustdocs.
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

### Recommendation

Ship (a) first as a `ttl` feature, but be explicit that the wrapper gives
logical expiration over the current `Cache` trait rather than a zero-overhead
embedded TTL.

For builder integration, prefer a **separate `DynExpiringCache<K, V>`
type** returned by a TTL-specific builder path over implementing
`Cache<K, V>` for `DynCache<K, V>` and wrapping it. The decisive reason
is that the first option permits `Expiring<Expiring<DynCache>>` to type-
check — two clocks, two indexes, surprising semantics — and we have no
clean way to disallow it at the type level once `DynCache: Cache`. A
distinct expiring type makes double-wrapping impossible by construction.
The cost is one extra public type and minor delegation boilerplate; the
benefit is that the only TTL surface is the one the builder hands out.

Then, where profiling justifies it, embed `expires_at` into specific
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

### A) Lazy min-heap of `(expires_at, key)`

`ds::LazyMinHeap<K, S>` already exists at
[`src/ds/lazy_heap.rs`](../../src/ds/lazy_heap.rs) and explicitly lists TTL
in its use cases. Insertion is O(log n); `pop_best` is amortized O(log n);
`update` is O(log n) with `maybe_rebuild` to bound staleness.

Used as a TTL index, this needs a thin `ExpirationIndex` wrapper over
`LazyMinHeap` rather than using the heap directly. The wrapper should expose:

```rust
pub struct ExpirationIndex<K> { /* LazyMinHeap<K, u64> */ }

impl<K> ExpirationIndex<K> {
    pub fn set_deadline(&mut self, key: K, expires_at: u64) -> Option<u64> {
        /* ... */
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<u64>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        /* ... */
    }

    pub fn peek_deadline(&mut self) -> Option<(&K, u64)> {
        /* ... */
    }

    pub fn pop_expired(&mut self, now: u64) -> Option<(K, u64)> {
        /* ... */
    }
}
```

`LazyMinHeap` currently has destructive `pop_best` but no non-destructive
live-minimum peek (verified against [`src/ds/lazy_heap.rs`](../../src/ds/lazy_heap.rs):
only `update`, `pop_best`, `with_auto_rebuild`, `maybe_rebuild` are public).
The first slice should **add a `peek_best` primitive to `LazyMinHeap`**
rather than reimplementing live-minimum logic inside `ExpirationIndex`.
The wrapper approach would have to inspect the heap's internal staleness
state to skip popped entries, which couples `ExpirationIndex` to
`LazyMinHeap`'s representation. A `peek_best(&mut self) -> Option<(&K, &S)>`
that drains stale-tombstoned roots in place (mutating because lazy
deletion may need to advance past tombstones, immutable observation
otherwise) is the right primitive and is reusable outside TTL.

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
  while `peek_deadline()` returns a deadline `<= now`, call
  `pop_expired(now)` and remove that key from the wrapped cache.
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

A `Clock` parameter on `Expiring<C, K, T>` (default `StdClock`) and on any
policy that embeds expiry. Mirrors how `RandomCore` keeps `rng_state`
rather than calling `rand::thread_rng()` directly.

### c) Builder integration

Two complementary surfaces:

```rust
let mut cache = CacheBuilder::new(1000)
    .with_default_ttl(Duration::from_secs(60))
    .build::<u64, String>(CachePolicy::Lru);

cache.insert_with_ttl(1, v, Duration::from_secs(5));
```

Internally, `with_default_ttl(Some(_))` switches the builder to produce
a `DynExpiringCache<K, V>` (separate public type) rather than a
`DynCache<K, V>`. The two paths considered were:

1. Add `impl Cache<K, V> for DynCache<K, V>` and store
   `CacheInner::Ttl(Expiring<BoxedOrDynCache<K, V>>)` / an equivalent wrapper
   around the already-built `DynCache`.
2. Introduce a separate `DynExpiringCache<K, V>` returned by a TTL-specific
   builder path, avoiding a recursive enum at the cost of another public type.

Option (2) is recommended (see §1 Recommendation). The deciding factor is
that option (1) lets `Expiring<Expiring<DynCache>>` type-check, which is
silently wrong (two clocks, two indexes). Option (2) makes double-
wrapping unrepresentable: `Expiring<C>` is only constructed through the
builder, and the builder never returns an inner expiring cache. The
document's "one new variant, not 18" goal still holds — the new type
delegates `Cache::insert` etc. via a single match arm per inner policy,
identical to the existing `DynCache` boilerplate but with the expiry
check threaded through. The duplication is real but bounded.

### d) Feature gating

A `ttl` feature; `chrono` is already a dev-dep (see [`Cargo.toml`](../../Cargo.toml)),
but TTL itself should depend only on `std::time` and the existing
`LazyMinHeap` / `SlotArena`. The `ExpirationIndex` lives at
`src/ds/expiration_index.rs` but is gated behind `#[cfg(feature = "ttl")]`
so the `ds` module does not grow a time abstraction when TTL is disabled.
The new `Clock` trait at `src/time.rs` is similarly gated. Keep `metrics`
integration lightweight: extend `LruMetrics` / `StoreMetrics` with
`expirations: u64` behind `metrics`, similar to how `evictions` is tracked
today (see [`src/store/traits.rs`](../../src/store/traits.rs) `StoreMetrics`).

### e) Concurrent variants

The existing `Concurrent*` wrappers (`ConcurrentLruCache` in
[`src/policy/lru.rs`](../../src/policy/lru.rs), `ConcurrentSlotArena` in
[`src/ds/slot_arena.rs`](../../src/ds/slot_arena.rs),
`ConcurrentClockRing` in [`src/ds/clock_ring.rs`](../../src/ds/clock_ring.rs))
wrap their single-threaded core in `parking_lot::RwLock`. TTL follows that
shape with two non-negotiable rules:

1. **Return owned/`Arc<V>`, not `&V`.** The `Cache::get(&mut self) -> Option<&V>`
   signature cannot be implemented safely on `ConcurrentExpiring<C>`
   without holding the write lock across the borrow, which serializes
   readers and defeats the point of `RwLock`. `ConcurrentExpiring<C>`
   therefore exposes `fn get(&self, key: &K) -> Option<Arc<V>>` (and the
   sibling mutators), and does **not** implement `Cache<K, V>`. It is a
   concrete type with its own API, mirroring how `ConcurrentLruCache`
   already deviates from `Cache<K, V>`.
2. **Atomic expiry-and-removal.** The expiry check, policy removal, and
   index removal must be one atomic write-locked operation. Splitting
   them allows a concurrent `set_ttl` or `insert` to race with a stale
   expiry decision and produce a "you remove an entry I just renewed"
   outcome. A read-locked fast path (check `expires_at <= now` under a
   read lock, escalate to write lock for the actual removal) is safe so
   long as the write-locked path re-checks the deadline before acting.

### f) `DynCache` touchpoint

With the `DynExpiringCache<K, V>` route chosen in §4(c), `DynCache` itself
is **untouched** by TTL. The new type lives next to `DynCache` and mirrors
its match-arm boilerplate one level out (the expiry check happens before
delegating to the inner policy's `Cache::insert`/`get`/etc.). The `Debug`
impl on `DynExpiringCache` should report TTL mode (default TTL, clock
type) without exposing keys or deadlines.

### g) `prelude.rs`

Re-export `ExpiringCache`, `Clock`, `StdClock`, `Expiring` so users get them
via `use cachekit::prelude::*;`.

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

## 6. Recommended First Slice

A pragmatic phased roadmap:

1. New module `src/policy/expiring.rs` with `Expiring<C, K, T = StdClock>`
   decorator. Define `peek` / `contains` as logical reads; `Cache::len`
   reports physical occupancy; add an inherent `Expiring::live_len(&mut self)`
   for callers that need the live count (see §1(a) Decision).
2. New ds module `src/ds/expiration_index.rs` backed by
   `LazyMinHeap<K, u64>` (cheap reuse) with auto-rebuild enabled to bound
   stale heap growth. Add a `peek_best` primitive to `LazyMinHeap`
   itself (see §3.A) so `ExpirationIndex` can implement
   `peek_deadline` / `pop_expired(now)` without coupling to heap
   internals. Leave the door open to swap in a timer wheel later. Both
   files are gated behind `#[cfg(feature = "ttl")]`.
3. `Clock` trait + `StdClock` / `MockClock` in a new `src/time.rs`.
4. `ExpiringCache<K, V>` capability trait in `src/traits.rs`, using
   `TtlStatus` for unambiguous status reporting.
5. `CacheBuilder::with_default_ttl(Duration)` returns a separate
   `DynExpiringCache<K, V>` (not `DynCache<K, V>`) — see §1 Recommendation
   and §4(c). This makes `Expiring<Expiring<…>>` structurally
   unrepresentable.
6. Feature flag `ttl`; metrics field `expirations` (gated on `metrics`);
   doctests + a fuzz seed; benchmark group `ttl_overhead` that compares
   plain LRU vs. `Expiring<LRU>` under the existing Zipfian and scan
   workloads.
7. **Phase 2:** profile (a) and, if the extra hash hit shows up in
   flamegraphs, embed `expires_at: u64` into `LruCore::Node` and
   `S3FifoCache::Node` (the two highest-traffic policies in the existing
   benches at [`benches/`](../../benches)) — but **opt-in per node**, not
   unconditionally. Two viable shapes:
   - A const generic `Node<K, V, const TTL: bool>` so non-TTL caches
     monomorphize to the slimmer layout.
   - A separate type `LruWithTtl<K, V>` (and `S3FifoWithTtl<K, V>`)
     that wraps the slot arena with a parallel `Vec<u64>` keyed by slot
     handle.
   Embedding `expires_at` unconditionally would add 8 bytes per node to
   every LRU and S3-FIFO instance — a 10–25% memory regression for the
   common case of fixed-size value caches — and would regress the very
   benchmarks step 6 is using as a gate. The `.cursorrules` "keep
   metadata tight" rule applies here.

This sequence preserves policy/storage separation and keeps TTL opt-in, but
the decorator does not preserve every hot-path invariant: inserts pay heap
maintenance, the index clones keys, and expired entries may remain physically
resident until a mutable operation purges them. The benchmark gate in step 6
is therefore part of the design, not optional cleanup.

---

## 7. Open Questions

- Should `purge_expired` be exposed publicly, run on a background thread,
  triggered on insert-when-full, or all three (configurable)?
- Should the `Clock` trait live in a top-level `time` module or inside `ds`?
  Step 6.3 currently picks `src/time.rs`; revisit if `no_std` support
  becomes a constraint.
- How should serialization (under `serde` feature) handle `expires_at` —
  the current recommendation is relative remaining duration, but restoring
  long-lived caches may need wall-clock deadlines. Open until a
  serialization API is proposed.
- Is there demand for *negative* TTL (entries that become valid only after
  a delay)? Probably no, but worth confirming before locking the API.
- Should `purge_expired` return a `usize` count, the evicted `(K, V)`
  pairs, or both (via separate methods)? The current trait sketch returns
  `usize`; users who need the values can iterate `pop_expired` directly
  through a lower-level API.

Resolved during this design pass (kept here for posterity):
- `len` reports physical occupancy (matches `Cache::len`'s `&self`
  constraint); add `live_len(&mut self)` if/when the wrapper grows a
  mutable counterpart — see §1(a).
- Builder integration uses a separate `DynExpiringCache<K, V>` rather
  than `impl Cache for DynCache` — see §1 Recommendation and §4(c).

---

## References

- [`docs/policies/roadmap/ttl.md`](../policies/roadmap/ttl.md) — high-level
  stub
- [`docs/policy-ds/lazy-heap.md`](../policy-ds/lazy-heap.md) — lazy heap
  primitive used as the index
- [`src/ds/lazy_heap.rs`](../../src/ds/lazy_heap.rs) — implementation that
  already lists TTL as a use case
- [`src/traits.rs`](../../src/traits.rs) — capability-trait pattern this
  design extends
- [`src/builder.rs`](../../src/builder.rs) — `DynCache` integration point
- [Wikipedia: Cache replacement policies](https://en.wikipedia.org/wiki/Cache_replacement_policies)
