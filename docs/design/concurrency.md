# Concurrency

> Status: design rationale for the concurrent surface that ships today
> behind the `concurrency` feature flag. Companion to the cross-cutting
> principles in [`docs/design/design.md`](design.md) §3 and the trait
> rationale in [`docs/design/trait-hierarchy.md`](trait-hierarchy.md).

cachekit's default surface is single-threaded. Concurrency is opt-in,
delivered through a parallel set of types and traits gated by the
`concurrency` Cargo feature. This document explains why the concurrent
surface looks the way it does, what invariants the wrappers promise,
and where the gaps are.

## Non-goals

- **`no_std`.** Concurrency relies on `parking_lot`, `std::sync::Arc`,
  and `std::sync::atomic`. No `loom`/`no_std` support is planned.
- **Lock-free policies.** Mostly-lock-free or strictly lock-free
  policies are out of scope today; see [Future directions](#future-directions).
- **Async-native traits.** `AsyncCacheFuture` is a Phase 2 placeholder
  ([`src/traits.rs`](../../src/traits.rs)); no policy implements it
  meaningfully yet.

## The dominant pattern: sequential core, concurrent wrapper

Every concurrent type in cachekit follows the same shape:

```text
ConcurrentX<K, V> { inner: Arc<RwLock<X<K, V>>> }
```

where `X` is the single-threaded core (`LruCore`, `FifoCache`,
`S3FifoCache`, `SlotArena`, `IntrusiveList`, `ClockRing`,
`HashMapStore`, `SlabStore`, `WeightStore`, `HandleStore`,
`FrequencyBuckets`). The wrapper:

1. holds the core behind an `Arc<RwLock<…>>`,
2. presents owned/`Arc<V>` returns instead of borrowed `&V`,
3. is `Clone` via `Arc::clone` so callers can hand copies to threads,
4. is `Send + Sync` because the inner core's `Send + Sync` impls
   auto-derive through `Arc<RwLock<…>>`.

The pattern is verbose but consistent and was chosen for three reasons:

- **No `&mut self` in the public API.** Sharing requires interior
  mutability; an `RwLock` is the cheapest tool that exposes both shared
  reads and exclusive writes through `&self`.
- **The sequential core stays unaware of locking.** Policy code under
  [`src/policy/`](../../src/policy) is single-threaded and easier to
  reason about. The locking discipline lives in one place per type.
- **The lock is replaceable.** Swapping `parking_lot::RwLock` for a
  different primitive (`std::sync::RwLock`, a sharded lock, a seqlock)
  is a local change because the inner core has no opinion on it.

The 11 concurrent wrappers shipped today live in:

- `ConcurrentLruCache` — [`src/policy/lru.rs`](../../src/policy/lru.rs)
- `ConcurrentFifoCache` — [`src/policy/fifo.rs`](../../src/policy/fifo.rs)
- `ConcurrentS3FifoCache` — [`src/policy/s3_fifo.rs`](../../src/policy/s3_fifo.rs)
- `ConcurrentHashMapStore`, `ShardedHashMapStore` — [`src/store/hashmap.rs`](../../src/store/hashmap.rs)
- `ConcurrentSlabStore` — [`src/store/slab.rs`](../../src/store/slab.rs)
- `ConcurrentWeightStore` — [`src/store/weight.rs`](../../src/store/weight.rs)
- `ConcurrentHandleStore` — [`src/store/handle.rs`](../../src/store/handle.rs)
- `ConcurrentSlotArena`, `ShardedSlotArena` — [`src/ds/slot_arena.rs`](../../src/ds/slot_arena.rs)
- `ConcurrentIntrusiveList` — [`src/ds/intrusive_list.rs`](../../src/ds/intrusive_list.rs)
- `ConcurrentClockRing` — [`src/ds/clock_ring.rs`](../../src/ds/clock_ring.rs)
- `ShardedFrequencyBuckets` — [`src/ds/frequency_buckets.rs`](../../src/ds/frequency_buckets.rs)

## Why `Concurrent*` does not implement `Cache<K, V>`

`Cache<K, V>` is the sequential trait. Its method signatures encode
sequential ownership:

```rust
fn peek(&self, key: &K) -> Option<&V>;
fn get(&mut self, key: &K) -> Option<&V>;
fn insert(&mut self, key: K, value: V) -> Option<V>;
```

Three of these are unimplementable on `Arc<RwLock<…>>`:

- **`peek` and `get` return `&V`.** A borrowed reference cannot
  outlive the `RwLockReadGuard`/`RwLockWriteGuard` it was extracted
  from. There is no safe lifetime that ties `&V` to `&self` rather
  than to the (anonymous) guard. Returning `&V` would force the
  caller to hold the lock across the borrow, which serializes readers
  and defeats `RwLock`.
- **`get` takes `&mut self`.** With shared ownership through
  `Arc<RwLock<…>>` the wrapper only ever holds `&self`. Forcing
  `&mut self` would require `Arc::make_mut` or external locking,
  defeating the point of the inner lock.

The concurrent wrappers therefore expose their own concrete API:

```rust
pub fn get(&self, key: &K) -> Option<Arc<V>>;
pub fn peek(&self, key: &K) -> Option<Arc<V>>;
pub fn insert(&self, key: K, value: V) -> Option<Arc<V>>;
pub fn insert_arc(&self, key: K, value: Arc<V>) -> Option<Arc<V>>;
pub fn remove(&self, key: &K) -> Option<Arc<V>>;
```

Returning `Arc<V>` is the contract. It costs one atomic refcount bump
on hit, which is cheap relative to the lock acquisition itself, and it
lets callers hold the value past lock release, send it across threads,
or stash it in another structure without lifetime gymnastics.

For uniformity across the store layer there is a parallel trait family
that **does** model the `&self` + `Arc<V>` shape:

| Sequential ([`src/store/traits.rs`](../../src/store/traits.rs)) | Concurrent ([`src/store/traits.rs`](../../src/store/traits.rs)) |
|---|---|
| `StoreRead` (`&mut self`, `&V`) | `ConcurrentStoreRead` (`&self`, `Arc<V>`) |
| `StoreMut` (`&mut self`) | `ConcurrentStore` (`&self`) |
| `StoreFactory` | `ConcurrentStoreFactory` |

The policy layer does not yet have a counterpart family — see
[Future directions](#future-directions).

## Lock primitive choice

Every concurrent wrapper uses **`parking_lot::RwLock`**. Two things
drove this:

- **Reader / writer split matches the access pattern.** `peek` /
  `contains` / `len` only need shared access. `get` (which mutates
  recency or frequency state) and `insert` / `remove` need exclusive
  access. `Mutex` would serialize all of these.
- **Fairness and uncontended speed.** `parking_lot::RwLock` is small
  (one `AtomicUsize` on 64-bit), uncontended-fast, and tunable via
  fairness traits. The `RwLock<HashMap<K, Arc<V>>>` and
  `RwLock<SlotArena<T>>` shapes throughout the codebase rely on this.

`Mutex` is intentionally absent from the wrappers. The few `Mutex`
references in the source tree are in doctests and rustdoc prose
describing how a user would wrap a non-concurrent cache themselves —
they are not on any hot path.

The `parking_lot` choice is **not** absolute. On Rust 1.85+ the
futex-based `std::sync::Mutex` is competitive for the uncontended
single-writer case on Linux/macOS, and revisiting this is reasonable
if `parking_lot` ever becomes a build burden. The `RwLock` advantage
is more durable: `std::sync::RwLock` still has writer-starvation
hazards on some platforms that `parking_lot` avoids by default.

## The `get` / `peek` lock-level asymmetry

`peek` and `get` both look up by key, but they differ in what they
mutate:

- **`peek`** is side-effect-free. The wrapper takes a **read lock**
  and clones the `Arc<V>`. Multiple readers proceed in parallel.
- **`get`** updates policy state (LRU recency, LFU frequency, Clock
  reference bit, …). The wrapper takes a **write lock**. Only one
  thread proceeds.

This asymmetry is the single most important reason `peek` and `get`
are distinct methods at all (see
[`trait-hierarchy.md`](trait-hierarchy.md) for the rationale at the
trait level). Without `peek`, every read would serialize through the
write lock. With `peek`, read-heavy workloads — buffer pools, immutable
metadata caches — scale linearly across cores.

The cost is that callers must choose, and choosing `get` on a
read-heavy workload silently kills scalability. The rustdoc on each
wrapper's `peek` and `get` says so explicitly; benchmarks under
[`benches/`](../../benches) compare the two.

## Atomic check-and-act

Compound operations must stay inside a single lock acquisition. The
rule is **check, decide, mutate, release** — all under the same write
lock. Splitting the steps across two acquisitions allows a concurrent
writer to invalidate the decision between them.

The pattern shows up in three places worth naming:

- **Insert-on-full.** Capacity check + eviction + insert must be one
  critical section. `WeightStore::try_insert` and the policy `insert`
  methods both follow this.
- **Replace-and-return.** `insert` returns the previous value if one
  existed. The "did this key exist?" check and the replace must
  happen under the same write lock; otherwise two concurrent inserts
  can both observe "key absent" and both return `None`.
- **Future: expiry + remove.** TTL (see
  [`docs/design/ttl.md`](ttl.md) §4(e)) requires the expiry check and
  the removal to be one atomic operation under a write lock. A
  read-locked fast path that observes `expires_at <= now` and
  escalates to a write lock is safe **only** if the write-locked
  path re-checks the deadline before acting, because a concurrent
  `set_ttl` may have renewed the entry in between.

The atomicity rule is a wrapper-level discipline, not a trait-level
one. The single-threaded core can't enforce it because it doesn't
know about locks.

## Cloning the wrapper

Every `Concurrent*` type implements `Clone` via `Arc::clone(&self.inner)`.
Cloning the wrapper is cheap (one atomic increment) and produces a
second handle to the **same** underlying cache. This is the intended
way to share a cache across threads:

```rust,ignore
let cache = ConcurrentLruCache::<u64, String>::new(1_000);
let cache2 = cache.clone();
std::thread::spawn(move || {
    cache2.insert(1, "hello".into());
});
cache.get(&1);
```

There is no separate `Arc<ConcurrentLruCache<…>>` wrapping needed;
the inner `Arc` is the sharing primitive. Callers who want
`Arc<dyn ConcurrentCache>` for type erasure are still free to wrap, but
in practice the concrete clone is what's used in the codebase.

## `ConcurrentCache`: marker trait, not capability trait

`ConcurrentCache` lives in [`src/traits.rs`](../../src/traits.rs) and
is declared `unsafe trait ConcurrentCache: Send + Sync {}`. It has
no methods. Its job is to **promise**, at the type system level,
that "this type is safe to share across threads in the cache sense" —
specifically that its `Cache`-like operations (whatever those happen
to be — concrete `Concurrent*` types do not implement `Cache<K, V>`)
take care of internal synchronization.

The `unsafe` is load-bearing. Implementing `ConcurrentCache`
incorrectly cannot be caught by the type system; it's an
implementer-side soundness claim, which is why only the wrappers
implement it (`ConcurrentFifoCache`, `ConcurrentS3FifoCache` today;
`ConcurrentLruCache` is a candidate but does not yet have the impl).

Users writing generic code that requires a thread-safe cache should
bound on `ConcurrentCache + Send + Sync`. They should **not** bound on
`Cache<K, V> + Send + Sync` and expect that to suffice — that bound
is satisfied by single-threaded caches whose user is responsible for
external locking.

## Sharded primitives

For data structures where a single `RwLock` becomes the bottleneck,
cachekit ships sharded variants:

- **`ShardedHashMapStore<K, V, S>`** — N independent shards, each its
  own `RwLock<HashMap<…>>`. Shard selected by hashing the key with
  the store's `BuildHasher`.
- **`ShardedSlotArena<T>`** — N independent arenas with sharded
  `SlotId`s. Same shape, applied to slab-style storage.
- **`ShardedFrequencyBuckets<K>`** — N independent frequency-bucket
  shards for LFU-family policies that want concurrent frequency
  updates.

Sharding lives at the **data-structure** layer, not the policy layer,
because the shard count, hash function, and shard-aware key type
(`ShardedSlotId`) all need to be visible to the policy that uses the
primitive. A `ShardedLruCache` does not yet exist as a single type;
it would be built by composing a `ShardedHashMapStore` with sharded
recency lists, and that composition is roadmap.

When sharding is **not** what you want:

- A single concurrent wrapper is simpler and faster for caches that
  fit on one or two cores' worth of contention.
- Sharding multiplies the working-set fragmentation across shards.
  A 1 M-entry cache split 16 ways has 16 caches of ~62 K each, and
  evictions on one shard cannot rescue items on another.
- Per-shard eviction is correct for capacity bookkeeping (each shard
  tracks its own capacity) but **not** globally optimal — a single-
  shard LRU strictly dominates a sharded LRU on hit rate.

## Concurrent policy coverage

Of the 17 implemented policies, **3 ship with a `Concurrent*` wrapper
today**: LRU, FIFO, S3-FIFO. The remaining 14 require external locking
by the caller — typically `Arc<parking_lot::RwLock<CacheCore>>`. The
relevant rustdoc on those policies (e.g. `LfuCache`, `HeapLfuCache`,
`MfuCache`) calls this out.

This is a coverage gap rather than a design choice. The pattern is
mechanical: wrap the sequential core in `Arc<RwLock<…>>`, expose the
`&self` API with `Arc<V>` returns, decide read-lock vs. write-lock per
method, implement `Clone` via `Arc::clone`, and implement
`unsafe impl ConcurrentCache`. The work is bounded; what's missing is
the discipline to do it consistently across all 17 policies.

## Failure modes

Three failure modes worth naming:

- **Poisoning.** `parking_lot` does **not** poison locks on panic.
  A panic inside a critical section unwinds, releases the lock, and
  leaves the inner core in whatever state the panic interrupted.
  The single-threaded cores are designed to be panic-safe for
  `Cache::insert` / `get` / `remove` — invariants are restored
  before any potentially-panicking operation (allocation, user
  hashing). This is a property of each core, not of the wrapper.
- **Deadlock.** Cachekit never holds two locks at once in the
  current code. Sharded primitives acquire exactly one shard lock
  per operation. Any future work that composes locks (e.g. a sharded
  LRU that touches a shared recency list) must document its locking
  order.
- **Starvation.** `parking_lot::RwLock` defaults to writer-friendly
  fairness; readers do not starve writers. Heavy `get`-dominated
  workloads still serialize through the write lock, which is the
  underlying constraint, not a fairness bug.

## Future directions

Tracked roughly in priority order:

1. **Coverage parity.** `Concurrent*` wrappers for the remaining 14
   policies (LFU, Heap-LFU, MFU, LRU-K, 2Q, ARC, CAR, Clock,
   Clock-PRO, NRU, SLRU, MRU, LIFO, Random). Mechanical work; the
   pattern is fixed.
2. **`ConcurrentExpiring<C>`.** TTL's concurrent wrapper, per
   [`docs/design/ttl.md`](ttl.md) §4(e). Distinct from `Concurrent*`
   policies because the expiry-check + remove must be atomic across
   *both* the inner cache and the expiration index.
3. **Sharded `Cache` wrappers.** A generic `Sharded<C: Cache<K, V>>`
   that hashes keys to N independent inner caches. The design
   question is how to model capacity: per-shard capacity (simple,
   imperfect global behaviour) vs. global capacity with cross-shard
   victim selection (correct, requires inter-shard locking).
4. **Lock-free reads.** `peek` and `contains` paths that avoid the
   `RwLock` entirely — `arc-swap` or seqlock-style techniques —
   for caches whose recency state can tolerate eventual consistency.
   Out of scope until benchmarks show the read lock is the bottleneck.
5. **Loom testing.** Once concurrent coverage stabilises, model-check
   the wrapper invariants under `loom`. Particularly valuable for the
   atomic check-and-act sequences in TTL and sharded composition.

## See also

- [Design overview](design.md) — §3 frames concurrency at the
  principles level
- [TTL design](ttl.md) — applied case for `ConcurrentExpiring<C>`
- [Cache trait hierarchy](trait-hierarchy.md) — read/mutate split and
  object-safety rationale
- [Stores](../stores/README.md) — `ConcurrentStoreRead` /
  `ConcurrentStore` trait family
- [`src/store/traits.rs`](../../src/store/traits.rs) — concurrent
  store traits
- [`src/traits.rs`](../../src/traits.rs) — `ConcurrentCache` marker
