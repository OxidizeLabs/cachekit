# Weighted Eviction

> Status: design rationale for [`WeightStore`](../../src/store/weight.rs)
> and [`ConcurrentWeightStore`](../../src/store/weight.rs). Companion to
> [`design.md`](design.md), [`concurrency.md`](concurrency.md), and the
> [`stores`](../stores/README.md) reference.

Entry-count caps are the wrong tool when entries vary in size. A cache
sized "max 1 000 entries" that holds a mix of 100-byte thumbnails and
10 MB blobs will either overshoot its memory budget by orders of
magnitude (when blobs dominate) or waste capacity (when thumbnails do).
`WeightStore` exists to give callers a second, byte-denominated budget
alongside the entry count.

This document explains the dual-limit model, the contract on the
user-supplied weight function, where weight integrates with eviction
policies today (it does not), and how it pre-stages GDS/GDSF on the
roadmap.

## The problem

A typical entry-count cache:

- Fails to bound memory when value sizes differ by orders of magnitude.
- Cannot answer "how many bytes am I caching?" without iterating.
- Treats a 1 KB and a 1 MB entry as equal eviction candidates, which
  is wrong when memory pressure is the binding constraint.

The complementary failure mode — a pure byte-budgeted cache — has its
own problems:

- Highly variable entry counts make per-entry metadata budgeting hard.
- A pathological "one giant entry fills the cache" case is the byte
  version of the "millions of one-byte entries fills the cache"
  problem in entry-count caches.
- Some policies (LFU bucket arrays, S3-FIFO ratios) are sized by entry
  count and need a stable upper bound.

`WeightStore` therefore enforces **both** an entry-count cap and a
weight cap — whichever is hit first triggers `StoreFull`. The user
picks the units of "weight" via a closure.

## Dual-limit model

```text
try_insert(key, value):
  │
  ├─► Existing key (update)
  │     │
  │     ├── new_weight    = weight_fn(&value)
  │     ├── next_total    = total_weight - old_weight + new_weight
  │     │
  │     └── next_total > capacity_weight? ──► Err(StoreFull)
  │                                       └──► Ok(Some(old_value))
  │
  └─► New key (insert)
        │
        ├── len() >= capacity_entries?         ──► Err(StoreFull)
        ├── new_weight = weight_fn(&value)
        ├── total_weight + new_weight > capacity_weight? ──► Err(StoreFull)
        │
        └── Ok(None)
```

Three properties worth naming:

- **Pre-checked, not retroactive.** `try_insert` returns
  `Err(StoreFull)` rather than silently evicting; the **store** is
  full, so the caller (or the policy layered above it) decides what
  to evict.
- **Updates can fail too.** Replacing a 1 MB value with a 2 MB value
  on a cache with 1.5 MB of remaining headroom returns `StoreFull` —
  the update is rejected and the original entry stays resident. This
  is the only sensible behaviour when an update can push the store
  past its budget.
- **Atomic weight bookkeeping.** `total_weight` is the live sum of
  every resident entry's weight. Every successful `try_insert` /
  `remove` / `clear` updates it; reads (`get`, `peek`) do not. The
  invariant `total_weight == sum(entries.weight)` is debug-asserted.

## The weight function: contract and hazards

```rust,ignore
F: Fn(&V) -> usize
```

The user supplies a closure. Three pieces of the contract matter:

- **Cheap.** Ideally O(1). The function is called on every insert and
  every update. A weight function that traverses the value to compute
  bytes (`|tree: &BTreeMap<K, V>| tree.iter().map(…).sum()`) makes
  insert latency proportional to value size.
- **Deterministic.** The same value must yield the same weight every
  time. A non-deterministic weight breaks `total_weight` accounting —
  the store remembers `old_weight` from the *previous* insert, so a
  changed weight on update leaks `(new_actual - old_recorded)` bytes
  of budget per update.
- **Non-panicking.** The function is invoked while a write lock is
  held in [`ConcurrentWeightStore`](../../src/store/weight.rs). A
  panicking weight function under `panic = "unwind"` poisons-by-
  unwind the inner state (the lock itself is `parking_lot`'s
  non-poisoning variant; what is "poisoned" is the call site,
  which never completes the insert). Under the crate's default
  `panic = "abort"` release profile this terminates the process.

Common shapes:

```rust,ignore
|v: &Vec<u8>|     v.len()
|s: &String|      s.len()
|img: &Image|     img.width * img.height * 4
|_: &T|           1                 // entry-count only
|v: &Cow<[u8]>|   v.len()           // works for borrowed/owned
```

The "weight = 1" specialization deserves a note: it makes
`WeightStore` behave exactly like a count-only store, at the cost of
an `Arc<V>` round-trip and per-entry weight slot. Use
`HashMapStore` for that case unless you specifically want the
`ConcurrentWeightStore` API.

## Precomputation: weight stored per entry

Each entry holds its weight in a small wrapper:

```rust,ignore
struct WeightEntry<V> {
    value: Arc<V>,
    weight: usize,
}
```

Weight is computed **once** at insert/update time and stored alongside
the value. Three consequences:

- Reads (`get`, `peek`, `contains`, `len`, `total_weight`) never
  invoke the weight function. They cannot — they only have a
  reference to the stored entry, and the stored entry already knows
  its weight.
- `remove` updates `total_weight` by subtracting the stored weight,
  with no recompute.
- Memory overhead per entry is `sizeof(usize)` + `sizeof(Arc<V>)` —
  one extra word plus the Arc header. Acceptable for variable-size
  caches where the value itself dominates the per-entry footprint.

The alternative — recomputing weight on every read for the sake of
"freshness" — would only matter if the weight function were
non-deterministic, which the contract forbids.

## `Arc<V>` everywhere

`WeightStore` stores `Arc<V>` even in the single-threaded variant:

```rust,ignore
pub fn try_insert(&mut self, key: K, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull>
pub fn get(&mut self, key: &K) -> Option<Arc<V>>
pub fn peek(&self, key: &K) -> Option<Arc<V>>
```

This is a deliberate divergence from `StoreCore` / `StoreMut` (which
return `V` directly). Three reasons:

- **Cheap shared ownership.** Large `V`s (images, blobs) are the
  target use case. Returning `Arc<V>` lets callers hold or share the
  value without forcing `V: Clone`.
- **Surface alignment with `ConcurrentWeightStore`.** The concurrent
  variant must return `Arc<V>` (the `&V`-across-lock problem from
  [`concurrency.md`](concurrency.md)). Keeping the single-threaded
  variant on the same shape lets callers swap between them by
  changing one type without re-plumbing returns.
- **`V: !Clone` is supported.** Callers who don't want to require
  `Clone` on their value type get the `Arc<V>` round-trip "for free."

The cost is that `WeightStore` does **not** implement `StoreCore` /
`StoreMut`. It is a sibling, not a subtype, of the entry-count stores
([`HashMapStore`](../../src/store/hashmap.rs),
[`SlabStore`](../../src/store/slab.rs)), and code generic over those
traits cannot accept a `WeightStore` without adaptation. This is the
single sharpest API edge in the store layer, called out explicitly in
the module documentation.

## Why weight is at the **store** layer, not the policy layer

The 17 implemented policies in `src/policy/` are all weight-unaware.
They count entries and evict by entry. `WeightStore` is below them in
the layering:

```text
   ┌─────────────────────────────┐
   │   policy (weight-unaware)   │   evicts by recency/frequency/etc
   └──────────────┬──────────────┘
                  │ Cache<K, V> uses store underneath
   ┌──────────────▼──────────────┐
   │  WeightStore (dual limits)  │   refuses inserts past weight cap
   └─────────────────────────────┘
```

This separation has two consequences worth understanding:

- **The policy decides who to evict; the store decides whether the
  result fits.** A policy operating over a `WeightStore` evicts its
  policy-chosen victim, then attempts the insert. If the insert
  still doesn't fit (one large value cannot be made room for by
  evicting a single small victim), the policy must evict again or
  surface `StoreFull` to the caller.
- **No policy in the tree today consumes `WeightStore` directly.**
  `WeightStore` is reachable only through its own concrete API, not
  through the `Cache<K, V>` trait or `DynCache`. Users who want a
  weight-aware cache today build one themselves on top of
  `WeightStore` plus a chosen eviction strategy.

The reason for this layering is forward compatibility. Weight-aware
**policies** (GDS, GDSF, LFU-DA, see roadmap) need this store as
their substrate. Coupling weight directly into a policy locks the
weight model to that policy; keeping it at the store layer keeps the
substrate reusable.

## Concurrent variant

`ConcurrentWeightStore<K, V, F>` follows the wrapper pattern from
[`concurrency.md`](concurrency.md):

```rust,ignore
pub struct ConcurrentWeightStore<K, V, F> {
    inner: Arc<RwLock<WeightStore<K, V, F>>>,
}
```

`parking_lot::RwLock`; `peek` / `contains` / `len` / `total_weight`
take the read lock; `try_insert` / `remove` / `clear` take the write
lock; metrics counters live in `AtomicU64` so the read-locked paths
can still increment them without escalating.

The weight function runs **inside the write lock** on every insert
and update. A slow `F` therefore stalls every reader and writer in
the cache — a DoS amplification vector when caching user-supplied
values. The mitigation is the cheapness contract; the rustdoc on
`ConcurrentWeightStore::try_insert` says so.

`ConcurrentWeightStore` implements `ConcurrentStoreRead<K, V>` and
`ConcurrentStore<K, V>`. Unlike the single-threaded variant — which
deliberately does not implement `StoreCore`/`StoreMut` — the
concurrent variant *does* fit the concurrent trait family because
both already use `Arc<V>` returns. The asymmetry is awkward but
honest: the trait family is shaped around the constraints the
concurrent path imposes, and the single-threaded variant happens to
borrow that shape rather than the sequential one.

## Lock-poisoning and total-weight integrity

Under `panic = "abort"` (the crate's release default) lock poisoning
is moot — the process exits. Under `panic = "unwind"`, the order of
operations in `clear()` matters:

```rust,ignore
fn clear(&mut self) {
    self.total_weight = 0;          // (1) reset first
    self.entries.clear();           // (2) then drop entries (may panic)
}
```

If (2) panics during entry drop, `total_weight = 0` and `len() == 0`
remain consistent post-panic. Individual values may leak through the
unwinding drop but the store's accounting cannot be corrupted into
"says it has 1 GB resident when actually empty" — which would
silently reject all future inserts. The module documentation calls
this out so callers who override `panic = "abort"` know what they
get.

## Failure mode: weight cap, not entry cap

When the weight budget is hit but the entry count is not:

- `try_insert` returns `StoreFull` for any new key whose value would
  push `total_weight` past `capacity_weight`.
- `len() < capacity_entries` — the entry budget has headroom that
  cannot be used.
- `total_weight == capacity_weight` (approximately, depending on
  insert sizes).

The reverse — entry cap hit, weight cap not — produces `StoreFull`
on any new insert regardless of weight, including tiny values.

Both are correct. The store does not silently demote either limit;
the caller's intent is "neither budget shall be exceeded," and the
store enforces it literally.

## Capacity tuning

The dual limits give callers two knobs:

| Setting | Effect |
|---|---|
| `capacity_entries` finite, `capacity_weight = usize::MAX` | Behaves like an entry-count store; weight is observable but unconstrained |
| `capacity_entries = usize::MAX`, `capacity_weight` finite | Behaves like a pure byte-budget store; entry count is observable but unconstrained |
| Both finite | Hard dual limit |

The first row is rarely what callers want (use `HashMapStore`
instead — no per-entry weight slot). The second is a legitimate
configuration for callers who genuinely want bytes-only accounting
and accept the per-entry overhead. The third is the design intent.

## Security considerations

The module rustdoc is unusually long on security; the points worth
naming at the design-doc level:

- **Hasher.** `WeightStore`'s key index uses `FxHashMap`, which is
  **not** HashDoS-resistant. Callers caching variable-size values
  keyed by request paths, tenant IDs, or filenames — i.e. exactly
  the use case `WeightStore` targets — should pre-hash keys with a
  keyed hash (`siphasher` with a per-process key) or migrate to
  `HashMapStore`'s `RandomState`-backed default.
- **Side channel.** `total_weight` is publicly readable. Callers
  with access to the counter can infer the size of other tenants'
  cached entries from before/after differentials. Avoid exposing
  `total_weight` across trust boundaries when caching tenant-keyed
  variable-size records.
- **Sensitive values.** Dropped `V`s are not zeroized. Wrap `V` in
  `zeroize::Zeroizing` (or equivalent) when caching credentials.
- **Counters.** Metrics use `Relaxed` ordering and wrap on overflow
  in release. Best-effort observability, not audit-grade.

## Pre-staging GDS/GDSF

GreedyDual-Size (GDS) and its frequency-aware variant GDSF evict by
**cost ÷ size** rather than recency or frequency alone. Both
require:

- A per-entry size (`WeightStore` already stores it).
- A per-entry cost (caller-supplied at insert time).
- An eviction priority queue ordered by `cost / size + age`.

`WeightStore` provides the size half today. The cost half and the
priority-queue substrate ([`LazyMinHeap`](../../src/ds/lazy_heap.rs)
is a natural fit) are the missing pieces. When GDS lands, the
expected shape is:

```rust,ignore
pub struct GdsCache<K, V, F> {
    store: WeightStore<K, V, F>,
    queue: LazyMinHeap<K, GdsPriority>,
    aging: AgingCounter,
}
```

The trait surface would be `Cache<K, V>` plus a future
`WeightTracking<K, V>` capability trait (sketched in
[`trait-hierarchy.md`](trait-hierarchy.md#future-capability-traits)),
giving generic code the ability to consult `weight(key)` and
`total_weight()` regardless of which policy is doing the evicting.

The non-trivial design question, when GDS lands, is whether the
priority queue stores cost / size at insert time (cheap, can become
stale if the value's "true" cost diverges from insert-time cost) or
recomputes on demand (more expensive, but always current). The
current expectation is "store at insert time, document the
staleness window" — matching the precomputed-weight discipline this
store already follows.

## When not to use `WeightStore`

- **Uniform value sizes.** Use `HashMapStore` or `SlabStore`. The
  weight slot is overhead with no benefit.
- **Hot-path latency dominates.** The weight function runs on every
  insert. If `F` is non-trivial, insert latency is `F`-dominated.
- **You need a policy.** `WeightStore` is a store; policies sit
  above it. A bare `WeightStore` evicts nothing on its own — it
  surfaces `StoreFull` and the caller decides what to remove. Use
  this directly only when the caller knows the eviction strategy
  better than any built-in policy would.

## See also

- [Design overview](design.md) — §2 (memory layout) and §5
  (eviction) frame the trade-offs at the principles level
- [Concurrency](concurrency.md) — `ConcurrentWeightStore` follows
  the standard wrapper pattern documented there
- [Cache trait hierarchy](trait-hierarchy.md) — future
  `WeightTracking` capability trait sketched in
  "Future capability traits"
- [Stores](../stores/README.md) and [`weight.md`](../stores/weight.md)
  — reference docs for the runtime behaviour
- [Error model](error-model.md) — `StoreFull` semantics
- [`src/store/weight.rs`](../../src/store/weight.rs) — the canonical
  implementation
- [Roadmap: GDS](../policies/roadmap/gds.md) and
  [GDSF](../policies/roadmap/gdsf.md) — the planned consumers
