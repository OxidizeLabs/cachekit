# Builder and Runtime Dispatch

> Status: design rationale for [`CacheBuilder`](../../src/builder.rs),
> [`CachePolicy`](../../src/builder.rs), and [`DynCache<K, V>`](../../src/builder.rs).
> Companion to [`design.md`](design.md) §13, [`trait-hierarchy.md`](trait-hierarchy.md),
> and [`concurrency.md`](concurrency.md).

cachekit ships 17 implemented eviction policies. Most application code
wants to pick one of them — possibly at runtime, based on configuration
— without writing 17 monomorphized call sites. This document explains
why that runtime choice is delivered through an enum dispatcher rather
than a `Box<dyn Cache>`, what the user-visible cost is, and how to
extend the surface when a new policy lands.

## The problem

A user with a `policy: String` configuration value wants to write:

```rust,ignore
let mut cache = build_cache_from_config(config);
cache.insert(key, value);
cache.get(&key);
```

without enumerating the 17 policies at every call site. The cache type
must therefore be **uniform across policies** — the concrete type the
caller holds cannot depend on which policy was chosen.

Two Rust mechanisms give a uniform type:

1. **Trait objects** — `Box<dyn Cache<K, V>>`, with dispatch through a
   vtable per method call.
2. **Enum dispatch** — a closed sum of every policy, with dispatch
   through a `match` per method call.

cachekit picks mechanism 2. The rest of this document explains why and
what it costs.

## Enum dispatch vs `Box<dyn Cache>`

`Cache<K, V>` is deliberately object-safe (see
[`trait-hierarchy.md`](trait-hierarchy.md#object-safety)) precisely so
`Box<dyn Cache<K, V>>` *can* be used; cachekit consumers can still take
that route in their own code. But the **library-provided** runtime
dispatcher is an enum, for five reasons:

| Property | `Box<dyn Cache<K, V>>` | `DynCache<K, V>` (enum) |
|---|---|---|
| Dispatch cost per call | Indirect call via vtable | Branch-predicted `match` |
| Devirtualization | No (opaque) | Yes (compiler sees the arm) |
| Inlining of policy body | No | Yes when the arm is statically reachable |
| Heap allocation per cache | One `Box` per cache | None (enum lives inline) |
| Closed vs open extension | Open (any `impl Cache`) | Closed (`#[non_exhaustive]` enum) |
| API stability for new policies | Adding a method is a breaking change | Adding a variant is a non-breaking change with `#[non_exhaustive]` |

The dominant terms are dispatch cost and devirtualization. A `match` on
an enum tag is a single branch that predicts well in tight loops; the
optimizer often hoists it out entirely when the enum tag is invariant
across a benchmark inner loop. A vtable call cannot be devirtualized
without inlining context and forces the policy body to live behind an
opaque indirection.

The cost is in extensibility. `Box<dyn Cache>` accepts any
out-of-tree policy that implements `Cache<K, V>`; `DynCache` does not.
Users with their own policy implementations still use them directly —
`MyCache::new(…)` returns a concrete `MyCache<K, V>` and works with any
code generic over `Cache<K, V>`. The enum is only the **library-provided
dispatcher**, not a general substrate.

## `CachePolicy` — config-carrying tag

`CachePolicy` ([`src/builder.rs`](../../src/builder.rs)) is the
user-facing enum that selects a policy. It is **separate** from the
internal `CacheInner` enum, and it carries per-policy configuration:

```rust,ignore
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CachePolicy {
    Fifo,
    Lru,
    FastLru,
    LruK { k: usize },
    Lfu { bucket_hint: Option<usize> },
    HeapLfu,
    TwoQ { probation_frac: f64 },
    S3Fifo { small_ratio: f64, ghost_ratio: f64 },
    Arc,
    Lifo,
    Mfu,
    Mru,
    Random,
    Slru { probationary_frac: f64 },
    Clock,
    ClockPro,
    Nru,
}
```

Three design decisions are worth naming:

- **`#[non_exhaustive]`.** Adding a new variant (e.g. when LIRS lands
  off the roadmap) is a **minor** version bump rather than a major
  one. Downstream `match` statements over `CachePolicy` must include a
  `_ =>` arm, which is the standard `non_exhaustive` discipline.
- **Config carried inline.** `LruK { k }` rather than `LruK` + separate
  `set_k`. The variant is the place where the parameter is
  type-checked, and `CachePolicy` stays `Copy` because every payload
  is `Copy`. This makes `let policy: CachePolicy = config.into();`
  trivial and lets callers pass `CachePolicy` by value without
  ceremony.
- **Tag separated from implementation.** `CachePolicy::Lru` is a
  user-facing intent; `CacheInner::Lru(LruCore<K, V>)` is the
  internal storage. Keeping them separate means the internal type can
  change (e.g. swap `LruCore` for a new implementation) without
  touching the public enum.

## `DynCache<K, V>` — uniform runtime type

The public dispatcher:

```rust,ignore
pub struct DynCache<K, V>
where
    K: Copy + Eq + Hash + Ord,
    V: Clone + Debug,
{
    inner: CacheInner<K, V>,
}

enum CacheInner<K, V> /* same bounds */ {
    #[cfg(feature = "policy-fifo")]    Fifo(FifoCache<K, V>),
    #[cfg(feature = "policy-lru")]     Lru(LruCore<K, V>),
    #[cfg(feature = "policy-fast-lru")] FastLru(FastLru<K, V>),
    #[cfg(feature = "policy-lru-k")]   LruK(LrukCache<K, V>),
    #[cfg(feature = "policy-lfu")]     Lfu(LfuCache<K, V>),
    #[cfg(feature = "policy-heap-lfu")] HeapLfu(HeapLfuCache<K, V>),
    #[cfg(feature = "policy-two-q")]   TwoQ(TwoQCore<K, V>),
    #[cfg(feature = "policy-s3-fifo")] S3Fifo(S3FifoCache<K, V>),
    #[cfg(feature = "policy-arc")]     Arc(ArcCore<K, V>),
    #[cfg(feature = "policy-lifo")]    Lifo(LifoCore<K, V>),
    #[cfg(feature = "policy-mfu")]     Mfu(MfuCore<K, V>),
    #[cfg(feature = "policy-mru")]     Mru(MruCore<K, V>),
    #[cfg(feature = "policy-random")]  Random(RandomCore<K, V>),
    #[cfg(feature = "policy-slru")]    Slru(SlruCore<K, V>),
    #[cfg(feature = "policy-clock")]   Clock(ClockCache<K, V>),
    #[cfg(feature = "policy-clock-pro")] ClockPro(ClockProCache<K, V>),
    #[cfg(feature = "policy-nru")]     Nru(NruCache<K, V>),
}
```

`CacheInner` is **private**. Users only see `DynCache`. Two consequences:

- Internal policy structs (`LruCore`, `S3FifoCache`, …) do not leak
  into the public type system through the dispatcher. They can be
  refactored without breaking SemVer.
- Pattern-matching on the variant from outside the crate is
  impossible, which forces feature requests through method additions
  rather than match-arm proliferation in user code.

## Type bounds: heavier than `Cache<K, V>`

`Cache<K, V>` requires only what each individual policy implementation
needs (typically `K: Eq + Hash`, sometimes `K: Copy`). `DynCache`
requires the **union** of all policies' bounds:

```rust,ignore
K: Copy + Eq + Hash + Ord
V: Clone + Debug
```

Each bound exists because at least one variant needs it:

- `K: Copy` — many policies rely on cheap key copies in eviction paths.
- `K: Eq + Hash` — every hashmap-backed lookup.
- `K: Ord` — `HeapLfuCache` orders keys in a min-heap.
- `V: Clone` — variants that store `Arc<V>` internally (LRU, LFU,
  HeapLFU) fall back to `(*arc).clone()` when `Arc::try_unwrap` fails
  on `insert` / `remove` (see below).
- `V: Debug` — `DynCache: Debug` delegates to the variant's `Debug`.

This is the **library-provided dispatcher tax**. Users who do not want
to pay `K: Ord` can call `LruCore::new(…)` directly and bypass
`DynCache`; the tax only applies when crossing the runtime-dispatch
boundary. The tax is documented at the `DynCache` doc comment so
users picking the dispatcher route know what to expect.

If a future policy adds a heavier bound (e.g. `K: Serialize` for a
persistent-cache policy), it forces every `DynCache` user to satisfy
that bound. The mitigation, when that happens, is a separate
dispatcher type (`DynPersistentCache<K, V>`) rather than tightening
the existing `DynCache` bounds — preserving SemVer for users who
don't need persistence.

## The `Arc<V>` round-trip

Three policies — `LruCore`, `LfuCache`, `HeapLfuCache` — internally
store `Arc<V>` rather than `V`. The rationale lives in those modules
(zero-copy sharing between `peek` and `get`, predictable eviction-time
move, alignment with the concurrent wrappers' `Arc<V>` returns). At
the `DynCache` boundary this creates a small impedance:

```rust,ignore
CacheInner::Lru(lru) => {
    let arc_value = Arc::new(value);
    lru.insert(key, arc_value)
        .map(|arc| Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone()))
},
```

`insert` wraps the value in `Arc` for the policy and tries to unwrap
the returned `Arc<V>` on the way out. `try_unwrap` is O(1) when the
refcount is 1 (the common case for sequential `DynCache`); it falls
back to `(*arc).clone()` only when another reference outlived the
caller's, which happens on iteration paths where the policy held a
secondary reference. The fallback is the reason `V: Clone` is required
on `DynCache`.

The cost is one `Arc::new` per insert and one branch (`try_unwrap`) per
return on Arc-storing variants. It does not affect FIFO, LIFO, MFU,
MRU, 2Q, S3-FIFO, ARC, Clock, Clock-PRO, NRU, Random, SLRU, LRU-K,
or FastLru, which store `V` directly. Users sensitive to this round
trip should pick a `V`-storing policy or use the concrete type
directly.

## Feature gating discipline

Every `CachePolicy` variant, every `CacheInner` variant, every match
arm in every `DynCache` method, every `CacheBuilder::build` arm, and
every `validate_policy` arm is gated by `#[cfg(feature = "policy-X")]`.
The discipline:

- A user building with `default-features = false, features = ["policy-lru"]`
  gets a `CachePolicy` enum with **one variant** and a `DynCache` enum
  with **one inner variant**. Match exhaustiveness still holds because
  every arm vanishes with its variant.
- The internal `match` in each `DynCache` method is **always
  exhaustive** at the active feature set, because every arm and every
  variant share the same set of `cfg` predicates.
- `policy-all` is a convenience feature that turns on every
  `policy-*` feature at once. The default is a curated subset
  (`policy-s3-fifo`, `policy-lru`, `policy-fast-lru`, `policy-lru-k`,
  `policy-clock`) chosen to cover the most-recommended workloads from
  [`docs/policies/README.md`](../policies/README.md).

The cost is that adding a new policy involves edits in *six*
synchronized locations (see [Adding a new policy](#adding-a-new-policy)).
The benefit is that a "policy-lru-only" build is genuinely small —
none of the other 16 policies appear in the resulting binary.

## Validation: panic vs `Result`

`CacheBuilder::build` panics on invalid configuration:

```rust,ignore
assert!(self.capacity > 0, "cache capacity must be greater than 0");
// …
match policy {
    CachePolicy::LruK { k } => assert!(*k > 0, "LruK: k must be greater than 0"),
    CachePolicy::TwoQ { probation_frac } =>
        check_frac("TwoQ: probation_frac", *probation_frac),
    // …
}
```

This is consistent with cachekit's broader error model
([`src/error.rs`](../../src/error.rs)): panics for **programming
errors** (programmer hands the builder a `k = 0`, which has no sensible
behavior), `Result<_, ConfigError>` reserved for **user-supplied
configuration** that arrives through deserialization or external
input.

Callers that need to validate untrusted configuration before calling
`build` should branch on the `CachePolicy` variant and inspect the
payload themselves, or use the per-policy fallible constructors
(`S3FifoCache::try_with_ratios`, future `LrukCache::try_with_k`)
directly. The builder deliberately does not provide a `try_build` —
adding one would split the API surface for marginal gain when the
panic path already catches the bug at the call site.

## `Send + Sync` is conditional

`DynCache<K, V>: Send + Sync` is **not** unconditional. The
`FastLru` policy uses `NonNull<Node>` for single-threaded performance
and is therefore `!Send + !Sync`. The test in
[`src/builder.rs`](../../src/builder.rs) encodes this:

```rust,ignore
#[cfg(all(feature = "policy-lru", not(feature = "policy-fast-lru")))]
const _: () = {
    fn assert_send<T: Send>() {}
    fn check() { assert_send::<DynCache<u64, String>>(); }
};
```

In words: `DynCache<K, V>` is `Send + Sync` whenever no
`!Send`-or-`!Sync` policy variant is enabled. With the default feature
set (which includes `policy-fast-lru`), `DynCache` is **not**
`Send + Sync`. Users who want a sendable `DynCache` should disable
`policy-fast-lru` and use `policy-lru` for the LRU path.

This is a known sharp edge. The alternative — making `FastLru: Send`
via an unsafe impl — would invalidate `FastLru`'s entire design
premise (raw-pointer recency list without atomics). The current
trade prioritises `FastLru`'s single-threaded speed over `DynCache`'s
universal sendability, on the grounds that callers wanting concurrent
access should use a `Concurrent*` wrapper directly (see
[`concurrency.md`](concurrency.md)), not `DynCache`.

## Maintenance cost

The dispatcher's runtime cost is small. The **maintenance** cost is
real:

- **17 inner variants** × **~10 `DynCache` methods** = **~170 match
  arms** that must stay in sync.
- A `Debug` impl, a `default()` (where applicable), and a
  `validate_policy` arm per variant.
- A `Cargo.toml` feature flag per variant.
- A documentation entry per variant in `docs/policies/`.

The mitigations in place:

1. **A single regression test** (`test_all_policies_basic_ops` in
   [`src/builder.rs`](../../src/builder.rs)) loops over every enabled
   policy and exercises `insert` / `get` / `contains` / `len` /
   update / `clear`. Adding a variant immediately surfaces if any arm
   was missed.
2. **Compile-time exhaustiveness** in the inner `match`. Forgetting an
   arm is a build error, not a runtime bug.
3. **`#[non_exhaustive]` on `CachePolicy`** keeps downstream code
   from depending on the full set of variants.

Even with those, the line count of `src/builder.rs` (~1300) is
disproportionate to its semantic content. A `macro_rules!` to generate
per-method dispatchers has been considered and rejected — the
explicit `match` is grep-friendly, readable in source review, and
each arm sometimes diverges from the boilerplate (the `Arc<V>` round
trip is the visible case; future TTL integration is another). Macros
would compress the file but obscure the points where the dispatcher
intervenes.

## Adding a new policy

Checklist for landing a new policy, ordered to minimise compile-time
churn:

1. Implement the policy core: `MyPolicyCache<K, V>` with a `Cache<K, V>`
   impl. Add `MyPolicyCache::new(capacity: usize)` and any config
   constructors. Land this with its own tests.
2. Add a `policy-my-policy` feature in [`Cargo.toml`](../../Cargo.toml).
   Add it to `policy-all`. Decide whether it joins `default = […]`.
3. Add the `CachePolicy::MyPolicy { … }` variant, gated by the new
   feature. Include any config fields as inline payload.
4. Add the `CacheInner::MyPolicy(MyPolicyCache<K, V>)` variant under
   the same `cfg`.
5. Add a match arm in every `DynCache` method (`insert`, `get`, `peek`,
   `contains`, `len`, `capacity`, `remove`, `clear`, `Debug` impl).
6. Add a `CachePolicy::MyPolicy { … } => CacheInner::MyPolicy(…)` arm
   in `CacheBuilder::build`. Add validation in `validate_policy` if
   the variant has constraints (frac in 0..=1, non-zero K, etc.).
7. Add the variant to `all_enabled_policies()` in the test module so
   the regression sweep covers it.
8. Document the policy in `docs/policies/my-policy.md`; if it's a
   roadmap policy graduating to implementation, move the doc from
   `docs/policies/roadmap/` per the rule in
   [`docs/policies/roadmap/README.md`](../policies/roadmap/README.md).
9. Update [`docs/policies/README.md`](../policies/README.md) and
   [`docs/guides/choosing-a-policy.md`](../guides/choosing-a-policy.md).

The work is mechanical. A CR template that lists these nine steps as
checkboxes would reduce the chance of missed updates further.

## Future: `DynExpiringCache<K, V>`

When the `ttl` feature lands ([`ttl.md`](ttl.md) §4(c)), TTL **does
not** modify `DynCache`. Instead, `with_default_ttl` on the builder
returns a sibling type:

```rust,ignore
let mut cache = CacheBuilder::new(1024)
    .with_default_ttl(Duration::from_secs(60))
    .build::<u64, String>(CachePolicy::Lru);
// `cache: DynExpiringCache<u64, String>`, not DynCache.
```

`DynExpiringCache<K, V>` mirrors `DynCache`'s match-arm boilerplate
one level out: each method threads the expiry check through the
inner policy's `Cache` call. The key design choice — argued in detail
in [`ttl.md`](ttl.md) §1, §4(c) — is that `DynExpiringCache` is a
**distinct type**, not `impl Cache for DynCache` plus a wrapper.
Distinctness makes `Expiring<Expiring<DynCache>>` structurally
unrepresentable, which prevents the "two clocks, two indexes"
double-wrapping bug at the type level.

The duplication is real: a parallel ~170 arms for the expiring
variant. It is bounded (one type per cross-cutting capability) and
the trade favours type-level safety over deduplication.

## When not to use `DynCache`

`DynCache` is the right tool when:

- The policy is chosen at runtime from configuration.
- The caller wants a single concrete type that can hold any policy.
- The dispatch cost is amortised over enough work that the `match`
  doesn't dominate.

It is the wrong tool when:

- The policy is known at compile time. Use the concrete type
  (`LruCache::new(…)`, `S3FifoCache::new(…)`) and let monomorphization
  do its work.
- The hottest inner loop is `get`-bound and devirtualization matters
  beyond what enum dispatch provides. Concrete types still win for
  raw throughput on benchmarks (see
  [`benches/comparison.rs`](../../benches/comparison.rs)).
- The caller needs `Send + Sync` and the build includes
  `policy-fast-lru`. See [`Send + Sync`](#send--sync-is-conditional)
  above; use the relevant `Concurrent*` wrapper instead.
- A user wants to plug in their own policy. `DynCache` is closed;
  generic code over `Cache<K, V>` is open.

## See also

- [Design overview](design.md) — §13 frames compile-time and runtime
  composition at the principles level
- [Cache trait hierarchy](trait-hierarchy.md) — kernel trait and
  capability traits
- [Concurrency](concurrency.md) — `Send + Sync` interaction, why
  `Concurrent*` is a separate path
- [TTL design](ttl.md) — `DynExpiringCache` as a worked extension of
  the dispatcher pattern
- [Error model](../../src/error.rs) — `ConfigError` vs panic discipline
- [`src/builder.rs`](../../src/builder.rs) — the canonical
  implementation
