# Error Model

> Status: design rationale for cachekit's panic-vs-`Result` discipline,
> the four error types in the public API, and the debug-only invariant
> checks. Companion to [`design.md`](design.md) and [`src/error.rs`](../../src/error.rs).

cachekit treats error handling as a design question, not an ergonomics
question. The rule is:

> **Panic on programming errors. Return `Result` for user-supplied
> input. Reserve invariant checks for `debug_assertions`.**

This document explains where each side of that rule applies, why the
four shipped error types each exist as separate types, and what
discipline a new error type needs to follow.

## The three tiers

cachekit divides every failure mode into one of three tiers, each with
its own response:

| Tier | Cause | Response | Example |
|---|---|---|---|
| 1. Programming error | Bug in the caller's code, statically detectable in principle | Panic | `LruK::with_k(10, 0)` (k = 0) |
| 2. User-supplied input | Configuration arriving from outside the program | `Result<_, ErrorType>` | `S3FifoCache::try_with_ratios(_, 2.0, _)` |
| 3. Invariant violation | Internal data-structure corruption (cannot reach in normal use) | `debug_assert` + `InvariantError` (test/debug only) | `pop_front` while queue length is zero |

The tiers are not opinions — they map to specific Rust constructs and
runtime behaviours. Mixing them (panicking on tier 2, returning
`Result` from tier 3) produces APIs that are either ergonomically
heavy or operationally unsafe.

## Tier 1: panic on programming errors

A "programming error" is a precondition violation the caller could
have prevented with a `if` or a type. cachekit panics in this case
rather than returning `Result`, because:

- The bug is in **the caller's code**, not in untrusted input the
  caller is forwarding.
- The right fix is for the caller to fix their code, not to handle
  an error path at the call site.
- Forcing every call site to handle `Result<_, "you passed 0 for capacity">`
  for a bug they could have prevented adds friction without
  catching anything new.

The shipped examples:

- `CacheBuilder::build` panics on `capacity == 0`, `k == 0` for LRU-K,
  and `probation_frac > 1.0` for 2Q. The validation is centralised in
  `validate_policy` ([`src/builder.rs`](../../src/builder.rs)).
- Direct constructors (`LruCore::new`, `S3FifoCache::new`) panic on
  invalid arguments. The fallible counterparts (`try_with_ratios`,
  `try_with_capacity`) exist for tier 2.
- `assert!(*k > 0, "LruK: k must be greater than 0")` in
  `CacheBuilder::validate_policy` is the canonical shape: a clear
  message that identifies the parameter and the constraint.

The cost is that a panicking call site terminates under the crate's
default `panic = "abort"` release profile. This is intentional —
cachekit's `panic = "abort"` is documented in the
[`Cargo.toml`](../../Cargo.toml) release profile, and the rationale
is that a panic in cache code under load is a bug worth surfacing
through the supervisor / restart strategy, not unwinding.

## Tier 2: `Result` for user-supplied input

When the failure mode is "user passes us configuration we don't
recognise as valid," return `Result`. The shipped error types each
cover a specific surface:

### `ConfigError` — invalid configuration parameters

```rust,ignore
pub struct ConfigError(String);
```

Defined in [`src/error.rs`](../../src/error.rs). Returned by fallible
constructors that accept user-tunable knobs:

- `S3FifoCache::try_with_ratios(capacity, small_ratio, ghost_ratio)`
- Future `try_build` variants on `CacheBuilder`

The contained `String` carries a human-readable description of which
parameter failed validation. By convention messages are lowercase,
unpunctuated, and identify the parameter: `"capacity must be greater
than zero"`, `"small_ratio must be in 0.0..=1.0"`.

`ConfigError`'s presence on a constructor signals that the parameter
set can legitimately come from outside the program — a config file,
a CLI flag, an HTTP request — and the caller should handle invalid
input gracefully rather than crashing the process.

### `StoreFull` — capacity-bound failure

```rust,ignore
pub struct StoreFull;
```

Zero-sized type defined in
[`src/store/traits.rs`](../../src/store/traits.rs). Returned by
`StoreMut::try_insert` and `ConcurrentStore::try_insert` when the
store is at capacity and the insert would exceed it. The contract:

- **`StoreFull` is not a panic.** A full store under capacity
  pressure is the **expected** outcome of `try_insert`. The caller —
  typically a policy layered on top — must respond by evicting and
  retrying.
- **The store does not evict on its own.** `StoreFull` is the
  signal that says "you, policy, decide who to evict." This is the
  core of the policy/storage separation rule from
  [`design.md`](design.md) §7.
- **The error carries no data.** The caller knows what they tried
  to insert; `StoreFull` adds nothing useful by retaining it.

`StoreFull` is **not** in `src/error.rs` despite being an error
type. It lives alongside the trait that returns it because the
two are co-evolving and the surface is small enough that the
co-location aids readability.

### `LazyMinHeapError` — `ds`-layer fallible construction

```rust,ignore
pub enum LazyMinHeapError {
    CapacityTooLarge { requested: usize, max: usize },
    Allocation(std::collections::TryReserveError),
}
```

Defined in [`src/ds/lazy_heap.rs`](../../src/ds/lazy_heap.rs).
Returned by `LazyMinHeap::try_with_capacity` when:

- The requested capacity exceeds the internal `MAX_CAPACITY` bound,
  or
- The allocator cannot satisfy the reservation.

The enum exposes both failure modes distinctly because a caller may
want to retry on `Allocation` (transient memory pressure) but not on
`CapacityTooLarge` (logic bug or genuinely-too-big request that
won't recover).

The pattern generalises: a future "fallible-construction" error type
on any `ds` primitive that pre-allocates should distinguish "you
asked for too much" from "we couldn't get what you asked for."

### `std::collections::TryReserveError` — passthrough

Some `try_new` constructors (`HashMapStore::try_new`,
`ConcurrentHashMapStore::try_new`) return the standard
`TryReserveError` directly rather than wrapping it. The reason: the
only failure mode is allocator pressure, and `TryReserveError`
already says exactly that. Wrapping it would add a layer for no
information.

The shape is: if cachekit has a distinct failure mode of its own
(`CapacityTooLarge`, `StoreFull`), wrap or define a new type; if the
only failure mode is "the allocator said no," return the standard
type and let the caller's error-handling stack absorb it.

## Tier 3: invariant checks (debug-only)

```rust,ignore
pub struct InvariantError(String);
```

Defined in [`src/error.rs`](../../src/error.rs). Returned by
`check_invariants` methods on internal data structures:

```rust,ignore
impl<K, V> S3FifoCache<K, V> {
    #[cfg(any(debug_assertions, test))]
    pub fn check_invariants(&self) -> Result<(), InvariantError> {
        if self.small.len() + self.main.len() != self.map.len() {
            return Err(InvariantError::new("queue length mismatch"));
        }
        // …
        Ok(())
    }
}
```

Three properties define the tier:

- **Off the hot path.** `check_invariants` is called from tests,
  fuzz harnesses, and `debug_assertions` paths. It is never called
  from normal `insert` / `get` / `evict`.
- **Internal-only.** The invariants are about data-structure
  integrity: "the queue length matches the map length", "the heap
  is in heap order", "the ghost list hasn't grown past its bound."
  No caller program would meaningfully react to one of these
  failing — the cache is corrupted, the right response is to
  capture state and bail.
- **Returns `Result`, not panics.** Counter-intuitive given the
  tier-1 rule. The reason: `check_invariants` is called by
  diagnostic code that wants to **report** the violation (in a test
  failure message, a fuzz reproducer, a debug-mode assertion's
  output) rather than crash. Returning `Result` lets the caller
  format the failure; if they want to panic, they `unwrap()`.

`InvariantError` carries the same `String`-message shape as
`ConfigError`, by the same convention: lowercase, unpunctuated,
identifying the specific invariant.

## Why four error types, not one

A single `CachekitError` enum could in principle subsume all four.
cachekit doesn't ship one, deliberately. Three reasons:

- **Each surface has different recovery semantics.** `StoreFull`
  means "evict and retry"; `ConfigError` means "fix your config";
  `LazyMinHeapError::Allocation` means "back off and retry";
  `InvariantError` means "we have a bug, capture state." A unified
  enum forces every caller to either match exhaustively (most of
  which can't happen at their call site) or use a catch-all that
  loses information.
- **Each lives near the trait that uses it.** `StoreFull` lives in
  `src/store/traits.rs`; `LazyMinHeapError` lives in
  `src/ds/lazy_heap.rs`; `ConfigError` and `InvariantError` live
  in `src/error.rs`. Co-location helps maintenance — adding a new
  failure mode to one surface doesn't ripple through the others.
- **Sum types compose poorly across abstractions.** A unified
  enum would propagate every variant up through every layer that
  touched it. The current shape lets a layer convert (or
  re-wrap) only the errors it cares about.

The cost is that downstream code wanting to catch "any cachekit
error" has to enumerate all four. The mitigation is that no
realistic downstream code wants that — each call site touches one
surface at a time and handles that surface's error.

## Operational contract: panic profile

The crate's release profile sets `panic = "abort"`:

```toml
[profile.release]
panic = "abort"
```

Two implications worth naming:

- **A panic terminates the process.** No unwind, no destructors,
  no observer recovery. A panicking weight function in
  `ConcurrentWeightStore` (see
  [`weighted-eviction.md`](weighted-eviction.md)) kills the
  process; a `parking_lot` lock-poisoning concern is moot under
  `panic = "abort"` because the process is gone before any
  observer can read poisoned state.
- **Callers who override the profile take on more contract.**
  Callers building with `panic = "unwind"` get unwind safety up
  to the documented invariants. The
  [`weighted-eviction.md`](weighted-eviction.md) clear-ordering
  rule and the
  [`concurrency.md`](concurrency.md#failure-modes) panic-safety
  notes apply only to this mode.

The interplay matters for error model design: under `abort`, tier 1
panics are terminal and need to be debugged at development time;
under `unwind`, they are catchable but should still be treated as
bugs because the cache may be in an unspecified-but-not-corrupt
state.

## What `Result` does **not** cover

Three failure modes are deliberately not represented as `Result`:

- **OOM in non-`try_*` constructors.** `LruCore::new(huge)` aborts
  on allocator failure. Use `try_with_capacity` to get a `Result`
  surface (where available).
- **Logic errors in policy code.** Eviction picking the wrong
  victim is a bug, not a return value. Detected (when detected) by
  `check_invariants` or by the policy's tests.
- **Concurrent contention.** `parking_lot::RwLock` doesn't poison,
  doesn't time out by default, and doesn't return `Result`. A
  contended cache blocks until it can proceed. Callers who need
  timeouts wrap the cache themselves with a wider locking
  discipline.

## Adding a new error

Checklist for a new failure mode:

1. **Decide the tier.** Programming error, user-supplied input, or
   internal invariant?
2. **Pick or define the type.**
   - Tier 1: use `assert!` / `debug_assert!` / `panic!`. No new
     type needed.
   - Tier 2: define a new type if the failure has data the caller
     needs and no existing type fits. Otherwise reuse `ConfigError`
     (with a clear message) or pass through `TryReserveError`.
   - Tier 3: add a `check_invariants` method on the affected type
     that returns `Result<(), InvariantError>`.
3. **Co-locate.** Types specific to a trait live with the trait
   (`StoreFull` in `src/store/traits.rs`). Types specific to a
   primitive live with the primitive (`LazyMinHeapError`).
   Cross-cutting types (`ConfigError`, `InvariantError`) live in
   `src/error.rs`.
4. **Implement `Display` and `Error`.** Both are required for
   `?` interop with `Box<dyn Error>`. The convention is:
   ```rust,ignore
   impl fmt::Display for MyError { … }
   impl std::error::Error for MyError {}
   ```
   `Display` writes the message; `Error` is empty unless the type
   wraps another error (then `source` returns the inner error).
5. **`Send + Sync + Clone`.** All existing error types satisfy this.
   The convention is `#[derive(Debug, Clone, PartialEq, Eq, Hash)]`
   for value types and matching impls for enums. Errors that flow
   between threads must be `Send + Sync`; errors that get cloned
   into snapshots / test fixtures must be `Clone`.

## Compatibility with `?` and `anyhow`/`thiserror`

The cachekit error types are intentionally **plain types, not
`thiserror`-derived**, to avoid forcing a `thiserror` dependency on
downstream users. They implement `std::error::Error` directly, so
they work with `?`, `Box<dyn Error>`, and any error-aggregation
crate (including `anyhow` and `thiserror::Error` in user code).

A downstream `thiserror`-derived enum that includes a `#[from]
cachekit::ConfigError` works. A downstream `anyhow::Result<_>` that
absorbs cachekit errors via `?` works. The choice not to bundle
either crate keeps the error layer dependency-free and gives
downstream the standard `From` and `Display` shape they expect.

## See also

- [Design overview](design.md) — §12 frames failure modes at the
  principles level
- [Concurrency](concurrency.md) — `parking_lot` non-poisoning,
  atomic check-and-act, lock-acquisition failure modes
- [Builder and runtime dispatch](builder-and-dyn-dispatch.md) —
  panic-in-`build` validation, `try_build`-deliberately-absent
  rationale
- [Weighted eviction](weighted-eviction.md) — `StoreFull`'s role
  and unwind-safety in `clear`
- [`src/error.rs`](../../src/error.rs) — `ConfigError`,
  `InvariantError`
- [`src/store/traits.rs`](../../src/store/traits.rs) — `StoreFull`
- [`src/ds/lazy_heap.rs`](../../src/ds/lazy_heap.rs) —
  `LazyMinHeapError`
