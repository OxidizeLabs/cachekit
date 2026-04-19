# Security Audit — `src/ds/intrusive_list.rs`

**Scope:** `IntrusiveList<T>`, `ConcurrentIntrusiveList<T>`, and their iterators,
as defined in `src/ds/intrusive_list.rs`. The audit also touches on
`SlotArena` where the two modules interact at the trust boundary of `SlotId`.

**Threat model:** Library consumers can provide arbitrary values of `T`, any
`SlotId`, and any closure passed to the `*_with` APIs of
`ConcurrentIntrusiveList`. They cannot directly mutate arena internals. The
module contains no `unsafe` — findings are logical/behavioural issues
(memory/state corruption, deadlock, DoS, panic) rather than memory-unsafety.

**Severity legend:** High (silent state corruption, panic on untrusted input,
deadlock), Medium (privilege/data confusion under realistic conditions, missing
defensive checks), Low (hardening / documentation gaps, edge cases requiring
self-inflicted misuse).

---

## Summary of findings

| ID    | Title                                                                                  | Severity |
|-------|----------------------------------------------------------------------------------------|----------|
| IL-01 | Cross-arena `SlotId` confusion silently corrupts an unrelated list                     | High     |
| IL-02 | `clear`/`clear_shrink` resets generations, letting stale `SlotId`s reassociate         | High     |
| IL-03 | `u32` generation counter wraps after 2³² removals, reviving stale `SlotId`s            | Medium   |
| IL-04 | `*_with` closures hold the lock and can deadlock `ConcurrentIntrusiveList`             | Medium   |
| IL-05 | Private `detach` can desync `head`/`tail` if ever called on an orphaned node           | Medium   |
| IL-06 | `move_to_front` / `move_to_back` ignore the `detach` return value                      | Low      |
| IL-07 | `approx_bytes` subtracts without a saturating guard (brittle, can underflow)           | Low      |
| IL-08 | `with_capacity` performs an unchecked allocation (DoS via over-sized requests)         | Low      |
| IL-09 | Panicking user closure inside `get_mut_with` is safe but undocumented                  | Info     |
| IL-10 | Iterators can be starved by contention but not forced into invalid states              | Info     |

---

## IL-01 — Cross-arena `SlotId` confusion (High)

**Where:** every public method taking `SlotId` — `get`, `get_mut`, `remove`,
`move_to_front`, `move_to_back`, `contains`, `epoch`, `set_epoch`, and the
`ConcurrentIntrusiveList` mirrors.

**Problem.** `SlotId` carries only `(index, generation)`. It does not identify
which `SlotArena` (and therefore which `IntrusiveList`) it was minted by. When
a `SlotId` from list `A` is fed to list `B`, nothing prevents it from
coincidentally matching a live slot in `B` — both arenas allocate `index = 0,
generation = 0` for their first element. All subsequent operations then
mutate the wrong node:

* `get`/`get_mut` return data belonging to an unrelated node (information
  disclosure / privilege confusion).
* `move_to_front(foreign_id)` runs `detach` + `attach_front` on whatever node
  in list `B` has the matching `(index, generation)`, silently reordering
  `B`.
* `remove(foreign_id)` deletes a live element of `B` and returns it to the
  caller.

In a library used as the ordering backbone for eviction/LRU policies this
becomes silent cache poisoning: policy decisions for cache `B` can be driven
by handles that were never associated with it.

**Impact.** State corruption, silent wrong-node access. Likelihood is
proportional to how often the same consumer holds two `IntrusiveList`
instances — which is the documented LRU-ghost pattern used elsewhere in the
repo (`docs/policy-ds/ghost-list.md`).

**Recommendations.**

1. Tag each list/arena with a process-unique `NonZeroU64` "arena id" generated
   at construction, store it inside `SlotId` (or a wrapper `ListId<T>` newtype
   around `SlotId`), and reject foreign ids early in every `SlotId`-taking
   method. This is the fix the
   [`slotmap`](https://crates.io/crates/slotmap) crate uses for exactly this
   class of bug.
2. At minimum, expose `IntrusiveList::contains` as the documented "you must
   check this first" guard and audit callers in the tree.

---

## IL-02 — `clear` resets generations and lets stale `SlotId`s come back to life (High)

**Where:** `IntrusiveList::clear`, `IntrusiveList::clear_shrink`,
`ConcurrentIntrusiveList::clear`, `ConcurrentIntrusiveList::clear_shrink` — all
of which delegate to `SlotArena::clear`:

```588:593:src/ds/slot_arena.rs
pub fn clear(&mut self) {
    self.slots.clear();
    self.generations.clear();
    self.free_list.clear();
    self.len = 0;
}
```

**Problem.** `generations` is truncated to length 0. After `clear()`, the
next `insert()` pushes a fresh generation `0` onto `generations` and returns
`SlotId { index: 0, generation: 0 }`. A `SlotId` the caller still holds from
before the `clear` has the same `(0, 0)` and is therefore accepted by
`arena.contains` / `arena.get` / `arena.get_mut` / `arena.remove`.

**Impact.** A caller that retains "expired" `SlotId`s across a `clear()` —
e.g. an external map keyed by application id — can read, mutate, reorder,
or delete newly inserted nodes that it was never intended to see. This is
functionally equivalent to a
[use-after-free](https://cwe.mitre.org/data/definitions/416.html) at the
logical level.

**Reproducer (conceptual):**

```rust
let mut list = IntrusiveList::new();
let stale = list.push_back("secret-A");   // SlotId { index: 0, generation: 0 }
list.clear();                              // generations wiped
let fresh = list.push_back("secret-B");   // SlotId { index: 0, generation: 0 }

assert_eq!(list.get(stale), list.get(fresh));  // stale id now leaks B
```

**Recommendations.**

1. In `SlotArena::clear`, bump every live generation (`g = g.wrapping_add(1)`)
   instead of truncating `generations`, then clear `slots` / `free_list`.
2. Equivalent (simpler) fix: replace `self.generations.clear()` with a bump of
   every entry and keep the vec length. Only shrink in `clear_shrink` (and even
   then, reissue a new "arena epoch" as suggested in IL-01 so previously
   observed ids cannot reassociate).
3. Document in `IntrusiveList::clear` / `clear_shrink` that all previously
   returned `SlotId`s are invalidated, including ids never before removed.

---

## IL-03 — `u32` generation counter wraps after 2³² removals (Medium)

**Where:** `SlotArena::remove`:

```373:377:src/ds/slot_arena.rs
let value = self.slots[idx].take()?;
self.generations[idx] = self.generations[idx].wrapping_add(1);
self.free_list.push(idx);
self.len -= 1;
```

**Problem.** `generations: Vec<u32>` with `wrapping_add` means a single slot
recycled 2³² times reissues the same `(index, generation)` as an ancient
`SlotId`. In a long-running cache with a hot slot and high churn this
eventually produces an ABA collision and the same class of silent corruption
as IL-02.

**Impact.** On a 1-GHz remove cycle this is ~4 seconds; in practice at
10 M ops/s per slot this is ~7 minutes. For caches running weeks/months in
production this is reachable.

**Recommendations.**

1. Widen generations to `u64` (the `wrapping_add` then represents ~584 years
   at 1 GHz), or
2. Panic / return an error once a slot's generation reaches `u32::MAX` rather
   than wrapping silently. This is an ancillary concern to IL-01 but should be
   fixed regardless.

---

## IL-04 — Closure-holding lock in `ConcurrentIntrusiveList::*_with` can deadlock (Medium)

**Where:** `get_with`, `get_mut_with`, `front_with`, `back_with`, and the
matching `try_*` variants in `ConcurrentIntrusiveList`.

**Problem.** These methods acquire an `RwLock` guard and then invoke the
user-supplied closure while holding it:

```1408:1411:src/ds/intrusive_list.rs
pub fn get_with<R>(&self, id: SlotId, f: impl FnOnce(&T) -> R) -> Option<R> {
    let list = self.inner.read();
    list.get(id).map(f)
}
```

`parking_lot::RwLock` is not reentrant. A closure that performs any operation
on the same `ConcurrentIntrusiveList` (directly or via an `Arc` clone captured
somewhere) will deadlock the current thread. `get_mut_with` uses a write lock,
so *any* other read or write in the closure deadlocks too. For multi-threaded
consumers this is a liveness hazard that converts an unrelated bug in user
code into a full-cache freeze.

**Recommendations.**

1. Document the contract explicitly ("the closure must not reacquire this
   list's lock — doing so will deadlock"). This matches the approach already
   taken in `docs/policy-ds/clock-ring.md` for the ClockRing audit.
2. Optionally catch closure panics with `std::panic::catch_unwind` when `R:
   UnwindSafe` so that a panic inside a user closure never leaves a writer
   guard held longer than necessary. `parking_lot` releases the guard on
   unwind today, but an explicit `catch_unwind` makes the contract
   inspectable.
3. For `get_with` / `front_with` / `back_with`, consider returning
   `Option<MappedRwLockReadGuard<'_, T>>` via `parking_lot::RwLockReadGuard::map`
   so the caller chooses when to drop the guard, rather than forcing them
   into an opaque closure.

---

## IL-05 — `detach` desynchronises `head`/`tail` on orphaned nodes (Medium)

**Where:** `IntrusiveList::detach`, lines 819–847.

```818:847:src/ds/intrusive_list.rs
fn detach(&mut self, id: SlotId) -> Option<()> {
    let (prev, next) = {
        let node = self.arena.get(id)?;
        (node.prev, node.next)
    };

    if let Some(prev_id) = prev {
        if let Some(prev_node) = self.arena.get_mut(prev_id) {
            prev_node.next = next;
        }
    } else {
        self.head = next;
    }

    if let Some(next_id) = next {
        if let Some(next_node) = self.arena.get_mut(next_id) {
            next_node.prev = prev;
        }
    } else {
        self.tail = prev;
    }
    ...
}
```

**Problem.** `detach` reads `(prev, next)` from the node and then unconditionally
mutates `self.head` / `self.tail` when either is `None`. There is no check
that `id` is actually the current `head` (when `prev == None`) or the current
`tail` (when `next == None`). If `detach` is ever invoked on a node that is
resident in the arena but not in the list's chain (`prev == next == None`
while `self.head != Some(id)`), the result is **`self.head = None;
self.tail = None`** — the rest of the linked list is leaked (the nodes stay in
the arena, unreachable from the list's head/tail) until `clear()`.

No public API currently produces such a node — `push_front`/`push_back` always
link immediately and `move_to_*` re-attach — but the function is defensive in
other respects (e.g. guarding the `arena.get_mut` calls) and this one
assumption is load-bearing across all mutating operations. It is also the
exact invariant that IL-01 / IL-02 can violate once a foreign or stale
`SlotId` is accepted.

**Recommendations.**

1. When `prev` is `None`, assert/branch `self.head == Some(id)` before writing
   `self.head = next`. Same for `tail`.
2. When the check fails, abort the detach (`return None`) and — at minimum —
   debug-assert so the failure is loud in test builds.

---

## IL-06 — Discarded `detach` return value in `move_to_front` / `move_to_back` (Low)

**Where:**

```700:711:src/ds/intrusive_list.rs
pub fn move_to_front(&mut self, id: SlotId) -> bool {
    if !self.arena.contains(id) {
        return false;
    }
    if Some(id) == self.head {
        return true;
    }
    self.detach(id);
    self.attach_front(id);
    true
}
```

**Problem.** `detach` and `attach_front` both return `Option<()>`, but the
return values are ignored. If `detach` returns `None` (IL-05 path) or
`attach_front` fails to find the node (arena race/future refactor), the list
ends up partially detached. Combined with the absent defensive check in IL-05,
this means malformed input can leave the list in a state that subsequent
iterators follow into infinite loops or `head`/`tail` `None`-but-not-empty.

**Recommendation.** Propagate the result:

```rust
self.detach(id)?;
self.attach_front(id)?;
```

or at minimum `debug_assert!(self.detach(id).is_some());` so invariant
breakage surfaces in tests.

---

## IL-07 — `approx_bytes` subtraction can underflow (Low)

**Where:**

```794:800:src/ds/intrusive_list.rs
pub fn approx_bytes(&self) -> usize {
    std::mem::size_of::<Self>() + self.arena.approx_bytes()
        - std::mem::size_of::<SlotArena<Node<T>>>()
}
```

**Problem.** Today `SlotArena::approx_bytes() >= size_of::<SlotArena<_>>()` by
construction, so this works. It is, however, a fragile cross-crate invariant:
a future refactor that lets `approx_bytes` drop below the inline size causes
a debug panic and a release-mode wrap-around to a gigantic value — a perfect
amplifier if the result is ever used as a capacity budget or reported to an
external controller.

**Recommendation.** Use `saturating_sub`:

```rust
(std::mem::size_of::<Self>() + self.arena.approx_bytes())
    .saturating_sub(std::mem::size_of::<SlotArena<Node<T>>>())
```

and add a regression test that exercises a list with `T` of varying size.

---

## IL-08 — `with_capacity` performs an unchecked allocation (Low)

**Where:** `IntrusiveList::with_capacity`, `ConcurrentIntrusiveList::with_capacity`.

**Problem.** A caller-controlled `usize` is forwarded straight to
`SlotArena::with_capacity`, which ultimately reserves a `Vec<Option<Node<T>>>`
of that capacity. A hostile or buggy caller can request
`usize::MAX / size_of::<Option<Node<T>>>()` and trigger an OOM abort.

**Impact.** DoS via resource exhaustion when the capacity originates from
untrusted input (e.g. a config file).

**Recommendation.** Mirror the `ClockRing` fix applied in commit `c4d7c87`:
introduce a `MAX_CAPACITY` constant and a `try_with_capacity` that returns a
typed error, so callers handling untrusted input can bound the request.

---

## IL-09 — Panic inside `get_mut_with` closure is safe but undocumented (Info)

`parking_lot` releases the write guard on panic unwind, so a panicking closure
does not poison the lock or leak the guard. The list's linked-structure
invariants are maintained because the closure only receives `&mut T`, not
access to `prev`/`next`. However, this behaviour is contractually relevant to
users and should be documented alongside the deadlock warning from IL-04.

---

## IL-10 — Iterator starvation under contention (Info)

`IntrusiveListIter` and friends hold `&IntrusiveList<T>` for their lifetime.
In the concurrent wrapper, the only way to iterate is to take the read lock
outside and then drive the iterator; heavy `write()` traffic can starve the
reader. This is a standard `RwLock` limitation, not a bug, but worth
documenting since the struct lacks an `iter_with` helper.

---

## Suggested remediation ordering

1. **IL-02** — small, contained, and eliminates the most surprising footgun.
2. **IL-01** — adopt a `ListId` newtype or arena tag; the largest change, but
   unblocks a large class of correctness bugs including IL-03's ABA concerns.
3. **IL-05 / IL-06** — add the invariant-preserving checks and propagate
   `detach` failures.
4. **IL-03** — widen generation to `u64` (trivial) while touching `SlotArena`.
5. **IL-04 / IL-09** — documentation + optional API that returns a mapped
   guard.
6. **IL-07 / IL-08** — defensive hardening alongside the existing ClockRing
   patterns.

## Out of scope / not a vulnerability

* No `unsafe` blocks; no memory unsafety was observed in this file.
* `#[repr(C)]` on `Node<T>` is benign — layout is used for cache-line tuning,
  not for FFI trust boundaries.
* `debug_validate_invariants` uses `std::collections::HashSet`, but the keys
  are local `SlotId`s and `HashMap/HashSet` already default to a
  DoS-resistant hasher.
* `parking_lot::RwLock` is non-poisoning by design; the audit does not treat
  this as a vulnerability.

## References

* CWE-416: Use After Free — relevant to IL-02 / IL-03.
* CWE-662: Improper Synchronization — relevant to IL-04.
* CWE-770: Allocation of Resources Without Limits or Throttling — IL-08.
* Prior art in this repo: `docs/policy-ds/clock-ring.md`, commit `c4d7c87`
  ("Fix/security audit clockring").
