use std::cell::Cell;

/// A metrics-only cell backed by [`Cell<u64>`].
///
/// `MetricsCell` exists so a policy's `&self` read paths can record
/// counters without forcing every embedding type to be `!Sync`. The
/// `unsafe impl Sync` below is sound **only** under the contract
/// documented on those `unsafe impl` blocks; callers that violate it
/// produce a data race.
///
/// Soundness contract (mirrored in `docs/design/metrics.md`):
///
/// - Increments must happen under **exclusive** external synchronization
///   (single-threaded, `&mut self`, behind a write lock, or behind a
///   `Mutex`). A shared `RwLock::read` guard does **not** serialize
///   readers and is **not** sufficient protection: concurrent `incr()`
///   calls behind a read lock are a data race even though every
///   individual increment uses a `Cell::set`.
/// - For counters incremented from a path that is reachable through a
///   shared read lock, use `AtomicU64` (or escalate to a write lock
///   before recording) instead. `MetricsCell` is the wrong primitive
///   for that path.
/// - Approximation is acceptable for metrics; data races are not.
///   "Best-effort observability" never justifies unsynchronized
///   `Cell` mutation.
#[repr(transparent)]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct MetricsCell(Cell<u64>);

impl MetricsCell {
    #[inline]
    pub fn get(&self) -> u64 {
        self.0.get()
    }

    #[inline]
    pub fn incr(&self) {
        self.0.set(self.0.get() + 1);
    }
}

// SAFETY: see the type-level "Soundness contract" doc comment above.
// Callers must ensure that every `incr` / `get` happens under
// exclusive external synchronization (single-threaded, `&mut self`,
// or behind a write lock / `Mutex`). A shared `RwLock::read` guard is
// not sufficient: multiple readers can race on the underlying `Cell`.
// Counters reachable through a read-locked path must use `AtomicU64`
// instead.
unsafe impl Sync for MetricsCell {}
// SAFETY: `Cell<u64>` is `Send` whenever `u64` is, and `MetricsCell`
// adds no extra non-`Send` state.
unsafe impl Send for MetricsCell {}
