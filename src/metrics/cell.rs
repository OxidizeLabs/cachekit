use std::cell::Cell;

/// A metrics-only cell backed by [`Cell<u64>`].
///
/// All accesses must be externally synchronized (e.g. by an `RwLock`).
/// This type is **not** safe for unsynchronized concurrent use.
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

// SAFETY:
// All access to MetricsCell is externally synchronized by an RwLock.
// Metrics are observational and do not affect correctness.
unsafe impl Sync for MetricsCell {}
unsafe impl Send for MetricsCell {}
