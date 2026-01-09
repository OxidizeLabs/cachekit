use std::cell::Cell;

/// A metrics-only cell.
///
/// # Safety
/// This type is only safe if all accesses are externally synchronized.
/// In this system, it is protected by an RwLock at a higher level.
#[repr(transparent)]
#[derive(Debug, Default)]
pub struct MetricsCell(Cell<u64>);

impl MetricsCell {
    #[inline]
    pub fn new() -> Self {
        Self(Cell::new(0))
    }

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
