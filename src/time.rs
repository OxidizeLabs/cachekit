//! Pluggable monotonic clock used by the TTL layer.
//!
//! ## Architecture
//!
//! ```text
//!     ┌─────────────────────────────────────────────────────────────┐
//!     │  Clock trait                                                │
//!     │    fn now(&self) -> u64                                     │
//!     │                                                             │
//!     │  Implementations                                            │
//!     │    StdClock(Instant anchor) ── ms since anchor              │
//!     │    MockClock(AtomicU64)     ── advance() / set() in tests   │
//!     └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! - [`Clock`] — minimal trait the TTL decorator consults on every operation.
//! - [`StdClock`] — anchored to a base [`Instant`] captured at construction;
//!   `now()` returns milliseconds since the anchor (saturating cast).
//! - [`MockClock`] — `AtomicU64` ticks for deterministic tests, proptest, and
//!   fuzz harnesses. Supports interior mutation through `&self`.
//!
//! ## Core Operations
//!
//! Tick unit is **milliseconds**. `u64::MAX` ms covers roughly 585 million
//! years, so saturating casts and additions never wrap in practice. The TTL
//! layer documents that `u64::MAX` is the "effectively never expires"
//! sentinel; do not return it from `Clock::now`.
//!
//! ## Performance Trade-offs
//!
//! - `StdClock::now()` calls `Instant::now()`, which on modern Linux is
//!   ~15–25 ns through the vDSO. The TTL decorator should call it once per
//!   operation, not multiple times.
//! - `MockClock::now()` is a single `Ordering::Relaxed` load.
//!
//! ## Thread Safety
//!
//! Every clock here is `Send + Sync`; `now` is `&self` so concurrent caches
//! can share a clock under a single read lock.
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::time::{Clock, MockClock};
//! use std::time::Duration;
//!
//! let clock = MockClock::new();
//! assert_eq!(clock.now(), 0);
//!
//! clock.advance(Duration::from_millis(150));
//! assert_eq!(clock.now(), 150);
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Saturating cast of `Duration` to millisecond ticks.
///
/// `Duration::as_millis()` returns `u128`; this clamps to `u64::MAX` rather
/// than truncating. With ms resolution the boundary is unreachable in
/// practice but the cast keeps the contract explicit.
#[inline]
pub(crate) fn duration_to_ticks(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// Monotonic millisecond clock consulted by the TTL layer.
///
/// Implementations must be monotonic non-decreasing across calls and should
/// never return `u64::MAX`, which the TTL layer reserves as the "effectively
/// never expires" sentinel.
///
/// This trait is **object-safe** and can be used as `dyn Clock`.
///
/// # Examples
///
/// ```
/// use cachekit::time::Clock;
///
/// #[derive(Debug)]
/// struct FixedClock(u64);
///
/// impl Clock for FixedClock {
///     fn now(&self) -> u64 { self.0 }
/// }
///
/// let clock = FixedClock(42);
/// assert_eq!(clock.now(), 42);
/// ```
pub trait Clock: Send + Sync + std::fmt::Debug {
    /// Returns the current tick.
    ///
    /// The tick unit is implementation-defined but is conventionally
    /// milliseconds in `cachekit`.
    fn now(&self) -> u64;
}

/// `Clock` backed by `std::time::Instant`.
///
/// `now()` returns milliseconds elapsed since the anchor captured at
/// construction. Two `StdClock` instances created at different times observe
/// different epochs; deadlines must therefore be computed against a single
/// clock instance.
///
/// # Example
///
/// ```
/// use cachekit::time::{Clock, StdClock};
/// use std::thread::sleep;
/// use std::time::Duration;
///
/// let clock = StdClock::new();
/// let a = clock.now();
/// sleep(Duration::from_millis(5));
/// let b = clock.now();
/// assert!(b >= a);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StdClock {
    anchor: Instant,
}

impl StdClock {
    /// Creates a clock anchored at `Instant::now()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cachekit::time::{Clock, StdClock};
    ///
    /// let clock = StdClock::new();
    /// let _tick = clock.now(); // ms since construction
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            anchor: Instant::now(),
        }
    }

    /// Creates a clock anchored at the supplied `Instant`.
    ///
    /// Useful for tests that need to derive several clocks from a shared
    /// anchor without depending on wall-clock timing.
    ///
    /// # Examples
    ///
    /// ```
    /// use cachekit::time::{Clock, StdClock};
    /// use std::time::Instant;
    ///
    /// let anchor = Instant::now();
    /// let clock_a = StdClock::with_anchor(anchor);
    /// let clock_b = StdClock::with_anchor(anchor);
    /// // Both clocks share the same epoch.
    /// assert!(clock_b.now() >= clock_a.now() || clock_a.now() == clock_b.now());
    /// ```
    #[must_use]
    pub fn with_anchor(anchor: Instant) -> Self {
        Self { anchor }
    }
}

impl From<Instant> for StdClock {
    fn from(anchor: Instant) -> Self {
        Self::with_anchor(anchor)
    }
}

impl Default for StdClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for StdClock {
    #[inline]
    fn now(&self) -> u64 {
        let elapsed = Instant::now().saturating_duration_since(self.anchor);
        duration_to_ticks(elapsed)
    }
}

/// `Clock` backed by an `AtomicU64` for deterministic tests.
///
/// The tick advances only via [`advance`](MockClock::advance) or
/// [`set`](MockClock::set); `now()` never changes on its own. `MockClock`
/// is `Send + Sync` so it can be shared by concurrent cache wrappers.
///
/// # Example
///
/// ```
/// use cachekit::time::{Clock, MockClock};
/// use std::time::Duration;
///
/// let clock = MockClock::new();
/// assert_eq!(clock.now(), 0);
///
/// clock.advance(Duration::from_secs(1));
/// assert_eq!(clock.now(), 1_000);
///
/// clock.set(42);
/// assert_eq!(clock.now(), 42);
/// ```
#[derive(Debug, Default)]
pub struct MockClock {
    now: AtomicU64,
}

impl Clone for MockClock {
    fn clone(&self) -> Self {
        Self {
            now: AtomicU64::new(self.now.load(Ordering::Relaxed)),
        }
    }
}

impl MockClock {
    /// Creates a clock anchored at tick `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cachekit::time::{Clock, MockClock};
    ///
    /// let clock = MockClock::new();
    /// assert_eq!(clock.now(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::with_tick(0)
    }

    /// Creates a clock anchored at the supplied tick.
    ///
    /// # Examples
    ///
    /// ```
    /// use cachekit::time::{Clock, MockClock};
    ///
    /// let clock = MockClock::with_tick(500);
    /// assert_eq!(clock.now(), 500);
    /// ```
    #[must_use]
    pub fn with_tick(tick: u64) -> Self {
        Self {
            now: AtomicU64::new(tick),
        }
    }

    /// Advances the clock by `delta`, saturating at `u64::MAX - 1`.
    ///
    /// Saturates one short of `u64::MAX` because the TTL layer reserves
    /// `u64::MAX` as the "effectively never expires" sentinel.
    ///
    /// # Examples
    ///
    /// ```
    /// use cachekit::time::{Clock, MockClock};
    /// use std::time::Duration;
    ///
    /// let clock = MockClock::new();
    /// clock.advance(Duration::from_millis(100));
    /// clock.advance(Duration::from_secs(1));
    /// assert_eq!(clock.now(), 1_100);
    /// ```
    pub fn advance(&self, delta: Duration) {
        let ticks = duration_to_ticks(delta);
        let mut current = self.now.load(Ordering::Relaxed);
        loop {
            let next = current.saturating_add(ticks).min(u64::MAX - 1);
            match self.now.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
    }

    /// Sets the clock to `tick` directly.
    ///
    /// Callers are responsible for keeping the clock monotonic; setting a
    /// value lower than the current tick violates the `Clock` contract.
    ///
    /// # Examples
    ///
    /// ```
    /// use cachekit::time::{Clock, MockClock};
    ///
    /// let clock = MockClock::new();
    /// clock.set(42);
    /// assert_eq!(clock.now(), 42);
    /// ```
    pub fn set(&self, tick: u64) {
        self.now.store(tick, Ordering::Relaxed);
    }
}

impl From<u64> for MockClock {
    fn from(tick: u64) -> Self {
        Self::with_tick(tick)
    }
}

impl Clock for MockClock {
    #[inline]
    fn now(&self) -> u64 {
        self.now.load(Ordering::Relaxed)
    }
}

const _: () = {
    fn _assert_send_sync<T: Send + Sync>() {}
    fn _check() {
        _assert_send_sync::<StdClock>();
        _assert_send_sync::<MockClock>();
    }
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn std_clock_is_monotonic_non_decreasing() {
        let clock = StdClock::new();
        let a = clock.now();
        let b = clock.now();
        assert!(b >= a);
    }

    #[test]
    fn mock_clock_advances_by_milliseconds() {
        let clock = MockClock::new();
        assert_eq!(clock.now(), 0);
        clock.advance(Duration::from_millis(100));
        assert_eq!(clock.now(), 100);
        clock.advance(Duration::from_secs(1));
        assert_eq!(clock.now(), 1_100);
    }

    #[test]
    fn mock_clock_set_overrides_value() {
        let clock = MockClock::new();
        clock.advance(Duration::from_millis(500));
        clock.set(42);
        assert_eq!(clock.now(), 42);
    }

    #[test]
    fn mock_clock_advance_saturates_below_max() {
        let clock = MockClock::with_tick(u64::MAX - 5);
        clock.advance(Duration::from_millis(100));
        // Saturates to one below MAX (the "effectively never" sentinel).
        assert_eq!(clock.now(), u64::MAX - 1);
        clock.advance(Duration::from_millis(50));
        assert_eq!(clock.now(), u64::MAX - 1);
    }

    #[test]
    fn duration_to_ticks_saturates_on_overflow() {
        assert_eq!(duration_to_ticks(Duration::ZERO), 0);
        assert_eq!(duration_to_ticks(Duration::from_millis(1)), 1);
        // Duration::MAX in ms would overflow u64.
        assert_eq!(duration_to_ticks(Duration::MAX), u64::MAX);
    }

    #[test]
    fn mock_clock_is_send_sync_across_threads() {
        let clock = Arc::new(MockClock::new());
        let mut handles = Vec::new();
        for _ in 0..4 {
            let c = Arc::clone(&clock);
            handles.push(thread::spawn(move || {
                c.advance(Duration::from_millis(10));
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        // Each thread added 10ms; total exactly 40ms (no losses).
        assert_eq!(clock.now(), 40);
    }
}
