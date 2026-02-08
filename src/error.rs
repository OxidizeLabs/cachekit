//! Error types for the cachekit library.
//!
//! ## Key Components
//!
//! - [`InvariantError`]: Returned when internal data-structure invariants are
//!   violated (debug-only `check_invariants` methods).
//! - [`ConfigError`]: Returned when cache configuration parameters are invalid
//!   (e.g. zero capacity, out-of-range ratios).
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::error::ConfigError;
//! use cachekit::policy::s3_fifo::S3FifoCache;
//!
//! // Fallible constructor for user-configurable parameters
//! let cache: Result<S3FifoCache<String, i32>, ConfigError> =
//!     S3FifoCache::try_with_ratios(100, 0.1, 0.9);
//! assert!(cache.is_ok());
//!
//! // Invalid ratio is caught without panicking
//! let bad = S3FifoCache::<String, i32>::try_with_ratios(100, 2.0, 0.9);
//! assert!(bad.is_err());
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// InvariantError
// ---------------------------------------------------------------------------

/// Error returned when internal cache invariants are violated.
///
/// Produced by debug-only `check_invariants` methods on cache types
/// (e.g. [`S3FifoCache::check_invariants`](crate::policy::s3_fifo::S3FifoCache::check_invariants)).
/// Carries a human-readable description of which invariant failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvariantError(String);

impl InvariantError {
    /// Creates a new `InvariantError` with the given description.
    #[inline]
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }

    /// Returns the error description.
    #[inline]
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for InvariantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InvariantError {}

// ---------------------------------------------------------------------------
// ConfigError
// ---------------------------------------------------------------------------

/// Error returned when cache configuration parameters are invalid.
///
/// Produced by fallible constructors such as
/// [`S3FifoCache::try_with_ratios`](crate::policy::s3_fifo::S3FifoCache::try_with_ratios)
/// and builder `try_build()` methods. Carries a human-readable description of
/// which parameter failed validation.
///
/// # Example
///
/// ```
/// use cachekit::error::ConfigError;
/// use cachekit::policy::s3_fifo::S3FifoCache;
///
/// let err = S3FifoCache::<u64, u64>::try_with_ratios(0, 0.1, 0.9).unwrap_err();
/// assert!(err.to_string().contains("capacity"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(String);

impl ConfigError {
    /// Creates a new `ConfigError` with the given description.
    #[inline]
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }

    /// Returns the error description.
    #[inline]
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- InvariantError ---------------------------------------------------

    #[test]
    fn invariant_display_shows_message() {
        let err = InvariantError::new("queue length mismatch");
        assert_eq!(err.to_string(), "queue length mismatch");
    }

    #[test]
    fn invariant_debug_includes_message() {
        let err = InvariantError::new("bad pointer");
        let dbg = format!("{:?}", err);
        assert!(dbg.contains("bad pointer"));
    }

    #[test]
    fn invariant_message_accessor() {
        let err = InvariantError::new("test");
        assert_eq!(err.message(), "test");
    }

    #[test]
    fn invariant_clone_and_eq() {
        let a = InvariantError::new("x");
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn invariant_implements_std_error() {
        fn assert_error<T: std::error::Error>() {}
        assert_error::<InvariantError>();
    }

    // -- ConfigError ------------------------------------------------------

    #[test]
    fn config_display_shows_message() {
        let err = ConfigError::new("capacity must be > 0");
        assert_eq!(err.to_string(), "capacity must be > 0");
    }

    #[test]
    fn config_debug_includes_message() {
        let err = ConfigError::new("bad ratio");
        let dbg = format!("{:?}", err);
        assert!(dbg.contains("bad ratio"));
    }

    #[test]
    fn config_message_accessor() {
        let err = ConfigError::new("test");
        assert_eq!(err.message(), "test");
    }

    #[test]
    fn config_clone_and_eq() {
        let a = ConfigError::new("x");
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn config_implements_std_error() {
        fn assert_error<T: std::error::Error>() {}
        assert_error::<ConfigError>();
    }
}
