//! Convenience re-exports for the most common cachekit types.
//!
//! Import everything with:
//!
//! ```
//! use cachekit::prelude::*;
//! ```
//!
//! This gives you the core traits ([`CoreCache`], [`ReadOnlyCache`],
//! [`MutableCache`]), the policy-specific traits, and the [`CacheBuilder`]
//! entry point. Internal data structures and concrete policy types are
//! available from their respective modules ([`ds`](crate::ds),
//! [`policy`](crate::policy)).

pub use crate::builder::{Cache, CacheBuilder, CachePolicy};
pub use crate::traits::{
    ConcurrentCache, CoreCache, FifoCacheTrait, LfuCacheTrait, LruCacheTrait, LrukCacheTrait,
    MutableCache, ReadOnlyCache,
};

#[cfg(feature = "metrics")]
pub use crate::metrics::snapshot::CacheMetricsSnapshot;
