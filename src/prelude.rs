//! Convenience re-exports for the most common cachekit types.
//!
//! Import everything with:
//!
//! ```
//! use cachekit::prelude::*;
//! ```
//!
//! This gives you the main [`Cache`] trait, the optional capability traits,
//! and the [`CacheBuilder`] entry point. Concrete policy types are available
//! from their respective modules ([`policy`](crate::policy)).

pub use crate::builder::{CacheBuilder, CachePolicy, DynCache};
pub use crate::traits::{
    Cache, ConcurrentCache, EvictingCache, FrequencyTracking, HistoryTracking, RecencyTracking,
    VictimInspectable,
};

#[cfg(feature = "metrics")]
pub use crate::metrics::snapshot::CacheMetricsSnapshot;

#[cfg(feature = "ttl")]
pub use crate::{
    builder::{DynExpiringCache, ExpiringBuilder},
    time::{Clock, MockClock, StdClock},
    traits::{ExpiringCache, Tick, TtlStatus},
};
