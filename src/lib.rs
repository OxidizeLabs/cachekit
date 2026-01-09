//! cachekit: cache policies and tiered cache management primitives.
//!
//! See `docs/design.md` for internal architecture and invariants.

pub mod error;
pub mod policy;
pub mod manager;

#[cfg(feature = "metrics")]
pub mod metrics;

pub mod prelude;
pub mod traits;
