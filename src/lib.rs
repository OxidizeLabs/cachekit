//! cachekit: cache policies and tiered cache management primitives.
//!
//! See `docs/design.md` for internal architecture and invariants.

pub mod ds;
pub mod policy;
pub mod store;

#[cfg(feature = "metrics")]
pub mod metrics;

pub mod builder;
pub mod prelude;
pub mod traits;
