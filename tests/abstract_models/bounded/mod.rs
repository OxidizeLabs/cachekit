//! Bounded reference models for adaptive and scan-resistant policies.
//!
//! When a victim is not uniquely determined from residency alone (ARC, CAR, Clock-PRO,
//! S3-FIFO), tests use **invariant-only** checks rather than a `PolicyModel` dual-run.
//!
//! Sibling modules are **documentation stubs** only; real checks live in
//! `policy_semantics/*_tests.rs`.

#[cfg(feature = "policy-arc")]
pub mod arc;
#[cfg(feature = "policy-car")]
pub mod car;
#[cfg(feature = "policy-clock-pro")]
pub mod clock_pro;
#[cfg(feature = "policy-s3-fifo")]
pub mod s3_fifo;
