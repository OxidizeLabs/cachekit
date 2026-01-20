//! Benchmark support utilities for cachekit.
//!
//! This crate provides shared infrastructure for benchmarking cache policies,
//! including workload generation, metrics collection, and result formatting.

pub mod json_results;
pub mod metrics;
pub mod registry;
pub mod workload;

// Note: for_each_policy macro is automatically exported at crate root via #[macro_export]
