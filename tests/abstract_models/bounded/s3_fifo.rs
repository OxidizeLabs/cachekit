//! S3-FIFO bounded oracle — residency and structural invariant checks.
//!
//! **Tier:** bounded. Three-queue scan resistance; victim not uniquely determined here.
//! **Checks:** `len <= capacity`, `S3FifoCache::check_invariants` after every op.
//! **Tests:** `policy_semantics/s3_fifo_tests.rs`.
//! **Op strategy:** [`op_strategy_with_get_mut`](super::super::op_strategy_with_get_mut).
