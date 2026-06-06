//! Clock-PRO bounded oracle — structural invariant checks.
//!
//! **Source:** [`docs/testing/specs/policies/bounded/clock-pro.md`](../../../docs/testing/specs/policies/bounded/clock-pro.md) ·
//! [matrix.md](../../../docs/testing/specs/matrix.md)
//! **Tier:** bounded. Hot/cold/non-resident lists make exact victim prediction impractical.
//! **Checks:** `len <= capacity`, `ClockProCache::debug_validate_invariants` after every op.
//! **Tests:** `policy_semantics/clock_pro_tests.rs`.
//! **Op strategy:** [`op_strategy`](super::super::op_strategy) (`GetMut`/`Touch`/`EvictOne` no-op
//! in adapter).
