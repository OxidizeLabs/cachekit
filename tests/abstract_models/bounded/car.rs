//! CAR bounded oracle — structural invariant checks on `CarCore`.
//!
//! **Tier:** bounded. Clock-with-adaptation victim is not modeled exactly.
//! **Checks:** `len <= capacity`, `CarCore::debug_validate_invariants` after every op.
//! **Tests:** `policy_semantics/car_tests.rs`.
//! **Op strategy:** [`op_strategy`](super::super::op_strategy) (`GetMut`/`Touch`/`EvictOne` no-op
//! in adapter).
