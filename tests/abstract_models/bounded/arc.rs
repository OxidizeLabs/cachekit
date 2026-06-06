//! ARC bounded oracle — structural invariant checks, not a full adaptive model.
//!
//! **Source:** [`docs/testing/specs/arc.md`](../../../docs/testing/specs/arc.md) ·
//! [matrix.md](../../../docs/testing/specs/matrix.md)
//! **Tier:** bounded. Victim selection is adaptive; this module documents the test contract.
//! **Checks:** `len <= capacity`, `ArcCore::debug_validate_invariants` after every op.
//! **Tests:** `policy_semantics/arc_tests.rs`.
//! **Op strategy:** [`op_strategy`](super::super::op_strategy) (`GetMut`/`Touch`/`EvictOne` no-op
//! in adapter).
