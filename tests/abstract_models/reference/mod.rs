//! Spec-derived reference models (independent formulation from operational specs).
//!
//! These models are transcribed from [`docs/testing/specs/`](../../../docs/testing/specs/)
//! only — not from reading `src/policy/`. Cross-model tests in `policy_semantics/` assert
//! agreement with [`exact`](../exact/) models on the same traces.
//!
//! **Not performance references:** use `HashSet`, `VecDeque`, `HashMap`, and timestamps;
//! production code uses slabs, rings, and intrusive lists.

#[cfg(feature = "policy-fifo")]
pub mod fifo;

#[cfg(feature = "policy-heap-lfu")]
pub mod heap_lfu;

#[cfg(feature = "policy-lfu")]
pub mod lfu;

#[cfg(feature = "policy-lifo")]
pub mod lifo;

#[cfg(feature = "policy-mfu")]
pub mod mfu;

#[cfg(feature = "policy-mru")]
pub mod mru;

#[cfg(any(feature = "policy-lru", feature = "policy-fast-lru", feature = "ttl"))]
pub mod lru;

#[cfg(feature = "policy-lru-k")]
pub mod lru_k;
