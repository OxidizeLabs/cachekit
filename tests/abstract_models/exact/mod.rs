//! Exact and mirror reference models.
//!
//! **Exact** models encode the policy rule directly (e.g. LRU deque, FIFO insertion order).
//! Victims and residency must match the implementation on every trace step.
//!
//! **Mirror** models transcribe internal state from the real data structure (`ClockRing`,
//! `TwoQCore`, `SlruCore`) rather than a simplified abstract rule. Use when behavior is
//! defined by the DS layout.
//!
//! See the [policy matrix](../../../docs/testing/specs/matrix.md) for per-policy model types.
//!
//! Each submodule is gated by the matching `policy-*` feature (plus `ttl` for `lru`).

#[cfg(feature = "policy-clock")]
pub mod clock;
#[cfg(feature = "policy-fifo")]
pub mod fifo;
#[cfg(feature = "policy-heap-lfu")]
pub mod heap_lfu;
#[cfg(feature = "policy-lfu")]
pub mod lfu;
#[cfg(feature = "policy-lifo")]
pub mod lifo;
#[cfg(any(feature = "policy-lru", feature = "policy-fast-lru", feature = "ttl"))]
pub mod lru;
#[cfg(feature = "policy-lru-k")]
pub mod lru_k;
#[cfg(feature = "policy-mfu")]
pub mod mfu;
#[cfg(feature = "policy-mru")]
pub mod mru;
#[cfg(feature = "policy-nru")]
pub mod nru;
#[cfg(feature = "policy-slru")]
pub mod slru;
#[cfg(feature = "policy-two-q")]
pub mod two_q;
