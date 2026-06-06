//! Exact and mirror reference models.
//!
//! **Exact** models encode the policy rule directly (e.g. LRU deque, FIFO insertion order).
//! Victims and residency must match the implementation on every trace step.
//!
//! **Mirror** models transcribe internal state from the real data structure (`ClockRing`,
//! `TwoQCore`, `SlruCore`) rather than a simplified abstract rule. Use when behavior is
//! defined by the DS layout.
//!
//! See the [policy matrix](README.md#policy-coverage) for per-policy model types.

#![allow(dead_code)]

pub mod clock;
pub mod fifo;
pub mod heap_lfu;
pub mod lfu;
pub mod lifo;
pub mod lru;
pub mod lru_k;
pub mod mfu;
pub mod mru;
pub mod nru;
pub mod slru;
pub mod two_q;
