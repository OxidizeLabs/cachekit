//! Policy semantic proptests (abstract interpretation oracles).
//!
//! Run: `cargo test --test policy_semantics --all-features`
//! High cases: `PROPTEST_CASES=1000 cargo test --test policy_semantics --all-features`

#[path = "../abstract_models/mod.rs"]
mod abstract_models;

#[cfg(feature = "policy-arc")]
mod arc_tests;
#[cfg(feature = "policy-car")]
mod car_tests;
#[cfg(feature = "policy-clock-pro")]
mod clock_pro_tests;
#[cfg(feature = "policy-clock")]
mod clock_tests;
mod dual_impl_tests;
#[cfg(feature = "policy-fast-lru")]
mod fast_lru_tests;
#[cfg(feature = "policy-fifo")]
mod fifo_tests;
#[cfg(feature = "policy-heap-lfu")]
mod heap_lfu_tests;
#[cfg(feature = "policy-lfu")]
mod lfu_tests;
#[cfg(feature = "policy-lifo")]
mod lifo_tests;
#[cfg(feature = "policy-lru-k")]
mod lru_k_tests;
#[cfg(feature = "policy-lru")]
mod lru_tests;
#[cfg(feature = "policy-mfu")]
mod mfu_tests;
#[cfg(feature = "policy-mru")]
mod mru_tests;
#[cfg(feature = "policy-nru")]
mod nru_tests;
#[cfg(feature = "policy-s3-fifo")]
mod s3_fifo_tests;
#[cfg(feature = "policy-slru")]
mod slru_tests;
#[cfg(feature = "policy-two-q")]
mod two_q_tests;
