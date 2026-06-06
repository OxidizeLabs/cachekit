//! Shared assertion helpers for dual-run cache vs model tests.
//!
//! These functions compare a real cache (implementing capability traits from
//! [`cachekit::traits`](../../src/traits.rs)) against a [`PolicyModel`] step produced by
//! [`PolicyModel::apply`].
//!
//! Typical usage in `policy_semantics/*_tests.rs`:
//!
//! ```ignore
//! let step = model.apply(op.clone());
//! // … apply op to cache …
//! assert_peek_victim(cache, model);
//! assert_recency_rank(cache, model.model_recency_rank(k), k);
//! ```

use std::collections::HashSet;
use std::hash::Hash;

use cachekit::traits::{RecencyTracking, VictimInspectable};

use crate::abstract_models::PolicyModel;

/// Assert `peek_victim` matches model when cache is non-empty.
pub fn assert_peek_victim<K, V, C, M>(cache: &C, model: &M)
where
    K: Clone + Eq + Hash + std::fmt::Debug,
    C: VictimInspectable<K, V>,
    M: PolicyModel<K>,
{
    match model.peek_victim_key() {
        Some(expected) => {
            let (key, _) = cache.peek_victim().expect("model has victim");
            assert_eq!(*key, expected);
        },
        None => assert!(cache.peek_victim().is_none()),
    }
}

/// Assert recency ranks match between cache and LRU model.
pub fn assert_recency_rank<K, V, C>(cache: &C, model_rank: Option<usize>, key: &K)
where
    K: Eq + Hash,
    C: RecencyTracking<K, V>,
{
    assert_eq!(cache.recency_rank(key), model_rank);
}

/// Residency set from probing the `u8` key space (`0..=255`).
///
/// Canonical helper for dual-run tests; replaces inline `(0..=255u8).filter(…).collect()`.
pub fn probe_resident<K>(contains: impl Fn(&K) -> bool) -> HashSet<K>
where
    K: Clone + From<u8> + Eq + Hash,
{
    (0..=255u8).map(K::from).filter(|k| contains(k)).collect()
}
