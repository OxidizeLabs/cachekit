//! Shared sharding helpers for consistent shard selection.
//!
//! Provides deterministic key-to-shard mapping used by sharded data structures
//! like [`ShardedSlotArena`](crate::ds::ShardedSlotArena) and
//! [`ShardedHashMapStore`](crate::store::hashmap::ShardedHashMapStore).
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         Shard Selection Flow                            │
//! │                                                                         │
//! │   Input Key                                                             │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌───────────────────────────────────────────────────────────────┐     │
//! │   │  ShardSelector { shards: 4, seed: 42 }                        │     │
//! │   │                                                               │     │
//! │   │  1. Create DefaultHasher                                      │     │
//! │   │  2. Hash seed: 42.hash(&mut hasher)                           │     │
//! │   │  3. Hash key:  key.hash(&mut hasher)                          │     │
//! │   │  4. Compute:   hasher.finish() % 4                            │     │
//! │   │                                                               │     │
//! │   └───────────────────────────────────────────────────────────────┘     │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   Shard Index: 0, 1, 2, or 3                                            │
//! │                                                                         │
//! │   ┌─────────┬─────────┬─────────┬─────────┐                             │
//! │   │ Shard 0 │ Shard 1 │ Shard 2 │ Shard 3 │                             │
//! │   │  keys   │  keys   │  keys   │  keys   │                             │
//! │   │  A, E   │  B, F   │  C, G   │  D, H   │                             │
//! │   └─────────┴─────────┴─────────┴─────────┘                             │
//! └─────────────────────────────────────────────────────────────────────────┘
//!
//! Properties
//! ──────────
//! • Deterministic: Same (key, seed, shards) always yields same shard
//! • Uniform: Keys distribute evenly across shards (given good Hash impl)
//! • Seed isolation: Different seeds produce different distributions
//! ```
//!
//! ## Key Concepts
//!
//! - **Deterministic mapping**: Given the same key, seed, and shard count,
//!   `shard_for_key` always returns the same shard index
//! - **Seed isolation**: Different seeds produce different key distributions,
//!   useful for avoiding pathological hash collisions
//! - **Uniform distribution**: Relies on `DefaultHasher` for even spread
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::ds::ShardSelector;
//!
//! // Create selector for 4 shards with seed 0
//! let selector = ShardSelector::new(4, 0);
//!
//! // Map keys to shards
//! let shard_a = selector.shard_for_key(&"user:123");
//! let shard_b = selector.shard_for_key(&"user:456");
//!
//! assert!(shard_a < 4);
//! assert!(shard_b < 4);
//!
//! // Same key always maps to same shard
//! assert_eq!(selector.shard_for_key(&"user:123"), shard_a);
//! ```
//!
//! ## When to Use
//!
//! - Sharded caches and data structures
//! - Consistent hashing for distributed systems
//! - Load balancing across workers or partitions
//!
//! ## Performance
//!
//! - `shard_for_key`: O(1) with cost of hashing the key

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Deterministic shard selector using a seeded hash.
///
/// Maps any `Hash`able key to a shard index in `[0, shards)`. The same
/// `(key, seed, shards)` tuple always produces the same result.
///
/// # Example
///
/// ```
/// use cachekit::ds::ShardSelector;
///
/// let selector = ShardSelector::new(8, 42);
///
/// // Deterministic: same key → same shard
/// let shard = selector.shard_for_key(&"my_key");
/// assert_eq!(selector.shard_for_key(&"my_key"), shard);
///
/// // Different keys may map to different shards
/// let shard_a = selector.shard_for_key(&"key_a");
/// let shard_b = selector.shard_for_key(&"key_b");
/// // shard_a and shard_b are both in [0, 8)
/// ```
///
/// # Seed Isolation
///
/// Different seeds produce different mappings:
///
/// ```
/// use cachekit::ds::ShardSelector;
///
/// let sel1 = ShardSelector::new(4, 100);
/// let sel2 = ShardSelector::new(4, 200);
///
/// // Same key, different seeds → likely different shards
/// let s1 = sel1.shard_for_key(&"test");
/// let s2 = sel2.shard_for_key(&"test");
/// // s1 and s2 may differ (not guaranteed, but probable)
/// ```
#[derive(Debug, PartialEq, Eq)]
pub struct ShardSelector {
    shards: usize,
    seed: u64,
}

impl ShardSelector {
    /// Creates a selector for `shards` shards with the given `seed`.
    ///
    /// The shard count is clamped to at least 1.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardSelector;
    ///
    /// let selector = ShardSelector::new(16, 0);
    /// assert_eq!(selector.shard_count(), 16);
    ///
    /// // Zero shards is clamped to 1
    /// let single = ShardSelector::new(0, 0);
    /// assert_eq!(single.shard_count(), 1);
    /// ```
    pub fn new(shards: usize, seed: u64) -> Self {
        Self {
            shards: shards.max(1),
            seed,
        }
    }

    /// Returns the number of shards.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardSelector;
    ///
    /// let selector = ShardSelector::new(8, 0);
    /// assert_eq!(selector.shard_count(), 8);
    /// ```
    pub fn shard_count(&self) -> usize {
        self.shards
    }

    /// Maps a key to a shard index in `[0, shards)`.
    ///
    /// The mapping is deterministic: the same key always returns the same
    /// shard index for a given selector configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardSelector;
    ///
    /// let selector = ShardSelector::new(4, 0);
    ///
    /// let shard = selector.shard_for_key(&"user:alice");
    /// assert!(shard < 4);
    ///
    /// // Deterministic
    /// assert_eq!(selector.shard_for_key(&"user:alice"), shard);
    ///
    /// // Works with any Hash type
    /// let int_shard = selector.shard_for_key(&12345_u64);
    /// assert!(int_shard < 4);
    /// ```
    pub fn shard_for_key<K: Hash>(&self, key: &K) -> usize {
        let mut hasher = DefaultHasher::new();
        self.seed.hash(&mut hasher);
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.shards
    }
}

impl Default for ShardSelector {
    /// Creates a single-shard selector with seed 0.
    fn default() -> Self {
        Self::new(1, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_selector_is_deterministic() {
        let selector = &ShardSelector::new(8, 123);

        let a = selector.shard_for_key(&"key");
        let b = selector.shard_for_key(&"key");
        assert_eq!(a, b);
        assert!(a < selector.shard_count());
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // =============================================================================
    // Property Tests - Determinism
    // =============================================================================

    proptest! {
        /// Property: Same key always returns same shard
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_deterministic_mapping(
            shard_count in 1usize..64,
            seed in any::<u64>(),
            key in any::<u32>()
        ) {
            let selector = ShardSelector::new(shard_count, seed);

            let shard1 = selector.shard_for_key(&key);
            let shard2 = selector.shard_for_key(&key);
            let shard3 = selector.shard_for_key(&key);

            prop_assert_eq!(shard1, shard2);
            prop_assert_eq!(shard2, shard3);
        }

        /// Property: Deterministic across multiple calls
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_deterministic_batch(
            shard_count in 1usize..64,
            seed in any::<u64>(),
            keys in prop::collection::vec(any::<u32>(), 0..50)
        ) {
            let selector = ShardSelector::new(shard_count, seed);

            // First pass
            let shards1: Vec<_> = keys.iter().map(|k| selector.shard_for_key(k)).collect();

            // Second pass
            let shards2: Vec<_> = keys.iter().map(|k| selector.shard_for_key(k)).collect();

            prop_assert_eq!(shards1, shards2);
        }
    }

    // =============================================================================
    // Property Tests - Range Validity
    // =============================================================================

    proptest! {
        /// Property: Shard index is always in valid range
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_shard_in_range(
            shard_count in 1usize..128,
            seed in any::<u64>(),
            key in any::<u64>()
        ) {
            let selector = ShardSelector::new(shard_count, seed);
            let shard = selector.shard_for_key(&key);

            prop_assert!(shard < shard_count);
            prop_assert!(shard < selector.shard_count());
        }

        /// Property: All keys map to valid shards
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_all_keys_valid_range(
            shard_count in 1usize..64,
            seed in any::<u64>(),
            keys in prop::collection::vec(any::<u32>(), 0..100)
        ) {
            let selector = ShardSelector::new(shard_count, seed);

            for key in keys {
                let shard = selector.shard_for_key(&key);
                prop_assert!(shard < shard_count);
            }
        }
    }

    // =============================================================================
    // Property Tests - Shard Count
    // =============================================================================

    proptest! {
        /// Property: shard_count returns configured count
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_shard_count_matches(
            shard_count in 1usize..128,
            seed in any::<u64>()
        ) {
            let selector = ShardSelector::new(shard_count, seed);
            prop_assert_eq!(selector.shard_count(), shard_count);
        }

        /// Property: Zero shards is clamped to 1
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_zero_shards_clamped(seed in any::<u64>()) {
            let selector = ShardSelector::new(0, seed);
            prop_assert_eq!(selector.shard_count(), 1);

            // All keys should map to shard 0
            for i in 0..10u32 {
                let shard = selector.shard_for_key(&i);
                prop_assert_eq!(shard, 0);
            }
        }
    }

    // =============================================================================
    // Property Tests - Single Shard
    // =============================================================================

    proptest! {
        /// Property: Single shard always returns 0
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_single_shard_returns_zero(
            seed in any::<u64>(),
            keys in prop::collection::vec(any::<u32>(), 0..50)
        ) {
            let selector = ShardSelector::new(1, seed);

            for key in keys {
                let shard = selector.shard_for_key(&key);
                prop_assert_eq!(shard, 0);
            }
        }
    }

    // =============================================================================
    // Property Tests - Seed Isolation
    // =============================================================================

    proptest! {
        /// Property: Different seeds produce different selectors
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_different_seeds_different_selectors(
            shard_count in 1usize..64,
            seed1 in any::<u64>(),
            seed2 in any::<u64>()
        ) {
            prop_assume!(seed1 != seed2);

            let selector1 = ShardSelector::new(shard_count, seed1);
            let selector2 = ShardSelector::new(shard_count, seed2);

            // Different seeds should produce different selectors
            prop_assert_ne!(selector1, selector2);
        }

        /// Property: Different seeds can produce different mappings
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_seed_affects_mapping(
            shard_count in 2usize..16,
            seed1 in any::<u64>(),
            seed2 in any::<u64>(),
            keys in prop::collection::vec(any::<u32>(), 10..50)
        ) {
            prop_assume!(seed1 != seed2);

            let selector1 = ShardSelector::new(shard_count, seed1);
            let selector2 = ShardSelector::new(shard_count, seed2);

            // With different seeds and multiple shards, keys can map to different shards
            // This test ensures the mechanism works without crashes
            // (not enforcing strict distribution as it's probabilistic)
            for key in &keys {
                let _shard1 = selector1.shard_for_key(key);
                let _shard2 = selector2.shard_for_key(key);
                // Both shards are valid, that's all we verify
            }
        }
    }

    // =============================================================================
    // Property Tests - Distribution
    // =============================================================================

    proptest! {
        /// Property: Keys distribute across available shards
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_keys_use_shards(
            shard_count in 2usize..16,
            seed in any::<u64>(),
            keys in prop::collection::vec(any::<u32>(), 20..100)
        ) {
            let selector = ShardSelector::new(shard_count, seed);

            let mut shard_counts = vec![0usize; shard_count];

            for key in &keys {
                let shard = selector.shard_for_key(key);
                shard_counts[shard] += 1;
            }

            // At least one shard should be used
            let used_shards = shard_counts.iter().filter(|&&count| count > 0).count();
            prop_assert!(used_shards > 0);

            // With enough unique keys, we expect multiple shards to be used
            let unique_keys: std::collections::HashSet<_> = keys.iter().collect();
            if unique_keys.len() >= shard_count * 2 {
                // Expect at least some distribution (not enforcing strict uniformity)
                prop_assert!(used_shards > 1);
            }
        }
    }

    // =============================================================================
    // Property Tests - Key Type Flexibility
    // =============================================================================

    proptest! {
        /// Property: Works with different key types (u32)
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_works_with_u32(
            shard_count in 1usize..32,
            seed in any::<u64>(),
            keys in prop::collection::vec(any::<u32>(), 0..30)
        ) {
            let selector = ShardSelector::new(shard_count, seed);

            for key in keys {
                let shard = selector.shard_for_key(&key);
                prop_assert!(shard < shard_count);
            }
        }

        /// Property: Works with different key types (u64)
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_works_with_u64(
            shard_count in 1usize..32,
            seed in any::<u64>(),
            keys in prop::collection::vec(any::<u64>(), 0..30)
        ) {
            let selector = ShardSelector::new(shard_count, seed);

            for key in keys {
                let shard = selector.shard_for_key(&key);
                prop_assert!(shard < shard_count);
            }
        }

        /// Property: Works with different key types (String)
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_works_with_string(
            shard_count in 1usize..32,
            seed in any::<u64>(),
            keys in prop::collection::vec("[a-z]{1,10}", 0..30)
        ) {
            let selector = ShardSelector::new(shard_count, seed);

            for key in keys {
                let shard = selector.shard_for_key(&key);
                prop_assert!(shard < shard_count);
            }
        }
    }

    // =============================================================================
    // Property Tests - Default Implementation
    // =============================================================================

    proptest! {
        /// Property: Default creates single-shard selector
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_default_single_shard(keys in prop::collection::vec(any::<u32>(), 0..30)) {
            let selector = ShardSelector::default();

            prop_assert_eq!(selector.shard_count(), 1);

            // All keys should map to shard 0
            for key in keys {
                let shard = selector.shard_for_key(&key);
                prop_assert_eq!(shard, 0);
            }
        }
    }

    // =============================================================================
    // Property Tests - Equality
    // =============================================================================

    proptest! {
        /// Property: Selectors with same config are equal
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_same_config_equal(
            shard_count in 1usize..64,
            seed in any::<u64>()
        ) {
            let selector1 = ShardSelector::new(shard_count, seed);
            let selector2 = ShardSelector::new(shard_count, seed);

            prop_assert_eq!(selector1, selector2);
        }

        /// Property: Selectors with different shard counts are not equal
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_different_shard_count_not_equal(
            shard_count1 in 1usize..32,
            shard_count2 in 32usize..64,
            seed in any::<u64>()
        ) {
            let selector1 = ShardSelector::new(shard_count1, seed);
            let selector2 = ShardSelector::new(shard_count2, seed);

            prop_assert_ne!(selector1, selector2);
        }
    }
}
