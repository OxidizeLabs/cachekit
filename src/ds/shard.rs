//! Shared sharding helpers for consistent shard selection.
//!
//! Provides deterministic key-to-shard mapping used by sharded data structures
//! like [`ShardedSlotArena`](crate::ds::ShardedSlotArena) and
//! [`ShardedHashMapStore`](crate::store::ShardedHashMapStore).
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         Shard Selection Flow                           │
//! │                                                                         │
//! │   Input Key                                                             │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌───────────────────────────────────────────────────────────────┐   │
//! │   │  ShardSelector { shards: 4, seed: 42 }                        │   │
//! │   │                                                               │   │
//! │   │  1. Create DefaultHasher                                      │   │
//! │   │  2. Hash seed: 42.hash(&mut hasher)                          │   │
//! │   │  3. Hash key:  key.hash(&mut hasher)                         │   │
//! │   │  4. Compute:   hasher.finish() % 4                           │   │
//! │   │                                                               │   │
//! │   └───────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   Shard Index: 0, 1, 2, or 3                                           │
//! │                                                                         │
//! │   ┌─────────┬─────────┬─────────┬─────────┐                           │
//! │   │ Shard 0 │ Shard 1 │ Shard 2 │ Shard 3 │                           │
//! │   │  keys   │  keys   │  keys   │  keys   │                           │
//! │   │  A, E   │  B, F   │  C, G   │  D, H   │                           │
//! │   └─────────┴─────────┴─────────┴─────────┘                           │
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
