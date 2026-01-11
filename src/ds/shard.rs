//! Shared sharding helpers for consistent shard selection.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Deterministic shard selector based on a hash seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardSelector {
    shards: usize,
    seed: u64,
}

impl ShardSelector {
    /// Creates a selector for `shards` with a given `seed`.
    pub fn new(shards: usize, seed: u64) -> Self {
        Self {
            shards: shards.max(1),
            seed,
        }
    }

    /// Returns the number of shards.
    pub fn shard_count(self) -> usize {
        self.shards
    }

    /// Maps a key to a shard index in `[0, shards)`.
    pub fn shard_for_key<K: Hash>(self, key: &K) -> usize {
        let mut hasher = DefaultHasher::new();
        self.seed.hash(&mut hasher);
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.shards
    }
}

impl Default for ShardSelector {
    fn default() -> Self {
        Self::new(1, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_selector_is_deterministic() {
        let selector = ShardSelector::new(8, 123);
        let a = selector.shard_for_key(&"key");
        let b = selector.shard_for_key(&"key");
        assert_eq!(a, b);
        assert!(a < selector.shard_count());
    }
}
