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
//! │   │  ShardSelector { shards: 4, k0, k1 }                          │     │
//! │   │                                                               │     │
//! │   │  1. SipHasher13::new_with_keys(k0, k1)                        │     │
//! │   │  2. key.hash(&mut hasher)                                     │     │
//! │   │  3. Reduce to [0, 4) via fastrange                            │     │
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
//! • Deterministic: Same (key, keys, shards) always yields same shard
//! • Uniform: Keys distribute evenly across shards (SipHash-1-3 PRF)
//! • Keyed: Per-selector SipHash keys defeat hash-flood DoS when kept secret
//! ```
//!
//! ## Key Concepts
//!
//! - **Deterministic mapping**: Given the same key, seed, and shard count,
//!   `shard_for_key` always returns the same shard index
//! - **Keyed hasher**: SipHash-1-3 is instantiated with per-selector keys
//!   derived from the caller-supplied seed, so an attacker who does not know
//!   the seed cannot craft keys that all collide on one shard
//! - **Uniform distribution**: SipHash-1-3 is a cryptographic PRF; combined
//!   with [fastrange] reduction the output is unbiased across `shards`
//!
//! [fastrange]: https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
//!
//! ## Security Considerations
//!
//! `ShardSelector` is the routing primitive for every sharded structure in
//! this crate, so hash collisions on the selector degrade into single-shard
//! contention and defeat the point of sharding entirely. Prior releases used
//! [`std::collections::hash_map::DefaultHasher`], which is **not keyed** — its
//! SipHash keys are fixed at `(0, 0)` — so any attacker who can influence
//! cache keys could pin every entry to one shard regardless of the
//! caller-supplied seed. The hasher is now SipHash-1-3 keyed with per-selector
//! material:
//!
//! - [`ShardSelector::randomized`] draws the key material from the stdlib's
//!   [`RandomState`](std::collections::hash_map::RandomState), which seeds
//!   itself from the OS CSPRNG. Use this whenever shard assignment is *not*
//!   required to be reproducible across processes (the common case).
//! - [`ShardSelector::new`] keeps the deterministic `(shards, seed)`
//!   constructor for reproducible routing. The `seed` value is treated as
//!   secret key material and is never exposed through `Debug`, accessors, or
//!   `Hash`. For HashDoS resistance to hold, `seed` must be kept secret from
//!   attackers who can influence keys; if reproducibility is not required,
//!   prefer `randomized`.
//!
//! The `shards` argument is clamped to `[1, MAX_SHARDS]` to prevent an
//! attacker-controlled configuration value from triggering an oversized
//! allocation in downstream sharded structures that allocate one lock/vector
//! per shard.
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::ds::ShardSelector;
//!
//! // Randomized, HashDoS-safe (keys seeded from the OS CSPRNG).
//! let selector = ShardSelector::randomized(4);
//!
//! // Map keys to shards
//! let shard_a = selector.shard_for_key(&"user:123");
//! let shard_b = selector.shard_for_key(&"user:456");
//!
//! assert!(shard_a < 4);
//! assert!(shard_b < 4);
//!
//! // Same key always maps to same shard *within a single selector*.
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
//! - `shard_for_key`: O(1) with cost of one SipHash-1-3 pass over the key

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};

use siphasher::sip::SipHasher13;

/// Upper bound on the number of shards a `ShardSelector` will accept.
///
/// Caps the per-shard allocation footprint of sharded data structures
/// (arenas, lock arrays, histogram vectors) when the shard count is derived
/// from untrusted configuration. `65_536` shards is well above any
/// practically useful parallelism on current hardware while keeping the
/// per-shard overhead of downstream structures bounded.
pub const MAX_SHARDS: usize = 1 << 16;

/// Deterministic shard selector using a seeded, keyed SipHash-1-3 hasher.
///
/// Maps any `Hash`able key to a shard index in `[0, shards)`. The same key
/// always produces the same result *for a given selector*. Different
/// selectors with different keys produce independent, uncorrelated mappings.
///
/// See the [module-level security
/// considerations](crate::ds::shard#security-considerations) for guidance on
/// choosing between [`ShardSelector::new`] and [`ShardSelector::randomized`].
///
/// # Example
///
/// ```
/// use cachekit::ds::ShardSelector;
///
/// let selector = ShardSelector::randomized(8);
///
/// // Deterministic: same key → same shard (within a selector)
/// let shard = selector.shard_for_key(&"my_key");
/// assert_eq!(selector.shard_for_key(&"my_key"), shard);
/// assert!(shard < 8);
/// ```
#[must_use]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ShardSelector {
    shards: usize,
    k0: u64,
    k1: u64,
}

impl ShardSelector {
    /// Creates a selector for `shards` shards keyed by the caller-supplied
    /// `seed`.
    ///
    /// The shard count is clamped to `[1, MAX_SHARDS]`. The `seed` is expanded
    /// into a SipHash-1-3 key pair; the expansion is deterministic, so the
    /// same `(shards, seed)` pair always yields the same routing.
    ///
    /// `seed` is treated as secret key material: it is not returned by any
    /// accessor, not included in `Debug`, and does not participate in `Hash`
    /// (the type no longer derives `Hash`). For HashDoS resistance to hold,
    /// callers must keep `seed` secret from attackers who can influence the
    /// keys being routed; if reproducibility across processes is not
    /// required, prefer [`ShardSelector::randomized`].
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardSelector;
    ///
    /// let selector = ShardSelector::new(16, 0xC0FFEE);
    /// assert_eq!(selector.shard_count(), 16);
    ///
    /// // Zero shards is clamped to 1
    /// let single = ShardSelector::new(0, 0);
    /// assert_eq!(single.shard_count(), 1);
    /// ```
    pub fn new(shards: usize, seed: u64) -> Self {
        let (k0, k1) = derive_keys(seed);
        Self {
            shards: clamp_shards(shards),
            k0,
            k1,
        }
    }

    /// Creates a selector whose SipHash keys are drawn from the stdlib's
    /// [`RandomState`](std::collections::hash_map::RandomState), which seeds
    /// itself from the OS CSPRNG.
    ///
    /// Use this for HashDoS-resistant sharding when the shard assignment does
    /// not need to be reproducible across processes. The resulting mapping
    /// is stable for the lifetime of the selector but differs between
    /// selectors (including between process restarts).
    ///
    /// The shard count is clamped to `[1, MAX_SHARDS]`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardSelector;
    ///
    /// let selector = ShardSelector::randomized(8);
    /// assert_eq!(selector.shard_count(), 8);
    ///
    /// let shard = selector.shard_for_key(&"user:alice");
    /// assert!(shard < 8);
    /// ```
    pub fn randomized(shards: usize) -> Self {
        let (k0, k1) = random_key_pair();
        Self {
            shards: clamp_shards(shards),
            k0,
            k1,
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
    /// The mapping is deterministic within a selector and uniform (up to
    /// SipHash-1-3's PRF guarantees combined with fastrange reduction).
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
    /// // Deterministic within a selector
    /// assert_eq!(selector.shard_for_key(&"user:alice"), shard);
    ///
    /// // Works with any Hash type
    /// let int_shard = selector.shard_for_key(&12345_u64);
    /// assert!(int_shard < 4);
    /// ```
    pub fn shard_for_key<K: Hash + ?Sized>(&self, key: &K) -> usize {
        let mut hasher = SipHasher13::new_with_keys(self.k0, self.k1);
        key.hash(&mut hasher);
        let h = hasher.finish();
        // Fastrange reduction: `((h * shards) >> 64)` gives a uniform-enough
        // mapping without modulo bias, and preserves full 64 bits of entropy
        // on 32-bit targets (where `h as usize` would otherwise truncate).
        ((u128::from(h) * self.shards as u128) >> 64) as usize
    }
}

impl Default for ShardSelector {
    /// Creates a single-shard selector with a randomized key pair.
    fn default() -> Self {
        Self::randomized(1)
    }
}

impl core::fmt::Debug for ShardSelector {
    /// Redacts the SipHash key material.
    ///
    /// The derived `Debug` would have printed `k0` / `k1` verbatim, turning
    /// any `tracing::debug!`, `dbg!`, or panic-unwind backtrace that touched
    /// a `ShardSelector` into a disclosure channel for the HashDoS key. The
    /// hand-written impl reports only the shard count.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ShardSelector")
            .field("shards", &self.shards)
            .field("keys", &"<redacted>")
            .finish()
    }
}

/// Clamps a caller-supplied shard count to `[1, MAX_SHARDS]`.
///
/// Both bounds matter: `0` would trigger division/modulo by zero (and
/// downstream empty-shard-array indexing), while an unbounded upper value
/// lets an attacker-controlled configuration value coerce downstream
/// sharded structures into allocating `shards` locks / vectors / lanes.
#[inline]
fn clamp_shards(shards: usize) -> usize {
    shards.clamp(1, MAX_SHARDS)
}

/// Expands a 64-bit seed into a pair of SipHash-1-3 keys.
///
/// Uses two different odd-multiplier / xor-constant splats (taken from
/// SplitMix64) so `seed == 0` still produces a non-zero `(k0, k1)` pair, and
/// `k0 != k1` for every seed. This is a *deterministic* derivation: callers
/// that want HashDoS resistance must either keep `seed` secret or use
/// [`ShardSelector::randomized`].
#[inline]
fn derive_keys(seed: u64) -> (u64, u64) {
    let k0 = splitmix64(seed ^ 0x9E37_79B9_7F4A_7C15);
    let k1 = splitmix64(seed ^ 0xBF58_476D_1CE4_E5B9);
    (k0, k1)
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Draws a `(k0, k1)` key pair from the stdlib's `RandomState`, which is
/// seeded from the OS CSPRNG on construction.
///
/// Hashing two distinct sentinels through the same `RandomState` yields two
/// independent-looking 64-bit outputs (SipHash-1-3 is a PRF), giving us 128
/// bits of per-selector entropy without pulling in a `rand` dependency.
fn random_key_pair() -> (u64, u64) {
    let rs = RandomState::new();
    let k0 = rs.hash_one(0x0000_0000_0000_0000_u64);
    let k1 = rs.hash_one(0xFFFF_FFFF_FFFF_FFFF_u64);
    (k0, k1)
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

    #[test]
    fn shards_clamped_to_max_shards() {
        let selector = ShardSelector::new(usize::MAX, 0);
        assert_eq!(selector.shard_count(), MAX_SHARDS);
    }

    #[test]
    fn randomized_selectors_differ_across_construction() {
        // Probability of collision across 128 bits of entropy is negligible,
        // but we construct many selectors to make the test effectively
        // zero-flake even under an adversarial `RandomState` implementation.
        let mut keys = std::collections::HashSet::new();
        for _ in 0..32 {
            let s = ShardSelector::randomized(8);
            keys.insert((s.k0, s.k1));
        }
        assert!(
            keys.len() > 1,
            "randomized selectors produced identical keys across 32 constructions"
        );
    }

    #[test]
    fn debug_impl_does_not_leak_key_material() {
        // Use a recognizable sentinel seed; if the impl ever regresses to
        // printing `k0` / `k1` directly, the derived keys are produced by
        // splitmix64 over the seed, so we check for the seed itself (which
        // only a naive `Debug` would ever reveal).
        let selector = ShardSelector::new(4, 0xDEAD_BEEF_CAFE_F00D);
        let rendered = format!("{selector:?}");
        assert!(rendered.contains("<redacted>"));
        assert!(!rendered.contains("DEAD"));
        assert!(!rendered.contains("dead"));
        assert!(!rendered.contains("3735928559")); // decimal(0xDEADBEEF)
        assert!(!rendered.contains(&selector.k0.to_string()));
        assert!(!rendered.contains(&selector.k1.to_string()));
    }

    #[test]
    fn derive_keys_is_non_trivial_for_zero_seed() {
        let (k0, k1) = derive_keys(0);
        assert_ne!(k0, 0, "seed=0 must not produce k0=0");
        assert_ne!(k1, 0, "seed=0 must not produce k1=0");
        assert_ne!(k0, k1, "derived keys must differ from each other");
    }

    #[test]
    fn fastrange_reduction_covers_full_range() {
        // Deterministic seed so the test is not flaky; with 16 shards and
        // 1024 diverse keys, SipHash-1-3 + fastrange should hit every bucket.
        let selector = ShardSelector::new(16, 0x1234_5678_9ABC_DEF0);
        let mut hits = [false; 16];
        for i in 0..1024u32 {
            hits[selector.shard_for_key(&i)] = true;
        }
        assert!(
            hits.iter().all(|&h| h),
            "fastrange did not cover all shards: {hits:?}"
        );
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
        /// Property: shard_count returns configured count (within [1, MAX_SHARDS])
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

        /// Property: Oversized shard count is clamped to MAX_SHARDS
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_oversized_shards_clamped(
            shard_count in (MAX_SHARDS + 1)..=usize::MAX,
            seed in any::<u64>()
        ) {
            let selector = ShardSelector::new(shard_count, seed);
            prop_assert_eq!(selector.shard_count(), MAX_SHARDS);
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
