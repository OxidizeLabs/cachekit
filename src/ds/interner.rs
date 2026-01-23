//! Simple key interner for mapping external keys to compact handles.
//!
//! Assigns monotonically increasing `u64` handles to unique keys, enabling
//! fast lookups while avoiding repeated key cloning in hot paths.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                         KeyInterner Layout                                  │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐  │
//! │   │  index: HashMap<K, u64>              keys: Vec<K>                   │  │
//! │   │                                                                     │  │
//! │   │  ┌────────────────────────┐          ┌─────────────────────────┐   │  │
//! │   │  │  Key         Handle    │          │ Index   Key             │   │  │
//! │   │  ├────────────────────────┤          ├─────────────────────────┤   │  │
//! │   │  │  "user:123"  → 0       │          │   0     "user:123"      │   │  │
//! │   │  │  "user:456"  → 1       │          │   1     "user:456"      │   │  │
//! │   │  │  "session:a" → 2       │          │   2     "session:a"     │   │  │
//! │   │  └────────────────────────┘          └─────────────────────────┘   │  │
//! │   │                                                                     │  │
//! │   │  intern("user:123") ──► lookup in index ──► return 0               │  │
//! │   │  resolve(1) ──► keys[1] ──► "user:456"                             │  │
//! │   └─────────────────────────────────────────────────────────────────────┘  │
//! │                                                                             │
//! │   Data Flow                                                                │
//! │   ─────────                                                                │
//! │     intern(key):                                                           │
//! │       1. Check index for existing handle                                   │
//! │       2. If found: return handle                                           │
//! │       3. If not: assign handle = keys.len(), store in both structures      │
//! │                                                                             │
//! │     resolve(handle):                                                       │
//! │       1. Direct index into keys vector: O(1)                               │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! - [`KeyInterner`]: Maps keys to compact `u64` handles
//!
//! ## Operations
//!
//! | Operation    | Description                          | Complexity |
//! |--------------|--------------------------------------|------------|
//! | `intern`     | Get or create handle for key         | O(1) avg   |
//! | `get_handle` | Lookup handle without inserting      | O(1) avg   |
//! | `resolve`    | Convert handle back to key reference | O(1)       |
//!
//! ## Use Cases
//!
//! - **Handle-based caches**: Avoid cloning large keys on every access
//! - **Frequency tracking**: Use compact handles as frequency map keys
//! - **Deduplication**: Ensure each unique key has exactly one handle
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::ds::KeyInterner;
//!
//! let mut interner = KeyInterner::new();
//!
//! // Intern keys to get compact handles
//! let h1 = interner.intern(&"long_key_name_1".to_string());
//! let h2 = interner.intern(&"long_key_name_2".to_string());
//!
//! // Same key returns same handle
//! let h1_again = interner.intern(&"long_key_name_1".to_string());
//! assert_eq!(h1, h1_again);
//!
//! // Resolve handle back to key
//! assert_eq!(interner.resolve(h1), Some(&"long_key_name_1".to_string()));
//! ```
//!
//! ## Use Case: Handle-Based Cache
//!
//! ```
//! use cachekit::ds::KeyInterner;
//! use std::collections::HashMap;
//!
//! // External keys are strings, internal cache uses u64 handles
//! let mut interner = KeyInterner::new();
//! let mut cache: HashMap<u64, Vec<u8>> = HashMap::new();
//!
//! fn put(interner: &mut KeyInterner<String>, cache: &mut HashMap<u64, Vec<u8>>,
//!        key: &str, value: Vec<u8>) {
//!     let handle = interner.intern(&key.to_string());
//!     cache.insert(handle, value);
//! }
//!
//! fn get<'a>(interner: &KeyInterner<String>, cache: &'a HashMap<u64, Vec<u8>>,
//!            key: &str) -> Option<&'a Vec<u8>> {
//!     let handle = interner.get_handle(&key.to_string())?;
//!     cache.get(&handle)
//! }
//!
//! put(&mut interner, &mut cache, "session:abc", vec![1, 2, 3]);
//! assert!(get(&interner, &cache, "session:abc").is_some());
//! ```
//!
//! ## Thread Safety
//!
//! `KeyInterner` is not thread-safe. For concurrent use, wrap in
//! `parking_lot::RwLock` or similar synchronization primitive.
//!
//! ## Implementation Notes
//!
//! - Handles are assigned monotonically starting at 0
//! - Keys are never removed (append-only design)
//! - Both `index` and `keys` store copies of the key

use rustc_hash::FxHashMap;
use std::hash::Hash;

/// Monotonic key interner that assigns a `u64` handle to each unique key.
///
/// Maps external keys to compact `u64` handles for efficient storage and lookup.
/// Handles are assigned sequentially starting from 0 and never reused.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Eq + Hash + Clone`
///
/// # Example
///
/// ```
/// use cachekit::ds::KeyInterner;
///
/// let mut interner = KeyInterner::new();
///
/// // Intern returns a handle
/// let handle = interner.intern(&"my_key");
/// assert_eq!(handle, 0);  // First key gets handle 0
///
/// // Same key returns same handle
/// assert_eq!(interner.intern(&"my_key"), 0);
///
/// // Different key gets next handle
/// assert_eq!(interner.intern(&"other_key"), 1);
///
/// // Resolve handle back to key
/// assert_eq!(interner.resolve(0), Some(&"my_key"));
/// ```
///
/// # Use Case: Frequency Tracking
///
/// ```
/// use cachekit::ds::KeyInterner;
/// use std::collections::HashMap;
///
/// let mut interner = KeyInterner::new();
/// let mut freq: HashMap<u64, u32> = HashMap::new();
///
/// // Track access frequency using handles (cheaper than cloning keys)
/// fn access(interner: &mut KeyInterner<String>, freq: &mut HashMap<u64, u32>, key: &str) {
///     let handle = interner.intern(&key.to_string());
///     *freq.entry(handle).or_insert(0) += 1;
/// }
///
/// access(&mut interner, &mut freq, "page_a");
/// access(&mut interner, &mut freq, "page_a");
/// access(&mut interner, &mut freq, "page_b");
///
/// let handle_a = interner.get_handle(&"page_a".to_string()).unwrap();
/// assert_eq!(freq[&handle_a], 2);
/// ```
#[derive(Debug, Default)]
pub struct KeyInterner<K> {
    index: FxHashMap<K, u64>,
    keys: Vec<K>,
}

impl<K> KeyInterner<K>
where
    K: Eq + Hash + Clone,
{
    /// Creates an empty interner.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::KeyInterner;
    ///
    /// let interner: KeyInterner<String> = KeyInterner::new();
    /// assert!(interner.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            index: FxHashMap::default(),
            keys: Vec::new(),
        }
    }

    /// Creates an interner with pre-allocated capacity.
    ///
    /// Pre-allocates space for `capacity` keys to avoid rehashing during growth.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::KeyInterner;
    ///
    /// let interner: KeyInterner<String> = KeyInterner::with_capacity(1000);
    /// assert!(interner.is_empty());
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            index: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            keys: Vec::with_capacity(capacity),
        }
    }

    /// Returns the handle for `key`, inserting it if missing.
    ///
    /// If the key is already interned, returns the existing handle.
    /// Otherwise, assigns the next sequential handle and stores the key.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::KeyInterner;
    ///
    /// let mut interner = KeyInterner::new();
    ///
    /// // First key gets handle 0
    /// let h1 = interner.intern(&"key_a");
    /// assert_eq!(h1, 0);
    ///
    /// // Second key gets handle 1
    /// let h2 = interner.intern(&"key_b");
    /// assert_eq!(h2, 1);
    ///
    /// // Same key returns same handle (no new entry)
    /// let h1_again = interner.intern(&"key_a");
    /// assert_eq!(h1_again, 0);
    /// assert_eq!(interner.len(), 2);  // Still only 2 keys
    /// ```
    pub fn intern(&mut self, key: &K) -> u64 {
        if let Some(&id) = self.index.get(key) {
            return id;
        }
        let id = self.keys.len() as u64;
        self.keys.push(key.clone());
        self.index.insert(key.clone(), id);
        id
    }

    /// Returns the handle for `key` if it exists.
    ///
    /// Does not insert the key if missing.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::KeyInterner;
    ///
    /// let mut interner = KeyInterner::new();
    /// let handle = interner.intern(&"existing");
    ///
    /// assert_eq!(interner.get_handle(&"existing"), Some(handle));
    /// assert_eq!(interner.get_handle(&"missing"), None);
    /// ```
    pub fn get_handle(&self, key: &K) -> Option<u64> {
        self.index.get(key).copied()
    }

    /// Resolves a handle to its original key.
    ///
    /// Returns `None` if the handle is out of bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::KeyInterner;
    ///
    /// let mut interner = KeyInterner::new();
    /// let handle = interner.intern(&"my_key");
    ///
    /// assert_eq!(interner.resolve(handle), Some(&"my_key"));
    /// assert_eq!(interner.resolve(999), None);  // Invalid handle
    /// ```
    pub fn resolve(&self, handle: u64) -> Option<&K> {
        self.keys.get(handle as usize)
    }

    /// Returns the number of interned keys.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::KeyInterner;
    ///
    /// let mut interner = KeyInterner::new();
    /// assert_eq!(interner.len(), 0);
    ///
    /// interner.intern(&"a");
    /// interner.intern(&"b");
    /// assert_eq!(interner.len(), 2);
    ///
    /// // Re-interning same key doesn't increase count
    /// interner.intern(&"a");
    /// assert_eq!(interner.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Returns `true` if no keys are interned.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::KeyInterner;
    ///
    /// let mut interner: KeyInterner<&str> = KeyInterner::new();
    /// assert!(interner.is_empty());
    ///
    /// interner.intern(&"key");
    /// assert!(!interner.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Clears all interned keys and shrinks internal storage.
    ///
    /// After calling this, all previously returned handles become invalid.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::KeyInterner;
    ///
    /// let mut interner = KeyInterner::new();
    /// let handle = interner.intern(&"key");
    /// assert_eq!(interner.resolve(handle), Some(&"key"));
    ///
    /// interner.clear_shrink();
    /// assert!(interner.is_empty());
    /// assert_eq!(interner.resolve(handle), None);  // Handle now invalid
    /// ```
    pub fn clear_shrink(&mut self) {
        self.index.clear();
        self.keys.clear();
        self.index.shrink_to_fit();
        self.keys.shrink_to_fit();
    }

    /// Returns an approximate memory footprint in bytes.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::KeyInterner;
    ///
    /// let mut interner: KeyInterner<String> = KeyInterner::new();
    /// let base_bytes = interner.approx_bytes();
    ///
    /// // Add some keys
    /// for i in 0..100 {
    ///     interner.intern(&format!("key_{}", i));
    /// }
    ///
    /// assert!(interner.approx_bytes() > base_bytes);
    /// ```
    pub fn approx_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.index.capacity() * std::mem::size_of::<(K, u64)>()
            + self.keys.capacity() * std::mem::size_of::<K>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_interner_basic_flow() {
        let mut interner = KeyInterner::new();
        assert!(interner.is_empty());
        let a = interner.intern(&"a".to_string());
        let b = interner.intern(&"b".to_string());
        let a2 = interner.intern(&"a".to_string());
        assert_eq!(a, a2);
        assert_ne!(a, b);
        assert_eq!(interner.len(), 2);
        assert_eq!(interner.get_handle(&"b".to_string()), Some(b));
        assert_eq!(interner.resolve(a), Some(&"a".to_string()));
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // =============================================================================
    // Property Tests - Handle Assignment
    // =============================================================================

    proptest! {
        /// Property: Handles start at 0 and increment sequentially
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_handles_sequential_from_zero(
            keys in prop::collection::vec(any::<u32>(), 1..50)
        ) {
            let mut interner: KeyInterner<u32> = KeyInterner::new();
            let mut unique_keys = Vec::new();

            for key in keys {
                if !unique_keys.contains(&key) {
                    unique_keys.push(key);
                    let handle = interner.intern(&key);
                    let expected_handle = (unique_keys.len() - 1) as u64;
                    prop_assert_eq!(handle, expected_handle);
                }
            }
        }

        /// Property: First key gets handle 0
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_first_key_gets_zero(key in any::<u32>()) {
            let mut interner: KeyInterner<u32> = KeyInterner::new();
            let handle = interner.intern(&key);
            prop_assert_eq!(handle, 0);
        }

        /// Property: Different keys get different handles
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_different_keys_different_handles(
            key1 in any::<u32>(),
            key2 in any::<u32>()
        ) {
            prop_assume!(key1 != key2);
            let mut interner: KeyInterner<u32> = KeyInterner::new();

            let h1 = interner.intern(&key1);
            let h2 = interner.intern(&key2);

            prop_assert_ne!(h1, h2);
        }
    }

    // =============================================================================
    // Property Tests - Idempotency
    // =============================================================================

    proptest! {
        /// Property: intern is idempotent - same key always returns same handle
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_intern_idempotent(
            key in any::<u32>(),
            repeat_count in 1usize..10
        ) {
            let mut interner: KeyInterner<u32> = KeyInterner::new();

            let first_handle = interner.intern(&key);

            for _ in 0..repeat_count {
                let handle = interner.intern(&key);
                prop_assert_eq!(handle, first_handle);
            }

            // Length should be 1 (only one unique key)
            prop_assert_eq!(interner.len(), 1);
        }

        /// Property: Re-interning doesn't increase length
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_reintern_no_length_increase(
            keys in prop::collection::vec(any::<u32>(), 1..30)
        ) {
            let mut interner: KeyInterner<u32> = KeyInterner::new();

            // Intern all keys once
            for &key in &keys {
                interner.intern(&key);
            }

            let len_after_first = interner.len();

            // Intern all keys again
            for &key in &keys {
                interner.intern(&key);
            }

            prop_assert_eq!(interner.len(), len_after_first);
        }
    }

    // =============================================================================
    // Property Tests - Bidirectional Mapping
    // =============================================================================

    proptest! {
        /// Property: intern -> resolve roundtrip returns same key
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_intern_resolve_roundtrip(
            keys in prop::collection::vec(any::<u32>(), 0..30)
        ) {
            let mut interner: KeyInterner<u32> = KeyInterner::new();

            for key in keys {
                let handle = interner.intern(&key);
                prop_assert_eq!(interner.resolve(handle), Some(&key));
            }
        }

        /// Property: get_handle -> resolve roundtrip is consistent
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_get_handle_resolve_consistent(
            keys in prop::collection::vec(any::<u32>(), 1..30)
        ) {
            let mut interner: KeyInterner<u32> = KeyInterner::new();

            // Intern keys
            for &key in &keys {
                interner.intern(&key);
            }

            // Verify consistency
            for &key in &keys {
                if let Some(handle) = interner.get_handle(&key) {
                    prop_assert_eq!(interner.resolve(handle), Some(&key));
                }
            }
        }

        /// Property: All handles from 0..len are valid
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_all_handles_valid_up_to_len(
            keys in prop::collection::vec(0u32..50, 1..30)
        ) {
            let mut interner: KeyInterner<u32> = KeyInterner::new();

            for key in keys {
                interner.intern(&key);
            }

            let len = interner.len() as u64;

            // All handles from 0 to len-1 should resolve to something
            for handle in 0..len {
                prop_assert!(interner.resolve(handle).is_some());
            }

            // Handles >= len should return None
            for handle in len..(len + 10) {
                prop_assert_eq!(interner.resolve(handle), None);
            }
        }
    }

    // =============================================================================
    // Property Tests - get_handle
    // =============================================================================

    proptest! {
        /// Property: get_handle returns None for keys not yet interned
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_get_handle_missing_returns_none(
            interned_keys in prop::collection::vec(0u32..20, 1..10),
            query_key in 20u32..40
        ) {
            let mut interner: KeyInterner<u32> = KeyInterner::new();

            for key in interned_keys {
                interner.intern(&key);
            }

            // Query key not in range should return None
            prop_assert_eq!(interner.get_handle(&query_key), None);
        }

        /// Property: get_handle doesn't modify state (read-only)
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_get_handle_read_only(
            keys in prop::collection::vec(any::<u32>(), 1..20),
            query_key in any::<u32>()
        ) {
            let mut interner: KeyInterner<u32> = KeyInterner::new();

            for key in keys {
                interner.intern(&key);
            }

            let len_before = interner.len();
            let _ = interner.get_handle(&query_key);
            let len_after = interner.len();

            prop_assert_eq!(len_before, len_after);
        }
    }

    // =============================================================================
    // Property Tests - Length and Empty State
    // =============================================================================

    proptest! {
        /// Property: len equals number of unique interned keys
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_len_equals_unique_keys(
            keys in prop::collection::vec(any::<u32>(), 0..50)
        ) {
            let mut interner: KeyInterner<u32> = KeyInterner::new();

            for key in &keys {
                interner.intern(key);
            }

            let unique_count = {
                let mut unique = std::collections::HashSet::new();
                for key in keys {
                    unique.insert(key);
                }
                unique.len()
            };

            prop_assert_eq!(interner.len(), unique_count);
        }

        /// Property: is_empty is consistent with len
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_is_empty_consistent_with_len(
            keys in prop::collection::vec(any::<u32>(), 0..30)
        ) {
            let mut interner: KeyInterner<u32> = KeyInterner::new();
            let mut unique_keys = std::collections::HashSet::new();

            for key in keys {
                interner.intern(&key);
                unique_keys.insert(key);

                // Check consistency: is_empty() matches whether we have any unique keys
                prop_assert_eq!(interner.is_empty(), unique_keys.is_empty());
                prop_assert_eq!(interner.len(), unique_keys.len());
            }
        }
    }

    // =============================================================================
    // Property Tests - Clear Operation
    // =============================================================================

    proptest! {
        /// Property: clear_shrink resets to empty state
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_clear_resets_state(
            keys in prop::collection::vec(any::<u32>(), 1..30)
        ) {
            let mut interner: KeyInterner<u32> = KeyInterner::new();

            for key in keys {
                interner.intern(&key);
            }

            interner.clear_shrink();

            prop_assert!(interner.is_empty());
            prop_assert_eq!(interner.len(), 0);
        }

        /// Property: clear invalidates all previous handles
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_clear_invalidates_handles(
            keys in prop::collection::vec(any::<u32>(), 1..20)
        ) {
            let mut interner: KeyInterner<u32> = KeyInterner::new();

            let mut handles = Vec::new();
            for key in &keys {
                let handle = interner.intern(key);
                handles.push(handle);
            }

            interner.clear_shrink();

            // All previous handles should now be invalid
            for handle in handles {
                prop_assert_eq!(interner.resolve(handle), None);
            }

            // All previous keys should not have handles
            for key in keys {
                prop_assert_eq!(interner.get_handle(&key), None);
            }
        }

        /// Property: usable after clear - handles restart from 0
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_usable_after_clear(
            keys1 in prop::collection::vec(any::<u32>(), 1..20),
            keys2 in prop::collection::vec(any::<u32>(), 1..20)
        ) {
            let mut interner: KeyInterner<u32> = KeyInterner::new();

            for key in keys1 {
                interner.intern(&key);
            }

            interner.clear_shrink();

            // After clear, handles should restart from 0
            if let Some(&first_key) = keys2.first() {
                let handle = interner.intern(&first_key);
                prop_assert_eq!(handle, 0);
            }
        }
    }

    // =============================================================================
    // Property Tests - Reference Implementation Equivalence
    // =============================================================================

    proptest! {
        /// Property: Behavior matches reference HashMap implementation
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_matches_reference_implementation(
            keys in prop::collection::vec(0u32..50, 0..50)
        ) {
            let mut interner: KeyInterner<u32> = KeyInterner::new();
            let mut reference: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
            let mut next_handle: u64 = 0;

            for key in keys {
                let handle = interner.intern(&key);

                // Update reference
                let ref_handle = *reference.entry(key).or_insert_with(|| {
                    let h = next_handle;
                    next_handle += 1;
                    h
                });

                // Verify handle matches reference
                prop_assert_eq!(handle, ref_handle);

                // Verify length matches
                prop_assert_eq!(interner.len(), reference.len());

                // Verify all keys in reference have correct handles
                for (&ref_key, &ref_handle) in &reference {
                    prop_assert_eq!(interner.get_handle(&ref_key), Some(ref_handle));
                    prop_assert_eq!(interner.resolve(ref_handle), Some(&ref_key));
                }
            }
        }
    }

    // =============================================================================
    // Property Tests - Memory and Capacity
    // =============================================================================

    proptest! {
        /// Property: approx_bytes increases as keys are added
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_approx_bytes_increases(
            keys in prop::collection::vec(any::<u32>(), 10..30)
        ) {
            let mut interner: KeyInterner<u32> = KeyInterner::new();
            let base_bytes = interner.approx_bytes();

            for key in keys {
                interner.intern(&key);
            }

            let after_bytes = interner.approx_bytes();
            prop_assert!(after_bytes >= base_bytes);
        }
    }
}
