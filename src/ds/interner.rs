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
