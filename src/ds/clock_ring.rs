//! Clock-sweep ring for second-chance eviction.
//!
//! Uses a fixed-size slot array and a hand pointer to evict the first
//! unreferenced entry encountered. Accesses set a referenced bit that
//! grants a second chance before eviction.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                           ClockRing<K, V>                                   │
//! │                                                                             │
//! │   ┌─────────────────────────────┐   ┌─────────────────────────────────┐   │
//! │   │  index: HashMap<K, usize>   │   │  slots: Vec<Option<Entry>>      │   │
//! │   │                             │   │                                 │   │
//! │   │  ┌───────────┬──────────┐  │   │  ┌─────┬─────┬─────┬─────┐     │   │
//! │   │  │    Key    │  Index   │  │   │  │  0  │  1  │  2  │  3  │     │   │
//! │   │  ├───────────┼──────────┤  │   │  ├─────┼─────┼─────┼─────┤     │   │
//! │   │  │  "key_a"  │    0     │──┼───┼──►│ A,1 │ B,0 │ C,1 │None │     │   │
//! │   │  │  "key_b"  │    1     │──┼───┼──►│     │     │     │     │     │   │
//! │   │  │  "key_c"  │    2     │──┼───┼──►│     │     │     │     │     │   │
//! │   │  └───────────┴──────────┘  │   │  └─────┴─────┴─────┴─────┘     │   │
//! │   └─────────────────────────────┘   │                ▲               │   │
//! │                                     │                │ hand = 1      │   │
//! │                                     └────────────────┼───────────────┘   │
//! │                                                      │                   │
//! │   Entry: { key, value, referenced: bool }            │                   │
//! │   "A,1" = Entry { key: "key_a", referenced: true }   │                   │
//! │   "B,0" = Entry { key: "key_b", referenced: false } ◄┘ (next victim)     │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────────┘
//!
//! Clock Sweep Algorithm
//! ─────────────────────
//!
//!   Insert "key_d" when full (hand at slot 1):
//!
//!     Step 1: Check slot[1] ("B", ref=0)
//!             ref=0 → EVICT, insert here
//!
//!     Result: slot[1] = Entry { key: "key_d", ref=false }
//!             hand advances to 2
//!             Return: Some(("key_b", value_b))
//!
//!   Insert "key_e" when full (hand at slot 2):
//!
//!     Step 1: Check slot[2] ("C", ref=1)
//!             ref=1 → clear ref, advance hand
//!     Step 2: Check slot[3] (None)
//!             Empty → insert here
//!
//!     Result: slot[3] = Entry { key: "key_e", ref=false }
//!             hand advances to 0
//!             Return: None (used empty slot)
//!
//! Second-Chance Behavior
//! ──────────────────────
//!
//!   ┌────────────────────────────────────────────────────────────────────────┐
//!   │  Access via get()/touch() → Sets referenced = true                    │
//!   │                                                                        │
//!   │  Eviction scan:                                                        │
//!   │    ref=1 → Clear to ref=0, skip (second chance granted)               │
//!   │    ref=0 → Evict this entry                                           │
//!   │                                                                        │
//!   │  Effect: Recently accessed entries survive longer                     │
//!   └────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! - [`ClockRing`]: Single-threaded CLOCK cache
//! - [`ConcurrentClockRing`]: Thread-safe wrapper with `RwLock`
//!
//! ## Operations
//!
//! | Operation     | Time        | Notes                                  |
//! |---------------|-------------|----------------------------------------|
//! | `insert`      | O(1) amort. | Bounded scan with reference clearing   |
//! | `get`         | O(1)        | Returns value, sets reference bit      |
//! | `peek`        | O(1)        | Returns value, does NOT set ref bit    |
//! | `touch`       | O(1)        | Sets reference bit only                |
//! | `remove`      | O(1)        | Clears slot + index entry              |
//! | `pop_victim`  | O(1) amort. | Evicts next unreferenced entry         |
//! | `peek_victim` | O(n) worst  | Finds next victim without modifying    |
//!
//! ## Use Cases
//!
//! - **Page replacement**: Classic use case for CLOCK algorithm
//! - **Buffer pool**: Database buffer management
//! - **Web cache**: Approximate LRU with lower overhead
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::ds::ClockRing;
//!
//! let mut cache = ClockRing::new(3);
//!
//! // Insert entries
//! cache.insert("page_1", "content_1");
//! cache.insert("page_2", "content_2");
//! cache.insert("page_3", "content_3");
//!
//! // Access sets the reference bit (second chance)
//! cache.get(&"page_1");  // page_1 now has ref=1
//!
//! // Insert when full - evicts unreferenced entry
//! let evicted = cache.insert("page_4", "content_4");
//! // page_2 or page_3 evicted (page_1 protected by ref bit)
//!
//! assert!(cache.contains(&"page_1"));  // Survived due to reference bit
//! assert!(cache.contains(&"page_4"));  // Newly inserted
//! ```
//!
//! ## Use Case: Page Buffer
//!
//! ```
//! use cachekit::ds::ClockRing;
//!
//! struct PageBuffer {
//!     cache: ClockRing<u64, Vec<u8>>,  // page_id -> page_data
//! }
//!
//! impl PageBuffer {
//!     fn new(capacity: usize) -> Self {
//!         Self { cache: ClockRing::new(capacity) }
//!     }
//!
//!     fn read_page(&mut self, page_id: u64) -> Option<&Vec<u8>> {
//!         // get() sets reference bit, giving this page a second chance
//!         self.cache.get(&page_id)
//!     }
//!
//!     fn load_page(&mut self, page_id: u64, data: Vec<u8>) {
//!         if let Some((evicted_id, _evicted_data)) = self.cache.insert(page_id, data) {
//!             // Could write back dirty page here
//!             println!("Evicted page {}", evicted_id);
//!         }
//!     }
//! }
//!
//! let mut buffer = PageBuffer::new(100);
//! buffer.load_page(1, vec![0; 4096]);
//! buffer.load_page(2, vec![0; 4096]);
//! assert!(buffer.read_page(1).is_some());
//! ```
//!
//! ## Thread Safety
//!
//! - [`ClockRing`]: Not thread-safe, use in single-threaded contexts
//! - [`ConcurrentClockRing`]: Thread-safe via `parking_lot::RwLock`
//!
//! ## Implementation Notes
//!
//! - Fixed-size slot array; no reallocation during operation
//! - Keys mapped to slot indices via HashMap
//! - Hand pointer advances after each insert/eviction
//! - `debug_validate_invariants()` available in debug/test builds

use rustc_hash::FxHashMap;
use std::hash::Hash;

#[cfg(feature = "concurrency")]
use parking_lot::RwLock;

/// Clock entry with cache-line optimized layout.
#[derive(Debug)]
#[repr(C)]
struct Entry<K, V> {
    // Hot field - checked/modified on every clock sweep
    referenced: bool,
    // Key needed for index removal during eviction
    key: K,
    // Value accessed on get
    value: V,
}

/// Fixed-size ring implementing the CLOCK (second-chance) eviction algorithm.
///
/// Provides O(1) amortized insertion with automatic eviction of unreferenced
/// entries. Accessed entries receive a "second chance" via a reference bit.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Eq + Hash + Clone`
/// - `V`: Value type
///
/// # Example
///
/// ```
/// use cachekit::ds::ClockRing;
///
/// let mut ring = ClockRing::new(2);
///
/// // Insert two entries
/// ring.insert("a", 1);
/// ring.insert("b", 2);
///
/// // Access "a" - sets reference bit
/// ring.get(&"a");
///
/// // Insert "c" - "b" is evicted (no reference bit)
/// let evicted = ring.insert("c", 3);
/// assert_eq!(evicted, Some(("b", 2)));
///
/// assert!(ring.contains(&"a"));
/// assert!(ring.contains(&"c"));
/// ```
///
/// # Use Case: Simple Cache with Eviction Callback
///
/// ```
/// use cachekit::ds::ClockRing;
///
/// let mut cache = ClockRing::new(3);
/// let mut eviction_count = 0;
///
/// for i in 0..10 {
///     if let Some((_key, _value)) = cache.insert(i, format!("value_{}", i)) {
///         eviction_count += 1;
///     }
/// }
///
/// assert_eq!(cache.len(), 3);
/// assert_eq!(eviction_count, 7);  // 10 inserts - 3 capacity = 7 evictions
/// ```
#[derive(Debug)]
pub struct ClockRing<K, V> {
    slots: Vec<Option<Entry<K, V>>>,
    index: FxHashMap<K, usize>,
    hand: usize,
    len: usize,
}

/// Thread-safe wrapper around [`ClockRing`] using `parking_lot::RwLock`.
///
/// Provides the same functionality as [`ClockRing`] but safe for concurrent
/// access. Uses closure-based value access since references cannot outlive
/// lock guards.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
/// use cachekit::ds::ConcurrentClockRing;
///
/// let cache = Arc::new(ConcurrentClockRing::new(100));
///
/// let handles: Vec<_> = (0..4).map(|t| {
///     let cache = Arc::clone(&cache);
///     thread::spawn(move || {
///         for i in 0..25 {
///             let key = t * 25 + i;
///             cache.insert(key, key * 10);
///         }
///     })
/// }).collect();
///
/// for h in handles {
///     h.join().unwrap();
/// }
///
/// assert!(cache.len() <= 100);
/// ```
///
/// # Non-blocking Operations
///
/// All operations have `try_*` variants that return `None` if the lock
/// cannot be acquired immediately:
///
/// ```
/// use cachekit::ds::ConcurrentClockRing;
///
/// let cache = ConcurrentClockRing::new(10);
/// cache.insert("key", 42);
///
/// // Non-blocking read
/// if let Some(Some(val)) = cache.try_get_with(&"key", |v| *v) {
///     assert_eq!(val, 42);
/// }
/// ```
#[cfg(feature = "concurrency")]
#[derive(Debug)]
pub struct ConcurrentClockRing<K, V> {
    inner: RwLock<ClockRing<K, V>>,
}

#[cfg(feature = "concurrency")]
impl<K, V> ConcurrentClockRing<K, V>
where
    K: Eq + Hash + Clone,
{
    /// Creates a new ring with `capacity` slots.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentClockRing;
    ///
    /// let cache: ConcurrentClockRing<String, i32> = ConcurrentClockRing::new(100);
    /// assert_eq!(cache.capacity(), 100);
    /// assert!(cache.is_empty());
    /// ```
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: RwLock::new(ClockRing::new(capacity)),
        }
    }

    /// Returns the configured capacity (number of slots).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentClockRing;
    ///
    /// let cache: ConcurrentClockRing<&str, i32> = ConcurrentClockRing::new(50);
    /// assert_eq!(cache.capacity(), 50);
    /// ```
    pub fn capacity(&self) -> usize {
        let ring = self.inner.read();
        ring.capacity()
    }

    /// Returns the number of occupied slots.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentClockRing;
    ///
    /// let cache = ConcurrentClockRing::new(10);
    /// assert_eq!(cache.len(), 0);
    ///
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    /// assert_eq!(cache.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        let ring = self.inner.read();
        ring.len()
    }

    /// Returns `true` if there are no entries.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentClockRing;
    ///
    /// let cache: ConcurrentClockRing<&str, i32> = ConcurrentClockRing::new(10);
    /// assert!(cache.is_empty());
    ///
    /// cache.insert("key", 42);
    /// assert!(!cache.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        let ring = self.inner.read();
        ring.is_empty()
    }

    /// Returns `true` if `key` is present.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentClockRing;
    ///
    /// let cache = ConcurrentClockRing::new(10);
    /// cache.insert("key", 42);
    ///
    /// assert!(cache.contains(&"key"));
    /// assert!(!cache.contains(&"missing"));
    /// ```
    pub fn contains(&self, key: &K) -> bool {
        let ring = self.inner.read();
        ring.contains(key)
    }

    /// Accesses value without setting the reference bit.
    ///
    /// Unlike [`get_with`](Self::get_with), this does not grant a second chance.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentClockRing;
    ///
    /// let cache = ConcurrentClockRing::new(10);
    /// cache.insert("key", vec![1, 2, 3]);
    ///
    /// let sum = cache.peek_with(&"key", |v| v.iter().sum::<i32>());
    /// assert_eq!(sum, Some(6));
    /// ```
    pub fn peek_with<R>(&self, key: &K, f: impl FnOnce(&V) -> R) -> Option<R> {
        let ring = self.inner.read();
        ring.peek(key).map(f)
    }

    /// Accesses value and sets the reference bit (grants second chance).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentClockRing;
    ///
    /// let cache = ConcurrentClockRing::new(2);
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    ///
    /// // Access "a" - sets reference bit
    /// cache.get_with(&"a", |v| *v);
    ///
    /// // Insert "c" - "b" evicted (no ref bit), "a" survives
    /// cache.insert("c", 3);
    /// assert!(cache.contains(&"a"));
    /// assert!(!cache.contains(&"b"));
    /// ```
    pub fn get_with<R>(&self, key: &K, f: impl FnOnce(&V) -> R) -> Option<R> {
        let mut ring = self.inner.write();
        ring.get(key).map(f)
    }

    /// Sets the reference bit for `key`; returns `false` if missing.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentClockRing;
    ///
    /// let cache = ConcurrentClockRing::new(10);
    /// cache.insert("key", 42);
    ///
    /// assert!(cache.touch(&"key"));       // Sets reference bit
    /// assert!(!cache.touch(&"missing"));  // Key not found
    /// ```
    pub fn touch(&self, key: &K) -> bool {
        let mut ring = self.inner.write();
        ring.touch(key)
    }

    /// Updates the value for an existing key, returning the old value.
    ///
    /// Sets the reference bit on update. Returns `None` if the key doesn't exist.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentClockRing;
    ///
    /// let cache = ConcurrentClockRing::new(2);
    /// cache.insert("a", 1);
    ///
    /// assert_eq!(cache.update(&"a", 10), Some(1));
    /// assert_eq!(cache.update(&"missing", 99), None);
    /// ```
    pub fn update(&self, key: &K, value: V) -> Option<V> {
        let mut ring = self.inner.write();
        ring.update(key, value)
    }

    /// Inserts or updates `key`, evicting if necessary.
    ///
    /// Returns `Some((evicted_key, evicted_value))` if an entry was evicted.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentClockRing;
    ///
    /// let cache = ConcurrentClockRing::new(2);
    /// assert_eq!(cache.insert("a", 1), None);  // No eviction
    /// assert_eq!(cache.insert("b", 2), None);  // No eviction
    ///
    /// // Full - must evict
    /// let evicted = cache.insert("c", 3);
    /// assert!(evicted.is_some());
    /// ```
    pub fn insert(&self, key: K, value: V) -> Option<(K, V)> {
        let mut ring = self.inner.write();
        ring.insert(key, value)
    }

    /// Removes `key` and returns its value, if present.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentClockRing;
    ///
    /// let cache = ConcurrentClockRing::new(10);
    /// cache.insert("key", 42);
    ///
    /// assert_eq!(cache.remove(&"key"), Some(42));
    /// assert_eq!(cache.remove(&"key"), None);  // Already removed
    /// ```
    pub fn remove(&self, key: &K) -> Option<V> {
        let mut ring = self.inner.write();
        ring.remove(key)
    }

    /// Peeks the next eviction candidate without modifying state.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentClockRing;
    ///
    /// let cache = ConcurrentClockRing::new(2);
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    /// cache.touch(&"a");  // "a" now has ref bit
    ///
    /// // "b" is next victim (no ref bit)
    /// let victim_key = cache.peek_victim_with(|k, _v| *k);
    /// assert_eq!(victim_key, Some("b"));
    /// ```
    pub fn peek_victim_with<R>(&self, f: impl FnOnce(&K, &V) -> R) -> Option<R> {
        let ring = self.inner.read();
        ring.peek_victim().map(|(key, value)| f(key, value))
    }

    /// Evicts the next candidate and returns it.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentClockRing;
    ///
    /// let cache = ConcurrentClockRing::new(3);
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    /// cache.insert("c", 3);
    ///
    /// let evicted = cache.pop_victim();
    /// assert!(evicted.is_some());
    /// assert_eq!(cache.len(), 2);
    /// ```
    pub fn pop_victim(&self) -> Option<(K, V)> {
        let mut ring = self.inner.write();
        ring.pop_victim()
    }

    /// Non-blocking version of [`update`](Self::update).
    pub fn try_update(&self, key: &K, value: V) -> Option<Option<V>> {
        let mut ring = self.inner.try_write()?;
        Some(ring.update(key, value))
    }

    /// Non-blocking version of [`insert`](Self::insert).
    pub fn try_insert(&self, key: K, value: V) -> Option<Option<(K, V)>> {
        let mut ring = self.inner.try_write()?;
        Some(ring.insert(key, value))
    }

    /// Non-blocking version of [`remove`](Self::remove).
    pub fn try_remove(&self, key: &K) -> Option<Option<V>> {
        let mut ring = self.inner.try_write()?;
        Some(ring.remove(key))
    }

    /// Non-blocking version of [`touch`](Self::touch).
    pub fn try_touch(&self, key: &K) -> Option<bool> {
        let mut ring = self.inner.try_write()?;
        Some(ring.touch(key))
    }

    /// Non-blocking version of [`peek_with`](Self::peek_with).
    pub fn try_peek_with<R>(&self, key: &K, f: impl FnOnce(&V) -> R) -> Option<Option<R>> {
        let ring = self.inner.try_read()?;
        Some(ring.peek(key).map(f))
    }

    /// Non-blocking version of [`get_with`](Self::get_with).
    pub fn try_get_with<R>(&self, key: &K, f: impl FnOnce(&V) -> R) -> Option<Option<R>> {
        let mut ring = self.inner.try_write()?;
        Some(ring.get(key).map(f))
    }

    /// Non-blocking version of [`peek_victim_with`](Self::peek_victim_with).
    pub fn try_peek_victim_with<R>(&self, f: impl FnOnce(&K, &V) -> R) -> Option<Option<R>> {
        let ring = self.inner.try_read()?;
        Some(ring.peek_victim().map(|(key, value)| f(key, value)))
    }

    /// Non-blocking version of [`pop_victim`](Self::pop_victim).
    pub fn try_pop_victim(&self) -> Option<Option<(K, V)>> {
        let mut ring = self.inner.try_write()?;
        Some(ring.pop_victim())
    }

    /// Non-blocking clear. Returns `true` if successful.
    pub fn try_clear(&self) -> bool {
        if let Some(mut ring) = self.inner.try_write() {
            ring.clear();
            true
        } else {
            false
        }
    }

    /// Non-blocking clear and shrink. Returns `true` if successful.
    pub fn try_clear_shrink(&self) -> bool {
        if let Some(mut ring) = self.inner.try_write() {
            ring.clear_shrink();
            true
        } else {
            false
        }
    }
}
impl<K, V> ClockRing<K, V>
where
    K: Eq + Hash + Clone,
{
    /// Creates a new ring with `capacity` slots.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ClockRing;
    ///
    /// let ring: ClockRing<String, i32> = ClockRing::new(100);
    /// assert_eq!(ring.capacity(), 100);
    /// assert!(ring.is_empty());
    /// ```
    pub fn new(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        slots.resize_with(capacity, || None);
        Self {
            slots,
            index: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            hand: 0,
            len: 0,
        }
    }

    /// Returns the configured capacity (number of slots).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ClockRing;
    ///
    /// let ring: ClockRing<&str, i32> = ClockRing::new(50);
    /// assert_eq!(ring.capacity(), 50);
    /// ```
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Reserves capacity for at least `additional` more index entries.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ClockRing;
    ///
    /// let mut ring: ClockRing<String, i32> = ClockRing::new(10);
    /// ring.reserve_index(100);  // Pre-allocate index capacity
    /// ```
    pub fn reserve_index(&mut self, additional: usize) {
        self.index.reserve(additional);
    }

    /// Shrinks internal storage to fit current contents.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ClockRing;
    ///
    /// let mut ring: ClockRing<&str, i32> = ClockRing::new(100);
    /// ring.insert("a", 1);
    /// ring.shrink_to_fit();
    /// ```
    pub fn shrink_to_fit(&mut self) {
        self.index.shrink_to_fit();
        self.slots.shrink_to_fit();
    }

    /// Clears all entries without releasing capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ClockRing;
    ///
    /// let mut ring = ClockRing::new(10);
    /// ring.insert("a", 1);
    /// ring.insert("b", 2);
    ///
    /// ring.clear();
    /// assert!(ring.is_empty());
    /// assert!(!ring.contains(&"a"));
    /// ```
    pub fn clear(&mut self) {
        self.index.clear();
        for slot in &mut self.slots {
            *slot = None;
        }
        self.len = 0;
        self.hand = 0;
    }

    /// Clears all entries and shrinks internal storage.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ClockRing;
    ///
    /// let mut ring = ClockRing::new(100);
    /// ring.insert("a", 1);
    ///
    /// ring.clear_shrink();
    /// assert!(ring.is_empty());
    /// ```
    pub fn clear_shrink(&mut self) {
        self.clear();
        self.index.shrink_to_fit();
        self.slots.shrink_to_fit();
    }

    /// Returns an approximate memory footprint in bytes.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ClockRing;
    ///
    /// let ring: ClockRing<u64, u64> = ClockRing::new(100);
    /// let bytes = ring.approx_bytes();
    /// assert!(bytes > 0);
    /// ```
    pub fn approx_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.index.capacity() * std::mem::size_of::<(K, usize)>()
            + self.slots.capacity() * std::mem::size_of::<Option<Entry<K, V>>>()
    }

    /// Returns the number of occupied slots.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ClockRing;
    ///
    /// let mut ring = ClockRing::new(10);
    /// assert_eq!(ring.len(), 0);
    ///
    /// ring.insert("a", 1);
    /// ring.insert("b", 2);
    /// assert_eq!(ring.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if there are no entries.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ClockRing;
    ///
    /// let mut ring: ClockRing<&str, i32> = ClockRing::new(10);
    /// assert!(ring.is_empty());
    ///
    /// ring.insert("key", 42);
    /// assert!(!ring.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` if `key` is present.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ClockRing;
    ///
    /// let mut ring = ClockRing::new(10);
    /// ring.insert("key", 42);
    ///
    /// assert!(ring.contains(&"key"));
    /// assert!(!ring.contains(&"missing"));
    /// ```
    pub fn contains(&self, key: &K) -> bool {
        self.index.contains_key(key)
    }

    /// Returns the value without setting the reference bit.
    ///
    /// Unlike [`get`](Self::get), this does not grant a second chance.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ClockRing;
    ///
    /// let mut ring = ClockRing::new(10);
    /// ring.insert("key", 42);
    ///
    /// // peek doesn't set reference bit
    /// assert_eq!(ring.peek(&"key"), Some(&42));
    /// assert_eq!(ring.peek(&"missing"), None);
    /// ```
    pub fn peek(&self, key: &K) -> Option<&V> {
        let idx = *self.index.get(key)?;
        self.slots.get(idx)?.as_ref().map(|entry| &entry.value)
    }

    /// Returns the value and sets the reference bit (grants second chance).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ClockRing;
    ///
    /// let mut ring = ClockRing::new(2);
    /// ring.insert("a", 1);
    /// ring.insert("b", 2);
    ///
    /// // Access "a" - sets reference bit
    /// assert_eq!(ring.get(&"a"), Some(&1));
    ///
    /// // Insert "c" - "b" evicted (no ref bit), "a" survives
    /// ring.insert("c", 3);
    /// assert!(ring.contains(&"a"));
    /// assert!(!ring.contains(&"b"));
    /// ```
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let idx = *self.index.get(key)?;
        let entry = self.slots.get_mut(idx)?.as_mut()?;
        entry.referenced = true;
        Some(&entry.value)
    }

    /// Sets the reference bit for `key`; returns `false` if missing.
    ///
    /// Use this to mark an entry as recently accessed without retrieving its value.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ClockRing;
    ///
    /// let mut ring = ClockRing::new(2);
    /// ring.insert("a", 1);
    /// ring.insert("b", 2);
    ///
    /// // Touch "a" without reading value
    /// assert!(ring.touch(&"a"));
    /// assert!(!ring.touch(&"missing"));
    ///
    /// // "a" survives eviction due to reference bit
    /// let evicted = ring.insert("c", 3);
    /// assert_eq!(evicted, Some(("b", 2)));
    /// ```
    pub fn touch(&mut self, key: &K) -> bool {
        let idx = match self.index.get(key) {
            Some(idx) => *idx,
            None => return false,
        };
        if let Some(entry) = self.slots.get_mut(idx).and_then(|slot| slot.as_mut()) {
            entry.referenced = true;
            return true;
        }
        false
    }

    /// Updates the value for an existing key, returning the old value.
    ///
    /// Sets the reference bit on update. Returns `None` if the key doesn't exist.
    /// This method never evicts - use [`insert`](Self::insert) for that.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ClockRing;
    ///
    /// let mut ring = ClockRing::new(2);
    /// ring.insert("a", 1);
    /// ring.insert("b", 2);
    ///
    /// // Update existing key
    /// assert_eq!(ring.update(&"a", 10), Some(1));
    /// assert_eq!(ring.peek(&"a"), Some(&10));
    ///
    /// // Key doesn't exist - returns None
    /// assert_eq!(ring.update(&"missing", 99), None);
    /// ```
    pub fn update(&mut self, key: &K, value: V) -> Option<V> {
        let idx = *self.index.get(key)?;
        let entry = self.slots.get_mut(idx)?.as_mut()?;
        let old = std::mem::replace(&mut entry.value, value);
        entry.referenced = true;
        Some(old)
    }

    /// Inserts or updates `key`, evicting if necessary.
    ///
    /// If inserting into a full ring, evicts and returns `(evicted_key, evicted_value)`.
    /// Updating an existing key sets its reference bit but does not evict.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ClockRing;
    ///
    /// let mut ring = ClockRing::new(2);
    ///
    /// // Insert into empty slots - no eviction
    /// assert_eq!(ring.insert("a", 1), None);
    /// assert_eq!(ring.insert("b", 2), None);
    ///
    /// // Update existing - no eviction
    /// assert_eq!(ring.insert("a", 10), None);
    /// assert_eq!(ring.peek(&"a"), Some(&10));
    ///
    /// // Insert new when full - evicts unreferenced entry
    /// let evicted = ring.insert("c", 3);
    /// assert!(evicted.is_some());
    /// ```
    pub fn insert(&mut self, key: K, value: V) -> Option<(K, V)> {
        if self.capacity() == 0 {
            return None;
        }

        if let Some(&idx) = self.index.get(&key) {
            if let Some(entry) = self.slots.get_mut(idx).and_then(|slot| slot.as_mut()) {
                entry.value = value;
                entry.referenced = true;
            }
            return None;
        }

        loop {
            let idx = self.hand;
            if let Some(entry) = self.slots.get_mut(idx).and_then(|slot| slot.as_mut()) {
                if entry.referenced {
                    entry.referenced = false;
                    self.advance_hand();
                    continue;
                }

                let evicted = self.slots[idx].take().expect("occupied slot missing");
                self.index.remove(&evicted.key);

                let entry_key = key.clone();
                self.slots[idx] = Some(Entry {
                    key: entry_key,
                    value,
                    referenced: false,
                });
                self.index.insert(key, idx);
                self.advance_hand();
                return Some((evicted.key, evicted.value));
            }

            let entry_key = key.clone();
            self.slots[idx] = Some(Entry {
                key: entry_key,
                value,
                referenced: false,
            });
            self.index.insert(key, idx);
            self.len += 1;
            self.advance_hand();
            return None;
        }
    }

    /// Peeks the next eviction candidate without modifying state.
    ///
    /// Scans from the hand position to find the first unreferenced entry.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ClockRing;
    ///
    /// let mut ring = ClockRing::new(2);
    /// ring.insert("a", 1);
    /// ring.insert("b", 2);
    /// ring.touch(&"a");  // "a" now has reference bit
    ///
    /// // "b" is the next victim (no reference bit)
    /// let victim = ring.peek_victim();
    /// assert_eq!(victim, Some((&"b", &2)));
    /// ```
    pub fn peek_victim(&self) -> Option<(&K, &V)> {
        if self.capacity() == 0 || self.len == 0 {
            return None;
        }
        let cap = self.capacity();
        for offset in 0..cap {
            let idx = (self.hand + offset) % cap;
            if let Some(entry) = self.slots.get(idx).and_then(|slot| slot.as_ref()) {
                if !entry.referenced {
                    return Some((&entry.key, &entry.value));
                }
            }
        }
        None
    }

    /// Evicts the next candidate (first unreferenced slot) and returns it.
    ///
    /// Clears reference bits as it scans, giving referenced entries a second chance.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ClockRing;
    ///
    /// let mut ring = ClockRing::new(3);
    /// ring.insert("a", 1);
    /// ring.insert("b", 2);
    /// ring.insert("c", 3);
    ///
    /// // Evict one entry
    /// let evicted = ring.pop_victim();
    /// assert!(evicted.is_some());
    /// assert_eq!(ring.len(), 2);
    /// ```
    pub fn pop_victim(&mut self) -> Option<(K, V)> {
        if self.capacity() == 0 || self.len == 0 {
            return None;
        }
        let cap = self.capacity();
        for _ in 0..cap {
            let idx = self.hand;
            if let Some(entry) = self.slots.get_mut(idx).and_then(|slot| slot.as_mut()) {
                if entry.referenced {
                    entry.referenced = false;
                    self.advance_hand();
                    continue;
                }

                let evicted = self.slots[idx].take().expect("occupied slot missing");
                self.index.remove(&evicted.key);
                self.len -= 1;
                self.advance_hand();
                return Some((evicted.key, evicted.value));
            }
            self.advance_hand();
        }
        None
    }

    #[cfg(any(test, debug_assertions))]
    /// Returns a debug snapshot of slot occupancy in ring order.
    pub fn debug_snapshot_slots(&self) -> Vec<Option<(&K, bool)>> {
        self.slots
            .iter()
            .map(|slot| slot.as_ref().map(|entry| (&entry.key, entry.referenced)))
            .collect()
    }

    /// Removes `key` and returns its value, if present.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ClockRing;
    ///
    /// let mut ring = ClockRing::new(10);
    /// ring.insert("key", 42);
    ///
    /// assert_eq!(ring.remove(&"key"), Some(42));
    /// assert_eq!(ring.remove(&"key"), None);  // Already removed
    /// assert!(!ring.contains(&"key"));
    /// ```
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let idx = self.index.remove(key)?;
        let entry = self.slots.get_mut(idx)?.take()?;
        self.len -= 1;
        Some(entry.value)
    }

    fn advance_hand(&mut self) {
        let cap = self.capacity();
        if cap == 0 {
            self.hand = 0;
        } else {
            self.hand = (self.hand + 1) % cap;
        }
    }

    #[cfg(any(test, debug_assertions))]
    pub fn debug_validate_invariants(&self) {
        let slot_count = self.slots.iter().filter(|slot| slot.is_some()).count();
        assert_eq!(self.len, slot_count);
        assert_eq!(self.len, self.index.len());

        if self.capacity() == 0 {
            assert_eq!(self.hand, 0);
        } else {
            assert!(self.hand < self.capacity());
        }

        for (key, &idx) in &self.index {
            assert!(idx < self.slots.len());
            let entry = self.slots[idx]
                .as_ref()
                .expect("index points to empty slot");
            assert!(&entry.key == key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_ring_eviction_prefers_unreferenced() {
        let mut ring = ClockRing::new(2);
        ring.insert("a", 1);
        ring.insert("b", 2);
        ring.touch(&"a");
        let evicted = ring.insert("c", 3);

        assert_eq!(evicted, Some(("b", 2)));
        assert!(ring.contains(&"a"));
        assert!(ring.contains(&"c"));
    }

    #[test]
    fn clock_ring_zero_capacity_is_noop() {
        let mut ring = ClockRing::<&str, i32>::new(0);
        assert!(ring.is_empty());
        assert_eq!(ring.capacity(), 0);
        assert_eq!(ring.insert("a", 1), None);
        assert!(ring.is_empty());
        assert!(ring.peek(&"a").is_none());
        assert!(ring.get(&"a").is_none());
        assert!(!ring.contains(&"a"));
    }

    #[test]
    fn clock_ring_insert_and_peek_no_eviction() {
        let mut ring = ClockRing::new(3);
        assert_eq!(ring.insert("a", 1), None);
        assert_eq!(ring.insert("b", 2), None);
        assert_eq!(ring.insert("c", 3), None);
        assert_eq!(ring.len(), 3);
        assert!(ring.contains(&"a"));
        assert!(ring.contains(&"b"));
        assert!(ring.contains(&"c"));
        assert_eq!(ring.peek(&"a"), Some(&1));
        assert_eq!(ring.peek(&"b"), Some(&2));
        assert_eq!(ring.peek(&"c"), Some(&3));
    }

    #[test]
    fn clock_ring_update_existing_key_does_not_grow() {
        let mut ring = ClockRing::new(2);
        assert_eq!(ring.insert("a", 1), None);
        assert_eq!(ring.insert("b", 2), None);
        assert_eq!(ring.len(), 2);

        assert_eq!(ring.insert("a", 10), None);
        assert_eq!(ring.len(), 2);
        assert_eq!(ring.peek(&"a"), Some(&10));
    }

    #[test]
    fn clock_ring_update_returns_old_value() {
        let mut ring = ClockRing::new(3);
        ring.insert("a", 1);
        ring.insert("b", 2);

        // Update existing key returns old value
        assert_eq!(ring.update(&"a", 10), Some(1));
        assert_eq!(ring.peek(&"a"), Some(&10));
        assert_eq!(ring.len(), 2);

        // Update non-existent key returns None
        assert_eq!(ring.update(&"missing", 99), None);
        assert_eq!(ring.len(), 2);

        // Update sets reference bit - verify by eviction
        let mut ring2 = ClockRing::new(2);
        ring2.insert("x", 1);
        ring2.insert("y", 2);
        ring2.update(&"x", 10); // Sets ref bit on x
        let evicted = ring2.insert("z", 3);
        assert_eq!(evicted, Some(("y", 2))); // y evicted, not x
    }

    #[test]
    fn clock_ring_get_sets_referenced_and_eviction_skips_it() {
        let mut ring = ClockRing::new(2);
        ring.insert("a", 1);
        ring.insert("b", 2);
        assert_eq!(ring.get(&"a"), Some(&1));
        let evicted = ring.insert("c", 3);

        assert_eq!(evicted, Some(("b", 2)));
        assert!(ring.contains(&"a"));
        assert!(ring.contains(&"c"));
        assert!(!ring.contains(&"b"));
    }

    #[test]
    fn clock_ring_touch_marks_referenced() {
        let mut ring = ClockRing::new(2);
        ring.insert("a", 1);
        ring.insert("b", 2);
        assert!(ring.touch(&"b"));

        let evicted = ring.insert("c", 3);
        assert_eq!(evicted, Some(("a", 1)));
        assert!(ring.contains(&"b"));
        assert!(ring.contains(&"c"));
    }

    #[test]
    fn clock_ring_remove_clears_slot_and_updates_len() {
        let mut ring = ClockRing::new(3);
        ring.insert("a", 1);
        ring.insert("b", 2);
        ring.insert("c", 3);
        assert_eq!(ring.len(), 3);

        assert_eq!(ring.remove(&"b"), Some(2));
        assert_eq!(ring.len(), 2);
        assert!(!ring.contains(&"b"));
        assert!(ring.peek(&"b").is_none());

        let evicted = ring.insert("d", 4);
        assert!(ring.contains(&"d"));
        assert!(!ring.contains(&"b"));
        if evicted.is_some() {
            assert_eq!(ring.len(), 2);
        } else {
            assert_eq!(ring.len(), 3);
        }
    }

    #[test]
    fn clock_ring_eviction_cycles_with_hand_wrap() {
        let mut ring = ClockRing::new(2);
        ring.insert("a", 1);
        ring.insert("b", 2);

        let evicted1 = ring.insert("c", 3);
        assert!(matches!(evicted1, Some(("a", 1)) | Some(("b", 2))));
        assert_eq!(ring.len(), 2);

        let evicted2 = ring.insert("d", 4);
        assert!(matches!(
            evicted2,
            Some(("a", 1)) | Some(("b", 2)) | Some(("c", 3))
        ));
        assert_eq!(ring.len(), 2);
        assert!(ring.contains(&"d"));
    }

    #[test]
    fn clock_ring_debug_invariants_hold() {
        let mut ring = ClockRing::new(3);
        ring.insert("a", 1);
        ring.insert("b", 2);
        ring.get(&"a");
        ring.insert("c", 3);
        ring.remove(&"b");
        ring.debug_validate_invariants();
    }

    #[test]
    fn clock_ring_peek_and_pop_victim() {
        let mut ring = ClockRing::new(3);
        ring.insert("a", 1);
        ring.insert("b", 2);
        ring.insert("c", 3);

        // All entries are inserted unreferenced; any is a valid victim.
        let peeked = ring.peek_victim();
        assert!(matches!(
            peeked,
            Some((&"a", &1)) | Some((&"b", &2)) | Some((&"c", &3))
        ));

        let evicted = ring.pop_victim();
        assert!(matches!(
            evicted,
            Some(("a", 1)) | Some(("b", 2)) | Some(("c", 3))
        ));
        assert_eq!(ring.len(), 2);
        ring.debug_validate_invariants();
    }

    #[test]
    fn clock_ring_peek_skips_referenced_entries() {
        let mut ring = ClockRing::new(2);
        ring.insert("a", 1);
        ring.insert("b", 2);
        ring.touch(&"a");

        let peeked = ring.peek_victim();
        assert_eq!(peeked, Some((&"b", &2)));
    }

    #[test]
    fn clock_ring_pop_victim_clears_referenced_then_eviction() {
        let mut ring = ClockRing::new(2);
        ring.insert("a", 1);
        ring.insert("b", 2);
        ring.touch(&"a");
        ring.touch(&"b");

        let first = ring.pop_victim();
        if first.is_none() {
            let second = ring.pop_victim();
            assert!(matches!(second, Some(("a", 1)) | Some(("b", 2))));
            assert_eq!(ring.len(), 1);
        } else {
            assert!(matches!(first, Some(("a", 1)) | Some(("b", 2))));
            assert_eq!(ring.len(), 1);
        }
    }

    #[test]
    fn clock_ring_debug_snapshot_slots() {
        let mut ring = ClockRing::new(2);
        ring.insert("a", 1);
        ring.insert("b", 2);
        ring.touch(&"a");
        let snapshot = ring.debug_snapshot_slots();
        assert_eq!(snapshot.len(), 2);
        assert!(
            snapshot
                .iter()
                .any(|slot| matches!(slot, &Some((&"a", true))))
        );
    }

    #[cfg(feature = "concurrency")]
    #[test]
    fn concurrent_clock_ring_try_ops() {
        let ring = ConcurrentClockRing::new(2);
        assert_eq!(ring.try_insert("a", 1), Some(None));
        assert_eq!(ring.try_peek_with(&"a", |v| *v), Some(Some(1)));
        assert_eq!(ring.try_get_with(&"a", |v| *v), Some(Some(1)));
        assert_eq!(ring.try_touch(&"a"), Some(true));
        assert!(ring.try_clear());
        assert!(ring.is_empty());
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // =============================================================================
    // Property Tests - Core Invariants
    // =============================================================================

    proptest! {
        /// Property: len() never exceeds capacity
        #[test]
        fn prop_len_within_capacity(
            capacity in 1usize..100,
            ops in prop::collection::vec((0u32..1000, 0u32..100), 0..200)
        ) {
            let mut ring = ClockRing::new(capacity);

            for (key, value) in ops {
                ring.insert(key, value);
                prop_assert!(ring.len() <= ring.capacity());
            }
        }

        /// Property: Index and slot consistency - every indexed key has matching slot
        #[test]
        fn prop_index_slot_consistency(
            capacity in 1usize..50,
            ops in prop::collection::vec((0u32..100, 0u32..100), 0..100)
        ) {
            let mut ring = ClockRing::new(capacity);

            for (key, value) in ops {
                ring.insert(key, value);
                ring.debug_validate_invariants();
            }
        }

        /// Property: Get after insert returns the correct value
        #[test]
        fn prop_get_after_insert(
            capacity in 1usize..50,
            key in 0u32..100,
            value in 0u32..1000
        ) {
            let mut ring = ClockRing::new(capacity);
            ring.insert(key, value);

            if ring.contains(&key) {
                prop_assert_eq!(ring.peek(&key), Some(&value));
            }
        }

        /// Property: Insert on full ring returns eviction or None
        #[test]
        fn prop_insert_eviction_behavior(
            capacity in 1usize..20,
            keys in prop::collection::vec(0u32..50, 0..100)
        ) {
            let mut ring = ClockRing::new(capacity);

            for key in keys {
                let len_before = ring.len();
                let evicted = ring.insert(key, key * 10);

                if len_before < capacity {
                    // Not full - no eviction expected
                    if evicted.is_some() {
                        // Unless we're updating an existing key
                        prop_assert!(len_before == ring.len());
                    }
                } else {
                    // Full - eviction expected unless updating existing key
                    prop_assert!(ring.len() <= capacity);
                }

                ring.debug_validate_invariants();
            }
        }

        /// Property: Remove decreases length and makes key absent
        #[test]
        fn prop_remove_behavior(
            capacity in 1usize..50,
            keys in prop::collection::vec(0u32..100, 1..50)
        ) {
            let mut ring = ClockRing::new(capacity);

            // Insert all keys
            for &key in &keys {
                ring.insert(key, key * 10);
            }

            // Remove half
            for &key in &keys[0..keys.len()/2] {
                let len_before = ring.len();
                let removed = ring.remove(&key);

                if removed.is_some() {
                    prop_assert_eq!(ring.len(), len_before - 1);
                    prop_assert!(!ring.contains(&key));
                }

                ring.debug_validate_invariants();
            }
        }

        /// Property: Update doesn't change len, only value
        #[test]
        fn prop_update_preserves_len(
            capacity in 1usize..50,
            ops in prop::collection::vec((0u32..50, 0u32..100), 1..50)
        ) {
            let mut ring = ClockRing::new(capacity);

            for (key, value) in &ops {
                ring.insert(*key, *value);
            }

            let len_before = ring.len();

            // Update existing keys
            for (key, new_value) in ops {
                if ring.contains(&key) {
                    ring.update(&key, new_value + 1000);
                    prop_assert_eq!(ring.len(), len_before);
                }
            }

            ring.debug_validate_invariants();
        }

        /// Property: Referenced entries survive longer than unreferenced
        #[test]
        fn prop_second_chance_behavior(
            capacity in 2usize..10
        ) {
            let mut ring = ClockRing::new(capacity);

            // Fill ring
            for i in 0..capacity {
                ring.insert(i as u32, i as u32);
            }

            // Mark first entry as referenced
            ring.touch(&0);

            // Insert new entry - should not evict entry 0
            ring.insert(capacity as u32, capacity as u32);

            // Entry 0 should still be present due to reference bit
            prop_assert!(ring.contains(&0) || ring.len() < capacity);
        }

        /// Property: peek doesn't modify state (can be called multiple times)
        #[test]
        fn prop_peek_idempotent(
            capacity in 1usize..50,
            keys in prop::collection::vec(0u32..100, 1..50)
        ) {
            let mut ring = ClockRing::new(capacity);

            for key in keys {
                ring.insert(key, key * 10);
            }

            // Take snapshot
            let snapshot_before = ring.debug_snapshot_slots();

            // Peek all entries multiple times
            for i in 0..100 {
                ring.peek(&i);
            }

            let snapshot_after = ring.debug_snapshot_slots();

            // State should be unchanged
            prop_assert_eq!(snapshot_before, snapshot_after);
        }

        /// Property: Hand position stays within bounds
        #[test]
        fn prop_hand_within_bounds(
            capacity in 1usize..50,
            ops in prop::collection::vec((0u32..100, 0u32..100), 0..200)
        ) {
            let mut ring = ClockRing::new(capacity);

            for (key, value) in ops {
                ring.insert(key, value);
                ring.debug_validate_invariants();
            }
        }

        /// Property: pop_victim decreases length
        #[test]
        fn prop_pop_victim_decreases_len(
            capacity in 1usize..50,
            keys in prop::collection::vec(0u32..100, 1..50)
        ) {
            let mut ring = ClockRing::new(capacity);

            for key in keys {
                ring.insert(key, key * 10);
            }

            while !ring.is_empty() {
                let len_before = ring.len();
                let evicted = ring.pop_victim();

                if evicted.is_some() {
                    prop_assert_eq!(ring.len(), len_before - 1);
                    let (evicted_key, _) = evicted.unwrap();
                    prop_assert!(!ring.contains(&evicted_key));
                }

                ring.debug_validate_invariants();
            }
        }

        /// Property: Clear empties the ring
        #[test]
        fn prop_clear_empties(
            capacity in 1usize..50,
            keys in prop::collection::vec(0u32..100, 1..50)
        ) {
            let mut ring = ClockRing::new(capacity);

            for key in keys {
                ring.insert(key, key * 10);
            }

            ring.clear();

            prop_assert!(ring.is_empty());
            prop_assert_eq!(ring.len(), 0);
            ring.debug_validate_invariants();
        }
    }

    // =============================================================================
    // Property Tests - Sequential Operation Consistency
    // =============================================================================

    #[derive(Debug, Clone)]
    enum Operation {
        Insert(u32, u32),
        Get(u32),
        Peek(u32),
        Touch(u32),
        Update(u32, u32),
        Remove(u32),
        PopVictim,
    }

    fn operation_strategy() -> impl Strategy<Value = Operation> {
        prop_oneof![
            (0u32..50, 0u32..100).prop_map(|(k, v)| Operation::Insert(k, v)),
            (0u32..50).prop_map(Operation::Get),
            (0u32..50).prop_map(Operation::Peek),
            (0u32..50).prop_map(Operation::Touch),
            (0u32..50, 0u32..100).prop_map(|(k, v)| Operation::Update(k, v)),
            (0u32..50).prop_map(Operation::Remove),
            Just(Operation::PopVictim),
        ]
    }

    proptest! {
        /// Property: Arbitrary operation sequences maintain all invariants
        #[test]
        fn prop_arbitrary_ops_maintain_invariants(
            capacity in 1usize..30,
            ops in prop::collection::vec(operation_strategy(), 0..200)
        ) {
            let mut ring = ClockRing::new(capacity);

            for op in ops {
                match op {
                    Operation::Insert(k, v) => {
                        ring.insert(k, v);
                    }
                    Operation::Get(k) => {
                        ring.get(&k);
                    }
                    Operation::Peek(k) => {
                        ring.peek(&k);
                    }
                    Operation::Touch(k) => {
                        ring.touch(&k);
                    }
                    Operation::Update(k, v) => {
                        ring.update(&k, v);
                    }
                    Operation::Remove(k) => {
                        ring.remove(&k);
                    }
                    Operation::PopVictim => {
                        ring.pop_victim();
                    }
                }

                // All invariants must hold after every operation
                ring.debug_validate_invariants();
                prop_assert!(ring.len() <= ring.capacity());
            }
        }

        /// Property: Interleaved inserts/removes maintain consistency
        #[test]
        fn prop_interleaved_insert_remove(
            capacity in 1usize..30,
            ops in prop::collection::vec((0u32..50, any::<bool>()), 0..200)
        ) {
            let mut ring = ClockRing::new(capacity);

            for (key, should_insert) in ops {
                if should_insert {
                    ring.insert(key, key * 10);
                } else {
                    ring.remove(&key);
                }

                ring.debug_validate_invariants();
                prop_assert!(ring.len() <= ring.capacity());
            }
        }
    }

    // =============================================================================
    // Property Tests - Edge Cases
    // =============================================================================

    proptest! {
        /// Property: Zero capacity ring stays empty
        #[test]
        fn prop_zero_capacity_noop(
            ops in prop::collection::vec((0u32..100, 0u32..100), 0..50)
        ) {
            let mut ring = ClockRing::<u32, u32>::new(0);

            for (key, value) in ops {
                ring.insert(key, value);
                prop_assert!(ring.is_empty());
                prop_assert_eq!(ring.len(), 0);
                prop_assert!(!ring.contains(&key));
            }
        }

        /// Property: Capacity 1 ring never exceeds 1 entry
        #[test]
        fn prop_capacity_one_single_entry(
            keys in prop::collection::vec(0u32..100, 1..50)
        ) {
            let mut ring = ClockRing::new(1);

            for key in keys {
                ring.insert(key, key * 10);
                prop_assert!(ring.len() <= 1);
                ring.debug_validate_invariants();
            }
        }

        /// Property: Duplicate inserts don't grow the ring
        #[test]
        fn prop_duplicate_inserts_no_growth(
            capacity in 1usize..30,
            key in 0u32..50,
            values in prop::collection::vec(0u32..100, 1..50)
        ) {
            let mut ring = ClockRing::new(capacity);

            ring.insert(key, 0);
            let len_after_first = ring.len();

            for value in values {
                ring.insert(key, value);
                prop_assert_eq!(ring.len(), len_after_first);
            }
        }
    }

    // =============================================================================
    // Property Tests - Concurrent Wrapper (if enabled)
    // =============================================================================

    #[cfg(feature = "concurrency")]
    proptest! {
        /// Property: Concurrent wrapper maintains same invariants
        #[test]
        fn prop_concurrent_maintains_invariants(
            capacity in 1usize..30,
            ops in prop::collection::vec((0u32..50, 0u32..100), 0..100)
        ) {
            let ring = ConcurrentClockRing::new(capacity);

            for (key, value) in ops {
                ring.insert(key, value);
                prop_assert!(ring.len() <= ring.capacity());
            }
        }
    }
}

#[cfg(test)]
mod fuzz_tests {
    use super::*;

    /// Fuzz target: arbitrary operation sequences
    ///
    /// This can be used with cargo-fuzz or as a regular test with
    /// generated inputs.
    pub fn fuzz_arbitrary_operations(data: &[u8]) {
        if data.len() < 2 {
            return;
        }

        let capacity = (data[0] as usize % 50).max(1);
        let mut ring = ClockRing::new(capacity);

        let mut idx = 1;
        while idx < data.len() {
            if idx + 2 >= data.len() {
                break;
            }

            let op = data[idx] % 7;
            let key = data[idx + 1] as u32;
            let value = data[idx + 2] as u32;

            match op {
                0 => {
                    ring.insert(key, value);
                },
                1 => {
                    ring.get(&key);
                },
                2 => {
                    ring.peek(&key);
                },
                3 => {
                    ring.touch(&key);
                },
                4 => {
                    ring.update(&key, value);
                },
                5 => {
                    ring.remove(&key);
                },
                6 => {
                    ring.pop_victim();
                },
                _ => unreachable!(),
            }

            // Validate invariants after each operation
            ring.debug_validate_invariants();
            assert!(ring.len() <= ring.capacity());

            idx += 3;
        }
    }

    /// Fuzz target: stress test with many inserts
    pub fn fuzz_insert_stress(data: &[u8]) {
        if data.len() < 4 {
            return;
        }

        let capacity = (data[0] as usize % 100).max(1);
        let mut ring = ClockRing::new(capacity);

        for chunk in data[1..].chunks(2) {
            if chunk.len() < 2 {
                break;
            }
            let key = chunk[0] as u32;
            let value = chunk[1] as u32;
            ring.insert(key, value);

            assert!(ring.len() <= ring.capacity());
        }

        ring.debug_validate_invariants();
    }

    /// Fuzz target: eviction patterns with reference bits
    pub fn fuzz_eviction_patterns(data: &[u8]) {
        if data.len() < 3 {
            return;
        }

        let capacity = (data[0] as usize % 20).max(2);
        let mut ring = ClockRing::new(capacity);

        // Fill the ring
        for i in 0..capacity {
            ring.insert(i as u32, i as u32);
        }

        let mut idx = 1;
        while idx < data.len() {
            if idx + 1 >= data.len() {
                break;
            }

            let key = data[idx] as u32 % capacity as u32;
            let should_touch = data[idx + 1] % 2 == 0;

            if should_touch {
                ring.touch(&key);
            }

            // Insert new entry to trigger eviction
            let new_key = capacity as u32 + (idx as u32);
            ring.insert(new_key, new_key);

            ring.debug_validate_invariants();
            assert!(ring.len() <= ring.capacity());

            idx += 2;
        }
    }

    #[test]
    fn test_fuzz_arbitrary_operations_smoke() {
        // Smoke test with some sample inputs
        let inputs = vec![
            vec![5, 0, 1, 2, 1, 3, 4, 2, 5, 6],
            vec![10, 6, 7, 8, 5, 9, 10, 0, 1, 2],
            vec![1, 0, 0, 0, 1, 1, 1, 2, 2, 2],
        ];

        for input in inputs {
            fuzz_arbitrary_operations(&input);
        }
    }

    #[test]
    fn test_fuzz_insert_stress_smoke() {
        let inputs = vec![
            vec![5, 1, 2, 3, 4, 5, 6, 7, 8],
            vec![1, 0, 0, 0, 0, 0, 0],
            vec![20; 100],
        ];

        for input in inputs {
            fuzz_insert_stress(&input);
        }
    }

    #[test]
    fn test_fuzz_eviction_patterns_smoke() {
        let inputs = vec![
            vec![5, 0, 1, 1, 0, 2, 1, 3, 0],
            vec![3, 0, 0, 1, 1, 2, 0, 0, 1],
            vec![10, 1, 0, 2, 1, 3, 0, 4, 1],
        ];

        for input in inputs {
            fuzz_eviction_patterns(&input);
        }
    }
}
