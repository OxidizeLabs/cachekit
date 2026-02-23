//! Fast LRU cache optimized for single-threaded performance.
//!
//! This implementation prioritizes raw speed over flexibility by:
//! - Storing values directly (no `Arc` wrapping)
//! - Using FxHashMap for fast hashing (same hasher used in rustc)
//! - Cache-line optimized node layout
//! - Aggressive inlining in hot paths
//!
//! ## When to Use
//!
//! Use `FastLru` when:
//! - Maximum single-threaded performance is critical
//! - Values don't need to be shared after eviction
//! - You don't need the policy/storage separation
//!
//! Use `LruCore` when:
//! - Values need to outlive eviction (via `Arc`)
//! - You need concurrent access (wrap in lock)
//! - You need pluggable storage backends
//!
//! ## Performance
//!
//! Compared to `LruCore`, `FastLru` is ~7-10x faster for get/insert operations
//! due to FxHash, reduced indirection, and no atomic reference counting.

use rustc_hash::FxHashMap;
use std::hash::Hash;
use std::mem;
use std::ptr::NonNull;

#[cfg(feature = "metrics")]
use crate::metrics::metrics_impl::LruMetrics;
#[cfg(feature = "metrics")]
use crate::metrics::snapshot::LruMetricsSnapshot;
#[cfg(feature = "metrics")]
use crate::metrics::traits::{
    CoreMetricsRecorder, LruMetricsReadRecorder, LruMetricsRecorder, MetricsSnapshotProvider,
};

/// A fast, single-threaded LRU cache.
///
/// Values are stored directly without `Arc` wrapping for maximum performance.
/// All operations are O(1) average case.
///
/// # Example
///
/// ```
/// use cachekit::policy::fast_lru::FastLru;
///
/// let mut cache = FastLru::new(3);
///
/// cache.insert(1, "one");
/// cache.insert(2, "two");
/// cache.insert(3, "three");
///
/// assert_eq!(cache.get(&1), Some(&"one"));
///
/// // Inserting a 4th item evicts the LRU (key 2, since 1 was just accessed)
/// cache.insert(4, "four");
/// assert_eq!(cache.get(&2), None);
/// ```
pub struct FastLru<K, V> {
    map: FxHashMap<K, NonNull<Node<K, V>>>,
    head: Option<NonNull<Node<K, V>>>,
    tail: Option<NonNull<Node<K, V>>>,
    capacity: usize,
    #[cfg(feature = "metrics")]
    metrics: LruMetrics,
}

/// Node in the LRU linked list.
///
/// Layout is optimized for cache locality:
/// - Linked list pointers (prev/next) are at the start for fast traversal
/// - Key and value follow, keeping the hot metadata together
/// - `#[repr(C)]` ensures predictable field ordering
#[repr(C)]
struct Node<K, V> {
    // Hot fields first - accessed during every list operation
    prev: Option<NonNull<Node<K, V>>>,
    next: Option<NonNull<Node<K, V>>>,
    // Key needed for map removal during eviction
    key: K,
    // Value accessed on get/peek
    value: V,
}

impl<K, V> FastLru<K, V>
where
    K: Eq + Hash + Clone,
{
    /// Creates a new cache with the specified capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::fast_lru::FastLru;
    ///
    /// let cache: FastLru<u64, String> = FastLru::new(100);
    /// assert_eq!(cache.capacity(), 100);
    /// assert!(cache.is_empty());
    /// ```
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self {
            map: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            head: None,
            tail: None,
            capacity,
            #[cfg(feature = "metrics")]
            metrics: LruMetrics::default(),
        }
    }

    /// Returns the number of entries in the cache.
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns the maximum capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns `true` if the key exists in the cache.
    ///
    /// Does not update LRU order.
    #[inline]
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Gets a reference to a value, updating LRU order.
    ///
    /// Returns `None` if the key doesn't exist.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::fast_lru::FastLru;
    ///
    /// let mut cache = FastLru::new(10);
    /// cache.insert(1, "value");
    ///
    /// assert_eq!(cache.get(&1), Some(&"value"));
    /// assert_eq!(cache.get(&2), None);
    /// ```
    #[inline(always)]
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let node_ptr = match self.map.get(key) {
            Some(&ptr) => ptr,
            None => {
                #[cfg(feature = "metrics")]
                self.metrics.record_get_miss();
                return None;
            },
        };

        #[cfg(feature = "metrics")]
        self.metrics.record_get_hit();

        // Move to front (MRU position)
        self.detach(node_ptr);
        self.attach_front(node_ptr);

        // SAFETY: node_ptr is valid as long as it's in the map
        Some(unsafe { &(*node_ptr.as_ptr()).value })
    }

    /// Gets a mutable reference to a value, updating LRU order.
    #[inline(always)]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let node_ptr = *self.map.get(key)?;

        self.detach(node_ptr);
        self.attach_front(node_ptr);

        // SAFETY: node_ptr is valid as long as it's in the map
        Some(unsafe { &mut (*node_ptr.as_ptr()).value })
    }

    /// Peeks at a value without updating LRU order.
    #[inline(always)]
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.map
            .get(key)
            .map(|node_ptr| unsafe { &(*node_ptr.as_ptr()).value })
    }

    /// Inserts a key-value pair, returning the previous value if the key existed.
    ///
    /// If the cache is at capacity, evicts the least recently used item.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::fast_lru::FastLru;
    ///
    /// let mut cache = FastLru::new(2);
    ///
    /// assert_eq!(cache.insert(1, "a"), None);
    /// assert_eq!(cache.insert(2, "b"), None);
    /// assert_eq!(cache.insert(1, "A"), Some("a")); // Update returns old value
    ///
    /// cache.insert(3, "c"); // Evicts key 2 (LRU)
    /// assert!(!cache.contains(&2));
    /// ```
    #[inline(always)]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        #[cfg(feature = "metrics")]
        self.metrics.record_insert_call();

        // Check for existing key
        if let Some(&node_ptr) = self.map.get(&key) {
            #[cfg(feature = "metrics")]
            self.metrics.record_insert_update();

            // Update existing value
            let old_value = unsafe {
                let node = node_ptr.as_ptr();
                mem::replace(&mut (*node).value, value)
            };

            // Move to front
            self.detach(node_ptr);
            self.attach_front(node_ptr);

            return Some(old_value);
        }

        // Evict if at capacity
        if self.capacity > 0 && self.map.len() >= self.capacity {
            #[cfg(feature = "metrics")]
            self.metrics.record_evict_call();

            if self.pop_lru().is_some() {
                #[cfg(feature = "metrics")]
                self.metrics.record_evicted_entry();
            }
        }

        // Don't insert if capacity is 0
        if self.capacity == 0 {
            return None;
        }

        #[cfg(feature = "metrics")]
        self.metrics.record_insert_new();

        // Create new node with optimized field order
        let node = Box::new(Node {
            prev: None,
            next: None,
            key: key.clone(),
            value,
        });
        let node_ptr = NonNull::new(Box::into_raw(node)).unwrap();

        // Add to map and list
        self.map.insert(key, node_ptr);
        self.attach_front(node_ptr);

        None
    }

    /// Removes a key from the cache, returning its value.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::fast_lru::FastLru;
    ///
    /// let mut cache = FastLru::new(10);
    /// cache.insert(1, "value");
    ///
    /// assert_eq!(cache.remove(&1), Some("value"));
    /// assert_eq!(cache.remove(&1), None);
    /// ```
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let node_ptr = self.map.remove(key)?;

        self.detach(node_ptr);

        // SAFETY: We own the node after removing from map
        let node = unsafe { Box::from_raw(node_ptr.as_ptr()) };
        Some(node.value)
    }

    /// Removes and returns the least recently used item.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::fast_lru::FastLru;
    ///
    /// let mut cache = FastLru::new(10);
    /// cache.insert(1, "one");
    /// cache.insert(2, "two");
    ///
    /// // Access key 1 to make it MRU
    /// cache.get(&1);
    ///
    /// // Key 2 is now LRU
    /// assert_eq!(cache.pop_lru(), Some((2, "two")));
    /// ```
    pub fn pop_lru(&mut self) -> Option<(K, V)> {
        #[cfg(feature = "metrics")]
        self.metrics.record_pop_lru_call();

        let tail_ptr = self.tail?;

        // SAFETY: tail is valid if Some
        let key = unsafe { (*tail_ptr.as_ptr()).key.clone() };

        self.map.remove(&key);
        self.detach(tail_ptr);

        let node = unsafe { Box::from_raw(tail_ptr.as_ptr()) };

        #[cfg(feature = "metrics")]
        self.metrics.record_pop_lru_found();

        Some((node.key, node.value))
    }

    /// Peeks at the least recently used item without removing it.
    pub fn peek_lru(&self) -> Option<(&K, &V)> {
        #[cfg(feature = "metrics")]
        (&self.metrics).record_peek_lru_call();

        self.tail.map(|node_ptr| {
            #[cfg(feature = "metrics")]
            (&self.metrics).record_peek_lru_found();

            unsafe {
                let node = node_ptr.as_ptr();
                (&(*node).key, &(*node).value)
            }
        })
    }

    /// Clears all entries from the cache.
    pub fn clear(&mut self) {
        #[cfg(feature = "metrics")]
        self.metrics.record_clear();

        while self.pop_lru().is_some() {}
    }

    /// Moves an existing entry to MRU position without returning its value.
    ///
    /// Returns `true` if the key existed and was touched.
    #[inline(always)]
    pub fn touch(&mut self, key: &K) -> bool {
        #[cfg(feature = "metrics")]
        self.metrics.record_touch_call();

        if let Some(&node_ptr) = self.map.get(key) {
            self.detach(node_ptr);
            self.attach_front(node_ptr);

            #[cfg(feature = "metrics")]
            self.metrics.record_touch_found();

            true
        } else {
            false
        }
    }

    // =========================================================================
    // Internal linked-list operations
    // =========================================================================

    /// Detaches a node from its current position in the list.
    #[inline(always)]
    fn detach(&mut self, node_ptr: NonNull<Node<K, V>>) {
        unsafe {
            let node = node_ptr.as_ptr();
            let prev = (*node).prev;
            let next = (*node).next;

            match prev {
                Some(prev_ptr) => (*prev_ptr.as_ptr()).next = next,
                None => self.head = next,
            }

            match next {
                Some(next_ptr) => (*next_ptr.as_ptr()).prev = prev,
                None => self.tail = prev,
            }

            (*node).prev = None;
            (*node).next = None;
        }
    }

    /// Attaches a node at the front (MRU position) of the list.
    #[inline(always)]
    fn attach_front(&mut self, node_ptr: NonNull<Node<K, V>>) {
        unsafe {
            let node = node_ptr.as_ptr();
            (*node).prev = None;
            (*node).next = self.head;

            match self.head {
                Some(head_ptr) => (*head_ptr.as_ptr()).prev = Some(node_ptr),
                None => self.tail = Some(node_ptr),
            }

            self.head = Some(node_ptr);
        }
    }
}

#[cfg(feature = "metrics")]
impl<K, V> FastLru<K, V>
where
    K: Eq + Hash + Clone,
{
    pub fn metrics_snapshot(&self) -> LruMetricsSnapshot {
        LruMetricsSnapshot {
            get_calls: self.metrics.get_calls,
            get_hits: self.metrics.get_hits,
            get_misses: self.metrics.get_misses,
            insert_calls: self.metrics.insert_calls,
            insert_updates: self.metrics.insert_updates,
            insert_new: self.metrics.insert_new,
            evict_calls: self.metrics.evict_calls,
            evicted_entries: self.metrics.evicted_entries,
            pop_lru_calls: self.metrics.pop_lru_calls,
            pop_lru_found: self.metrics.pop_lru_found,
            peek_lru_calls: self.metrics.peek_lru_calls.get(),
            peek_lru_found: self.metrics.peek_lru_found.get(),
            touch_calls: self.metrics.touch_calls,
            touch_found: self.metrics.touch_found,
            recency_rank_calls: self.metrics.recency_rank_calls.get(),
            recency_rank_found: self.metrics.recency_rank_found.get(),
            recency_rank_scan_steps: self.metrics.recency_rank_scan_steps.get(),
            cache_len: self.map.len(),
            capacity: self.capacity,
        }
    }
}

#[cfg(feature = "metrics")]
impl<K, V> MetricsSnapshotProvider<LruMetricsSnapshot> for FastLru<K, V>
where
    K: Eq + Hash + Clone,
{
    fn snapshot(&self) -> LruMetricsSnapshot {
        self.metrics_snapshot()
    }
}

impl<K, V> Drop for FastLru<K, V> {
    fn drop(&mut self) {
        // Free all nodes
        let mut current = self.head;
        while let Some(node_ptr) = current {
            unsafe {
                let node = Box::from_raw(node_ptr.as_ptr());
                current = node.next;
            }
        }
    }
}

// SAFETY: FastLru is Send if K and V are Send
unsafe impl<K: Send, V: Send> Send for FastLru<K, V> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut cache = FastLru::new(3);

        assert!(cache.is_empty());
        assert_eq!(cache.capacity(), 3);

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&1), Some(&"one"));
        assert_eq!(cache.get(&2), Some(&"two"));
        assert_eq!(cache.get(&3), Some(&"three"));
    }

    #[test]
    fn test_eviction() {
        let mut cache = FastLru::new(2);

        cache.insert(1, "one");
        cache.insert(2, "two");

        // Access 1 to make it MRU
        cache.get(&1);

        // Insert 3, should evict 2 (LRU)
        cache.insert(3, "three");

        assert!(cache.contains(&1));
        assert!(!cache.contains(&2));
        assert!(cache.contains(&3));
    }

    #[test]
    fn test_update() {
        let mut cache = FastLru::new(10);

        cache.insert(1, "one");
        let old = cache.insert(1, "ONE");

        assert_eq!(old, Some("one"));
        assert_eq!(cache.get(&1), Some(&"ONE"));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_remove() {
        let mut cache = FastLru::new(10);

        cache.insert(1, "one");
        cache.insert(2, "two");

        assert_eq!(cache.remove(&1), Some("one"));
        assert_eq!(cache.remove(&1), None);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_pop_lru() {
        let mut cache = FastLru::new(10);

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        // 1 is LRU (inserted first, not accessed)
        assert_eq!(cache.pop_lru(), Some((1, "one")));
        assert_eq!(cache.pop_lru(), Some((2, "two")));
        assert_eq!(cache.pop_lru(), Some((3, "three")));
        assert_eq!(cache.pop_lru(), None);
    }

    #[test]
    fn test_touch() {
        let mut cache = FastLru::new(3);

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        // Touch 1 to make it MRU
        assert!(cache.touch(&1));

        // Insert 4, evicts 2 (now LRU)
        cache.insert(4, "four");

        assert!(cache.contains(&1));
        assert!(!cache.contains(&2));
    }

    #[test]
    fn test_zero_capacity() {
        let mut cache: FastLru<i32, &str> = FastLru::new(0);

        assert_eq!(cache.insert(1, "one"), None);
        assert!(cache.is_empty());
        assert_eq!(cache.get(&1), None);
    }

    #[test]
    fn test_clear() {
        let mut cache = FastLru::new(10);

        cache.insert(1, "one");
        cache.insert(2, "two");

        cache.clear();

        assert!(cache.is_empty());
        assert_eq!(cache.get(&1), None);
    }

    #[test]
    fn test_peek() {
        let mut cache = FastLru::new(3);

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        // Peek doesn't change order
        assert_eq!(cache.peek(&1), Some(&"one"));

        // Insert 4, should evict 1 (still LRU because peek doesn't update)
        cache.insert(4, "four");

        assert!(!cache.contains(&1));
    }
}
