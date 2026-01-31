//! Storage backends for cache policies.
//!
//! Stores focus on key/value ownership and lookup semantics, while policies
//! manage eviction order and metadata. This keeps policy logic independent
//! of how values are stored (e.g., HashMap, concurrent map, arena).
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                           Cache System                                  │
//! │                                                                         │
//! │  ┌─────────────────────┐              ┌─────────────────────┐           │
//! │  │       Policy        │              │        Store        │           │
//! │  │  (eviction order)   │◄────────────►│   (key/value data)  │           │
//! │  └─────────────────────┘   notifies   └─────────────────────┘           │
//! │           │                                     │                       │
//! │           │ manages                             │ implements            │
//! │           ▼                                     ▼                       │
//! │  ┌─────────────────────┐              ┌─────────────────────┐           │
//! │  │  Policy Metadata    │              │   Store Traits      │           │
//! │  │  - access order     │              │   - StoreCore       │           │
//! │  │  - frequency counts │              │   - StoreMut        │           │
//! │  │  - eviction hints   │              │   - ConcurrentStore │           │
//! │  └─────────────────────┘              └─────────────────────┘           │
//! └─────────────────────────────────────────────────────────────────────────┘
//!
//! Trait Hierarchy
//! ───────────────
//!
//! Single-threaded (direct ownership):
//!
//!     ┌──────────────────┐
//!     │    StoreCore     │  get(&K) -> &V
//!     │   (read-only)    │  contains, len, capacity
//!     └────────┬─────────┘
//!              │ extends
//!              ▼
//!     ┌──────────────────┐
//!     │    StoreMut      │  try_insert(K, V) -> Result<Option<V>>
//!     │   (read-write)   │  remove(&K) -> Option<V>
//!     └──────────────────┘
//!
//! Concurrent (Arc-based ownership):
//!
//!     ┌──────────────────────┐
//!     │ ConcurrentStoreRead  │  get(&K) -> Arc<V>
//!     │     (read-only)      │  contains, len, capacity
//!     └──────────┬───────────┘
//!                │ extends
//!                ▼
//!     ┌──────────────────────┐
//!     │   ConcurrentStore    │  try_insert(K, Arc<V>) -> Result<Option<Arc<V>>>
//!     │    (read-write)      │  remove(&K) -> Option<Arc<V>>
//!     └──────────────────────┘
//!
//! Factory Traits
//! ──────────────
//!
//!     StoreFactory<K, V>           ConcurrentStoreFactory<K, V>
//!            │                                │
//!            ▼                                ▼
//!     create(capacity)              create(capacity)
//!            │                                │
//!            ▼                                ▼
//!     impl StoreMut<K, V>          impl ConcurrentStore<K, V>
//! ```
//!
//! ## Ownership Model
//!
//! Single-threaded stores (`StoreCore` + `StoreMut`) use direct ownership:
//! - Store owns `K` and `V` after insertion
//! - `get` returns `&V` (zero overhead)
//! - `remove` returns owned `V`
//!
//! Concurrent stores (`ConcurrentStore`) use shared ownership:
//! - Store holds `Arc<V>` internally
//! - `get` returns `Arc<V>` (can outlive the lock)
//! - Required because references can't outlive lock guards

use std::sync::Arc;

/// Snapshot of store-level metrics.
///
/// Provides a point-in-time view of store activity counters. All fields
/// are cumulative since store creation.
///
/// # Example
///
/// ```
/// use cachekit::store::traits::StoreMetrics;
///
/// let metrics = StoreMetrics {
///     hits: 150,
///     misses: 50,
///     inserts: 100,
///     updates: 20,
///     removes: 10,
///     evictions: 40,
/// };
///
/// let hit_rate = metrics.hits as f64 / (metrics.hits + metrics.misses) as f64;
/// assert!((hit_rate - 0.75).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StoreMetrics {
    /// Number of successful lookups.
    pub hits: u64,
    /// Number of failed lookups.
    pub misses: u64,
    /// Number of new key insertions.
    pub inserts: u64,
    /// Number of value updates for existing keys.
    pub updates: u64,
    /// Number of explicit removals via `remove()`.
    pub removes: u64,
    /// Number of entries evicted by the policy.
    pub evictions: u64,
}

/// Error returned when inserting into a store at capacity.
///
/// This error occurs when calling `try_insert` with a new key on a store
/// that has reached its maximum capacity. Updates to existing keys never
/// trigger this error.
///
/// # Example
///
/// ```
/// use cachekit::store::traits::StoreFull;
///
/// fn handle_insert_result<V>(result: Result<Option<V>, StoreFull>) {
///     match result {
///         Ok(Some(old)) => println!("Updated existing entry"),
///         Ok(None) => println!("Inserted new entry"),
///         Err(StoreFull) => println!("Store is full, consider evicting"),
///     }
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreFull;

// =============================================================================
// Single-threaded store traits
// =============================================================================

/// Read-only store operations for single-threaded backends.
///
/// Provides zero-cost access to stored values via borrowed references.
/// Implementors manage key-value storage without concerning themselves
/// with eviction logic—that's the policy's responsibility.
///
/// # Type Parameters
///
/// - `K`: Key type, typically requires `Hash + Eq` for map-based stores
/// - `V`: Value type, no trait bounds at the trait level
///
/// # Example
///
/// ```
/// use cachekit::store::traits::{StoreCore, StoreMut, StoreMetrics};
///
/// // A minimal in-memory store implementation
/// struct VecStore<K, V> {
///     entries: Vec<(K, V)>,
///     capacity: usize,
/// }
///
/// impl<K: PartialEq, V> StoreCore<K, V> for VecStore<K, V> {
///     fn get(&self, key: &K) -> Option<&V> {
///         self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
///     }
///
///     fn contains(&self, key: &K) -> bool {
///         self.entries.iter().any(|(k, _)| k == key)
///     }
///
///     fn len(&self) -> usize {
///         self.entries.len()
///     }
///
///     fn capacity(&self) -> usize {
///         self.capacity
///     }
/// }
/// ```
pub trait StoreCore<K, V> {
    /// Returns a reference to the value for the given key.
    ///
    /// Returns `None` if the key is not present. Does not update access
    /// metadata—that's handled by the policy layer.
    fn get(&self, key: &K) -> Option<&V>;

    /// Returns `true` if the key exists in the store.
    fn contains(&self, key: &K) -> bool;

    /// Returns the current number of entries.
    fn len(&self) -> usize;

    /// Returns `true` if the store contains no entries.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the maximum number of entries this store can hold.
    fn capacity(&self) -> usize;

    /// Returns a snapshot of the store's metrics.
    ///
    /// Default implementation returns zeroed metrics. Override to provide
    /// actual tracking.
    fn metrics(&self) -> StoreMetrics {
        StoreMetrics::default()
    }
}

/// Mutable store operations for single-threaded backends.
///
/// Extends [`StoreCore`] with write operations. Takes and returns owned
/// values directly—no `Arc` overhead. The store does not perform eviction;
/// it signals capacity via [`StoreFull`] and lets the policy decide what
/// to evict.
///
/// # Example
///
/// ```
/// use cachekit::store::traits::{StoreCore, StoreMut, StoreFull};
///
/// struct VecStore<K, V> {
///     entries: Vec<(K, V)>,
///     capacity: usize,
/// }
///
/// impl<K, V> VecStore<K, V> {
///     fn new(capacity: usize) -> Self {
///         Self { entries: Vec::with_capacity(capacity), capacity }
///     }
/// }
///
/// impl<K: PartialEq, V> StoreCore<K, V> for VecStore<K, V> {
///     fn get(&self, key: &K) -> Option<&V> {
///         self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
///     }
///     fn contains(&self, key: &K) -> bool {
///         self.entries.iter().any(|(k, _)| k == key)
///     }
///     fn len(&self) -> usize { self.entries.len() }
///     fn capacity(&self) -> usize { self.capacity }
/// }
///
/// impl<K: PartialEq, V> StoreMut<K, V> for VecStore<K, V> {
///     fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, StoreFull> {
///         // Check for existing key first
///         if let Some(pos) = self.entries.iter().position(|(k, _)| k == &key) {
///             let old = std::mem::replace(&mut self.entries[pos].1, value);
///             return Ok(Some(old));
///         }
///         // New key—check capacity
///         if self.entries.len() >= self.capacity {
///             return Err(StoreFull);
///         }
///         self.entries.push((key, value));
///         Ok(None)
///     }
///
///     fn remove(&mut self, key: &K) -> Option<V> {
///         self.entries.iter().position(|(k, _)| k == key)
///             .map(|pos| self.entries.remove(pos).1)
///     }
///
///     fn clear(&mut self) {
///         self.entries.clear();
///     }
/// }
///
/// let mut store = VecStore::new(2);
/// assert!(store.try_insert("a", 1).is_ok());
/// assert!(store.try_insert("b", 2).is_ok());
/// assert_eq!(store.try_insert("c", 3), Err(StoreFull));
/// ```
pub trait StoreMut<K, V>: StoreCore<K, V> {
    /// Inserts a key-value pair, returning the previous value if the key existed.
    ///
    /// # Errors
    ///
    /// Returns [`StoreFull`] if the store is at capacity and `key` is new.
    /// Updates to existing keys always succeed.
    fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, StoreFull>;

    /// Removes and returns the value for the given key.
    ///
    /// Returns `None` if the key was not present.
    fn remove(&mut self, key: &K) -> Option<V>;

    /// Removes all entries from the store.
    fn clear(&mut self);
}

// =============================================================================
// Concurrent store traits
// =============================================================================

/// Read-only store operations for concurrent backends.
///
/// Returns `Arc<V>` because borrowed references cannot outlive lock guards
/// in concurrent contexts. The `Arc` overhead is acceptable here since
/// concurrent access already involves synchronization costs.
///
/// # Thread Safety
///
/// Implementors must be `Send + Sync`. Internal synchronization (mutex,
/// RwLock, lock-free structures) is an implementation detail.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use std::sync::{Arc, RwLock};
/// use cachekit::store::traits::{ConcurrentStoreRead, StoreMetrics};
///
/// struct ConcurrentMapStore<K, V> {
///     inner: RwLock<HashMap<K, Arc<V>>>,
///     capacity: usize,
/// }
///
/// impl<K, V> ConcurrentMapStore<K, V> {
///     fn new(capacity: usize) -> Self {
///         Self {
///             inner: RwLock::new(HashMap::with_capacity(capacity)),
///             capacity,
///         }
///     }
/// }
///
/// impl<K: Eq + std::hash::Hash, V> ConcurrentStoreRead<K, V> for ConcurrentMapStore<K, V>
/// where
///     K: Send + Sync,
///     V: Send + Sync,
/// {
///     fn get(&self, key: &K) -> Option<Arc<V>> {
///         self.inner.read().unwrap().get(key).cloned()
///     }
///
///     fn contains(&self, key: &K) -> bool {
///         self.inner.read().unwrap().contains_key(key)
///     }
///
///     fn len(&self) -> usize {
///         self.inner.read().unwrap().len()
///     }
///
///     fn capacity(&self) -> usize {
///         self.capacity
///     }
/// }
/// ```
pub trait ConcurrentStoreRead<K, V>: Send + Sync {
    /// Returns a shared handle to the value for the given key.
    ///
    /// The returned `Arc<V>` can be held across await points or passed
    /// to other threads without lifetime concerns.
    fn get(&self, key: &K) -> Option<Arc<V>>;

    /// Returns `true` if the key exists in the store.
    fn contains(&self, key: &K) -> bool;

    /// Returns the current number of entries.
    ///
    /// Note: In concurrent contexts, this value may be stale by the time
    /// it's used.
    fn len(&self) -> usize;

    /// Returns `true` if the store contains no entries.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the maximum number of entries this store can hold.
    fn capacity(&self) -> usize;

    /// Returns a snapshot of the store's metrics.
    fn metrics(&self) -> StoreMetrics {
        StoreMetrics::default()
    }
}

/// Mutable store operations for concurrent backends.
///
/// Extends [`ConcurrentStoreRead`] with write operations using interior
/// mutability. Methods take `&self` (not `&mut self`) enabling shared
/// access across threads.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use std::sync::{Arc, RwLock};
/// use cachekit::store::traits::{
///     ConcurrentStore, ConcurrentStoreRead, StoreFull, StoreMetrics,
/// };
///
/// struct ConcurrentMapStore<K, V> {
///     inner: RwLock<HashMap<K, Arc<V>>>,
///     capacity: usize,
/// }
///
/// impl<K, V> ConcurrentMapStore<K, V> {
///     fn new(capacity: usize) -> Self {
///         Self {
///             inner: RwLock::new(HashMap::with_capacity(capacity)),
///             capacity,
///         }
///     }
/// }
///
/// impl<K: Eq + std::hash::Hash, V> ConcurrentStoreRead<K, V> for ConcurrentMapStore<K, V>
/// where
///     K: Send + Sync,
///     V: Send + Sync,
/// {
///     fn get(&self, key: &K) -> Option<Arc<V>> {
///         self.inner.read().unwrap().get(key).cloned()
///     }
///     fn contains(&self, key: &K) -> bool {
///         self.inner.read().unwrap().contains_key(key)
///     }
///     fn len(&self) -> usize {
///         self.inner.read().unwrap().len()
///     }
///     fn capacity(&self) -> usize {
///         self.capacity
///     }
/// }
///
/// impl<K: Eq + std::hash::Hash, V> ConcurrentStore<K, V> for ConcurrentMapStore<K, V>
/// where
///     K: Send + Sync,
///     V: Send + Sync,
/// {
///     fn try_insert(&self, key: K, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull> {
///         let mut guard = self.inner.write().unwrap();
///         if guard.contains_key(&key) {
///             return Ok(guard.insert(key, value));
///         }
///         if guard.len() >= self.capacity {
///             return Err(StoreFull);
///         }
///         Ok(guard.insert(key, value))
///     }
///
///     fn remove(&self, key: &K) -> Option<Arc<V>> {
///         self.inner.write().unwrap().remove(key)
///     }
///
///     fn clear(&self) {
///         self.inner.write().unwrap().clear();
///     }
/// }
///
/// let store = ConcurrentMapStore::<&str, i32>::new(2);
/// assert!(store.try_insert("a", Arc::new(1)).is_ok());
/// assert!(store.try_insert("b", Arc::new(2)).is_ok());
/// assert_eq!(store.try_insert("c", Arc::new(3)), Err(StoreFull));
/// ```
pub trait ConcurrentStore<K, V>: ConcurrentStoreRead<K, V> {
    /// Inserts a key-value pair, returning the previous value if the key existed.
    ///
    /// Uses interior mutability—takes `&self` not `&mut self`.
    ///
    /// # Errors
    ///
    /// Returns [`StoreFull`] if the store is at capacity and `key` is new.
    fn try_insert(&self, key: K, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull>;

    /// Removes and returns the value for the given key.
    fn remove(&self, key: &K) -> Option<Arc<V>>;

    /// Removes all entries from the store.
    fn clear(&self);
}

// =============================================================================
// Factory traits
// =============================================================================

/// Factory for creating single-threaded store instances.
///
/// Enables generic cache construction without hard-coding a specific store
/// implementation. Policies use this to create their backing storage.
///
/// # Example
///
/// ```
/// use cachekit::store::traits::{StoreCore, StoreMut, StoreFactory, StoreFull};
///
/// // A simple vec-backed store
/// struct VecStore<K, V> {
///     entries: Vec<(K, V)>,
///     capacity: usize,
/// }
///
/// impl<K: PartialEq, V> StoreCore<K, V> for VecStore<K, V> {
///     fn get(&self, key: &K) -> Option<&V> {
///         self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
///     }
///     fn contains(&self, key: &K) -> bool {
///         self.entries.iter().any(|(k, _)| k == key)
///     }
///     fn len(&self) -> usize { self.entries.len() }
///     fn capacity(&self) -> usize { self.capacity }
/// }
///
/// impl<K: PartialEq, V> StoreMut<K, V> for VecStore<K, V> {
///     fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, StoreFull> {
///         if let Some(pos) = self.entries.iter().position(|(k, _)| k == &key) {
///             let old = std::mem::replace(&mut self.entries[pos].1, value);
///             return Ok(Some(old));
///         }
///         if self.entries.len() >= self.capacity {
///             return Err(StoreFull);
///         }
///         self.entries.push((key, value));
///         Ok(None)
///     }
///     fn remove(&mut self, key: &K) -> Option<V> {
///         self.entries.iter().position(|(k, _)| k == key)
///             .map(|pos| self.entries.remove(pos).1)
///     }
///     fn clear(&mut self) { self.entries.clear(); }
/// }
///
/// // Factory implementation
/// struct VecStoreFactory;
///
/// impl<K: PartialEq, V> StoreFactory<K, V> for VecStoreFactory {
///     type Store = VecStore<K, V>;
///
///     fn create(capacity: usize) -> Self::Store {
///         VecStore {
///             entries: Vec::with_capacity(capacity),
///             capacity,
///         }
///     }
/// }
///
/// // Generic function that works with any store factory
/// fn make_store<F, K, V>(cap: usize) -> F::Store
/// where
///     F: StoreFactory<K, V>,
/// {
///     F::create(cap)
/// }
///
/// let store: VecStore<String, i32> = make_store::<VecStoreFactory, _, _>(100);
/// assert_eq!(store.capacity(), 100);
/// ```
pub trait StoreFactory<K, V> {
    /// The concrete store type produced by this factory.
    type Store: StoreCore<K, V>;

    /// Creates a new store with the specified capacity.
    fn create(capacity: usize) -> Self::Store;
}

/// Factory for creating concurrent store instances.
///
/// Same pattern as [`StoreFactory`] but for thread-safe stores.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use std::sync::{Arc, RwLock};
/// use cachekit::store::traits::{
///     ConcurrentStore, ConcurrentStoreFactory, ConcurrentStoreRead, StoreFull,
/// };
///
/// struct ConcurrentMapStore<K, V> {
///     inner: RwLock<HashMap<K, Arc<V>>>,
///     capacity: usize,
/// }
///
/// // ... trait impls omitted for brevity (see ConcurrentStore example)
/// # impl<K: Eq + std::hash::Hash + Send + Sync, V: Send + Sync> ConcurrentStoreRead<K, V>
/// #     for ConcurrentMapStore<K, V>
/// # {
/// #     fn get(&self, key: &K) -> Option<Arc<V>> {
/// #         self.inner.read().unwrap().get(key).cloned()
/// #     }
/// #     fn contains(&self, key: &K) -> bool {
/// #         self.inner.read().unwrap().contains_key(key)
/// #     }
/// #     fn len(&self) -> usize { self.inner.read().unwrap().len() }
/// #     fn capacity(&self) -> usize { self.capacity }
/// # }
/// # impl<K: Eq + std::hash::Hash + Send + Sync, V: Send + Sync> ConcurrentStore<K, V>
/// #     for ConcurrentMapStore<K, V>
/// # {
/// #     fn try_insert(&self, key: K, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull> {
/// #         let mut guard = self.inner.write().unwrap();
/// #         if guard.len() >= self.capacity && !guard.contains_key(&key) {
/// #             return Err(StoreFull);
/// #         }
/// #         Ok(guard.insert(key, value))
/// #     }
/// #     fn remove(&self, key: &K) -> Option<Arc<V>> {
/// #         self.inner.write().unwrap().remove(key)
/// #     }
/// #     fn clear(&self) { self.inner.write().unwrap().clear(); }
/// # }
///
/// struct ConcurrentMapStoreFactory;
///
/// impl<K, V> ConcurrentStoreFactory<K, V> for ConcurrentMapStoreFactory
/// where
///     K: Eq + std::hash::Hash + Send + Sync,
///     V: Send + Sync,
/// {
///     type Store = ConcurrentMapStore<K, V>;
///
///     fn create(capacity: usize) -> Self::Store {
///         ConcurrentMapStore {
///             inner: RwLock::new(HashMap::with_capacity(capacity)),
///             capacity,
///         }
///     }
/// }
/// ```
pub trait ConcurrentStoreFactory<K, V> {
    /// The concrete store type produced by this factory.
    type Store: ConcurrentStore<K, V>;

    /// Creates a new store with the specified capacity.
    fn create(capacity: usize) -> Self::Store;
}
