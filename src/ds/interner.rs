//! Simple key interner for mapping external keys to compact handles.
//!
//! Assigns monotonically increasing `u64` handles to unique keys, enabling
//! fast lookups while avoiding repeated key cloning in hot paths.
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │                         KeyInterner Layout                                 │
//! │                                                                            │
//! │   ┌─────────────────────────────────────────────────────────────────────┐  │
//! │   │  index: HashMap<K, u64, S>           keys: Vec<K>                   │  │
//! │   │                                                                     │  │
//! │   │  ┌────────────────────────┐          ┌─────────────────────────┐    │  │
//! │   │  │  Key         Handle    │          │ Index   Key             │    │  │
//! │   │  ├────────────────────────┤          ├─────────────────────────┤    │  │
//! │   │  │  "user:123"  → 0       │          │   0     "user:123"      │    │  │
//! │   │  │  "user:456"  → 1       │          │   1     "user:456"      │    │  │
//! │   │  │  "session:a" → 2       │          │   2     "session:a"     │    │  │
//! │   │  └────────────────────────┘          └─────────────────────────┘    │  │
//! │   │                                                                     │  │
//! │   │  intern("user:123") ──► lookup in index ──► return 0                │  │
//! │   │  resolve(1) ──► keys[1] ──► "user:456"                              │  │
//! │   └─────────────────────────────────────────────────────────────────────┘  │
//! │                                                                            │
//! │   Data Flow                                                                │
//! │   ─────────                                                                │
//! │     intern(key):                                                           │
//! │       1. Check index for existing handle                                   │
//! │       2. If found: return handle                                           │
//! │       3. If not: assign handle = keys.len(), store in both structures      │
//! │                                                                            │
//! │     resolve(handle):                                                       │
//! │       1. Direct index into keys vector: O(1)                               │
//! │                                                                            │
//! └────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! - [`KeyInterner`]: Maps keys to compact `u64` handles
//!
//! ## Operations
//!
//! | Operation               | Description                                  | Complexity |
//! |-------------------------|----------------------------------------------|------------|
//! | `new`                   | Create an empty interner                     | O(1)       |
//! | `with_capacity`         | Create an empty interner with reserved space | O(1)       |
//! | `with_hasher`           | Create an empty interner with a custom hasher| O(1)       |
//! | `intern`                | Get or create handle for key                 | O(1) avg   |
//! | `try_intern`            | Fallible [`intern`], respects `MAX_CAPACITY` | O(1) avg   |
//! | `get_handle`            | Lookup handle without inserting              | O(1) avg   |
//! | `get_handle_borrowed`   | Lookup handle using a borrowed key form      | O(1) avg   |
//! | `resolve`               | Convert handle back to key reference         | O(1)       |
//! | `len`                   | Return number of interned keys               | O(1)       |
//! | `is_empty`              | Check whether any keys are interned          | O(1)       |
//! | `clear`                 | Remove all interned keys (bumps generation)  | O(n)       |
//! | `shrink_to_fit`         | Shrink backing storage to fit length         | O(n)       |
//! | `clear_shrink`          | Clear all keys and release spare capacity    | O(n)       |
//! | `approx_bytes`          | Estimate memory footprint                    | O(1)       |
//! | `generation`            | Epoch counter bumped on every `clear`        | O(1)       |
//! | `iter`                  | Iterate over `(handle, key)` pairs           | O(n) total |
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
//! let mut interner: KeyInterner<String> = KeyInterner::new();
//!
//! // Intern keys to get compact handles
//! let h1 = interner.intern(&"long_key_name_1".to_owned());
//! let _h2 = interner.intern(&"long_key_name_2".to_owned());
//!
//! // Same key returns same handle
//! let h1_again = interner.intern(&"long_key_name_1".to_owned());
//! assert_eq!(h1, h1_again);
//!
//! // Resolve handle back to key
//! assert_eq!(interner.resolve(h1).map(String::as_str), Some("long_key_name_1"));
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
//!     let handle = interner.intern(&key.to_owned());
//!     cache.insert(handle, value);
//! }
//!
//! fn get<'a>(interner: &KeyInterner<String>, cache: &'a HashMap<u64, Vec<u8>>,
//!            key: &str) -> Option<&'a Vec<u8>> {
//!     let handle = interner.get_handle_borrowed(key)?;
//!     cache.get(&handle)
//! }
//!
//! put(&mut interner, &mut cache, "session:abc", vec![1, 2, 3]);
//! assert!(get(&interner, &cache, "session:abc").is_some());
//! ```
//!
//! ## Thread Safety
//!
//! `KeyInterner<K, S>` is `Send + Sync` when `K` and `S` are, but provides no
//! internal synchronization. For shared mutable access, wrap in
//! `parking_lot::RwLock` or similar synchronization primitive.
//!
//! ## Security
//!
//! The default hasher is [`rustc_hash::FxBuildHasher`], chosen for speed on
//! trusted input. **`FxHash` is non-cryptographic and is not resistant to
//! hash-flooding / HashDoS attacks.** If a `KeyInterner` may observe keys
//! derived from untrusted input (for example, cache keys sourced from HTTP
//! URLs, user IDs, request parameters, or bearer tokens), callers should
//! either:
//!
//! - construct it with a DoS-resistant hasher via
//!   [`KeyInterner::with_hasher`] /
//!   [`KeyInterner::with_capacity_and_hasher`] (for example
//!   [`std::collections::hash_map::RandomState`]), or
//! - preprocess keys into a form the attacker cannot control the hash of.
//!
//! `KeyInterner` is also **append-only**: keys are never removed except via
//! [`KeyInterner::clear`] / [`KeyInterner::clear_shrink`], so an attacker
//! who can trigger interning of unique keys can drive memory usage without
//! bound. Two mitigations are provided:
//!
//! 1. [`KeyInterner::MAX_CAPACITY`] caps the total number of unique keys.
//!    [`KeyInterner::intern`] panics when the cap is reached;
//!    [`KeyInterner::try_intern`] returns
//!    [`InternerError::CapacityExceeded`] instead. When keys may come from
//!    untrusted input, always use `try_intern` and enforce a smaller
//!    admission-control bound of your own on top.
//! 2. [`KeyInterner::try_with_capacity`] refuses oversized preallocations
//!    rather than aborting the process with an allocator error.
//!
//! **Handles are not capability tokens.** They are plain sequential
//! `u64`s; there is no cryptographic or structural guarantee that prevents
//! a handle minted by one `KeyInterner` from accidentally resolving on
//! another. Two specific hazards:
//!
//! - *Cross-instance confusion.* A handle from `interner_a` used against
//!   `interner_b` will silently resolve to whatever happens to live at
//!   that index in `interner_b`, not to `None`.
//! - *Clear-cycle reuse.* After [`KeyInterner::clear`] /
//!   [`KeyInterner::clear_shrink`], handles restart from `0`. A caller
//!   who stored handles externally and then interns new keys will see
//!   the stored handles silently resolve to unrelated keys. Use
//!   [`KeyInterner::generation`] to detect staleness: capture
//!   `generation()` when you store a handle, compare before using it,
//!   and treat mismatches as "handle invalidated".
//!
//! If you are exposing handles across a trust boundary, wrap
//! `(generation, handle)` together and validate both before `resolve`.
//!
//! `KeyInterner`'s [`std::fmt::Debug`] impl deliberately omits the
//! interned keys themselves and only reports length / capacity /
//! generation. Keys are frequently sensitive (URLs with query strings,
//! user IDs, auth material), and a single `eprintln!("{:?}", interner)`
//! would otherwise spill every interned key to logs. Use
//! [`KeyInterner::iter`] explicitly when you need the full contents.
//!
//! ## Implementation Notes
//!
//! - Handles are assigned monotonically starting at 0
//! - Keys are never removed (append-only design); use [`KeyInterner::clear`]
//!   to drop the whole table and restart from handle `0`
//! - Both `index` and `keys` store copies of the key
//! - `K`'s `Hash`, `Eq`, and `Clone` impls must be mutually consistent:
//!   `a == b` implies `hash(a) == hash(b)`, and `a.clone() == a`. Violating
//!   this contract either leaks entries (one stored handle, another new
//!   handle minted for the same key on next `intern`) or, for panicking
//!   `Hash` / `Eq` impls during `HashMap::insert`, leaves the interner in
//!   an internally consistent but reduced state (see
//!   [`KeyInterner::intern`] for the full exception-safety contract).

use rustc_hash::FxBuildHasher;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::collections::TryReserveError;
use std::hash::{BuildHasher, Hash};

/// Error type returned by [`KeyInterner::try_intern`] and
/// [`KeyInterner::try_with_capacity`] /
/// [`KeyInterner::try_with_capacity_and_hasher`].
///
/// Exists to give callers a non-panicking path when keys or capacities come
/// from untrusted input. See the module-level [security
/// notes](crate::ds::interner#security).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum InternerError {
    /// The interner is already at [`KeyInterner::MAX_CAPACITY`] and cannot
    /// accept another unique key.
    CapacityExceeded,
    /// The underlying allocator refused a growth request. Carries the
    /// original [`TryReserveError`] so callers can inspect the cause.
    AllocationFailed(TryReserveError),
}

impl std::fmt::Display for InternerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CapacityExceeded => write!(
                f,
                "KeyInterner is at MAX_CAPACITY; refusing to intern another unique key"
            ),
            Self::AllocationFailed(e) => write!(f, "KeyInterner allocation failed: {e}"),
        }
    }
}

impl std::error::Error for InternerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::AllocationFailed(e) => Some(e),
            _ => None,
        }
    }
}

impl From<TryReserveError> for InternerError {
    fn from(e: TryReserveError) -> Self {
        Self::AllocationFailed(e)
    }
}

/// Monotonic key interner that assigns a `u64` handle to each unique key.
///
/// Maps external keys to compact `u64` handles for efficient storage and lookup.
/// Handles are assigned sequentially starting from 0 and never reused **within
/// a single generation**; see [`KeyInterner::generation`] for detecting reuse
/// across [`KeyInterner::clear`] cycles.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Eq + Hash + Clone` for [`intern`](Self::intern)
/// - `S`: Hash builder; defaults to [`FxBuildHasher`]. Swap for a
///   DoS-resistant builder (e.g. [`std::collections::hash_map::RandomState`])
///   when keys may come from untrusted input — see the module-level
///   [security notes](self#security).
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
///     let handle = interner.intern(&key.to_owned());
///     *freq.entry(handle).or_insert(0) += 1;
/// }
///
/// access(&mut interner, &mut freq, "page_a");
/// access(&mut interner, &mut freq, "page_a");
/// access(&mut interner, &mut freq, "page_b");
///
/// let handle_a = interner.get_handle_borrowed("page_a").unwrap();
/// assert_eq!(freq[&handle_a], 2);
/// ```
pub struct KeyInterner<K, S = FxBuildHasher> {
    index: HashMap<K, u64, S>,
    keys: Vec<K>,
    /// Monotonic epoch counter bumped every time [`clear`](Self::clear) /
    /// [`clear_shrink`](Self::clear_shrink) invalidates existing handles.
    /// Callers that hold handles across a clear can compare
    /// [`generation`](Self::generation) before and after use to detect
    /// the staleness described in the module-level security notes.
    generation: u64,
}

// Manual `Debug` impl: only print aggregate state, not the interned keys
// themselves. Keys commonly contain sensitive data (URLs with query strings,
// user IDs, auth tokens); the derived `Debug` would dump every one of them
// on any `{:?}` or `panic!` formatting, which is an avoidable information-
// disclosure vector. See the module-level security notes.
//
// `finish_non_exhaustive` communicates to readers that internal storage is
// intentionally hidden rather than forgotten.
impl<K, S> std::fmt::Debug for KeyInterner<K, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyInterner")
            .field("len", &self.keys.len())
            .field("capacity", &self.keys.capacity())
            .field("index_capacity", &self.index.capacity())
            .field("generation", &self.generation)
            .finish_non_exhaustive()
    }
}

impl<K, S> Clone for KeyInterner<K, S>
where
    K: Clone + Eq + Hash,
    S: BuildHasher + Clone,
{
    fn clone(&self) -> Self {
        // Clone with tight capacity (len, not source capacity) so a cleared-
        // but-unshrunk interner doesn't propagate its oversized allocation
        // through every subsequent `.clone()`. This is a defensive measure
        // against memory-DoS amplification when clones fan out.
        let len = self.keys.len();
        let hasher = self.index.hasher().clone();
        let mut new_index = HashMap::with_capacity_and_hasher(len, hasher);
        for (k, &v) in self.index.iter() {
            new_index.insert(k.clone(), v);
        }
        let new_keys = self.keys.clone();
        Self {
            index: new_index,
            keys: new_keys,
            generation: self.generation,
        }
    }
}

impl<K, S> Default for KeyInterner<K, S>
where
    S: Default,
{
    fn default() -> Self {
        Self {
            index: HashMap::with_hasher(S::default()),
            keys: Vec::new(),
            generation: 0,
        }
    }
}

impl<K> KeyInterner<K, FxBuildHasher> {
    /// Creates an empty interner with the default [`FxBuildHasher`].
    ///
    /// # Security
    ///
    /// The default hasher is **not** DoS-resistant. If keys may be
    /// attacker-controlled, prefer [`KeyInterner::with_hasher`] with a
    /// randomised builder such as
    /// [`std::collections::hash_map::RandomState`]. See the module-level
    /// [security notes](self#security).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::KeyInterner;
    ///
    /// let interner: KeyInterner<String> = KeyInterner::new();
    /// assert!(interner.is_empty());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an interner with pre-allocated capacity, using the default
    /// [`FxBuildHasher`].
    ///
    /// The requested `capacity` is silently clamped to
    /// [`KeyInterner::MAX_CAPACITY`] and, beyond that, the allocator is
    /// allowed to refuse the reservation without aborting. This keeps
    /// configuration-derived capacities from crashing the process.
    /// Use [`try_with_capacity`](Self::try_with_capacity) to observe
    /// both the clamp and the allocator failure explicitly.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::KeyInterner;
    ///
    /// let interner: KeyInterner<String> = KeyInterner::with_capacity(1000);
    /// assert!(interner.is_empty());
    /// ```
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self
    where
        K: Eq + Hash,
    {
        Self::with_capacity_and_hasher(capacity, FxBuildHasher)
    }

    /// Fallible version of [`with_capacity`](Self::with_capacity): returns an
    /// error instead of clamping if `capacity > MAX_CAPACITY`, and
    /// surfaces allocator failures rather than aborting.
    ///
    /// # Errors
    ///
    /// - [`InternerError::CapacityExceeded`] if `capacity > MAX_CAPACITY`.
    /// - [`InternerError::AllocationFailed`] if the underlying allocator
    ///   refuses the reservation.
    pub fn try_with_capacity(capacity: usize) -> Result<Self, InternerError>
    where
        K: Eq + Hash,
    {
        Self::try_with_capacity_and_hasher(capacity, FxBuildHasher)
    }
}

impl<K, S> KeyInterner<K, S> {
    /// Maximum number of unique keys a single `KeyInterner` will hold.
    ///
    /// Chosen to bound per-instance memory at a clearly "not-usually-legit"
    /// ceiling while leaving several orders of magnitude of headroom over
    /// realistic cache-key cardinalities. Even at the cap, the interner
    /// occupies at least `MAX_CAPACITY * (size_of::<(K, u64)>() + size_of::<K>())`
    /// bytes — for `K = String` the hidden heap-allocated payloads dominate
    /// this figure, so callers exposing `KeyInterner` to untrusted input
    /// should impose a much smaller admission-control cap of their own.
    ///
    /// Derived from `isize::MAX as usize / 64` to stay well below the
    /// allocation limit on the target platform.
    pub const MAX_CAPACITY: usize = (isize::MAX as usize) / 64;

    /// Creates an empty interner with the given hash builder.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::KeyInterner;
    /// use std::collections::hash_map::RandomState;
    ///
    /// // DoS-resistant hasher for untrusted keys.
    /// let interner: KeyInterner<String, RandomState> =
    ///     KeyInterner::with_hasher(RandomState::new());
    /// assert!(interner.is_empty());
    /// ```
    #[must_use]
    pub fn with_hasher(hash_builder: S) -> Self {
        Self {
            index: HashMap::with_hasher(hash_builder),
            keys: Vec::new(),
            generation: 0,
        }
    }

    /// Creates an interner with pre-allocated capacity and a custom hash builder.
    ///
    /// `capacity` is silently clamped to [`KeyInterner::MAX_CAPACITY`],
    /// and any further allocator refusal is absorbed rather than aborting
    /// the process. Use
    /// [`try_with_capacity_and_hasher`](Self::try_with_capacity_and_hasher)
    /// to observe the clamp or allocator failure.
    #[must_use]
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self
    where
        K: Eq + Hash,
        S: BuildHasher,
    {
        // Clamp + best-effort reserve: `with_capacity` is the infallible
        // constructor, so it must never abort the process on an oversized
        // request. We first clamp to `MAX_CAPACITY` (which defends against
        // `usize::MAX` arithmetic), then use `try_reserve` so the
        // allocator is free to refuse any remaining too-large request
        // without aborting. Callers who need to observe that path
        // should use `try_with_capacity_and_hasher` instead.
        let clamped = capacity.min(Self::MAX_CAPACITY);
        let mut keys: Vec<K> = Vec::new();
        let mut index: HashMap<K, u64, S> = HashMap::with_hasher(hash_builder);
        let _ = keys.try_reserve(clamped);
        let _ = index.try_reserve(clamped);
        Self {
            index,
            keys,
            generation: 0,
        }
    }

    /// Fallible version of
    /// [`with_capacity_and_hasher`](Self::with_capacity_and_hasher).
    ///
    /// # Errors
    ///
    /// - [`InternerError::CapacityExceeded`] if `capacity > MAX_CAPACITY`.
    /// - [`InternerError::AllocationFailed`] if the allocator refuses.
    pub fn try_with_capacity_and_hasher(
        capacity: usize,
        hash_builder: S,
    ) -> Result<Self, InternerError>
    where
        K: Eq + Hash,
        S: BuildHasher,
    {
        if capacity > Self::MAX_CAPACITY {
            return Err(InternerError::CapacityExceeded);
        }
        let mut keys: Vec<K> = Vec::new();
        keys.try_reserve(capacity)?;
        let mut index: HashMap<K, u64, S> = HashMap::with_hasher(hash_builder);
        index.try_reserve(capacity)?;
        Ok(Self {
            index,
            keys,
            generation: 0,
        })
    }

    /// Returns a reference to the hash builder used by this interner.
    #[must_use]
    pub fn hasher(&self) -> &S {
        self.index.hasher()
    }

    /// Returns the current generation counter.
    ///
    /// The counter starts at `0` and is incremented on every
    /// [`clear`](Self::clear) / [`clear_shrink`](Self::clear_shrink). Callers
    /// that persist handles across a possible clear can store
    /// `(generation, handle)` pairs and reject handles whose recorded
    /// generation no longer matches the live value — this is the documented
    /// mitigation for the clear-cycle handle-reuse hazard described in the
    /// module-level [security notes](self#security).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::KeyInterner;
    ///
    /// let mut interner: KeyInterner<String> = KeyInterner::new();
    /// let gen_before = interner.generation();
    /// let h = interner.intern(&"k".to_owned());
    ///
    /// // Later, after a clear_shrink elsewhere:
    /// interner.clear_shrink();
    /// let gen_after = interner.generation();
    /// assert_ne!(gen_before, gen_after);
    ///
    /// // The stored handle is stale.
    /// let stored = (gen_before, h);
    /// assert_ne!(stored.0, gen_after);
    /// ```
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

impl<K, S> KeyInterner<K, S>
where
    K: Eq + Hash + Clone,
    S: BuildHasher,
{
    /// Returns the handle for `key`, inserting it if missing.
    ///
    /// If the key is already interned, returns the existing handle.
    /// Otherwise, assigns the next sequential handle and stores the key.
    ///
    /// # Panics
    ///
    /// Panics if the interner is already holding [`MAX_CAPACITY`]
    /// unique keys, or if the allocator refuses to grow the backing
    /// storage. Use [`try_intern`](Self::try_intern) for a non-panicking
    /// variant — callers that process untrusted keys should always prefer
    /// `try_intern`, since this method is a trivial DoS vector otherwise.
    ///
    /// [`MAX_CAPACITY`]: Self::MAX_CAPACITY
    ///
    /// # Exception safety
    ///
    /// `intern` is designed so that, assuming `K`'s `Hash` / `Eq` / `Clone`
    /// impls do not panic, a panic from `HashMap::insert` cannot leave the
    /// interner with `len(keys) > len(index)` — which would otherwise
    /// cause a subsequent `intern(&same_key)` to mint a *second* handle
    /// for the same key and permanently strand the first one. In detail:
    ///
    /// 1. Capacity is reserved via `try_reserve` up front, so neither
    ///    the `Vec::push` nor the `HashMap::insert` below can fail for
    ///    allocator reasons.
    /// 2. The key is then inserted into `index` **first**. If `K::hash`
    ///    or `K::eq` panics (a pathological impl), the `keys` vector is
    ///    untouched, so the interner retains its invariant.
    /// 3. Only after the insert returns do we push onto `keys`.
    ///
    /// Panicking `Clone` / `Drop` impls for `K` can still violate the
    /// invariant; the standard-library `HashMap` makes no stronger
    /// guarantee either.
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
    #[track_caller]
    pub fn intern(&mut self, key: &K) -> u64 {
        match self.try_intern(key) {
            Ok(id) => id,
            Err(InternerError::CapacityExceeded) => panic!(
                "KeyInterner::intern: reached MAX_CAPACITY ({}); use try_intern to handle this gracefully",
                Self::MAX_CAPACITY
            ),
            Err(InternerError::AllocationFailed(e)) => {
                panic!("KeyInterner::intern: allocation failed: {e}")
            },
        }
    }

    /// Fallible counterpart to [`intern`](Self::intern).
    ///
    /// Returns the existing handle if `key` has already been interned;
    /// otherwise assigns the next sequential handle and stores the key.
    /// Returns [`InternerError::CapacityExceeded`] if the interner is
    /// already at [`MAX_CAPACITY`], and [`InternerError::AllocationFailed`]
    /// if the allocator refuses growth. **Preferred over `intern` whenever
    /// keys can come from untrusted input.**
    ///
    /// [`MAX_CAPACITY`]: Self::MAX_CAPACITY
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::KeyInterner;
    ///
    /// let mut interner: KeyInterner<String> = KeyInterner::new();
    /// let h = interner.try_intern(&"k".to_owned()).unwrap();
    /// assert_eq!(h, 0);
    /// // Idempotent just like `intern`.
    /// assert_eq!(interner.try_intern(&"k".to_owned()).unwrap(), h);
    /// ```
    pub fn try_intern(&mut self, key: &K) -> Result<u64, InternerError> {
        if let Some(&id) = self.index.get(key) {
            return Ok(id);
        }
        if self.keys.len() >= Self::MAX_CAPACITY {
            return Err(InternerError::CapacityExceeded);
        }
        // Reserve up front so neither push nor insert can fail for
        // allocator reasons below. This is the load-bearing precondition
        // for the exception-safety argument documented on `intern`.
        self.keys.try_reserve(1)?;
        self.index.try_reserve(1)?;

        let id = self.keys.len() as u64;
        // Clone twice up front so any panic from `K::clone` happens
        // before we mutate either container.
        let k_for_index = key.clone();
        let k_for_keys = key.clone();

        // Insert into the index FIRST. If `K::hash` / `K::eq` panics
        // here, the vector is untouched and the interner's
        // len(keys) == len(index) invariant holds.
        self.index.insert(k_for_index, id);
        // Now push. After `try_reserve`, this is infallible for the
        // `Vec` itself; `Drop` of the pushed value cannot run on push,
        // so no user code executes between these two statements.
        self.keys.push(k_for_keys);

        Ok(id)
    }
}

impl<K, S> KeyInterner<K, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
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
    #[must_use]
    pub fn get_handle(&self, key: &K) -> Option<u64> {
        self.get_handle_borrowed(key)
    }

    /// Returns the handle for a borrowed form of `K` if it exists.
    ///
    /// This enables allocation-free lookups for owned key types like `String`
    /// by querying with `&str`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::KeyInterner;
    ///
    /// let mut interner: KeyInterner<String> = KeyInterner::new();
    /// interner.intern(&"hello".to_string());
    ///
    /// // Lookup by &str without allocating a String
    /// assert_eq!(interner.get_handle_borrowed("hello"), Some(0));
    /// assert_eq!(interner.get_handle_borrowed("missing"), None);
    /// ```
    #[must_use]
    pub fn get_handle_borrowed<Q>(&self, key: &Q) -> Option<u64>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.index.get(key).copied()
    }

    /// Shrinks internal storage to fit current length.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::KeyInterner;
    ///
    /// let mut interner = KeyInterner::new();
    /// for i in 0..100u32 {
    ///     interner.intern(&i);
    /// }
    /// interner.clear();
    /// interner.shrink_to_fit();
    /// ```
    pub fn shrink_to_fit(&mut self) {
        self.index.shrink_to_fit();
        self.keys.shrink_to_fit();
    }

    /// Clears all interned keys and shrinks internal storage.
    ///
    /// After calling this, **all previously returned handles become
    /// invalid** and the [`generation`](Self::generation) counter is
    /// bumped. Callers holding handles across the clear should compare
    /// `generation()` to detect staleness — see the module-level
    /// [security notes](self#security).
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
        self.clear();
        self.shrink_to_fit();
    }
}

impl<K, S> KeyInterner<K, S> {
    /// Resolves a handle to its original key.
    ///
    /// Returns `None` if the handle is out of bounds.
    ///
    /// **Note:** handles are not capability-safe across
    /// [`clear`](Self::clear) cycles or across distinct `KeyInterner`
    /// instances. See the module-level [security notes](self#security).
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
    #[must_use]
    pub fn resolve(&self, handle: u64) -> Option<&K> {
        let index = usize::try_from(handle).ok()?;
        self.keys.get(index)
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
    #[must_use]
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
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Clears all interned keys and bumps [`generation`](Self::generation).
    ///
    /// After calling this, all previously returned handles become invalid.
    /// The internal allocations are retained; use
    /// [`clear_shrink`](Self::clear_shrink) to also release spare capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::KeyInterner;
    ///
    /// let mut interner = KeyInterner::new();
    /// interner.intern(&"key");
    /// assert!(!interner.is_empty());
    ///
    /// let gen_before = interner.generation();
    /// interner.clear();
    /// assert!(interner.is_empty());
    /// assert_ne!(gen_before, interner.generation());
    /// ```
    pub fn clear(&mut self) {
        self.index.clear();
        self.keys.clear();
        // Use wrapping_add defensively: a caller who clears 2^64 times
        // has bigger problems than generation-counter rollover, but we
        // refuse to panic in release (overflow-checks = false) either
        // way.
        self.generation = self.generation.wrapping_add(1);
    }

    /// Returns an approximate memory footprint in bytes.
    ///
    /// Uses saturating arithmetic internally so that pathological
    /// capacities cannot under-report the footprint via `usize` overflow.
    /// Under-reporting here would let an attacker bypass any admission-
    /// control check that consults `approx_bytes`.
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
    #[must_use]
    pub fn approx_bytes(&self) -> usize {
        let entry_size = std::mem::size_of::<(K, u64)>();
        let key_size = std::mem::size_of::<K>();
        let index_bytes = self.index.capacity().saturating_mul(entry_size);
        let keys_bytes = self.keys.capacity().saturating_mul(key_size);
        std::mem::size_of::<Self>()
            .saturating_add(index_bytes)
            .saturating_add(keys_bytes)
    }

    /// Returns an iterator over (handle, key) pairs in insertion order.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::KeyInterner;
    ///
    /// let mut interner = KeyInterner::new();
    /// interner.intern(&"a");
    /// interner.intern(&"b");
    ///
    /// let pairs: Vec<_> = interner.iter().collect();
    /// assert_eq!(pairs, vec![(0, &"a"), (1, &"b")]);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = (u64, &K)> + '_ {
        self.keys.iter().enumerate().map(|(i, k)| (i as u64, k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_interner_basic_flow() {
        let mut interner: KeyInterner<String> = KeyInterner::new();
        assert!(interner.is_empty());
        let a = interner.intern(&"a".to_owned());
        let b = interner.intern(&"b".to_owned());
        let a2 = interner.intern(&"a".to_owned());
        assert_eq!(a, a2);
        assert_ne!(a, b);
        assert_eq!(interner.len(), 2);
        assert_eq!(interner.get_handle_borrowed("b"), Some(b));
        assert_eq!(interner.resolve(a).map(String::as_str), Some("a"));
    }

    #[test]
    fn key_interner_iter() {
        let mut interner: KeyInterner<String> = KeyInterner::new();
        interner.intern(&"x".to_owned());
        interner.intern(&"y".to_owned());

        let mut pairs = interner.iter();
        assert_eq!(pairs.next(), Some((0, &"x".to_owned())));
        assert_eq!(pairs.next(), Some((1, &"y".to_owned())));
        assert_eq!(pairs.next(), None);
    }

    // =========================================================================
    // Security hardening tests
    // =========================================================================

    #[test]
    fn debug_impl_does_not_leak_keys() {
        // Guard: the manual `Debug` impl must not dump interned keys.
        // Regressing this is an information-disclosure bug.
        let mut interner: KeyInterner<String> = KeyInterner::new();
        interner.intern(&"sensitive_token_abc123".to_owned());
        interner.intern(&"another_secret".to_owned());
        let formatted = format!("{interner:?}");
        assert!(
            !formatted.contains("sensitive_token_abc123"),
            "Debug must not include key contents; got: {formatted}"
        );
        assert!(
            !formatted.contains("another_secret"),
            "Debug must not include key contents; got: {formatted}"
        );
        // But it should still be useful for diagnostics.
        assert!(formatted.contains("len"), "Debug missing len: {formatted}");
        assert!(
            formatted.contains("generation"),
            "Debug missing generation: {formatted}"
        );
    }

    #[test]
    fn try_with_capacity_rejects_oversized() {
        // Cap enforcement: callers passing untrusted capacities should get
        // an error, not a process-wide abort.
        let err = KeyInterner::<String>::try_with_capacity(
            KeyInterner::<String>::MAX_CAPACITY.saturating_add(1),
        )
        .unwrap_err();
        assert_eq!(err, InternerError::CapacityExceeded);
    }

    // Skipped under Miri: the test intentionally feeds `usize::MAX` to
    // exercise the clamp-and-swallow-allocator-failure path. The real
    // allocator refuses the clamped `MAX_CAPACITY` request and
    // `try_reserve` converts that into an `Err` we drop. Miri instead
    // raises a "resource exhaustion" error that aborts the test, so the
    // invariant we're pinning (no process-wide abort) can't be
    // meaningfully validated under Miri.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn with_capacity_clamps_silently() {
        // `with_capacity` is documented to clamp rather than panic, so
        // configuration-derived capacities can't crash the process.
        let interner = KeyInterner::<u32>::with_capacity(usize::MAX);
        assert!(interner.is_empty());
        // And the clamped allocation shouldn't exceed MAX_CAPACITY.
        assert!(interner.approx_bytes() < usize::MAX / 2);
    }

    #[test]
    fn generation_bumps_on_clear() {
        // Callers use `generation` to detect the clear-cycle handle-reuse
        // hazard. This must strictly increase on every clear.
        let mut interner: KeyInterner<String> = KeyInterner::new();
        let g0 = interner.generation();
        interner.intern(&"a".to_owned());
        assert_eq!(interner.generation(), g0, "intern must not bump generation");
        interner.clear();
        let g1 = interner.generation();
        assert_ne!(g0, g1, "clear must bump generation");
        interner.clear_shrink();
        let g2 = interner.generation();
        assert_ne!(g1, g2, "clear_shrink must bump generation");
    }

    #[test]
    fn handles_reset_to_zero_after_clear_documented_behavior() {
        // Regression test for the documented handle-reuse hazard: after
        // clear, handle 0 is legitimately reused. Callers must guard
        // via `generation()`; this test pins the hazard so future
        // refactors don't silently remove the invariant callers were
        // promised.
        let mut interner: KeyInterner<String> = KeyInterner::new();
        let h_alice = interner.intern(&"alice".to_owned());
        assert_eq!(h_alice, 0);
        let g_alice = interner.generation();

        interner.clear();
        let h_bob = interner.intern(&"bob".to_owned());
        assert_eq!(h_bob, 0, "handle 0 must be reused after clear");
        assert_ne!(g_alice, interner.generation(), "generation must differ");
    }

    #[test]
    fn try_intern_happy_path_and_idempotent() {
        let mut interner: KeyInterner<u32> = KeyInterner::new();
        for i in 0..64u32 {
            assert!(interner.try_intern(&i).is_ok());
        }
        // Idempotent.
        assert_eq!(interner.try_intern(&0u32).unwrap(), 0);
    }

    #[test]
    fn custom_hasher_compiles_and_works() {
        // Ensures the generic hasher path is actually usable, which is
        // the mitigation for the HashDoS hazard.
        use std::collections::hash_map::RandomState;
        let mut interner: KeyInterner<String, RandomState> =
            KeyInterner::with_hasher(RandomState::new());
        let h = interner.intern(&"k".to_owned());
        assert_eq!(h, 0);
        assert_eq!(
            interner.get_handle_borrowed("k"),
            Some(0),
            "custom hasher must still support borrowed lookups"
        );
    }

    #[test]
    fn approx_bytes_saturates_on_overflow() {
        // Under-reporting here would let an attacker slip past an
        // admission-control check sized in bytes.
        let interner = KeyInterner::<u8>::new();
        let bytes = interner.approx_bytes();
        // Just asserting it doesn't panic / overflow with a realistic
        // input; the saturating_{mul,add} calls prevent wraparound even
        // for pathological capacities.
        assert!(bytes >= std::mem::size_of::<KeyInterner<u8>>());
    }

    #[test]
    fn clone_propagates_generation_and_contents() {
        let mut interner: KeyInterner<String> = KeyInterner::new();
        interner.intern(&"a".to_owned());
        interner.clear();
        interner.intern(&"b".to_owned());

        let cloned = interner.clone();
        assert_eq!(cloned.generation(), interner.generation());
        assert_eq!(cloned.len(), interner.len());
        assert_eq!(cloned.get_handle_borrowed("b"), Some(0));
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

        /// Property: generation strictly increases across clear cycles.
        /// This backs the module-level promise that callers can use
        /// `generation()` to detect handle-reuse after a clear.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_generation_strictly_increases_on_clear(
            rounds in 1usize..20
        ) {
            let mut interner: KeyInterner<u32> = KeyInterner::new();
            let mut last = interner.generation();
            for i in 0..rounds {
                interner.intern(&(i as u32));
                interner.clear();
                let now = interner.generation();
                prop_assert_ne!(now, last);
                last = now;
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
