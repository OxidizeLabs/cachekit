//! High-performance cache primitives with pluggable eviction policies.
//!
//! `cachekit` provides a trait-based cache framework with 17 eviction policies,
//! arena-backed data structures, and optional metrics — all designed for
//! allocation-free hot paths and predictable tail latency.
//!
//! # Quick Start
//!
//! The fastest way to get a cache is through the [`builder`] module:
//!
//! ```
//! use cachekit::builder::{CacheBuilder, CachePolicy};
//!
//! let mut cache = CacheBuilder::new(1000).build::<u64, String>(CachePolicy::Lru);
//! cache.insert(1, "hello".to_string());
//! assert_eq!(cache.get(&1), Some(&"hello".to_string()));
//! ```
//!
//! For direct access to policy-specific APIs, use the concrete types:
//!
//! ```
//! use cachekit::policy::lru_k::LrukCache;
//! use cachekit::traits::{CoreCache, LrukCacheTrait};
//!
//! let mut cache = LrukCache::with_k(1000, 2);
//! cache.insert(42, "value");
//! cache.get(&42); // second access — now scan-resistant
//! assert!(cache.k_distance(&42).is_some());
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                          cachekit                                    │
//! │                                                                      │
//! │   traits        Trait hierarchy (ReadOnlyCache → CoreCache → …)      │
//! │   builder       Unified CacheBuilder + Cache<K,V> wrapper            │
//! │   policy        17 eviction policies behind feature flags            │
//! │   ds            Arena, ring buffer, intrusive list, ghost list, …    │
//! │   store         Storage backends (HashMap, slab, weighted)           │
//! │   metrics       Hit/miss counters and snapshots (feature-gated)      │
//! │   error         ConfigError and InvariantError types                 │
//! └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! **Policy ↔ Storage separation.** Policies only manage metadata and eviction
//! ordering; the underlying storage is a separate concern. This lets each
//! policy use the most cache-friendly layout (contiguous arenas, ring buffers,
//! frequency buckets) without coupling to a single map implementation.
//!
//! # Trait Hierarchy
//!
//! All caches implement [`traits::CoreCache`], which extends
//! [`traits::ReadOnlyCache`]. Policy-specific behaviour is expressed through
//! additional traits:
//!
//! | Trait | Extends | Purpose |
//! |---|---|---|
//! | [`ReadOnlyCache`](traits::ReadOnlyCache) | — | `contains`, `len`, `capacity` (no side effects) |
//! | [`CoreCache`](traits::CoreCache) | `ReadOnlyCache` | `insert`, `get`, `clear` |
//! | [`MutableCache`](traits::MutableCache) | `CoreCache` | `remove` (not available on FIFO) |
//! | [`FifoCacheTrait`](traits::FifoCacheTrait) | `CoreCache` | `pop_oldest`, `age_rank` |
//! | [`LruCacheTrait`](traits::LruCacheTrait) | `MutableCache` | `pop_lru`, `touch`, `recency_rank` |
//! | [`LfuCacheTrait`](traits::LfuCacheTrait) | `MutableCache` | `pop_lfu`, `frequency` |
//! | [`LrukCacheTrait`](traits::LrukCacheTrait) | `MutableCache` | `pop_lru_k`, `k_distance` |
//!
//! Write generic code against the trait you need:
//!
//! ```
//! use cachekit::traits::{CoreCache, ReadOnlyCache};
//!
//! fn utilization<K, V, C: ReadOnlyCache<K, V>>(cache: &C) -> f64 {
//!     cache.len() as f64 / cache.capacity() as f64
//! }
//! ```
//!
//! # Eviction Policies
//!
//! Each policy is gated behind a `policy-*` feature flag. The `default` feature
//! set enables S3-FIFO, LRU, Fast-LRU, LRU-K, and Clock. Enable `policy-all`
//! for everything.
//!
//! | Policy | Feature | Eviction basis | Complexity | Best for |
//! |---|---|---|---|---|
//! | FIFO | `policy-fifo` | Insertion order | O(1) | Streaming, predictable eviction |
//! | LRU | `policy-lru` | Recency | O(1) | Temporal locality |
//! | Fast-LRU | `policy-fast-lru` | Recency (no Arc) | O(1) | Max single-threaded throughput |
//! | LRU-K | `policy-lru-k` | K-th access time | O(1) | Scan resistance (databases) |
//! | LFU | `policy-lfu` | Frequency (buckets) | O(1) | Stable hot spots |
//! | Heap-LFU | `policy-heap-lfu` | Frequency (heap) | O(log n) | Large caches, frequent eviction |
//! | 2Q | `policy-two-q` | Two-queue promotion | O(1) | Mixed workloads |
//! | S3-FIFO | `policy-s3-fifo` | Three-queue FIFO | O(1) | CDN, scan-heavy workloads |
//! | ARC | `policy-arc` | Adaptive recency/freq | O(1) | Unknown/changing workloads |
//! | CAR | `policy-car` | Clock + ARC | O(1) | Adaptive with low overhead |
//! | SLRU | `policy-slru` | Segmented LRU | O(1) | Buffer pools, scans |
//! | Clock | `policy-clock` | Reference bit | O(1) amortised | Low-overhead LRU approx |
//! | Clock-PRO | `policy-clock-pro` | Adaptive clock | O(1) amortised | Scan-resistant clock |
//! | NRU | `policy-nru` | Not-recently-used bit | O(n) worst case | Small caches, coarse recency |
//! | LIFO | `policy-lifo` | Reverse insertion | O(1) | Stack-like / undo buffers |
//! | MRU | `policy-mru` | Most recent access | O(1) | Cyclic / sequential scans |
//! | MFU | `policy-mfu` | Highest frequency | O(1) | Niche inverse-frequency |
//! | Random | `policy-random` | Uniform random | O(1) | Baselines |
//!
//! # Feature Flags
//!
//! | Flag | Default | Description |
//! |---|---|---|
//! | `policy-s3-fifo` | yes | S3-FIFO policy |
//! | `policy-lru` | yes | LRU policy |
//! | `policy-fast-lru` | yes | Fast-LRU (no Arc wrapping) |
//! | `policy-lru-k` | yes | LRU-K policy |
//! | `policy-clock` | yes | Clock (second-chance) policy |
//! | `policy-all` | no | Enable every policy |
//! | `metrics` | no | Hit/miss counters, [`metrics::snapshot::CacheMetricsSnapshot`] |
//! | `concurrency` | no | `parking_lot`-backed concurrent data structures |
//!
//! Disable defaults and cherry-pick for smaller builds:
//!
//! ```toml
//! [dependencies]
//! cachekit = { version = "0.4", default-features = false, features = ["policy-s3-fifo"] }
//! ```
//!
//! # Data Structures (`ds`)
//!
//! The [`ds`] module exposes the building blocks used by policies:
//!
//! - [`ClockRing`](ds::ClockRing) — fixed-capacity ring buffer with reference bits
//! - [`SlotArena`](ds::SlotArena) — index-addressed arena with O(1) alloc/free
//! - [`IntrusiveList`](ds::IntrusiveList) — doubly-linked list with arena-backed nodes
//! - [`FrequencyBuckets`](ds::FrequencyBuckets) — O(1) frequency counter buckets
//! - [`GhostList`](ds::GhostList) — bounded evicted-key history
//! - [`LazyMinHeap`](ds::LazyMinHeap) — lazy-deletion min-heap
//! - [`FixedHistory`](ds::FixedHistory) — fixed-size circular access history
//! - [`KeyInterner`](ds::KeyInterner) — deduplicating key storage
//! - [`ShardSelector`](ds::ShardSelector) — deterministic shard routing
//!
//! All structures pre-allocate and reuse memory; none allocate on the hot path.
//!
//! # Thread Safety
//!
//! Individual caches are **not** thread-safe by default. Options:
//!
//! 1. Wrap in `Arc<RwLock<Cache>>` for coarse-grained sharing.
//! 2. Enable the `concurrency` feature for `parking_lot`-backed concurrent
//!    variants ([`ConcurrentClockRing`](ds::clock_ring::ConcurrentClockRing),
//!    [`ConcurrentSlotArena`](ds::slot_arena::ConcurrentSlotArena), etc.).
//! 3. Use the [`ConcurrentCache`](traits::ConcurrentCache) marker trait to
//!    constrain generic code to thread-safe implementations.
//!
//! # Metrics
//!
//! Enable the `metrics` feature to get lightweight hit/miss/eviction counters.
//! Detailed snapshots are available via
//! [`CacheMetricsSnapshot`](metrics::snapshot::CacheMetricsSnapshot).
//!
//! # Error Handling
//!
//! Fallible constructors (e.g.
//! [`S3FifoCache::try_with_ratios`](policy::s3_fifo::S3FifoCache::try_with_ratios))
//! return [`ConfigError`](error::ConfigError) for invalid parameters. Debug-only
//! invariant checks produce [`InvariantError`](error::InvariantError).
//!
//! # Choosing a Policy
//!
//! ```text
//!                    ┌─ temporal locality? ──► LRU / Fast-LRU
//!                    │
//!  What does your ───┼─ frequency matters? ──► LFU / Heap-LFU
//!  workload look     │
//!  like?             ├─ scan-heavy / mixed? ─► S3-FIFO / 2Q / LRU-K / ARC
//!                    │
//!                    ├─ unknown / changing? ──► ARC / Clock-PRO
//!                    │
//!                    └─ simple / streaming? ──► FIFO / Clock
//! ```

pub mod ds;
pub mod error;
pub mod policy;
pub mod store;

#[cfg(feature = "metrics")]
pub mod metrics;

pub mod builder;
pub mod prelude;
pub mod traits;
