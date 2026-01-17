//! DHAT heap profiler for cachekit.
//!
//! Run with: cargo run --bin dhat_profile --release --features dhat-heap
//! View results: Open dhat-heap.json in <https://nnethercote.github.io/dh_view/dh_view.html>

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use std::sync::Arc;

use cachekit::policy::fifo::FifoCache;
use cachekit::policy::lfu::LfuCache;
use cachekit::policy::lru::LruCore;
use cachekit::policy::lru_k::LrukCache;
use cachekit::policy::two_q::TwoQCore;
use cachekit::traits::CoreCache;

/// Simple XorShift64 RNG for deterministic workloads.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / (u64::MAX as f64);
        (self.next_u64() as f64) * SCALE
    }
}

/// Run a hotset workload: 90% of accesses hit 10% of keys.
fn hotset_workload<C: CoreCache<u64, Arc<u64>>>(
    cache: &mut C,
    operations: usize,
    universe: u64,
    seed: u64,
) {
    let mut rng = XorShift64::new(seed);
    let hot_size = (universe as f64 * 0.1) as u64;

    for _ in 0..operations {
        let key = if rng.next_f64() < 0.9 {
            // Hot key (10% of universe, 90% of accesses)
            rng.next_u64() % hot_size
        } else {
            // Cold key
            hot_size + (rng.next_u64() % (universe - hot_size))
        };

        if cache.get(&key).is_none() {
            let _ = cache.insert(key, Arc::new(key));
        }
    }
}

/// Run a scan workload: sequential access pattern.
fn scan_workload<C: CoreCache<u64, Arc<u64>>>(cache: &mut C, operations: usize, universe: u64) {
    for i in 0..operations {
        let key = (i as u64) % universe;
        if cache.get(&key).is_none() {
            let _ = cache.insert(key, Arc::new(key));
        }
    }
}

/// Run eviction churn: insert more items than capacity.
fn eviction_churn<C: CoreCache<u64, Arc<u64>>>(cache: &mut C, operations: usize) {
    for i in 0..operations {
        let _ = cache.insert(i as u64, Arc::new(i as u64));
    }
}

fn profile_lru() {
    println!("=== Profiling LRU ===");
    let capacity = 4096;
    let operations = 100_000;
    let universe = 16_384;

    let mut cache = LruCore::new(capacity);

    // Warm up
    for i in 0..capacity as u64 {
        cache.insert(i, Arc::new(i));
    }

    // Hotset workload
    hotset_workload(&mut cache, operations, universe, 42);

    // Scan workload
    scan_workload(&mut cache, operations / 2, universe);

    // Eviction churn
    eviction_churn(&mut cache, operations / 4);

    println!("  Final size: {}", cache.len());
}

fn profile_lfu() {
    println!("=== Profiling LFU ===");
    let capacity = 4096;
    let operations = 100_000;
    let universe = 16_384;

    let mut cache = LfuCache::new(capacity);

    for i in 0..capacity as u64 {
        cache.insert(i, Arc::new(i));
    }

    hotset_workload(&mut cache, operations, universe, 42);
    scan_workload(&mut cache, operations / 2, universe);
    eviction_churn(&mut cache, operations / 4);

    println!("  Final size: {}", cache.len());
}

fn profile_fifo() {
    println!("=== Profiling FIFO ===");
    let capacity = 4096;
    let operations = 100_000;
    let universe = 16_384;

    let mut cache = FifoCache::new(capacity);

    for i in 0..capacity as u64 {
        cache.insert(i, Arc::new(i));
    }

    hotset_workload(&mut cache, operations, universe, 42);
    scan_workload(&mut cache, operations / 2, universe);
    eviction_churn(&mut cache, operations / 4);

    println!("  Final size: {}", cache.len());
}

fn profile_lru_k() {
    println!("=== Profiling LRU-K ===");
    let capacity = 4096;
    let operations = 100_000;
    let universe = 16_384;

    let mut cache = LrukCache::new(capacity);

    for i in 0..capacity as u64 {
        cache.insert(i, Arc::new(i));
    }

    hotset_workload(&mut cache, operations, universe, 42);
    scan_workload(&mut cache, operations / 2, universe);
    eviction_churn(&mut cache, operations / 4);

    println!("  Final size: {}", cache.len());
}

fn profile_two_q() {
    println!("=== Profiling 2Q ===");
    let capacity = 4096;
    let operations = 100_000;
    let universe = 16_384;

    // TwoQCore::new takes (protected_cap, a1_frac)
    // a1_frac is the fraction of capacity for A1in queue (typically 0.25)
    let mut cache = TwoQCore::new(capacity, 0.25);

    for i in 0..capacity as u64 {
        cache.insert(i, Arc::new(i));
    }

    hotset_workload(&mut cache, operations, universe, 42);
    scan_workload(&mut cache, operations / 2, universe);
    eviction_churn(&mut cache, operations / 4);

    println!("  Final size: {}", cache.len());
}

fn main() {
    let _profiler = dhat::Profiler::new_heap();

    println!("CacheKit DHAT Heap Profiling");
    println!("============================\n");

    profile_lru();
    profile_lfu();
    profile_fifo();
    profile_lru_k();
    profile_two_q();

    println!("\n============================");
    println!("Profiling complete!");
    println!(
        "View results: Open dhat-heap.json in <https://nnethercote.github.io/dh_view/dh_view.html>"
    );
}
