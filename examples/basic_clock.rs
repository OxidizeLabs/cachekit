//! Example demonstrating the Clock cache policy.
//!
//! Clock approximates LRU by giving referenced entries a second chance during
//! eviction instead of moving them through a linked list on every hit.
//!
//! Run with: cargo run --example basic_clock

use cachekit::policy::clock::ClockCache;
use cachekit::traits::Cache;

fn main() {
    let mut cache: ClockCache<u32, &str> = ClockCache::new(3);

    cache.insert(1, "alpha");
    cache.insert(2, "beta");
    cache.insert(3, "gamma");

    // Give key 1 a second chance before the next eviction sweep.
    cache.get(&1);
    cache.insert(4, "delta");

    println!("contains 1? {}", cache.contains(&1));
    println!("contains 2? {}", cache.contains(&2));
    println!("len: {}", cache.len());
}

// Expected output:
// contains 1? true
// contains 2? false
// len: 3
//
// Explanation: key 1 had its reference bit set before the cache filled, so the
// sweep clears that bit and skips it, then evicts key 2 as the first
// unreferenced entry.
