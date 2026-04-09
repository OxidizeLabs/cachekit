//! Example demonstrating the Clock-PRO cache policy.
//!
//! Clock-PRO starts new entries as cold pages, tracks evicted cold keys in a
//! ghost ring, and promotes ghost hits back into the cache as hot pages.
//!
//! Run with: cargo run --example basic_clock_pro --features policy-clock-pro

use cachekit::policy::clock_pro::ClockProCache;
use cachekit::traits::Cache;

fn main() {
    let mut cache: ClockProCache<&str, &str> = ClockProCache::with_ghost_capacity(2, 4);

    cache.insert("a", "alpha");
    cache.insert("b", "beta");
    cache.insert("c", "gamma");

    println!("contains a after eviction? {}", cache.contains(&"a"));
    println!("ghost entries after eviction: {}", cache.ghost_count());

    // Re-inserting a ghost key promotes it directly to hot status.
    cache.insert("a", "alpha-again");

    println!("contains a after ghost hit? {}", cache.contains(&"a"));
    println!("hot pages: {}", cache.hot_count());
    println!("cold pages: {}", cache.cold_count());
}

// Expected output:
// contains a after eviction? false
// ghost entries after eviction: 1
// contains a after ghost hit? true
// hot pages: 1
// cold pages: 1
//
// Explanation: after key "a" is evicted as a cold page, Clock-PRO keeps only its
// key in the ghost ring. Re-inserting "a" is treated as a ghost hit, so the new
// resident page is admitted as hot instead of cold.
