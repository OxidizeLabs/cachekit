//! Example demonstrating the Clock with Adaptive Replacement (CAR) policy.
//!
//! CAR combines Clock-style reference bits with ARC-style adaptation, keeping a
//! recent ring, a frequent ring, and ghost history for evicted keys.
//!
//! Run with: cargo run --example basic_car --features policy-car

use cachekit::policy::car::CarCore;
use cachekit::traits::Cache;

fn main() {
    let mut ghost_demo: CarCore<&str, &str> = CarCore::new(2);
    ghost_demo.insert("a", "alpha");
    ghost_demo.insert("b", "beta");
    ghost_demo.insert("c", "gamma");

    println!(
        "ghost demo: recent={}, ghost_recent={}",
        ghost_demo.recent_len(),
        ghost_demo.ghost_recent_len()
    );

    let mut cache: CarCore<&str, &str> = CarCore::new(3);

    cache.insert("a", "alpha");
    cache.insert("b", "beta");
    cache.insert("c", "gamma");

    println!(
        "before sweep: recent={}, frequent={}",
        cache.recent_len(),
        cache.frequent_len()
    );

    // Mark two entries as referenced. CAR turns that cheap bit-setting into
    // promotions while sweeping for a victim on the next insertion.
    cache.get(&"a");
    cache.get(&"b");
    cache.insert("d", "delta");

    println!(
        "after sweep: recent={}, frequent={}",
        cache.recent_len(),
        cache.frequent_len()
    );
    println!("ghost recent entries: {}", cache.ghost_recent_len());
    println!("target recent size: {}", cache.target_recent_size());
}

// Expected output:
// ghost demo: recent=2, ghost_recent=1
// before sweep: recent=3, frequent=0
// after sweep: recent=2, frequent=1
// ghost recent entries: 0
// target recent size: 1
//
// Explanation: the first section shows CAR recording a recent eviction in ghost
// history. The second section shows Clock-style hits becoming a frequent-ring
// promotion during the next eviction sweep.
