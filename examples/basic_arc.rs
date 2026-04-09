//! Example demonstrating the Adaptive Replacement Cache (ARC) policy.
//!
//! ARC keeps recently seen entries in T1, frequently reused entries in T2, and
//! ghost history for recently evicted keys to adapt between recency and
//! frequency automatically.
//!
//! Run with: cargo run --example basic_arc --features policy-arc

use cachekit::policy::arc::ArcCore;
use cachekit::traits::Cache;

fn main() {
    let mut cache: ArcCore<&str, &str> = ArcCore::new(2);

    cache.insert("a", "alpha");
    cache.insert("b", "beta");
    println!(
        "after inserts: T1={}, T2={}",
        cache.t1_len(),
        cache.t2_len()
    );

    // Re-access "a" so it moves from T1 (recent once) to T2 (frequent).
    cache.get(&"a");
    println!("after get(a): T1={}, T2={}", cache.t1_len(), cache.t2_len());

    let mut ghost_demo: ArcCore<&str, &str> = ArcCore::new(2);
    ghost_demo.insert("a", "alpha");
    ghost_demo.insert("b", "beta");
    ghost_demo.insert("c", "gamma");

    println!("ghost contains a? {}", ghost_demo.contains(&"a"));
    println!("ghost B1 entries: {}", ghost_demo.b1_len());
    println!("ghost B2 entries: {}", ghost_demo.b2_len());
    println!("adaptation p: {}", ghost_demo.p_value());
}

// Expected output:
// after inserts: T1=2, T2=0
// after get(a): T1=1, T2=1
// ghost contains a? false
// ghost B1 entries: 1
// ghost B2 entries: 0
// adaptation p: 1
//
// Explanation: the first half shows ARC promoting a reused key from T1 into T2.
// The second half shows a pure recency eviction: inserting "c" into a full cache
// pushes the oldest recent-once key into the B1 ghost list.
