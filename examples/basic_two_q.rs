//! Example demonstrating the 2Q (Two-Queue) cache policy.
//!
//! 2Q uses two queues: a probation queue for new entries and a protected queue
//! for entries that have been accessed more than once. This provides scan
//! resistance similar to LRU-K.
//!
//! Run with: cargo run --example basic_two_q

use cachekit::builder::{CacheBuilder, CachePolicy};

fn main() {
    println!("=== 2Q Cache Example ===\n");

    // Create a 2Q cache with capacity 10 and 25% probation queue
    // - Probation queue: ~2-3 slots (for new items, FIFO eviction)
    // - Protected queue: ~7-8 slots (for frequently accessed items, LRU eviction)
    let mut cache = CacheBuilder::new(10).build::<u64, String>(CachePolicy::TwoQ {
        probation_frac: 0.25,
    });

    println!("Created 2Q cache: capacity={}\n", cache.capacity());

    // Insert items (all start in probation queue)
    for i in 1..=5 {
        cache.insert(i, format!("value-{}", i));
    }
    println!("Inserted keys 1-5 (all in probation queue)");
    println!("  len: {}", cache.len());

    // Access keys 1 and 2 to promote them from probation to protected
    cache.get(&1);
    cache.get(&2);
    println!("\nAccessed keys 1 and 2 (promoted to protected queue)");

    // Insert more items to fill the cache
    for i in 6..=10 {
        cache.insert(i, format!("value-{}", i));
    }
    println!("Inserted keys 6-10");
    println!("  len: {}", cache.len());

    // Now insert more items - probation queue items should be evicted first
    cache.insert(11, "value-11".to_string());
    cache.insert(12, "value-12".to_string());

    println!("\nAfter inserting keys 11, 12:");
    println!(
        "  contains 1? {} (was promoted to protected)",
        cache.contains(&1)
    );
    println!(
        "  contains 2? {} (was promoted to protected)",
        cache.contains(&2)
    );
    println!("  len: {}", cache.len());

    // Demonstrate scan resistance
    println!("\n=== Scan Resistance Demo ===\n");

    let mut cache = CacheBuilder::new(10).build::<u64, String>(CachePolicy::TwoQ {
        probation_frac: 0.3, // 30% probation
    });

    // Insert "hot" items and access them to promote to protected queue
    cache.insert(1, "hot-1".to_string());
    cache.insert(2, "hot-2".to_string());
    cache.insert(3, "hot-3".to_string());

    // Multiple accesses promote items to protected queue
    cache.get(&1);
    cache.get(&2);
    cache.get(&3);
    println!("Inserted and promoted hot items: 1, 2, 3");

    // Simulate a scan with one-time accesses (stay in probation, get evicted first)
    println!("Simulating scan with items 100-120...");
    for i in 100..=120 {
        cache.insert(i, format!("scan-{}", i));
    }

    // Hot items should survive the scan because they're in protected queue
    println!("\nAfter scan (21 one-time insertions):");
    println!("  contains hot-1? {}", cache.contains(&1));
    println!("  contains hot-2? {}", cache.contains(&2));
    println!("  contains hot-3? {}", cache.contains(&3));
    println!("  len: {}", cache.len());

    // Show that scan items were evicted
    let scan_items_remaining: Vec<_> = (100..=120).filter(|&i| cache.contains(&i)).collect();
    println!("  scan items remaining: {:?}", scan_items_remaining);
}

// Expected output:
// === 2Q Cache Example ===
//
// Created 2Q cache: capacity=10
//
// Inserted keys 1-5 (all in probation queue)
//   len: 5
//
// Accessed keys 1 and 2 (promoted to protected queue)
// Inserted keys 6-10
//   len: 10
//
// After inserting keys 11, 12:
//   contains 1? true (was promoted to protected)
//   contains 2? true (was promoted to protected)
//   len: 10
//
// === Scan Resistance Demo ===
//
// Inserted and promoted hot items: 1, 2, 3
// Simulating scan with items 100-120...
//
// After scan (21 one-time insertions):
//   contains hot-1? true
//   contains hot-2? true
//   contains hot-3? true
//   len: 10
//   scan items remaining: [last ~7 scan items]
