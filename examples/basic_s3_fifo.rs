//! Example demonstrating the S3-FIFO (Simple, Scalable, Scan-resistant FIFO) cache policy.
//!
//! S3-FIFO uses three FIFO queues:
//! - Small: for new items (filters one-hit wonders)
//! - Main: for items that were accessed in Small (protected)
//! - Ghost: tracks recently evicted keys for smarter admission
//!
//! Run with: cargo run --example basic_s3_fifo

use cachekit::builder::{CacheBuilder, CachePolicy};

fn main() {
    println!("=== S3-FIFO Cache Example ===\n");

    // Create an S3-FIFO cache with capacity 10
    // - Small queue: 10% of capacity (1 slot)
    // - Ghost list: 90% of capacity (9 slots)
    let mut cache = CacheBuilder::new(10).build::<u64, String>(CachePolicy::S3Fifo {
        small_ratio: 0.1,
        ghost_ratio: 0.9,
    });

    println!("Created S3-FIFO cache: capacity={}\n", cache.capacity());

    // Insert items (all start in Small queue)
    for i in 1..=5 {
        cache.insert(i, format!("value-{}", i));
    }
    println!("Inserted keys 1-5 (all in Small queue)");
    println!("  len: {}", cache.len());

    // Access keys 1 and 2 to increase their frequency
    // When evicted from Small, they'll be promoted to Main
    cache.get(&1);
    cache.get(&2);
    println!("\nAccessed keys 1 and 2 (frequency increased)");

    // Insert more items to fill the cache
    for i in 6..=10 {
        cache.insert(i, format!("value-{}", i));
    }
    println!("Inserted keys 6-10");
    println!("  len: {}", cache.len());

    // Now insert more items - one-hit wonders should be evicted first
    cache.insert(11, "value-11".to_string());
    cache.insert(12, "value-12".to_string());

    println!("\nAfter inserting keys 11, 12:");
    println!(
        "  contains 1? {} (was accessed, promoted to Main)",
        cache.contains(&1)
    );
    println!(
        "  contains 2? {} (was accessed, promoted to Main)",
        cache.contains(&2)
    );
    println!("  len: {}", cache.len());

    // Demonstrate scan resistance
    println!("\n=== Scan Resistance Demo ===\n");

    let mut cache = CacheBuilder::new(10).build::<u64, String>(CachePolicy::S3Fifo {
        small_ratio: 0.1,
        ghost_ratio: 0.9,
    });

    // Insert "hot" items and access them to increase frequency
    cache.insert(1, "hot-1".to_string());
    cache.insert(2, "hot-2".to_string());
    cache.insert(3, "hot-3".to_string());

    // Multiple accesses increase frequency
    for _ in 0..3 {
        cache.get(&1);
        cache.get(&2);
        cache.get(&3);
    }
    println!("Inserted and accessed hot items: 1, 2, 3 (freq=3)");

    // Simulate a scan with one-time accesses (stay in Small, get evicted first)
    println!("Simulating scan with items 100-120...");
    for i in 100..=120 {
        cache.insert(i, format!("scan-{}", i));
    }

    // Hot items should survive the scan because they were promoted to Main
    println!("\nAfter scan (21 one-time insertions):");
    println!("  contains hot-1? {}", cache.contains(&1));
    println!("  contains hot-2? {}", cache.contains(&2));
    println!("  contains hot-3? {}", cache.contains(&3));
    println!("  len: {}", cache.len());

    // Show that scan items were evicted
    let scan_items_remaining: Vec<_> = (100..=120).filter(|&i| cache.contains(&i)).collect();
    println!("  scan items remaining: {:?}", scan_items_remaining);

    // Demonstrate ghost-guided admission
    println!("\n=== Ghost-Guided Admission Demo ===\n");

    let mut cache = CacheBuilder::new(5).build::<u64, String>(CachePolicy::S3Fifo {
        small_ratio: 0.2,
        ghost_ratio: 1.0, // 100% ghost list for demo
    });

    // Fill the cache
    for i in 1..=5 {
        cache.insert(i, format!("value-{}", i));
    }
    println!("Filled cache with keys 1-5");

    // Evict key 1 by inserting more items
    cache.insert(6, "value-6".to_string());
    println!("Inserted key 6, key 1 evicted (recorded in Ghost)");
    println!("  contains 1? {}", cache.contains(&1));

    // Now re-insert key 1 - it should go to Main (ghost-guided admission)
    cache.insert(1, "value-1-reinserted".to_string());
    println!("\nRe-inserted key 1 (should go to Main due to ghost hit)");
    println!("  contains 1? {}", cache.contains(&1));

    // Key 1 should survive more evictions since it's in Main
    for i in 10..=15 {
        cache.insert(i, format!("value-{}", i));
    }
    println!("\nAfter inserting keys 10-15:");
    println!(
        "  contains 1? {} (was promoted to Main via ghost)",
        cache.contains(&1)
    );
    println!("  len: {}", cache.len());
}

// Expected output:
// === S3-FIFO Cache Example ===
//
// Created S3-FIFO cache: capacity=10
//
// Inserted keys 1-5 (all in Small queue)
//   len: 5
//
// Accessed keys 1 and 2 (frequency increased)
// Inserted keys 6-10
//   len: 10
//
// After inserting keys 11, 12:
//   contains 1? true (was accessed, promoted to Main)
//   contains 2? true (was accessed, promoted to Main)
//   len: 10
//
// === Scan Resistance Demo ===
//
// Inserted and accessed hot items: 1, 2, 3 (freq=3)
// Simulating scan with items 100-120...
//
// After scan (21 one-time insertions):
//   contains hot-1? true
//   contains hot-2? true
//   contains hot-3? true
//   len: 10
//   scan items remaining: [last ~7 scan items]
//
// === Ghost-Guided Admission Demo ===
//
// Filled cache with keys 1-5
// Inserted key 6, key 1 evicted (recorded in Ghost)
//   contains 1? false
//
// Re-inserted key 1 (should go to Main due to ghost hit)
//   contains 1? true
//
// After inserting keys 10-15:
//   contains 1? true (was promoted to Main via ghost)
//   len: 5
