//! Example demonstrating the SLRU (Segmented LRU) cache policy.
//!
//! SLRU uses two LRU segments: a probationary segment for new entries and a protected segment
//! for entries that have been accessed more than once. This provides scan
//! resistance by preventing one-time accesses from polluting the protected segment.
//!
//! Run with: cargo run --example basic_slru

use cachekit::policy::slru::SlruCore;

fn main() {
    println!("=== SLRU Cache Example ===\n");

    // Create an SLRU cache with capacity 10 and 25% probationary segment
    // - Probationary segment: ~2-3 slots (for new items, LRU eviction)
    // - Protected segment: ~7-8 slots (for frequently accessed items, LRU eviction)
    let mut cache = SlruCore::new(10, 0.25);

    println!("Created SLRU cache: capacity={}\n", cache.capacity());

    // Insert items (all start in probationary segment)
    for i in 1..=5 {
        cache.insert(i, format!("value-{}", i));
    }
    println!("Inserted keys 1-5 (all in probationary segment)");
    println!("  len: {}", cache.len());

    // Access keys 1 and 2 to promote them from probationary to protected
    cache.get(&1);
    cache.get(&2);
    println!("\nAccessed keys 1 and 2 (promoted to protected segment)");

    // Insert more items to fill the cache
    for i in 6..=10 {
        cache.insert(i, format!("value-{}", i));
    }
    println!("Inserted keys 6-10");
    println!("  len: {}", cache.len());

    // Now insert more items - probationary segment items should be evicted first
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

    let mut cache = SlruCore::new(10, 0.3); // 30% probationary

    // Insert "hot" items and access them to promote to protected segment
    cache.insert(1, "hot-1".to_string());
    cache.insert(2, "hot-2".to_string());
    cache.insert(3, "hot-3".to_string());

    // Multiple accesses promote items to protected segment
    cache.get(&1);
    cache.get(&2);
    cache.get(&3);
    println!("Inserted and promoted hot items: 1, 2, 3");

    // Simulate a scan with one-time accesses (stay in probationary, get evicted first)
    println!("Simulating scan with items 100-120...");
    for i in 100..=120 {
        cache.insert(i, format!("scan-{}", i));
    }

    // Hot items should survive the scan because they're in protected segment
    println!("\nAfter scan (21 one-time insertions):");
    println!("  contains hot-1? {}", cache.contains(&1));
    println!("  contains hot-2? {}", cache.contains(&2));
    println!("  contains hot-3? {}", cache.contains(&3));
    println!("  len: {}", cache.len());

    // Show that scan items were evicted
    let scan_items_remaining: Vec<_> = (100..=120).filter(|&i| cache.contains(&i)).collect();
    println!("  scan items remaining: {:?}", scan_items_remaining);

    // Demonstrate the difference from regular LRU
    println!("\n=== SLRU vs LRU Comparison ===\n");
    println!("Key differences:");
    println!("  • SLRU: New items enter probationary segment (LRU eviction)");
    println!("  • SLRU: Re-accessed items move to protected segment (LRU eviction)");
    println!("  • LRU:  All items in single list (LRU eviction from tail)");
    println!();
    println!("Benefits of SLRU:");
    println!("  • Scan resistance: one-time accesses don't pollute protected segment");
    println!("  • Simple: just two LRU lists, no complex frequency tracking");
    println!("  • Tunable: adjust probationary/protected ratio for workload");
}

// Expected output:
// === SLRU Cache Example ===
//
// Created SLRU cache: capacity=10
//
// Inserted keys 1-5 (all in probationary segment)
//   len: 5
//
// Accessed keys 1 and 2 (promoted to protected segment)
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
//
// === SLRU vs LRU Comparison ===
//
// Key differences:
//   • SLRU: New items enter probationary segment (LRU eviction)
//   • SLRU: Re-accessed items move to protected segment (LRU eviction)
//   • LRU:  All items in single list (LRU eviction from tail)
//
// Benefits of SLRU:
//   • Scan resistance: one-time accesses don't pollute protected segment
//   • Simple: just two LRU lists, no complex frequency tracking
//   • Tunable: adjust probationary/protected ratio for workload
