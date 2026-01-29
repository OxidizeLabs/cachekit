//! Example demonstrating the Random cache eviction policy.
//!
//! Random eviction selects victims uniformly at random when capacity is reached.
//! This provides a baseline policy with minimal overhead and no access tracking.
//!
//! ⚠️  NOTE: Random is primarily useful as a baseline for benchmarking.
//!     It will typically underperform policies that track access patterns.
//!
//! Run with: cargo run --example basic_random

use cachekit::policy::random::RandomCore;

fn main() {
    println!("=== Random Eviction Cache Example ===\n");

    // Create a random eviction cache with capacity 10
    let mut cache = RandomCore::new(10);

    println!("Created random cache: capacity={}\n", cache.capacity());

    // Insert items 1-10
    for i in 1..=10 {
        cache.insert(i, format!("value-{}", i));
    }
    println!("Inserted keys 1-10");
    println!("  len: {}", cache.len());

    // Access key 5 multiple times (doesn't affect eviction in random policy!)
    for _ in 0..100 {
        cache.get(&5);
    }
    println!("\nAccessed key 5 one hundred times");
    println!("  (This doesn't affect eviction probability in random policy)");

    // Insert more items - random eviction will occur
    println!("\nInserting keys 11-20 (triggers 10 random evictions)...");
    for i in 11..=20 {
        cache.insert(i, format!("value-{}", i));
    }

    println!("\nAfter insertions:");
    println!("  len: {}", cache.len());

    // Count which items from 1-10 survived
    let mut survivors_1_10 = vec![];
    for i in 1..=10 {
        if cache.contains(&i) {
            survivors_1_10.push(i);
        }
    }

    // Count which items from 11-20 survived
    let mut survivors_11_20 = vec![];
    for i in 11..=20 {
        if cache.contains(&i) {
            survivors_11_20.push(i);
        }
    }

    println!("  survivors from 1-10: {:?}", survivors_1_10);
    println!("  survivors from 11-20: {:?}", survivors_11_20);
    println!();
    println!("Note: Key 5 was accessed 100 times, but has same eviction");
    println!("      probability as any other key (that's random eviction!)");

    // Demonstrate baseline property
    println!("\n=== Random as Baseline ===\n");

    println!("Random eviction properties:");
    println!("  • No access pattern tracking");
    println!("  • Minimal overhead (Vec + HashMap)");
    println!("  • All entries have equal eviction probability");
    println!("  • Hot items can be evicted (unpredictable hit rate)");
    println!();
    println!("Use cases:");
    println!("  ✓ Benchmark baseline (smarter policies should beat random)");
    println!("  ✓ Testing cache infrastructure");
    println!("  ✓ Truly random access patterns (rare)");
    println!("  ✓ Minimal overhead is critical");
    println!();
    println!("When to avoid:");
    println!("  ✗ Access patterns have temporal locality (use LRU/SLRU)");
    println!("  ✗ Access patterns have frequency skew (use LFU)");
    println!("  ✗ Need scan resistance (use S3-FIFO/LRU-K)");
    println!("  ✗ Need predictable performance");

    // Show get doesn't affect eviction
    println!("\n=== Get Doesn't Affect Eviction ===\n");

    let mut cache = RandomCore::new(5);

    // Insert 5 items
    for i in 1..=5 {
        cache.insert(i, i * 10);
    }

    // Access item 1 heavily
    println!("Accessing item 1 one thousand times...");
    for _ in 0..1000 {
        cache.get(&1);
    }

    // Never access item 2
    println!("Never accessing item 2");

    // Insert more items
    println!("\nInserting items 6-10 (triggers 5 random evictions)...");
    for i in 6..=10 {
        cache.insert(i, i * 10);
    }

    println!("\nResult:");
    println!("  contains 1? {} (heavily accessed)", cache.contains(&1));
    println!("  contains 2? {} (never accessed)", cache.contains(&2));
    println!();
    println!("Both items had ~50% chance of survival despite different access patterns.");
    println!("This demonstrates that random eviction ignores access frequency.");

    // Performance comparison context
    println!("\n=== Performance Context ===\n");

    println!("Expected performance on workload with 80/20 locality:");
    println!("  Random:  ~20% hit rate (baseline)");
    println!("  LRU:     ~60-80% hit rate (tracks recency)");
    println!("  LFU:     ~70-90% hit rate (tracks frequency)");
    println!();
    println!("Random provides the lower bound - any policy with access");
    println!("tracking should significantly outperform random on real workloads.");
}

// Expected output (note: exact survivors vary due to randomness):
// === Random Eviction Cache Example ===
//
// Created random cache: capacity=10
//
// Inserted keys 1-10
//   len: 10
//
// Accessed key 5 one hundred times
//   (This doesn't affect eviction probability in random policy)
//
// Inserting keys 11-20 (triggers 10 random evictions)...
//
// After insertions:
//   len: 10
//   survivors from 1-10: [varies]
//   survivors from 11-20: [varies]
//
// Note: Key 5 was accessed 100 times, but has same eviction
//       probability as any other key (that's random eviction!)
//
// === Random as Baseline ===
//
// Random eviction properties:
//   • No access pattern tracking
//   • Minimal overhead (Vec + HashMap)
//   • All entries have equal eviction probability
//   • Hot items can be evicted (unpredictable hit rate)
//
// Use cases:
//   ✓ Benchmark baseline (smarter policies should beat random)
//   ✓ Testing cache infrastructure
//   ✓ Truly random access patterns (rare)
//   ✓ Minimal overhead is critical
//
// When to avoid:
//   ✗ Access patterns have temporal locality (use LRU/SLRU)
//   ✗ Access patterns have frequency skew (use LFU)
//   ✗ Need scan resistance (use S3-FIFO/LRU-K)
//   ✗ Need predictable performance
//
// === Get Doesn't Affect Eviction ===
//
// Accessing item 1 one thousand times...
// Never accessing item 2
//
// Inserting items 6-10 (triggers 5 random evictions)...
//
// Result:
//   contains 1? [varies] (heavily accessed)
//   contains 2? [varies] (never accessed)
//
// Both items had ~50% chance of survival despite different access patterns.
// This demonstrates that random eviction ignores access frequency.
//
// === Performance Context ===
//
// Expected performance on workload with 80/20 locality:
//   Random:  ~20% hit rate (baseline)
//   LRU:     ~60-80% hit rate (tracks recency)
//   LFU:     ~70-90% hit rate (tracks frequency)
//
// Random provides the lower bound - any policy with access
// tracking should significantly outperform random on real workloads.
