//! Example demonstrating the MRU (Most Recently Used) cache policy.
//!
//! MRU evicts the **most** recently accessed item when capacity is reached.
//! This is the opposite of LRU and is useful for specific cyclic access patterns.
//!
//! ⚠️  WARNING: MRU is NOT a general-purpose cache policy!
//!     It only works well for specific cyclic or sequential patterns.
//!
//! Run with: cargo run --example basic_mru

use cachekit::policy::mru::MruCore;

fn main() {
    println!("=== MRU Cache Example ===\n");

    // Create an MRU cache with capacity 5
    let mut cache = MruCore::new(5);

    println!("Created MRU cache: capacity={}\n", cache.capacity());

    // Insert items 1-5
    for i in 1..=5 {
        cache.insert(i, format!("value-{}", i));
    }
    println!("Inserted keys 1-5");
    println!("  len: {}", cache.len());

    // Access key 5 (most recent) - moves it to MRU (head)
    cache.get(&5);
    println!("\nAccessed key 5 (moved to MRU/head position)");

    // Now insert key 6 - MRU evicts key 5 (the most recent!)
    cache.insert(6, "value-6".to_string());
    println!("Inserted key 6");

    println!("\nAfter inserting key 6:");
    println!(
        "  contains 5? {} (was most recent, got evicted!)",
        cache.contains(&5)
    );
    println!(
        "  contains 1? {} (oldest, still in cache)",
        cache.contains(&1)
    );
    println!("  contains 6? {} (newly inserted)", cache.contains(&6));
    println!("  len: {}", cache.len());

    // Demonstrate the difference from LRU
    println!("\n=== MRU vs LRU Comparison ===\n");

    println!("Key differences:");
    println!("  • MRU: Evicts from head (most recently used)");
    println!("  • LRU: Evicts from tail (least recently used)");
    println!();
    println!("Example with cache capacity 3:");
    println!("  Initial: [A, B, C]");
    println!("  Access A → A moves to MRU position: [A, B, C]");
    println!("  Insert D:");
    println!("    • MRU evicts A (most recent) → [D, B, C]");
    println!("    • LRU would evict C (least recent) → [D, A, B]");

    // Demonstrate cyclic pattern (where MRU might be useful)
    println!("\n=== Cyclic Pattern Demo ===\n");

    let mut cache = MruCore::new(3);

    println!("Simulating cyclic access pattern: 1, 2, 3, 1, 2, 3, ...");
    println!("(Each number represents a different page in a cycle)\n");

    // First cycle
    for i in 1..=3 {
        cache.insert(i, format!("page-{}", i));
    }
    println!("First cycle complete: keys 1, 2, 3 in cache");

    // Second cycle - each insert evicts the most recent
    cache.insert(1, "page-1-updated".to_string());
    println!("Accessed 1 again (evicted 3 - the most recent)");

    cache.insert(2, "page-2-updated".to_string());
    println!("Accessed 2 again (evicted 1 - the most recent)");

    cache.insert(3, "page-3-updated".to_string());
    println!("Accessed 3 again (evicted 2 - the most recent)");

    println!("\nFinal state:");
    println!("  contains 1? {}", cache.contains(&1));
    println!("  contains 2? {}", cache.contains(&2));
    println!("  contains 3? {}", cache.contains(&3));
    println!("  len: {}", cache.len());

    // Show when MRU is BAD (temporal locality)
    println!("\n=== When MRU Performs Poorly ===\n");

    let mut cache = MruCore::new(3);

    println!("Access pattern with temporal locality:");
    println!("  hot_page (many times), cold_page_1, cold_page_2\n");

    cache.insert("hot_page", "important");
    // Access hot_page multiple times
    for _ in 0..5 {
        cache.get(&"hot_page");
    }
    println!("Accessed 'hot_page' 6 times total");

    // Insert two more items
    cache.insert("cold_page_1", "data1");
    cache.insert("cold_page_2", "data2");
    println!("Inserted 'cold_page_1' and 'cold_page_2'");

    println!("\nResult:");
    println!("  contains hot_page? {}", cache.contains(&"hot_page"));
    println!("  contains cold_page_1? {}", cache.contains(&"cold_page_1"));
    println!("  contains cold_page_2? {}", cache.contains(&"cold_page_2"));
    println!();
    println!("⚠️  'hot_page' was evicted despite being accessed frequently!");
    println!("    This is why MRU is NOT recommended for general-purpose caching.");
    println!("    Use LRU, SLRU, or S3-FIFO for typical workloads.");
}

// Expected output:
// === MRU Cache Example ===
//
// Created MRU cache: capacity=5
//
// Inserted keys 1-5
//   len: 5
//
// Accessed key 5 (moved to MRU/head position)
// Inserted key 6
//
// After inserting key 6:
//   contains 5? false (was most recent, got evicted!)
//   contains 1? true (oldest, still in cache)
//   contains 6? true (newly inserted)
//   len: 5
//
// === MRU vs LRU Comparison ===
//
// Key differences:
//   • MRU: Evicts from head (most recently used)
//   • LRU: Evicts from tail (least recently used)
//
// Example with cache capacity 3:
//   Initial: [A, B, C]
//   Access A → A moves to MRU position: [A, B, C]
//   Insert D:
//     • MRU evicts A (most recent) → [D, B, C]
//     • LRU would evict C (least recent) → [D, A, B]
//
// === Cyclic Pattern Demo ===
//
// Simulating cyclic access pattern: 1, 2, 3, 1, 2, 3, ...
// (Each number represents a different page in a cycle)
//
// First cycle complete: keys 1, 2, 3 in cache
// Accessed 1 again (evicted 3 - the most recent)
// Accessed 2 again (evicted 1 - the most recent)
// Accessed 3 again (evicted 2 - the most recent)
//
// Final state:
//   contains 1? false
//   contains 2? false
//   contains 3? true
//   len: 1
//
// === When MRU Performs Poorly ===
//
// Access pattern with temporal locality:
//   hot_page (many times), cold_page_1, cold_page_2
//
// Accessed 'hot_page' 6 times total
// Inserted 'cold_page_1' and 'cold_page_2'
//
// Result:
//   contains hot_page? false
//   contains cold_page_1? true
//   contains cold_page_2? true
//
// ⚠️  'hot_page' was evicted despite being accessed frequently!
//     This is why MRU is NOT recommended for general-purpose caching.
//     Use LRU, SLRU, or S3-FIFO for typical workloads.
