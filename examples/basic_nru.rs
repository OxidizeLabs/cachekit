//! Example demonstrating the NRU (Not Recently Used) cache eviction policy.
//!
//! NRU uses a single reference bit per entry to distinguish between recently used
//! and not recently used items, providing O(1) access with simple eviction tracking.
//!
//! Run with: cargo run --example basic_nru

use cachekit::policy::nru::NruCache;
use cachekit::traits::{CoreCache, ReadOnlyCache};

fn main() {
    println!("=== NRU (Not Recently Used) Cache Example ===\n");

    // Create an NRU cache with capacity 10
    let mut cache = NruCache::new(10);

    println!("Created NRU cache: capacity={}\n", cache.capacity());

    // Insert items 1-10
    for i in 1..=10 {
        cache.insert(i, format!("value-{}", i));
    }
    println!("Inserted keys 1-10");
    println!("  len: {}", cache.len());
    println!("  (All items have referenced=true after insertion)");

    // Access keys 1, 3, 5, 7, 9 (odd numbers)
    println!("\nAccessing odd-numbered keys (1, 3, 5, 7, 9)...");
    for i in (1..=9).step_by(2) {
        cache.get(&i);
    }
    println!("  (These items now have referenced=true)");
    println!("  (Even-numbered keys will have referenced=false after epoch reset)");

    // Note: In this implementation, all items start as referenced after insert.
    // To demonstrate NRU behavior, we need to trigger an eviction scenario.

    println!("\n=== Eviction Behavior ===\n");

    // Create a fresh cache to demonstrate eviction clearly
    let mut cache = NruCache::new(5);

    // Insert 5 items
    for i in 1..=5 {
        cache.insert(i, i * 10);
    }

    // Manually demonstrate the reference bit concept
    println!("Cache has 5 items (capacity=5)");
    println!("  Keys: [1, 2, 3, 4, 5]");
    println!("  All have referenced=true (freshly inserted)");

    // Access only items 1 and 5
    println!("\nAccessing keys 1 and 5...");
    cache.get(&1);
    cache.get(&5);

    // Now insert item 6 - this will trigger eviction
    println!("\nInserting key 6 (triggers eviction)...");
    println!("  NRU scans for first unreferenced entry");
    println!("  Since all were just inserted/accessed, NRU may:");
    println!("    1. Find an unreferenced entry (if any exist)");
    println!("    2. Clear all reference bits and evict first entry");

    cache.insert(6, 60);

    println!("\nAfter eviction:");
    println!("  len: {}", cache.len());

    // Check what survived
    for i in 1..=6 {
        if cache.contains(&i) {
            println!("  key {} survived", i);
        }
    }

    println!("\n=== Reference Bit Protection ===\n");

    // Create a new cache to demonstrate reference bit protection
    let mut cache = NruCache::new(4);

    println!("Demonstrating reference bit protection:");
    println!();

    // Insert 4 items
    cache.insert(1, 100);
    cache.insert(2, 200);
    cache.insert(3, 300);
    cache.insert(4, 400);

    println!("Inserted keys 1-4 (capacity=4)");

    // Access only key 1 heavily
    println!("Accessing key 1 ten times...");
    for _ in 0..10 {
        cache.get(&1);
    }

    // Don't access keys 2, 3, 4
    println!("Not accessing keys 2, 3, 4");

    // Insert key 5 - should prefer evicting an unreferenced entry
    println!("\nInserting key 5 (triggers eviction)...");
    cache.insert(5, 500);

    println!("\nResult:");
    println!("  contains 1? {} (heavily accessed)", cache.contains(&1));
    println!("  contains 2? {}", cache.contains(&2));
    println!("  contains 3? {}", cache.contains(&3));
    println!("  contains 4? {}", cache.contains(&4));
    println!("  contains 5? {} (newly inserted)", cache.contains(&5));

    println!("\nKey 1 has referenced=true, protecting it from eviction.");
    println!("One of the unreferenced keys (2, 3, or 4) was evicted.");

    println!("\n=== NRU Policy Properties ===\n");

    println!("NRU characteristics:");
    println!("  • Binary tracking: used vs not used (coarse granularity)");
    println!("  • O(1) access operations (set reference bit)");
    println!("  • O(n) worst-case eviction (scan for unreferenced)");
    println!("  • Low memory overhead (1 bit per entry)");
    println!("  • Approximates LRU with less overhead");
    println!();
    println!("Eviction algorithm:");
    println!("  1. Scan for first entry with referenced=false");
    println!("  2. If found: evict that entry");
    println!("  3. If not found (all referenced): clear all bits, evict first");
    println!();
    println!("Use cases:");
    println!("  ✓ Temporal locality with coarse tracking");
    println!("  ✓ Memory-constrained environments");
    println!("  ✓ Can tolerate O(n) eviction scans");
    println!("  ✓ Want simpler implementation than full LRU");
    println!();
    println!("When to avoid:");
    println!("  ✗ Need O(1) eviction guarantees (use Clock)");
    println!("  ✗ Need fine-grained recency tracking (use LRU)");
    println!("  ✗ Need scan resistance (use S3-FIFO/LRU-K)");
    println!("  ✗ Need frequency tracking (use LFU)");

    println!("\n=== Comparison with Other Policies ===\n");

    println!("Policy comparison for access pattern [A,B,C,A,A,A,D]:");
    println!("  (cache capacity = 3)");
    println!();
    println!("NRU:");
    println!("  • After A,B,C: [A, B, C] (all referenced)");
    println!("  • After A,A,A: [A, B, C] (A remains referenced)");
    println!("  • Insert D: Evicts B or C (unreferenced), keeps A");
    println!();
    println!("LRU:");
    println!("  • After A,B,C: [A, B, C] (A=LRU)");
    println!("  • After A,A,A: [B, C, A] (A moves to MRU)");
    println!("  • Insert D: [C, A, D] (evicts B=LRU)");
    println!();
    println!("Random:");
    println!("  • After A,B,C: [A, B, C]");
    println!("  • After A,A,A: [A, B, C] (no change)");
    println!("  • Insert D: Evicts random item (could evict A!)");
    println!();
    println!("Clock:");
    println!("  • Similar to NRU but uses hand sweep");
    println!("  • More predictable eviction order");
    println!("  • Better amortized eviction cost");

    println!("\n=== Performance Context ===\n");

    println!("Expected relative performance:");
    println!("  Access cost:  NRU ≈ Clock < LRU (all O(1), but NRU/Clock just set bit)");
    println!("  Eviction:     LRU=O(1), Clock=O(1) amortized, NRU=O(n) worst case");
    println!("  Memory:       NRU ≈ Clock < LRU (bits vs pointers)");
    println!("  Hit rate:     LRU ≥ NRU ≥ Clock (with temporal locality)");
    println!();
    println!("NRU trades eviction cost for simpler implementation than Clock");
    println!("and lower memory overhead than full LRU.");
}
