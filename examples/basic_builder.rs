//! Example demonstrating the unified CacheBuilder API.
//!
//! Run with: cargo run --example basic_builder

use cachekit::builder::{CacheBuilder, CachePolicy};

fn main() {
    println!("=== CacheBuilder Examples ===\n");

    // Example 1: LRU Cache
    println!("1. LRU Cache");
    let mut lru = CacheBuilder::new(3).build::<u64, String>(CachePolicy::Lru);

    lru.insert(1, "one".to_string());
    lru.insert(2, "two".to_string());
    lru.insert(3, "three".to_string());

    // Access key 1 to make it MRU
    lru.get(&1);

    // Insert key 4, evicts LRU (key 2)
    lru.insert(4, "four".to_string());

    println!("   contains 1? {} (was accessed)", lru.contains(&1));
    println!("   contains 2? {} (evicted as LRU)", lru.contains(&2));
    println!("   contains 4? {} (just inserted)", lru.contains(&4));
    println!();

    // Example 2: FIFO Cache
    println!("2. FIFO Cache");
    let mut fifo = CacheBuilder::new(3).build::<u64, String>(CachePolicy::Fifo);

    fifo.insert(1, "one".to_string());
    fifo.insert(2, "two".to_string());
    fifo.insert(3, "three".to_string());

    // Access doesn't affect FIFO order
    fifo.get(&1);

    // Insert key 4, evicts oldest (key 1)
    fifo.insert(4, "four".to_string());

    println!("   contains 1? {} (evicted as oldest)", fifo.contains(&1));
    println!("   contains 2? {} (still present)", fifo.contains(&2));
    println!();

    // Example 3: LRU-K Cache (scan-resistant)
    println!("3. LRU-K Cache (K=2)");
    let mut lru_k = CacheBuilder::new(3).build::<u64, String>(CachePolicy::LruK { k: 2 });

    lru_k.insert(1, "one".to_string());
    lru_k.insert(2, "two".to_string());
    lru_k.insert(3, "three".to_string());

    // Access key 1 twice (reaches K=2 threshold)
    lru_k.get(&1);

    // Insert key 4, evicts cold entry (key 2 or 3)
    lru_k.insert(4, "four".to_string());

    println!(
        "   contains 1? {} (accessed twice, promoted)",
        lru_k.contains(&1)
    );
    println!("   len: {}", lru_k.len());
    println!();

    // Example 4: LFU Cache
    println!("4. LFU Cache");
    let mut lfu = CacheBuilder::new(3).build::<u64, String>(CachePolicy::Lfu);

    lfu.insert(1, "one".to_string());
    lfu.insert(2, "two".to_string());
    lfu.insert(3, "three".to_string());

    // Access key 1 multiple times
    lfu.get(&1);
    lfu.get(&1);
    lfu.get(&1);

    // Insert key 4, evicts LFU (key 2 or 3)
    lfu.insert(4, "four".to_string());

    println!("   contains 1? {} (highest frequency)", lfu.contains(&1));
    println!("   len: {}", lfu.len());
    println!();

    // Example 5: 2Q Cache
    println!("5. 2Q Cache (25% probation)");
    let mut two_q = CacheBuilder::new(4).build::<u64, String>(CachePolicy::TwoQ {
        probation_frac: 0.25,
    });

    two_q.insert(1, "one".to_string());
    two_q.insert(2, "two".to_string());
    two_q.insert(3, "three".to_string());
    two_q.insert(4, "four".to_string());

    // Access to promote from probation to protected
    two_q.get(&1);

    println!("   capacity: {}", two_q.capacity());
    println!("   len: {}", two_q.len());
    println!();

    // Example 6: Common operations
    println!("6. Common Operations");
    let mut cache = CacheBuilder::new(10).build::<u64, String>(CachePolicy::Lru);

    // Insert and update
    cache.insert(1, "original".to_string());
    let old = cache.insert(1, "updated".to_string());
    println!("   insert returned previous: {:?}", old);

    // Get
    if let Some(value) = cache.get(&1) {
        println!("   get(&1): {}", value);
    }

    // Contains (doesn't update access order)
    println!("   contains(&1): {}", cache.contains(&1));
    println!("   contains(&99): {}", cache.contains(&99));

    // Size
    println!(
        "   len: {}, capacity: {}, is_empty: {}",
        cache.len(),
        cache.capacity(),
        cache.is_empty()
    );

    // Clear
    cache.clear();
    println!("   after clear - is_empty: {}", cache.is_empty());
}

// Expected output:
// === CacheBuilder Examples ===
//
// 1. LRU Cache
//    contains 1? true (was accessed)
//    contains 2? false (evicted as LRU)
//    contains 4? true (just inserted)
//
// 2. FIFO Cache
//    contains 1? false (evicted as oldest)
//    contains 2? true (still present)
//
// 3. LRU-K Cache (K=2)
//    contains 1? true (accessed twice, promoted)
//    len: 3
//
// 4. LFU Cache
//    contains 1? true (highest frequency)
//    len: 3
//
// 5. 2Q Cache (25% probation)
//    capacity: 4
//    len: 4
//
// 6. Common Operations
//    insert returned previous: Some("original")
//    get(&1): updated
//    contains(&1): true
//    contains(&99): false
//    len: 1, capacity: 10, is_empty: false
//    after clear - is_empty: true
