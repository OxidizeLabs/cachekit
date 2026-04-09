//! Example demonstrating the Fast LRU cache.
//!
//! Fast LRU stores values directly instead of wrapping them in `Arc`, which keeps
//! the hot path lean for single-threaded workloads.
//!
//! Run with: cargo run --example basic_fast_lru

use cachekit::policy::fast_lru::FastLru;

fn main() {
    let mut cache: FastLru<u32, String> = FastLru::new(2);

    cache.insert(1, "alpha".to_string());
    cache.insert(2, "beta".to_string());

    // Touch key 1 without reading it so key 2 becomes the LRU entry.
    cache.touch(&1);

    if let Some((key, value)) = cache.peek_lru() {
        println!("lru before insert: {key} -> {value}");
    }

    cache.insert(3, "gamma".to_string());

    println!("contains 1? {}", cache.contains(&1));
    println!("contains 2? {}", cache.contains(&2));
}

// Expected output:
// lru before insert: 2 -> beta
// contains 1? true
// contains 2? false
//
// Explanation: touching key 1 moves it to the MRU position, so inserting key 3
// evicts key 2 as the least recently used entry.
