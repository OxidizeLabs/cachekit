use cachekit::prelude::FIFOCache;
use cachekit::traits::CoreCache;

fn main() {
    // Create a FIFO cache with a capacity of 100 entries
    let mut cache = FIFOCache::new(100);

    // Insert an item
    cache.insert("key1", "value1");

    // Retrieve an item
    if let Some(value) = cache.get(&"key1") {
        println!("Got from cache: {}", value);
    }
}

// Expected output:
// Got from cache: value1
//
// Explanation: FIFO preserves insertion order; the inserted key is still present.
