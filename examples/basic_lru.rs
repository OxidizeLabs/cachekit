use cachekit::policy::lru::LRUCore;
use cachekit::traits::CoreCache;
use std::sync::Arc;

fn main() {
    let mut cache: LRUCore<u32, String> = LRUCore::new(2);

    cache.insert(1, Arc::new("alpha".to_string()));
    cache.insert(2, Arc::new("beta".to_string()));

    if let Some(value) = cache.get(&1) {
        println!("hit 1: {}", value.as_str());
    }

    cache.insert(3, Arc::new("gamma".to_string()));

    println!("contains 2? {}", cache.contains(&2));
}

// Expected output:
// hit 1: alpha
// contains 2? false
//
// Explanation: capacity=2; after get(&1), key 1 is MRU and key 2 is LRU.
// Inserting key 3 evicts key 2, so contains(2) is false.
