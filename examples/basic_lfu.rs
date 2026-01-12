use cachekit::policy::lfu::LFUCache;
use cachekit::traits::CoreCache;
use std::sync::Arc;

fn main() {
    let mut cache: LFUCache<&str, String> = LFUCache::new(2);

    cache.insert("a", Arc::new("alpha".to_string()));
    cache.insert("b", Arc::new("beta".to_string()));

    cache.get(&"a");
    cache.insert("c", Arc::new("gamma".to_string()));

    println!("contains a? {}", cache.contains(&"a"));
    println!("contains b? {}", cache.contains(&"b"));
}

// Expected output:
// contains a? true
// contains b? false
//
// Explanation: capacity=2; "a" is accessed before inserting "c", so "b" is evicted.
