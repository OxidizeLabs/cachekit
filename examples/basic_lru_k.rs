use cachekit::policy::lru_k::LRUKCache;
use cachekit::traits::CoreCache;

fn main() {
    let mut cache: LRUKCache<&str, i32> = LRUKCache::with_k(2, 2);

    cache.insert("a", 10);
    cache.insert("b", 20);

    cache.get(&"a");
    cache.insert("c", 30);

    println!("contains a? {}", cache.contains(&"a"));
    println!("contains b? {}", cache.contains(&"b"));
}

// Expected output:
// contains a? true
// contains b? false
//
// Explanation: capacity=2; "a" is touched before inserting "c", so "b" is evicted.
