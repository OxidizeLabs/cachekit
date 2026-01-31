use cachekit::policy::lru_k::LrukCache;
use cachekit::traits::{CoreCache, ReadOnlyCache};

fn main() {
    let mut cache: LrukCache<&str, i32> = LrukCache::with_k(2, 2);

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
