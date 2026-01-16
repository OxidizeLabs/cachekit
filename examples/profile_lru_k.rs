use cachekit::policy::lru_k::LrukCache;
use cachekit::traits::CoreCache;

fn main() {
    let mut cache: LrukCache<&i32, &str> = LrukCache::with_k(100, 2);

    // Backing storage so references stay valid
    let mut keys: Vec<i32> = Vec::new();
    let mut values: Vec<String> = Vec::new();

    // Generate sample key–value pairs
    for i in 1..=20 {
        keys.push(i);
        values.push(format!("user_{}", i));
    }

    // Insert into cache
    for i in 0..keys.len() {
        let k = &keys[i];
        let v = values[i].as_str();
        cache.insert(k, v);
    }

    // Access some keys to affect LRU-K history
    cache.get(&&keys[1]);
    cache.get(&&keys[1]);
    cache.get(&&keys[5]);
    cache.get(&&keys[1]);
    cache.get(&&keys[10]);
    cache.get(&&keys[5]);

    println!("Cache populated and exercised with sample data.");
}
