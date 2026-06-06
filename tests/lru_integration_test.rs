use std::sync::Arc;

#[cfg(feature = "concurrency")]
use cachekit::policy::lru::ConcurrentLruCache;
use cachekit::policy::lru::LruCore;
use cachekit::traits::Cache;

#[test]
fn zero_copy_lru_core() {
    let mut cache = LruCore::new(3);

    cache.insert(1, Arc::new("one"));
    cache.insert(2, Arc::new("two"));
    cache.insert(3, Arc::new("three"));

    let value = cache.get(&1).unwrap();
    assert_eq!(**value, "one");

    let peeked = cache.peek(&2).unwrap();
    assert_eq!(*peeked, "two");

    cache.insert(4, Arc::new("four"));
    assert!(cache.get(&2).is_none());

    let removed = cache.remove(&3).unwrap();
    assert_eq!(*removed, "three");

    let (key, value) = cache.pop_lru().unwrap();
    assert_eq!(key, 1);
    assert_eq!(*value, "one");
}

#[cfg(feature = "concurrency")]
#[test]
fn concurrent_lru_single_threaded_smoke() {
    let cache = ConcurrentLruCache::new(3);
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    assert!(cache.peek(&1).is_some_and(|v| *v == "one"));
    assert_eq!(cache.len(), 2);
}
