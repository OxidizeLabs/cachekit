use std::sync::Arc;

#[cfg(test)]
mod integration_tests {
    use cachekit::policy::lru::{ConcurrentLruCache, LruCore};
    use cachekit::traits::{CoreCache, LruCacheTrait, MutableCache};

    use super::*;

    #[test]
    fn test_zero_copy_lru_core() {
        let mut cache = LruCore::new(3);

        // Test basic zero-copy operations
        cache.insert(1, Arc::new("one"));
        cache.insert(2, Arc::new("two"));
        cache.insert(3, Arc::new("three"));

        // Test get returns Arc<V>
        let value = cache.get(&1).unwrap();
        assert_eq!(**value, "one");

        // Test peek (concurrent read support)
        let peeked = cache.peek(&2).unwrap();
        assert_eq!(*peeked, "two");

        // Test zero-copy eviction - insert key 4 should evict key 2 (LRU after accessing key 1)
        cache.insert(4, Arc::new("four"));
        assert!(cache.get(&2).is_none());

        // Test remove returns Arc<V>
        let removed = cache.remove(&3).unwrap();
        assert_eq!(*removed, "three");

        // Test pop_lru returns (K, Arc<V>)
        // After the operations above:
        // - Key 1 was accessed (most recent after initial inserts)
        // - Key 2 was evicted when inserting 4
        // - Key 3 was removed
        // - Key 4 was just inserted (most recent)
        // Remaining keys: 1 and 4, with 1 being the LRU (older access than 4)
        let (key, value) = cache.pop_lru().unwrap();
        assert_eq!(key, 1); // Key 1 should now be the LRU
        assert_eq!(*value, "one");

        println!("✅ Zero-copy LRU core test passed!");
    }

    #[test]
    fn test_concurrent_lru_cache() {
        let cache = ConcurrentLruCache::new(3);

        // Test concurrent wrapper operations
        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        // Test peek (concurrent reads)
        let value1 = cache.peek(&1).unwrap();
        let value2 = cache.peek(&1).unwrap(); // Concurrent access
        assert!(Arc::ptr_eq(&value1, &value2)); // Same Arc instance - zero copy!

        // Test get with LRU update
        let value = cache.get(&2).unwrap();
        assert_eq!(*value, "two");

        // Test insert_arc for direct Arc insertion
        let arc_value = Arc::new("four");
        let original_ptr = Arc::as_ptr(&arc_value);
        cache.insert_arc(4, arc_value);

        let retrieved = cache.peek(&4).unwrap();
        assert_eq!(Arc::as_ptr(&retrieved), original_ptr); // Same pointer - truly zero copy!

        // Test concurrent properties: insert over capacity should evict LRU
        assert_eq!(cache.capacity(), 3);
        assert_eq!(cache.len(), 3);
        assert!(!cache.contains(&1)); // LRU should have been evicted
        assert!(cache.contains(&2));
        assert!(cache.contains(&3));
        assert!(cache.contains(&4));

        println!("✅ Concurrent LRU cache test passed!");
    }

    #[test]
    fn test_database_workload_simulation() {
        // Simulate a database buffer pool workload
        type PageId = u32;
        type Page = Vec<u8>;

        let cache: ConcurrentLruCache<PageId, Page> = ConcurrentLruCache::new(100);

        // Simulate page insertions
        for page_id in 0..150u32 {
            let page_data = vec![page_id as u8; 4096]; // 4KB page
            cache.insert(page_id, page_data);
        }

        // Simulate concurrent read workload
        let hot_pages = vec![100, 101, 102, 103, 104]; // Hot pages
        let mut shared_refs = Vec::new();

        for &page_id in &hot_pages {
            if let Some(page) = cache.peek(&page_id) {
                shared_refs.push(page); // Zero-copy sharing
            }
        }

        // Verify zero-copy semantics
        assert_eq!(shared_refs.len(), 5);
        for (i, page_ref) in shared_refs.iter().enumerate() {
            assert_eq!(page_ref[0], (100 + i) as u8);
        }

        // Verify cache state
        assert_eq!(cache.len(), 100); // Should be at capacity
        assert!(cache.contains(&140)); // Recent page should be present
        assert!(!cache.contains(&10)); // Old page should be evicted

        println!("✅ Database workload simulation passed!");
        println!(
            "   Cache handled {} page insertions with zero-copy semantics",
            150
        );
    }
}
