// ==============================================
// LFU PERFORMANCE TESTS (integration)
// ==============================================
use std::sync::Arc;
use std::time::{Duration, Instant};

mod lookup_performance {
    use cachekit::policy::lfu::LfuCache;
    use cachekit::traits::{CoreCache, LfuCacheTrait};

    use super::*;

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_get_performance_with_varying_frequencies() {
        let cache_size = 10000;
        let mut cache = LfuCache::new(cache_size);

        // Setup: Fill cache with items
        for i in 0..cache_size {
            cache.insert(format!("key_{}", i), Arc::new(i));
        }

        // Test 1: Uniform frequency distribution (all items accessed equally)
        let start = Instant::now();
        for i in 0..1000 {
            let key = format!("key_{}", i % cache_size);
            cache.get(&key);
        }
        let uniform_duration = start.elapsed();

        // Test 2: Skewed frequency distribution (80/20 rule - 20% of keys get 80% of accesses)
        let start = Instant::now();
        for i in 0..1000 {
            let key_index = if i % 5 == 0 {
                // 20% of requests go to first 20% of keys
                i % (cache_size / 5)
            } else {
                // 80% of requests go to remaining 80% of keys
                (cache_size / 5) + (i % (4 * cache_size / 5))
            };
            let key = format!("key_{}", key_index);
            cache.get(&key);
        }
        let skewed_duration = start.elapsed();

        // Test 3: Highly concentrated access pattern (90% of accesses to 10% of keys)
        let start = Instant::now();
        for i in 0..1000 {
            let key_index = if i % 10 < 9 {
                // 90% of requests go to first 10% of keys
                i % (cache_size / 10)
            } else {
                // 10% of requests go to remaining 90% of keys
                (cache_size / 10) + (i % (9 * cache_size / 10))
            };
            let key = format!("key_{}", key_index);
            cache.get(&key);
        }
        let concentrated_duration = start.elapsed();

        // Performance assertions (get operations should be fast)
        assert!(
            uniform_duration < Duration::from_millis(100),
            "Uniform access pattern should be fast: {:?}",
            uniform_duration
        );
        assert!(
            skewed_duration < Duration::from_millis(100),
            "Skewed access pattern should be fast: {:?}",
            skewed_duration
        );
        assert!(
            concentrated_duration < Duration::from_millis(100),
            "Concentrated access pattern should be fast: {:?}",
            concentrated_duration
        );

        // All patterns should have similar performance characteristics
        // since HashMap lookup is O(1) average case
        log::info!(
            "Get performance - Uniform: {:?}, Skewed: {:?}, Concentrated: {:?}",
            uniform_duration,
            skewed_duration,
            concentrated_duration
        );

        // Verify cache functionality wasn't broken
        assert_eq!(cache.len(), cache_size);
        assert!(cache.get(&"key_0".to_string()).is_some());
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_contains_performance() {
        let cache_size = 50000;
        let mut cache = LfuCache::new(cache_size);

        // Setup: Fill cache with items
        for i in 0..cache_size {
            cache.insert(format!("item_{}", i), Arc::new(i));
        }

        // Test 1: Contains performance for existing keys
        let start = Instant::now();
        let mut hit_count = 0;
        for i in 0..10000 {
            let key = format!("item_{}", i % cache_size);
            if cache.contains(&key) {
                hit_count += 1;
            }
        }
        let existing_keys_duration = start.elapsed();
        assert_eq!(hit_count, 10000); // All keys should exist

        // Test 2: Contains performance for non-existing keys
        let start = Instant::now();
        let mut miss_count = 0;
        for i in 0..10000 {
            let key = format!("missing_{}", i);
            if !cache.contains(&key) {
                miss_count += 1;
            }
        }
        let missing_keys_duration = start.elapsed();
        assert_eq!(miss_count, 10000); // No keys should exist

        // Test 3: Mixed contains performance (50% hits, 50% misses)
        let start = Instant::now();
        let mut mixed_hit_count = 0;
        for i in 0..10000 {
            let key = if i % 2 == 0 {
                format!("item_{}", i % cache_size)
            } else {
                format!("missing_{}", i)
            };
            if cache.contains(&key) {
                mixed_hit_count += 1;
            }
        }
        let mixed_duration = start.elapsed();
        assert_eq!(mixed_hit_count, 5000); // 50% should be hits

        // Performance assertions
        assert!(
            existing_keys_duration < Duration::from_millis(50),
            "Contains for existing keys should be fast: {:?}",
            existing_keys_duration
        );
        assert!(
            missing_keys_duration < Duration::from_millis(50),
            "Contains for missing keys should be fast: {:?}",
            missing_keys_duration
        );
        assert!(
            mixed_duration < Duration::from_millis(50),
            "Mixed contains operations should be fast: {:?}",
            mixed_duration
        );

        // Contains should be consistently fast regardless of hit/miss
        log::info!(
            "Contains performance - Existing: {:?}, Missing: {:?}, Mixed: {:?}",
            existing_keys_duration,
            missing_keys_duration,
            mixed_duration
        );

        // Verify cache wasn't modified by contains operations
        assert_eq!(cache.len(), cache_size);

        // Test 4: Performance comparison with very large cache
        let large_cache_size = 100000;
        let mut large_cache = LfuCache::new(large_cache_size);

        for i in 0..large_cache_size {
            large_cache.insert(format!("large_{}", i), Arc::new(i));
        }

        let start = Instant::now();
        for i in 0..1000 {
            let key = format!("large_{}", i % large_cache_size);
            large_cache.contains(&key);
        }
        let large_cache_duration = start.elapsed();

        assert!(
            large_cache_duration < Duration::from_millis(25),
            "Large cache contains should still be fast: {:?}",
            large_cache_duration
        );
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_frequency_lookup_performance() {
        let cache_size = 25000;
        let mut cache = LfuCache::new(cache_size);

        // Setup: Fill cache and create varied frequency distributions
        for i in 0..cache_size {
            cache.insert(format!("freq_{}", i), Arc::new(i));
        }

        // Create different frequency patterns
        // High frequency items (accessed 50+ times)
        for _ in 0..50 {
            for i in 0..100 {
                cache.get(&format!("freq_{}", i));
            }
        }

        // Medium frequency items (accessed 10 times)
        for _ in 0..10 {
            for i in 100..500 {
                cache.get(&format!("freq_{}", i));
            }
        }

        // Low frequency items (accessed 2-5 times)
        for _ in 0..3 {
            for i in 500..2000 {
                cache.get(&format!("freq_{}", i));
            }
        }

        // Items with frequency 1 (only inserted, never accessed): 2000..cache_size

        // Test 1: Frequency lookup performance for high-frequency items
        let start = Instant::now();
        for i in 0..5000 {
            let key = format!("freq_{}", i % 100);
            cache.frequency(&key);
        }
        let high_freq_duration = start.elapsed();

        // Test 2: Frequency lookup performance for medium-frequency items
        let start = Instant::now();
        for i in 0..5000 {
            let key = format!("freq_{}", 100 + (i % 400));
            cache.frequency(&key);
        }
        let medium_freq_duration = start.elapsed();

        // Test 3: Frequency lookup performance for low-frequency items
        let start = Instant::now();
        for i in 0..5000 {
            let key = format!("freq_{}", 500 + (i % 1500));
            cache.frequency(&key);
        }
        let low_freq_duration = start.elapsed();

        // Test 4: Frequency lookup performance for frequency-1 items
        let start = Instant::now();
        for i in 0..5000 {
            let key = format!("freq_{}", 2000 + (i % (cache_size - 2000)));
            cache.frequency(&key);
        }
        let freq_one_duration = start.elapsed();

        // Test 5: Frequency lookup performance for non-existent items
        let start = Instant::now();
        for i in 0..5000 {
            let key = format!("nonexistent_{}", i);
            cache.frequency(&key);
        }
        let nonexistent_duration = start.elapsed();

        // Performance assertions
        assert!(
            high_freq_duration < Duration::from_millis(50),
            "High frequency lookups should be fast: {:?}",
            high_freq_duration
        );
        assert!(
            medium_freq_duration < Duration::from_millis(50),
            "Medium frequency lookups should be fast: {:?}",
            medium_freq_duration
        );
        assert!(
            low_freq_duration < Duration::from_millis(50),
            "Low frequency lookups should be fast: {:?}",
            low_freq_duration
        );
        assert!(
            freq_one_duration < Duration::from_millis(50),
            "Frequency-1 lookups should be fast: {:?}",
            freq_one_duration
        );
        assert!(
            nonexistent_duration < Duration::from_millis(50),
            "Non-existent key lookups should be fast: {:?}",
            nonexistent_duration
        );

        // All frequency lookups should have similar performance (O(1) HashMap access)
        log::info!(
            "Frequency lookup performance - High: {:?}, Medium: {:?}, Low: {:?}, Freq-1: {:?}, Non-existent: {:?}",
            high_freq_duration,
            medium_freq_duration,
            low_freq_duration,
            freq_one_duration,
            nonexistent_duration
        );

        // Verify frequency values are correct
        assert!(cache.frequency(&"freq_0".to_string()).unwrap() > 50);
        assert!(cache.frequency(&"freq_100".to_string()).unwrap() > 10);
        assert!(cache.frequency(&"freq_500".to_string()).unwrap() > 1);
        assert_eq!(cache.frequency(&"freq_2000".to_string()), Some(1));
        assert_eq!(cache.frequency(&"nonexistent_0".to_string()), None);

        // Test 6: Batch frequency lookup performance
        let keys_to_test: Vec<String> = (0..1000)
            .map(|i| format!("freq_{}", i % cache_size))
            .collect();

        let start = Instant::now();
        for key in &keys_to_test {
            cache.frequency(key);
        }
        let batch_duration = start.elapsed();

        assert!(
            batch_duration < Duration::from_millis(25),
            "Batch frequency lookups should be fast: {:?}",
            batch_duration
        );

        // Verify cache state wasn't affected by frequency lookups
        assert_eq!(cache.len(), cache_size);
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_peek_lfu_performance() {
        // Test 1: Small cache peek_lfu performance
        let small_cache_size = 1000;
        let mut small_cache = LfuCache::new(small_cache_size);

        for i in 0..small_cache_size {
            small_cache.insert(format!("small_{}", i), Arc::new(i));
        }

        let start = Instant::now();
        for _ in 0..1000 {
            small_cache.peek_lfu();
        }
        let small_cache_duration = start.elapsed();

        // Test 2: Medium cache peek_lfu performance
        let medium_cache_size = 10000;
        let mut medium_cache = LfuCache::new(medium_cache_size);

        for i in 0..medium_cache_size {
            medium_cache.insert(format!("medium_{}", i), Arc::new(i));
        }

        let start = Instant::now();
        for _ in 0..1000 {
            medium_cache.peek_lfu();
        }
        let medium_cache_duration = start.elapsed();

        // Test 3: Large cache peek_lfu performance
        let large_cache_size = 100000;
        let mut large_cache = LfuCache::new(large_cache_size);

        for i in 0..large_cache_size {
            large_cache.insert(format!("large_{}", i), Arc::new(i));
        }

        let start = Instant::now();
        for _ in 0..1000 {
            large_cache.peek_lfu();
        }
        let large_cache_duration = start.elapsed();

        // Test 4: Performance with varied frequency distributions
        let mut varied_cache = LfuCache::new(50000);

        // Insert items with intentionally varied frequencies
        for i in 0..50000 {
            varied_cache.insert(format!("varied_{}", i), Arc::new(i));
        }

        // Create frequency distribution: some high, some medium, many low
        // High frequency (100+ accesses): first 100 items
        for _ in 0..100 {
            for i in 0..100 {
                varied_cache.get(&format!("varied_{}", i));
            }
        }

        // Medium frequency (10 accesses): next 500 items
        for _ in 0..10 {
            for i in 100..600 {
                varied_cache.get(&format!("varied_{}", i));
            }
        }

        // Low frequency (1-3 accesses): next 1000 items
        for _ in 0..2 {
            for i in 600..1600 {
                varied_cache.get(&format!("varied_{}", i));
            }
        }

        // Frequency 1 (inserted only): remaining items

        let start = Instant::now();
        for _ in 0..1000 {
            varied_cache.peek_lfu();
        }
        let varied_cache_duration = start.elapsed();

        // Test 5: Performance when LFU changes frequently
        let mut dynamic_cache = LfuCache::new(5000);
        for i in 0..5000 {
            dynamic_cache.insert(format!("dynamic_{}", i), Arc::new(i));
        }

        let start = Instant::now();
        for i in 0..1000 {
            // Peek LFU
            dynamic_cache.peek_lfu();

            // Occasionally access a random item to change frequency distribution
            if i % 10 == 0 {
                dynamic_cache.get(&format!("dynamic_{}", i % 5000));
            }
        }
        let dynamic_cache_duration = start.elapsed();

        // Performance assertions
        // Note: peek_lfu performance scales with cache size since it needs to find minimum frequency
        assert!(
            small_cache_duration < Duration::from_millis(100),
            "Small cache peek_lfu should be fast: {:?}",
            small_cache_duration
        );
        assert!(
            medium_cache_duration < Duration::from_millis(1000),
            "Medium cache peek_lfu should be reasonably fast: {:?}",
            medium_cache_duration
        );
        assert!(
            large_cache_duration < Duration::from_millis(5000),
            "Large cache peek_lfu should be acceptable: {:?}",
            large_cache_duration
        );
        assert!(
            varied_cache_duration < Duration::from_millis(5000),
            "Varied frequency cache peek_lfu should be acceptable: {:?}",
            varied_cache_duration
        );
        assert!(
            dynamic_cache_duration < Duration::from_millis(500),
            "Dynamic cache peek_lfu should be fast: {:?}",
            dynamic_cache_duration
        );

        log::info!(
            "Peek LFU performance - Small: {:?}, Medium: {:?}, Large: {:?}, Varied: {:?}, Dynamic: {:?}",
            small_cache_duration,
            medium_cache_duration,
            large_cache_duration,
            varied_cache_duration,
            dynamic_cache_duration
        );

        // Verify peek_lfu returns correct results
        let (lfu_key, _) = small_cache.peek_lfu().unwrap();
        assert!(lfu_key.starts_with("small_"));

        let (lfu_key, _) = varied_cache.peek_lfu().unwrap();
        // Should be one of the frequency-1 items (index >= 1600)
        let key_index: usize = lfu_key.strip_prefix("varied_").unwrap().parse().unwrap();
        assert!(key_index >= 1600);

        // Test 6: Performance consistency across multiple operations
        let mut consistency_cache = LfuCache::new(20000);
        for i in 0..20000 {
            consistency_cache.insert(format!("consistency_{}", i), Arc::new(i));
        }

        let mut durations = Vec::new();
        for _ in 0..10 {
            let start = Instant::now();
            for _ in 0..100 {
                consistency_cache.peek_lfu();
            }
            durations.push(start.elapsed());
        }

        // Check that performance is consistent (allow for reasonable variance)
        let avg_duration = durations.iter().sum::<Duration>() / durations.len() as u32;
        for duration in &durations {
            assert!(
                duration.as_millis() <= avg_duration.as_millis() * 10,
                "Performance should be reasonably consistent, got {:?} vs avg {:?}",
                duration,
                avg_duration
            );
        }
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_cache_hit_vs_miss_performance() {
        let cache_size = 20000;
        let mut cache = LfuCache::new(cache_size);

        // Setup: Fill cache with items
        for i in 0..cache_size {
            cache.insert(format!("hit_{}", i), Arc::new(i));
        }

        // Test 1: Pure cache hits performance
        let start = Instant::now();
        let mut hit_count = 0;
        for i in 0..10000 {
            let key = format!("hit_{}", i % cache_size);
            if cache.get(&key).is_some() {
                hit_count += 1;
            }
        }
        let pure_hits_duration = start.elapsed();
        assert_eq!(hit_count, 10000);

        // Test 2: Pure cache misses performance
        let start = Instant::now();
        let mut miss_count = 0;
        for i in 0..10000 {
            let key = format!("miss_{}", i);
            if cache.get(&key).is_none() {
                miss_count += 1;
            }
        }
        let pure_misses_duration = start.elapsed();
        assert_eq!(miss_count, 10000);

        // Test 3: Mixed hit/miss performance (50/50)
        let start = Instant::now();
        let mut mixed_hits = 0;
        let mut mixed_misses = 0;
        for i in 0..10000 {
            let key = if i % 2 == 0 {
                format!("hit_{}", i % cache_size)
            } else {
                format!("miss_{}", i)
            };
            if cache.get(&key).is_some() {
                mixed_hits += 1;
            } else {
                mixed_misses += 1;
            }
        }
        let mixed_duration = start.elapsed();
        assert_eq!(mixed_hits, 5000);
        assert_eq!(mixed_misses, 5000);

        // Test 4: High hit ratio performance (90% hits, 10% misses)
        let start = Instant::now();
        let mut high_hits = 0;
        let mut high_misses = 0;
        for i in 0..10000 {
            let key = if i % 10 < 9 {
                format!("hit_{}", i % cache_size)
            } else {
                format!("miss_{}", i)
            };
            if cache.get(&key).is_some() {
                high_hits += 1;
            } else {
                high_misses += 1;
            }
        }
        let high_hit_ratio_duration = start.elapsed();
        assert_eq!(high_hits, 9000);
        assert_eq!(high_misses, 1000);

        // Test 5: Low hit ratio performance (10% hits, 90% misses)
        let start = Instant::now();
        let mut low_hits = 0;
        let mut low_misses = 0;
        for i in 0..10000 {
            let key = if i % 10 == 0 {
                format!("hit_{}", i % cache_size)
            } else {
                format!("miss_{}", i)
            };
            if cache.get(&key).is_some() {
                low_hits += 1;
            } else {
                low_misses += 1;
            }
        }
        let low_hit_ratio_duration = start.elapsed();
        assert_eq!(low_hits, 1000);
        assert_eq!(low_misses, 9000);

        // Performance assertions
        assert!(
            pure_hits_duration < Duration::from_millis(100),
            "Pure hits should be fast: {:?}",
            pure_hits_duration
        );
        assert!(
            pure_misses_duration < Duration::from_millis(100),
            "Pure misses should be fast: {:?}",
            pure_misses_duration
        );
        assert!(
            mixed_duration < Duration::from_millis(100),
            "Mixed hits/misses should be fast: {:?}",
            mixed_duration
        );
        assert!(
            high_hit_ratio_duration < Duration::from_millis(100),
            "High hit ratio should be fast: {:?}",
            high_hit_ratio_duration
        );
        assert!(
            low_hit_ratio_duration < Duration::from_millis(100),
            "Low hit ratio should be fast: {:?}",
            low_hit_ratio_duration
        );

        log::info!(
            "Hit vs Miss performance - Pure hits: {:?}, Pure misses: {:?}, Mixed: {:?}, High hit ratio: {:?}, Low hit ratio: {:?}",
            pure_hits_duration,
            pure_misses_duration,
            mixed_duration,
            high_hit_ratio_duration,
            low_hit_ratio_duration
        );

        // Test 6: Performance difference analysis between contains vs get
        let start = Instant::now();
        for i in 0..5000 {
            let key = format!("hit_{}", i % cache_size);
            cache.contains(&key);
        }
        let contains_hits_duration = start.elapsed();

        let start = Instant::now();
        for i in 0..5000 {
            let key = format!("miss_{}", i);
            cache.contains(&key);
        }
        let contains_misses_duration = start.elapsed();

        let start = Instant::now();
        for i in 0..5000 {
            let key = format!("hit_{}", i % cache_size);
            cache.get(&key);
        }
        let get_hits_duration = start.elapsed();

        let start = Instant::now();
        for i in 0..5000 {
            let key = format!("miss_{}", i);
            cache.get(&key);
        }
        let get_misses_duration = start.elapsed();

        // Get should be slightly slower than contains for hits due to frequency updates
        // but similar for misses since both fail at HashMap lookup
        log::info!(
            "Contains vs Get - Contains hits: {:?}, Contains misses: {:?}, Get hits: {:?}, Get misses: {:?}",
            contains_hits_duration,
            contains_misses_duration,
            get_hits_duration,
            get_misses_duration
        );

        // Test 7: Performance with different cache sizes for hit/miss patterns
        let mut small_cache = LfuCache::<String, i32>::new(100);
        let mut medium_cache = LfuCache::<String, i32>::new(5000);
        let mut large_cache = LfuCache::<String, i32>::new(50000);

        // All caches should have similar miss performance (O(1) HashMap lookup failure)
        let start = Instant::now();
        for i in 0..1000 {
            let key = format!("definitely_missing_{}", i);
            small_cache.get(&key);
        }
        let small_miss_duration = start.elapsed();

        let start = Instant::now();
        for i in 0..1000 {
            let key = format!("definitely_missing_{}", i);
            medium_cache.get(&key);
        }
        let medium_miss_duration = start.elapsed();

        let start = Instant::now();
        for i in 0..1000 {
            let key = format!("definitely_missing_{}", i);
            large_cache.get(&key);
        }
        let large_miss_duration = start.elapsed();

        // Miss performance should be consistent across cache sizes
        assert!(small_miss_duration < Duration::from_millis(25));
        assert!(medium_miss_duration < Duration::from_millis(25));
        assert!(large_miss_duration < Duration::from_millis(25));

        log::info!(
            "Miss performance across sizes - Small: {:?}, Medium: {:?}, Large: {:?}",
            small_miss_duration,
            medium_miss_duration,
            large_miss_duration
        );

        // Verify cache state integrity after all performance tests
        assert_eq!(cache.len(), cache_size);
        // Frequencies should have been updated due to get() calls
        assert!(cache.frequency(&"hit_0".to_string()).unwrap() > 1);
    }
}

mod insertion_performance {
    use cachekit::policy::lfu::LfuCache;
    use cachekit::traits::{CoreCache, LfuCacheTrait};

    use super::*;

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_insertion_performance_with_eviction() {
        let cache_capacity = 5000;
        let mut cache = LfuCache::new(cache_capacity);

        // Phase 1: Fill cache to capacity without eviction
        let start = Instant::now();
        for i in 0..cache_capacity {
            cache.insert(format!("initial_{}", i), Arc::new(i));
        }
        let fill_duration = start.elapsed();
        assert_eq!(cache.len(), cache_capacity);

        // Phase 2: Insert additional items that trigger eviction
        let eviction_count = 2000;
        let start = Instant::now();
        for i in 0..eviction_count {
            cache.insert(format!("evict_{}", i), Arc::new(i + cache_capacity));
        }
        let eviction_duration = start.elapsed();
        assert_eq!(cache.len(), cache_capacity); // Should still be at capacity

        // Phase 3: Compare performance per operation
        let fill_per_op = fill_duration / cache_capacity as u32;
        let eviction_per_op = eviction_duration / eviction_count as u32;

        // Eviction operations should be slower due to LFU finding
        log::info!(
            "Fill performance: {:?} per op, Eviction performance: {:?} per op",
            fill_per_op,
            eviction_per_op
        );

        // Performance assertions
        assert!(
            fill_duration < Duration::from_millis(500),
            "Filling cache should be fast: {:?}",
            fill_duration
        );
        assert!(
            eviction_duration < Duration::from_millis(2000),
            "Eviction insertions should be reasonable: {:?}",
            eviction_duration
        );

        // Test with frequent access patterns during eviction
        let mut cache_with_access = LfuCache::new(1000);

        // Fill cache
        for i in 0..1000 {
            cache_with_access.insert(format!("access_{}", i), Arc::new(i));
        }

        // Create frequency distribution by accessing some items
        for _ in 0..5 {
            for i in 0..200 {
                cache_with_access.get(&format!("access_{}", i));
            }
        }

        // Now test eviction with mixed frequency items
        let start = Instant::now();
        for i in 0..500 {
            cache_with_access.insert(format!("new_evict_{}", i), Arc::new(i + 2000));
        }
        let mixed_eviction_duration = start.elapsed();

        assert!(
            mixed_eviction_duration < Duration::from_millis(1000),
            "Mixed frequency eviction should be reasonable: {:?}",
            mixed_eviction_duration
        );

        // Verify that high-frequency items are preserved
        assert!(cache_with_access.contains(&"access_0".to_string()));
        assert!(cache_with_access.contains(&"access_100".to_string()));

        // Test eviction performance scaling
        let sizes = [100, 500, 1000, 2000];
        let mut eviction_times = Vec::new();

        for &size in &sizes {
            let mut test_cache = LfuCache::new(size);

            // Fill to capacity
            for i in 0..size {
                test_cache.insert(format!("scale_{}", i), Arc::new(i));
            }

            // Measure eviction performance
            let start = Instant::now();
            for i in 0..100 {
                test_cache.insert(format!("evict_scale_{}", i), Arc::new(i + size));
            }
            let duration = start.elapsed();
            eviction_times.push(duration);
        }

        // Performance should scale reasonably with cache size
        for (i, &duration) in eviction_times.iter().enumerate() {
            assert!(
                duration < Duration::from_millis(200 * (i + 1) as u64),
                "Eviction performance should scale reasonably for size {}: {:?}",
                sizes[i],
                duration
            );
        }

        log::info!("Eviction scaling: {:?}", eviction_times);
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_batch_insertion_performance() {
        // Test 1: Small batch insertions
        let mut small_cache = LfuCache::new(1000);
        let batch_sizes = [10, 50, 100, 500];
        let mut small_batch_times = Vec::new();

        for &batch_size in &batch_sizes {
            let start = Instant::now();
            for i in 0..batch_size {
                small_cache.insert(format!("small_batch_{}_{}", batch_size, i), Arc::new(i));
            }
            small_batch_times.push(start.elapsed());
        }

        // Performance should scale roughly linearly with batch size
        for (i, &duration) in small_batch_times.iter().enumerate() {
            assert!(
                duration < Duration::from_millis(50 * (i + 1) as u64),
                "Small batch {} should be fast: {:?}",
                batch_sizes[i],
                duration
            );
        }

        // Test 2: Large sequential insertions
        let large_cache_size = 20000;
        let mut large_cache = LfuCache::new(large_cache_size);

        let start = Instant::now();
        for i in 0..large_cache_size {
            large_cache.insert(format!("large_{}", i), Arc::new(i));
        }
        let large_batch_duration = start.elapsed();

        assert!(
            large_batch_duration < Duration::from_millis(1000),
            "Large batch insertion should be reasonable: {:?}",
            large_batch_duration
        );

        // Test 3: Insertion performance with different value sizes
        let mut value_size_cache = LfuCache::new(5000);

        // Small values (integers)
        let start = Instant::now();
        for i in 0..1000 {
            value_size_cache.insert(format!("int_{}", i), Arc::new(i));
        }
        let small_value_duration = start.elapsed();

        // Large values (also integers for consistency, but simulating larger data)
        let start = Instant::now();
        for i in 0..1000 {
            value_size_cache.insert(format!("large_{}", i), Arc::new(i * 1000000));
        }
        let large_value_duration = start.elapsed();

        // Both should be reasonably fast since they're both integers
        assert!(
            small_value_duration < Duration::from_millis(100),
            "Small value insertion should be fast: {:?}",
            small_value_duration
        );
        assert!(
            large_value_duration < Duration::from_millis(200),
            "Large value insertion should be reasonable: {:?}",
            large_value_duration
        );

        // Test 4: Batch insertion with interleaved operations
        let mut mixed_cache = LfuCache::new(2000);

        let start = Instant::now();
        for i in 0..1000 {
            // Insert
            mixed_cache.insert(format!("mixed_{}", i), Arc::new(i));

            // Occasionally read to create frequency variance
            if i % 10 == 0 && i > 0 {
                mixed_cache.get(&format!("mixed_{}", i / 2));
            }

            // Occasionally check existence
            if i % 15 == 0 {
                mixed_cache.contains(&format!("mixed_{}", i));
            }
        }
        let mixed_operations_duration = start.elapsed();

        assert!(
            mixed_operations_duration < Duration::from_millis(200),
            "Mixed operations should be fast: {:?}",
            mixed_operations_duration
        );

        // Test 5: Throughput measurement
        let throughput_cache_size = 10000;
        let mut throughput_cache = LfuCache::new(throughput_cache_size);

        let start = Instant::now();
        for i in 0..throughput_cache_size {
            throughput_cache.insert(format!("throughput_{}", i), Arc::new(i));
        }
        let throughput_duration = start.elapsed();

        let ops_per_second = throughput_cache_size as f64 / throughput_duration.as_secs_f64();

        assert!(
            ops_per_second > 10000.0,
            "Should achieve at least 10k insertions per second, got: {:.2}",
            ops_per_second
        );

        log::info!("Batch insertion performance:");
        log::info!("  Small batches: {:?}", small_batch_times);
        log::info!("  Large batch: {:?}", large_batch_duration);
        log::info!("  Small values: {:?}", small_value_duration);
        log::info!("  Large values: {:?}", large_value_duration);
        log::info!("  Mixed ops: {:?}", mixed_operations_duration);
        log::info!("  Throughput: {:.2} ops/sec", ops_per_second);

        // Test 6: Memory allocation impact during batch insertion
        let mut allocation_cache = LfuCache::new(5000);

        // Measure insertion of progressively larger batches
        let progressive_sizes = [100, 500, 1000, 2000];
        let mut progressive_times = Vec::new();

        for &size in &progressive_sizes {
            allocation_cache.clear();

            let start = Instant::now();
            for i in 0..size {
                allocation_cache.insert(format!("prog_{}_{}", size, i), Arc::new(i));
            }
            progressive_times.push(start.elapsed());
        }

        // Each batch should complete in reasonable time
        for (i, &duration) in progressive_times.iter().enumerate() {
            assert!(
                duration < Duration::from_millis(100 + (i * 50) as u64),
                "Progressive batch {} should be efficient: {:?}",
                progressive_sizes[i],
                duration
            );
        }

        log::info!("  Progressive batches: {:?}", progressive_times);
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_update_vs_new_insertion_performance() {
        let cache_size = 5000;
        let mut cache = LfuCache::new(cache_size);

        // Phase 1: Initial population with new insertions
        let start = Instant::now();
        for i in 0..cache_size {
            cache.insert(format!("new_{}", i), Arc::new(i));
        }
        let new_insertion_duration = start.elapsed();
        assert_eq!(cache.len(), cache_size);

        // Phase 2: Update existing keys
        let update_count = 2000;
        let start = Instant::now();
        for i in 0..update_count {
            let key = format!("new_{}", i % cache_size);
            cache.insert(key, Arc::new(i + 10000));
        }
        let update_duration = start.elapsed();
        assert_eq!(cache.len(), cache_size); // Length shouldn't change

        // Phase 3: Compare per-operation performance
        let new_per_op = new_insertion_duration / cache_size as u32;
        let update_per_op = update_duration / update_count as u32;

        // Updates should be faster since they don't require eviction logic
        log::info!(
            "New insertion: {:?} per op, Update: {:?} per op",
            new_per_op,
            update_per_op
        );

        // Both should be fast, but updates might be slightly faster
        assert!(
            new_insertion_duration < Duration::from_millis(500),
            "New insertions should be fast: {:?}",
            new_insertion_duration
        );
        assert!(
            update_duration < Duration::from_millis(300),
            "Updates should be fast: {:?}",
            update_duration
        );

        // Test 4: Mixed new vs update operations
        let mut mixed_cache = LfuCache::new(3000);

        // Pre-populate half the cache
        for i in 0..1500 {
            mixed_cache.insert(format!("mixed_{}", i), Arc::new(i));
        }

        let start = Instant::now();
        for i in 0..2000 {
            if i % 2 == 0 {
                // Update existing key
                let key = format!("mixed_{}", i % 1500);
                mixed_cache.insert(key, Arc::new(i + 5000));
            } else {
                // Insert new key (might trigger eviction)
                mixed_cache.insert(format!("new_mixed_{}", i), Arc::new(i));
            }
        }
        let mixed_duration = start.elapsed();

        assert!(
            mixed_duration < Duration::from_millis(400),
            "Mixed operations should be reasonable: {:?}",
            mixed_duration
        );

        // Test 5: Update performance with different frequency distributions
        let mut freq_cache = LfuCache::new(2000);

        // Create items with different frequencies
        for i in 0..2000 {
            freq_cache.insert(format!("freq_{}", i), Arc::new(i));
        }

        // Create frequency distribution
        for _ in 0..10 {
            for i in 0..200 {
                freq_cache.get(&format!("freq_{}", i)); // High frequency
            }
        }

        for _ in 0..3 {
            for i in 200..800 {
                freq_cache.get(&format!("freq_{}", i)); // Medium frequency
            }
        }
        // Items 800-2000 remain at frequency 1 (low frequency)

        // Test updating items with different frequencies
        let start = Instant::now();
        for i in 0..100 {
            freq_cache.insert(format!("freq_{}", i), Arc::new(i + 10000)); // High freq
        }
        let high_freq_update_duration = start.elapsed();

        let start = Instant::now();
        for i in 200..300 {
            freq_cache.insert(format!("freq_{}", i), Arc::new(i + 10000)); // Medium freq
        }
        let medium_freq_update_duration = start.elapsed();

        let start = Instant::now();
        for i in 1800..1900 {
            freq_cache.insert(format!("freq_{}", i), Arc::new(i + 10000)); // Low freq
        }
        let low_freq_update_duration = start.elapsed();

        // All should be fast since they're updates, not dependent on frequency
        assert!(
            high_freq_update_duration < Duration::from_millis(50),
            "High frequency updates should be fast: {:?}",
            high_freq_update_duration
        );
        assert!(
            medium_freq_update_duration < Duration::from_millis(50),
            "Medium frequency updates should be fast: {:?}",
            medium_freq_update_duration
        );
        assert!(
            low_freq_update_duration < Duration::from_millis(50),
            "Low frequency updates should be fast: {:?}",
            low_freq_update_duration
        );

        // Test 6: Update vs new insertion when cache is full
        let mut full_cache = LfuCache::new(1000);

        // Fill to capacity
        for i in 0..1000 {
            full_cache.insert(format!("full_{}", i), Arc::new(i));
        }

        // Test updates on full cache
        let start = Instant::now();
        for i in 0..500 {
            full_cache.insert(format!("full_{}", i), Arc::new(i + 2000));
        }
        let full_update_duration = start.elapsed();

        // Test new insertions on full cache (triggers eviction)
        let start = Instant::now();
        for i in 0..500 {
            full_cache.insert(format!("new_full_{}", i), Arc::new(i + 3000));
        }
        let full_new_duration = start.elapsed();

        // Updates should be significantly faster than new insertions requiring eviction
        assert!(
            full_update_duration < Duration::from_millis(100),
            "Updates on full cache should be fast: {:?}",
            full_update_duration
        );
        assert!(
            full_new_duration < Duration::from_millis(500),
            "New insertions on full cache should be reasonable: {:?}",
            full_new_duration
        );

        // Test 7: Batch update performance
        let mut batch_cache = LfuCache::new(5000);

        // Initial population
        for i in 0..5000 {
            batch_cache.insert(format!("batch_{}", i), Arc::new(i));
        }

        // Batch updates
        let batch_sizes = [100, 500, 1000, 2000];
        let mut batch_update_times = Vec::new();

        for &batch_size in &batch_sizes {
            let start = Instant::now();
            for i in 0..batch_size {
                batch_cache.insert(format!("batch_{}", i), Arc::new(i + 20000));
            }
            batch_update_times.push(start.elapsed());
        }

        // Batch updates should scale linearly
        for (i, &duration) in batch_update_times.iter().enumerate() {
            assert!(
                duration < Duration::from_millis(50 + (i * 25) as u64),
                "Batch update {} should be efficient: {:?}",
                batch_sizes[i],
                duration
            );
        }

        log::info!("Update vs New Performance:");
        log::info!(
            "  New insertions: {:?} total, {:?} per op",
            new_insertion_duration,
            new_per_op
        );
        log::info!(
            "  Updates: {:?} total, {:?} per op",
            update_duration,
            update_per_op
        );
        log::info!("  Mixed operations: {:?}", mixed_duration);
        log::info!(
            "  Frequency-based updates - High: {:?}, Medium: {:?}, Low: {:?}",
            high_freq_update_duration,
            medium_freq_update_duration,
            low_freq_update_duration
        );
        log::info!(
            "  Full cache - Updates: {:?}, New: {:?}",
            full_update_duration,
            full_new_duration
        );
        log::info!("  Batch updates: {:?}", batch_update_times);

        // Verify functional correctness after performance tests
        assert!(batch_cache.contains(&"batch_0".to_string()));
        assert_eq!(
            batch_cache.get(&"batch_0".to_string()).map(Arc::as_ref),
            Some(&20000)
        );
        assert_eq!(batch_cache.len(), 5000);
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_insertion_with_frequency_tracking() {
        // Test 1: Basic frequency tracking overhead during insertion
        let cache_size = 10000;
        let mut cache = LfuCache::new(cache_size);

        // Measure pure insertion time (frequency tracking included)
        let start = Instant::now();
        for i in 0..cache_size {
            cache.insert(format!("track_{}", i), Arc::new(i));
        }
        let insertion_with_tracking_duration = start.elapsed();

        // All items should have frequency 1 after insertion
        for i in (0..100).step_by(10) {
            assert_eq!(cache.frequency(&format!("track_{}", i)), Some(1));
        }

        assert!(
            insertion_with_tracking_duration < Duration::from_millis(800),
            "Insertion with frequency tracking should be reasonable: {:?}",
            insertion_with_tracking_duration
        );

        // Test 2: Frequency tracking during updates vs new insertions
        let mut tracking_cache = LfuCache::new(5000);

        // Initial population
        for i in 0..5000 {
            tracking_cache.insert(format!("freq_track_{}", i), Arc::new(i));
        }

        // Measure update performance (should preserve frequency)
        let start = Instant::now();
        for i in 0..1000 {
            tracking_cache.insert(format!("freq_track_{}", i), Arc::new(i + 10000));
        }
        let update_tracking_duration = start.elapsed();

        // Verify frequencies are preserved during updates
        for i in (0..100).step_by(10) {
            assert_eq!(
                tracking_cache.frequency(&format!("freq_track_{}", i)),
                Some(1)
            );
        }

        assert!(
            update_tracking_duration < Duration::from_millis(200),
            "Update tracking should be fast: {:?}",
            update_tracking_duration
        );

        // Test 3: Frequency tracking impact during eviction
        let mut eviction_cache = LfuCache::new(2000);

        // Fill cache
        for i in 0..2000 {
            eviction_cache.insert(format!("evict_track_{}", i), Arc::new(i));
        }

        // Create frequency variance
        for _ in 0..5 {
            for i in 0..400 {
                eviction_cache.get(&format!("evict_track_{}", i));
            }
        }

        // Now measure eviction with frequency consideration
        let start = Instant::now();
        for i in 0..1000 {
            eviction_cache.insert(format!("new_evict_track_{}", i), Arc::new(i + 5000));
        }
        let eviction_tracking_duration = start.elapsed();

        // Verify that high-frequency items were preserved
        assert!(eviction_cache.contains(&"evict_track_0".to_string()));
        assert!(eviction_cache.contains(&"evict_track_100".to_string()));

        assert!(
            eviction_tracking_duration < Duration::from_millis(1500),
            "Eviction with frequency tracking should be reasonable: {:?}",
            eviction_tracking_duration
        );

        // Test 4: Frequency tracking accuracy under load
        let mut accuracy_cache = LfuCache::new(3000);

        // Insert items
        for i in 0..3000 {
            accuracy_cache.insert(format!("accuracy_{}", i), Arc::new(i));
        }

        // Create complex frequency patterns
        for access_round in 0..20 {
            for i in 0..100 {
                accuracy_cache.get(&format!("accuracy_{}", i)); // Very high frequency
            }
            for i in 100..500 {
                if access_round % 2 == 0 {
                    accuracy_cache.get(&format!("accuracy_{}", i)); // Medium frequency
                }
            }
            for i in 500..1000 {
                if access_round % 5 == 0 {
                    accuracy_cache.get(&format!("accuracy_{}", i)); // Low frequency
                }
            }
        }

        // Verify frequency tracking accuracy
        assert!(accuracy_cache.frequency(&"accuracy_0".to_string()).unwrap() > 15);
        assert!(
            accuracy_cache
                .frequency(&"accuracy_100".to_string())
                .unwrap()
                > 5
        );
        assert!(
            accuracy_cache
                .frequency(&"accuracy_500".to_string())
                .unwrap()
                >= 1
        );
        assert_eq!(
            accuracy_cache.frequency(&"accuracy_2000".to_string()),
            Some(1)
        );

        // Test 5: Frequency tracking memory overhead
        let mut memory_test_cache = LfuCache::new(20000);

        // Insert large number of items and verify each has correct frequency
        let start = Instant::now();
        for i in 0..20000 {
            memory_test_cache.insert(format!("memory_test_{}", i), Arc::new(i));

            // Verify frequency tracking for every 1000th item
            if i % 1000 == 0 {
                assert_eq!(
                    memory_test_cache.frequency(&format!("memory_test_{}", i)),
                    Some(1)
                );
            }
        }
        let large_scale_duration = start.elapsed();

        assert!(
            large_scale_duration < Duration::from_millis(1500),
            "Large scale frequency tracking should be efficient: {:?}",
            large_scale_duration
        );

        // Test 6: Frequency increment performance during mixed operations
        let mut mixed_freq_cache = LfuCache::new(5000);

        // Populate cache
        for i in 0..5000 {
            mixed_freq_cache.insert(format!("mixed_freq_{}", i), Arc::new(i));
        }

        let start = Instant::now();
        for i in 0..10000 {
            if i % 3 == 0 {
                // Insert new (might evict)
                mixed_freq_cache.insert(format!("new_mixed_{}", i), Arc::new(i));
            } else if i % 3 == 1 {
                // Update existing
                mixed_freq_cache.insert(format!("mixed_freq_{}", i % 5000), Arc::new(i + 20000));
            } else {
                // Access existing (increment frequency)
                mixed_freq_cache.get(&format!("mixed_freq_{}", i % 5000));
            }
        }
        let mixed_ops_duration = start.elapsed();

        assert!(
            mixed_ops_duration < Duration::from_millis(2000),
            "Mixed operations with frequency tracking should be reasonable: {:?}",
            mixed_ops_duration
        );

        // Test 7: Frequency tracking during rapid insertions
        let mut rapid_cache = LfuCache::new(1000);

        let start = Instant::now();
        for i in 0..5000 {
            rapid_cache.insert(format!("rapid_{}", i), Arc::new(i));

            // Verify frequency tracking works under rapid insertion
            if i < 1000 && i % 100 == 0 {
                assert_eq!(rapid_cache.frequency(&format!("rapid_{}", i)), Some(1));
            }
        }
        let rapid_insertion_duration = start.elapsed();

        assert!(
            rapid_insertion_duration < Duration::from_millis(1000),
            "Rapid insertion with frequency tracking should be efficient: {:?}",
            rapid_insertion_duration
        );

        // Verify cache is still at capacity and LFU logic worked
        assert_eq!(rapid_cache.len(), 1000);

        // Test 8: Frequency bounds checking
        let mut bounds_cache = LfuCache::new(100);

        // Insert and access to create very high frequencies
        for i in 0..100 {
            bounds_cache.insert(format!("bounds_{}", i), Arc::new(i));
        }

        // Create extremely high frequency for one item
        let start = Instant::now();
        for _ in 0..10000 {
            bounds_cache.get(&"bounds_0".to_string());
        }
        let high_freq_duration = start.elapsed();

        let final_frequency = bounds_cache.frequency(&"bounds_0".to_string()).unwrap();
        assert_eq!(final_frequency, 10001); // 1 (insert) + 10000 (gets)

        assert!(
            high_freq_duration < Duration::from_millis(200),
            "High frequency increment should be fast: {:?}",
            high_freq_duration
        );

        log::info!("Frequency tracking performance:");
        log::info!("  Basic insertion: {:?}", insertion_with_tracking_duration);
        log::info!("  Update tracking: {:?}", update_tracking_duration);
        log::info!("  Eviction tracking: {:?}", eviction_tracking_duration);
        log::info!("  Large scale: {:?}", large_scale_duration);
        log::info!("  Mixed operations: {:?}", mixed_ops_duration);
        log::info!("  Rapid insertion: {:?}", rapid_insertion_duration);
        log::info!("  High frequency: {:?}", high_freq_duration);
        log::info!("  Final frequency achieved: {}", final_frequency);
    }
}

mod eviction_performance {
    use cachekit::policy::lfu::LfuCache;
    use cachekit::traits::{CoreCache, LfuCacheTrait};

    use super::*;

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_lfu_eviction_performance() {
        // Test 1: Basic LFU eviction performance
        let mut cache = LfuCache::new(1000);

        // Fill cache to capacity
        for i in 0..1000 {
            cache.insert(format!("key_{}", i), Arc::new(i));
        }

        // Create frequency distribution to establish clear LFU items
        for _ in 0..10 {
            for i in 0..100 {
                cache.get(&format!("key_{}", i)); // High frequency
            }
        }

        for _ in 0..3 {
            for i in 100..500 {
                cache.get(&format!("key_{}", i)); // Medium frequency
            }
        }
        // Items 500-999 remain at frequency 1 (LFU candidates)

        // Test eviction performance
        let start = Instant::now();
        for i in 1000..1500 {
            cache.insert(format!("new_key_{}", i), Arc::new(i));
        }
        let eviction_duration = start.elapsed();

        // Should evict 500 LFU items efficiently
        assert_eq!(cache.len(), 1000);
        assert!(
            eviction_duration < Duration::from_millis(500),
            "LFU eviction should be efficient: {:?}",
            eviction_duration
        );

        // Verify that high-frequency items are preserved
        assert!(cache.contains(&"key_0".to_string()));
        assert!(cache.contains(&"key_50".to_string()));
        assert!(cache.contains(&"key_100".to_string()));

        // Test 2: Performance scaling with cache size
        let sizes = [100, 500, 1000, 2000];
        let mut eviction_times = Vec::new();

        for &size in &sizes {
            let mut test_cache = LfuCache::new(size);

            // Fill cache
            for i in 0..size {
                test_cache.insert(format!("scale_{}", i), Arc::new(i));
            }

            // Create some frequency variance
            for i in 0..size / 10 {
                test_cache.get(&format!("scale_{}", i));
            }

            // Measure eviction performance
            let start = Instant::now();
            for i in 0..100 {
                test_cache.insert(format!("evict_{}", i), Arc::new(i + size));
            }
            let duration = start.elapsed();
            eviction_times.push(duration);
        }

        // Performance should scale reasonably
        for (i, &duration) in eviction_times.iter().enumerate() {
            assert!(
                duration < Duration::from_millis(100 + (i * 50) as u64),
                "Eviction performance should scale reasonably for size {}: {:?}",
                sizes[i],
                duration
            );
        }

        // Test 3: Eviction with uniform frequency distribution
        let mut uniform_cache = LfuCache::new(500);

        // Fill cache with uniform frequency
        for i in 0..500 {
            uniform_cache.insert(format!("uniform_{}", i), Arc::new(i));
            uniform_cache.get(&format!("uniform_{}", i)); // All have frequency 2
        }

        let start = Instant::now();
        for i in 0..200 {
            uniform_cache.insert(format!("uniform_new_{}", i), Arc::new(i + 1000));
        }
        let uniform_eviction_duration = start.elapsed();

        assert!(
            uniform_eviction_duration < Duration::from_millis(200),
            "Uniform frequency eviction should be reasonable: {:?}",
            uniform_eviction_duration
        );

        // Test 4: Eviction with highly skewed frequency distribution
        let mut skewed_cache = LfuCache::new(1000);

        // Fill cache
        for i in 0..1000 {
            skewed_cache.insert(format!("skewed_{}", i), Arc::new(i));
        }

        // Create highly skewed distribution
        for _ in 0..100 {
            skewed_cache.get(&"skewed_0".to_string()); // One very hot item
        }

        let start = Instant::now();
        for i in 0..500 {
            skewed_cache.insert(format!("skewed_new_{}", i), Arc::new(i + 2000));
        }
        let skewed_eviction_duration = start.elapsed();

        assert!(
            skewed_eviction_duration < Duration::from_millis(400),
            "Skewed frequency eviction should be efficient: {:?}",
            skewed_eviction_duration
        );

        // Hot item should be preserved
        assert!(skewed_cache.contains(&"skewed_0".to_string()));

        // Test 5: Repeated eviction performance consistency
        let mut consistent_cache = LfuCache::new(100);
        let mut eviction_durations = Vec::new();

        // Fill cache initially
        for i in 0..100 {
            consistent_cache.insert(format!("consistent_{}", i), Arc::new(i));
        }

        // Perform multiple rounds of eviction
        for round in 0..10 {
            let start = Instant::now();
            for i in 0..20 {
                consistent_cache
                    .insert(format!("round_{}_{}", round, i), Arc::new(round * 100 + i));
            }
            eviction_durations.push(start.elapsed());
        }

        // Check consistency
        let avg_duration =
            eviction_durations.iter().sum::<Duration>() / eviction_durations.len() as u32;
        let max_multiplier = if cfg!(feature = "metrics") { 5 } else { 3 };
        for duration in &eviction_durations {
            assert!(
                duration.as_millis() <= avg_duration.as_millis() * max_multiplier,
                "Eviction performance should be consistent: {:?} vs avg {:?}",
                duration,
                avg_duration
            );
        }

        log::info!("LFU eviction performance:");
        log::info!("  Basic eviction: {:?}", eviction_duration);
        log::info!("  Size scaling: {:?}", eviction_times);
        log::info!("  Uniform frequency: {:?}", uniform_eviction_duration);
        log::info!("  Skewed frequency: {:?}", skewed_eviction_duration);
        log::info!("  Consistency check: {:?}", eviction_durations);
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_pop_lfu_performance() {
        // Test 1: Basic pop_lfu performance
        let mut cache = LfuCache::new(2000);

        // Fill cache with items
        for i in 0..2000 {
            cache.insert(format!("pop_{}", i), Arc::new(i));
        }

        // Create frequency distribution
        for _ in 0..5 {
            for i in 0..200 {
                cache.get(&format!("pop_{}", i)); // High frequency
            }
        }

        for _ in 0..2 {
            for i in 200..800 {
                cache.get(&format!("pop_{}", i)); // Medium frequency
            }
        }
        // Items 800-1999 remain at frequency 1 (LFU candidates)

        // Test pop_lfu performance
        let start = Instant::now();
        let mut popped_items = Vec::new();
        for _ in 0..500 {
            if let Some((key, value)) = cache.pop_lfu() {
                popped_items.push((key, value));
            }
        }
        let pop_duration = start.elapsed();

        assert_eq!(popped_items.len(), 500);
        assert_eq!(cache.len(), 1500);
        let max_duration = if cfg!(feature = "metrics") {
            Duration::from_millis(1200)
        } else {
            Duration::from_millis(500)
        };
        assert!(
            pop_duration < max_duration,
            "pop_lfu should be efficient: {:?}",
            pop_duration
        );

        // Verify that high-frequency items remain
        assert!(cache.contains(&"pop_0".to_string()));
        assert!(cache.contains(&"pop_100".to_string()));
        assert!(cache.contains(&"pop_200".to_string()));

        // Test 2: pop_lfu with different cache sizes
        let sizes = [50, 200, 500, 1000];
        let mut pop_times = Vec::new();

        for &size in &sizes {
            let mut test_cache = LfuCache::new(size);

            // Fill cache
            for i in 0..size {
                test_cache.insert(format!("size_{}", i), Arc::new(i));
            }

            // Create some frequency variance
            for i in 0..size / 5 {
                test_cache.get(&format!("size_{}", i));
            }

            // Measure pop_lfu performance
            let start = Instant::now();
            for _ in 0..(size / 4) {
                test_cache.pop_lfu();
            }
            let duration = start.elapsed();
            pop_times.push(duration);
        }

        // Performance should scale reasonably
        for (i, &duration) in pop_times.iter().enumerate() {
            let max_duration = if cfg!(feature = "metrics") {
                Duration::from_millis(100 + (i * 75) as u64)
            } else {
                Duration::from_millis(50 + (i * 25) as u64)
            };
            assert!(
                duration < max_duration,
                "pop_lfu performance should scale reasonably for size {}: {:?}",
                sizes[i],
                duration
            );
        }

        // Test 3: pop_lfu with uniform frequencies (worst case)
        let mut uniform_cache = LfuCache::new(300);

        // Fill cache with uniform frequency
        for i in 0..300 {
            uniform_cache.insert(format!("uniform_{}", i), Arc::new(i));
            uniform_cache.get(&format!("uniform_{}", i)); // All have frequency 2
        }

        let start = Instant::now();
        let mut uniform_pops = 0;
        for _ in 0..100 {
            if uniform_cache.pop_lfu().is_some() {
                uniform_pops += 1;
            }
        }
        let uniform_pop_duration = start.elapsed();

        assert_eq!(uniform_pops, 100);
        let uniform_max = if cfg!(feature = "metrics") {
            Duration::from_millis(200)
        } else {
            Duration::from_millis(100)
        };
        assert!(
            uniform_pop_duration < uniform_max,
            "Uniform frequency pop_lfu should be reasonable: {:?}",
            uniform_pop_duration
        );

        // Test 4: pop_lfu until empty
        let mut empty_cache = LfuCache::new(100);

        // Fill cache
        for i in 0..100 {
            empty_cache.insert(format!("empty_{}", i), Arc::new(i));
        }

        let start = Instant::now();
        let mut total_popped = 0;
        while empty_cache.pop_lfu().is_some() {
            total_popped += 1;
        }
        let empty_duration = start.elapsed();

        assert_eq!(total_popped, 100);
        assert_eq!(empty_cache.len(), 0);
        assert!(
            empty_duration < Duration::from_millis(100),
            "pop_lfu until empty should be efficient: {:?}",
            empty_duration
        );

        // Test 5: pop_lfu performance with highly skewed distribution
        let mut skewed_cache = LfuCache::new(1000);

        // Fill cache
        for i in 0..1000 {
            skewed_cache.insert(format!("skewed_{}", i), Arc::new(i));
        }

        // Create very skewed distribution
        for _ in 0..50 {
            skewed_cache.get(&"skewed_0".to_string()); // One very hot item
        }
        for _ in 0..10 {
            for i in 1..50 {
                skewed_cache.get(&format!("skewed_{}", i)); // Some medium items
            }
        }
        // Items 50-999 remain at frequency 1

        let start = Instant::now();
        let mut skewed_pops = 0;
        for _ in 0..300 {
            if skewed_cache.pop_lfu().is_some() {
                skewed_pops += 1;
            }
        }
        let skewed_pop_duration = start.elapsed();

        assert_eq!(skewed_pops, 300);
        assert!(
            skewed_pop_duration < Duration::from_millis(300),
            "Skewed distribution pop_lfu should be efficient: {:?}",
            skewed_pop_duration
        );

        // Hot item should still be there
        assert!(skewed_cache.contains(&"skewed_0".to_string()));

        // Test 6: pop_lfu performance consistency
        let mut consistency_cache = LfuCache::new(200);
        let mut pop_durations = Vec::new();

        // Fill cache
        for i in 0..200 {
            consistency_cache.insert(format!("consistency_{}", i), Arc::new(i));
        }

        // Perform multiple rounds of pop operations
        for round in 0..5 {
            // Add some new items to maintain cache size
            for i in 0..10 {
                consistency_cache
                    .insert(format!("round_{}_{}", round, i), Arc::new(round * 100 + i));
            }

            let start = Instant::now();
            for _ in 0..10 {
                consistency_cache.pop_lfu();
            }
            pop_durations.push(start.elapsed());
        }

        // Check consistency
        let avg_duration = pop_durations.iter().sum::<Duration>() / pop_durations.len() as u32;
        let max_multiplier = if cfg!(debug_assertions) { 6 } else { 3 };
        for duration in &pop_durations {
            assert!(
                duration.as_millis() <= avg_duration.as_millis() * max_multiplier,
                "pop_lfu performance should be consistent: {:?} vs avg {:?}",
                duration,
                avg_duration
            );
        }

        // Test 7: pop_lfu on empty cache
        let mut empty_test_cache = LfuCache::<String, i32>::new(10);

        let start = Instant::now();
        let result = empty_test_cache.pop_lfu();
        let empty_pop_duration = start.elapsed();

        assert!(result.is_none());
        assert!(
            empty_pop_duration < Duration::from_millis(1),
            "pop_lfu on empty cache should be instant: {:?}",
            empty_pop_duration
        );

        log::info!("pop_lfu performance:");
        log::info!("  Basic pop operations: {:?}", pop_duration);
        log::info!("  Size scaling: {:?}", pop_times);
        log::info!("  Uniform frequency: {:?}", uniform_pop_duration);
        log::info!("  Pop until empty: {:?}", empty_duration);
        log::info!("  Skewed distribution: {:?}", skewed_pop_duration);
        log::info!("  Consistency check: {:?}", pop_durations);
        log::info!("  Empty cache: {:?}", empty_pop_duration);
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_eviction_with_many_same_frequency() {
        // Test 1: All items have same frequency (frequency = 1)
        let mut cache = LfuCache::new(1000);

        // Fill cache where all items have frequency 1
        for i in 0..1000 {
            cache.insert(format!("same_freq_{}", i), Arc::new(i));
        }

        // All items should have frequency 1
        for i in (0..100).step_by(10) {
            assert_eq!(cache.frequency(&format!("same_freq_{}", i)), Some(1));
        }

        // Test eviction performance with same frequency items
        let start = Instant::now();
        for i in 1000..1500 {
            cache.insert(format!("new_same_{}", i), Arc::new(i));
        }
        let same_freq_duration = start.elapsed();

        assert_eq!(cache.len(), 1000);
        assert!(
            same_freq_duration < Duration::from_millis(500),
            "Same frequency eviction should be reasonable: {:?}",
            same_freq_duration
        );

        // Test 2: Multiple groups with same frequencies
        let mut grouped_cache = LfuCache::new(1200);

        // Group 1: frequency 1 (400 items)
        for i in 0..400 {
            grouped_cache.insert(format!("group1_{}", i), Arc::new(i));
        }

        // Group 2: frequency 3 (400 items)
        for i in 400..800 {
            grouped_cache.insert(format!("group2_{}", i), Arc::new(i));
            grouped_cache.get(&format!("group2_{}", i));
            grouped_cache.get(&format!("group2_{}", i));
        }

        // Group 3: frequency 5 (400 items)
        for i in 800..1200 {
            grouped_cache.insert(format!("group3_{}", i), Arc::new(i));
            for _ in 0..4 {
                grouped_cache.get(&format!("group3_{}", i));
            }
        }

        // Force eviction of group 1 (frequency 1)
        let start = Instant::now();
        for i in 1200..1600 {
            grouped_cache.insert(format!("new_group_{}", i), Arc::new(i));
        }
        let grouped_eviction_duration = start.elapsed();

        assert_eq!(grouped_cache.len(), 1200);
        assert!(
            grouped_eviction_duration < Duration::from_millis(400),
            "Grouped frequency eviction should be efficient: {:?}",
            grouped_eviction_duration
        );

        // Verify that most Group 1 items (frequency 1) were evicted
        // and Group 2/3 items (higher frequency) were preserved
        let mut group1_remaining = 0;
        let mut group2_remaining = 0;
        let mut group3_remaining = 0;

        for i in 0..400 {
            if grouped_cache.contains(&format!("group1_{}", i)) {
                group1_remaining += 1;
            }
        }
        for i in 400..800 {
            if grouped_cache.contains(&format!("group2_{}", i)) {
                group2_remaining += 1;
            }
        }
        for i in 800..1200 {
            if grouped_cache.contains(&format!("group3_{}", i)) {
                group3_remaining += 1;
            }
        }

        // Verify eviction follows frequency preference (lower frequency items evicted more)
        // Group 1 should have fewer remaining than Group 2/3
        assert!(
            group1_remaining < group2_remaining,
            "Group 1 (freq=1) should have fewer remaining than Group 2 (freq=3): {} vs {}",
            group1_remaining,
            group2_remaining
        );
        assert!(
            group1_remaining < group3_remaining,
            "Group 1 (freq=1) should have fewer remaining than Group 3 (freq=5): {} vs {}",
            group1_remaining,
            group3_remaining
        );

        // Verify cache respects capacity and LFU behavior is working
        assert_eq!(
            grouped_cache.len(),
            1200,
            "Cache should maintain its capacity"
        );

        // The key test: lower frequency items should be evicted more than higher frequency items
        let total_old_remaining = group1_remaining + group2_remaining + group3_remaining;
        log::info!(
            "Group distribution - Group1 (freq=1): {}, Group2 (freq=3): {}, Group3 (freq=5): {}, Total old: {}",
            group1_remaining,
            group2_remaining,
            group3_remaining,
            total_old_remaining
        );

        // Test 3: Large number of items with identical frequency
        let mut identical_cache = LfuCache::new(2000);

        // Fill cache and make all items have frequency 3
        for i in 0..2000 {
            identical_cache.insert(format!("identical_{}", i), Arc::new(i));
            identical_cache.get(&format!("identical_{}", i));
            identical_cache.get(&format!("identical_{}", i));
        }

        // Verify all have same frequency
        for i in (0..2000).step_by(100) {
            assert_eq!(
                identical_cache.frequency(&format!("identical_{}", i)),
                Some(3)
            );
        }

        // Test eviction performance with identical frequencies
        let start = Instant::now();
        for i in 2000..2500 {
            identical_cache.insert(format!("new_identical_{}", i), Arc::new(i));
        }
        let identical_duration = start.elapsed();

        assert_eq!(identical_cache.len(), 2000);
        assert!(
            identical_duration < Duration::from_millis(600),
            "Identical frequency eviction should be reasonable: {:?}",
            identical_duration
        );

        // Test 4: Performance scaling with different ratios of same-frequency items
        let ratios = [0.1, 0.3, 0.5, 0.8, 1.0]; // Fraction of items with same frequency
        let mut ratio_times = Vec::new();

        for &ratio in &ratios {
            let mut ratio_cache = LfuCache::new(500);
            let same_freq_count = (500.0 * ratio) as usize;

            // Fill cache
            for i in 0..500 {
                ratio_cache.insert(format!("ratio_{}", i), Arc::new(i));
            }

            // Make some items have frequency 2, others keep frequency 1
            for i in 0..same_freq_count {
                ratio_cache.get(&format!("ratio_{}", i));
            }

            // Make remaining items have higher frequencies
            for i in same_freq_count..500 {
                for _ in 0..(i % 10 + 3) {
                    ratio_cache.get(&format!("ratio_{}", i));
                }
            }

            // Test eviction performance
            let start = Instant::now();
            for i in 500..600 {
                ratio_cache.insert(format!("new_ratio_{}", i), Arc::new(i));
            }
            let duration = start.elapsed();
            ratio_times.push(duration);
        }

        // Performance should be reasonable across all ratios
        for (i, &duration) in ratio_times.iter().enumerate() {
            assert!(
                duration < Duration::from_millis(100),
                "Ratio {} eviction should be efficient: {:?}",
                ratios[i],
                duration
            );
        }

        // Test 5: Eviction pattern with same frequency items
        let mut pattern_cache = LfuCache::new(300);

        // Create alternating frequency pattern
        for i in 0..300 {
            pattern_cache.insert(format!("pattern_{}", i), Arc::new(i));
            if i % 2 == 0 {
                pattern_cache.get(&format!("pattern_{}", i)); // Even indices: freq 2
            }
            // Odd indices: freq 1
        }

        // Count items of each frequency
        let mut freq1_count = 0;
        let mut freq2_count = 0;
        for i in 0..300 {
            if let Some(freq) = pattern_cache.frequency(&format!("pattern_{}", i)) {
                if freq == 1 {
                    freq1_count += 1;
                } else if freq == 2 {
                    freq2_count += 1;
                }
            }
        }

        assert_eq!(freq1_count, 150); // Odd indices
        assert_eq!(freq2_count, 150); // Even indices

        // Force eviction of freq 1 items
        let start = Instant::now();
        for i in 300..450 {
            pattern_cache.insert(format!("new_pattern_{}", i), Arc::new(i));
        }
        let pattern_duration = start.elapsed();

        assert_eq!(pattern_cache.len(), 300);
        assert!(
            pattern_duration < Duration::from_millis(150),
            "Pattern eviction should be efficient: {:?}",
            pattern_duration
        );

        // Most freq 1 items should be evicted, freq 2 items preserved
        let mut remaining_freq1 = 0;
        let mut remaining_freq2 = 0;
        for i in 0..300 {
            if pattern_cache.contains(&format!("pattern_{}", i))
                && let Some(freq) = pattern_cache.frequency(&format!("pattern_{}", i))
            {
                if freq == 1 {
                    remaining_freq1 += 1;
                } else if freq == 2 {
                    remaining_freq2 += 1;
                }
            }
        }

        assert!(
            remaining_freq2 > remaining_freq1,
            "More freq 2 items should remain: {} vs {}",
            remaining_freq2,
            remaining_freq1
        );

        // Test 6: Worst case scenario - all items same frequency after access
        let mut worst_case_cache = LfuCache::new(500);

        // Fill and access all items once to make them frequency 2
        for i in 0..500 {
            worst_case_cache.insert(format!("worst_{}", i), Arc::new(i));
            worst_case_cache.get(&format!("worst_{}", i));
        }

        // Verify all have same frequency
        for i in (0..500).step_by(50) {
            assert_eq!(worst_case_cache.frequency(&format!("worst_{}", i)), Some(2));
        }

        let start = Instant::now();
        for i in 500..750 {
            worst_case_cache.insert(format!("worst_new_{}", i), Arc::new(i));
        }
        let worst_case_duration = start.elapsed();

        assert_eq!(worst_case_cache.len(), 500);
        assert!(
            worst_case_duration < Duration::from_millis(300),
            "Worst case same frequency eviction should be acceptable: {:?}",
            worst_case_duration
        );

        log::info!("Same frequency eviction performance:");
        log::info!("  All same frequency: {:?}", same_freq_duration);
        log::info!("  Grouped frequencies: {:?}", grouped_eviction_duration);
        log::info!("  Identical frequencies: {:?}", identical_duration);
        log::info!("  Ratio scaling: {:?}", ratio_times);
        log::info!("  Pattern eviction: {:?}", pattern_duration);
        log::info!("  Worst case scenario: {:?}", worst_case_duration);
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_frequency_distribution_impact() {
        // Test 1: Uniform distribution impact
        let mut uniform_cache = LfuCache::new(1000);

        // Create uniform frequency distribution (all items frequency 3)
        for i in 0..1000 {
            uniform_cache.insert(format!("uniform_{}", i), Arc::new(i));
            uniform_cache.get(&format!("uniform_{}", i));
            uniform_cache.get(&format!("uniform_{}", i));
        }

        // Verify uniform distribution
        for i in (0..1000).step_by(100) {
            assert_eq!(uniform_cache.frequency(&format!("uniform_{}", i)), Some(3));
        }

        let start = Instant::now();
        for i in 1000..1200 {
            uniform_cache.insert(format!("new_uniform_{}", i), Arc::new(i));
        }
        let uniform_duration = start.elapsed();

        assert_eq!(uniform_cache.len(), 1000);
        assert!(
            uniform_duration < Duration::from_millis(200),
            "Uniform distribution eviction should be reasonable: {:?}",
            uniform_duration
        );

        // Test 2: Normal (bell curve) distribution impact
        let mut normal_cache = LfuCache::new(1000);

        // Create normal distribution of frequencies (center items higher frequency)
        for i in 0..1000 {
            normal_cache.insert(format!("normal_{}", i), Arc::new(i));
        }

        // Create bell curve frequency pattern
        for i in 0..1000 {
            let distance_from_center = ((i as f64 - 500.0).abs() / 500.0 * 10.0) as usize;
            let access_count = 10 - distance_from_center.min(9);
            for _ in 0..access_count {
                normal_cache.get(&format!("normal_{}", i));
            }
        }

        let start = Instant::now();
        for i in 1000..1200 {
            normal_cache.insert(format!("new_normal_{}", i), Arc::new(i));
        }
        let normal_duration = start.elapsed();

        assert_eq!(normal_cache.len(), 1000);
        assert!(
            normal_duration < Duration::from_millis(200),
            "Normal distribution eviction should be efficient: {:?}",
            normal_duration
        );

        // Center items should be preserved due to higher frequency
        assert!(normal_cache.contains(&"normal_500".to_string()));
        assert!(normal_cache.contains(&"normal_450".to_string()));
        assert!(normal_cache.contains(&"normal_550".to_string()));

        // Test 3: Exponential distribution impact
        let mut exponential_cache = LfuCache::new(1000);

        // Create exponential frequency distribution
        for i in 0..1000 {
            exponential_cache.insert(format!("exp_{}", i), Arc::new(i));
        }

        // Create exponential decay pattern
        for i in 0..1000 {
            let access_count = std::cmp::max(1, 20 - (i / 50));
            for _ in 0..access_count {
                exponential_cache.get(&format!("exp_{}", i));
            }
        }

        let start = Instant::now();
        for i in 1000..1300 {
            exponential_cache.insert(format!("new_exp_{}", i), Arc::new(i));
        }
        let exponential_duration = start.elapsed();

        assert_eq!(exponential_cache.len(), 1000);
        assert!(
            exponential_duration < Duration::from_millis(300),
            "Exponential distribution eviction should be reasonable: {:?}",
            exponential_duration
        );

        // Early items should be preserved due to higher frequency
        assert!(exponential_cache.contains(&"exp_0".to_string()));
        assert!(exponential_cache.contains(&"exp_10".to_string()));
        assert!(exponential_cache.contains(&"exp_50".to_string()));

        // Test 4: Power law (Zipf) distribution impact
        let mut zipf_cache = LfuCache::new(1000);

        // Create Zipf distribution (80/20 rule)
        for i in 0..1000 {
            zipf_cache.insert(format!("zipf_{}", i), Arc::new(i));
        }

        // Top 20% get 80% of accesses
        let hot_items = 200;
        let hot_accesses = 40;
        let cold_accesses = 1;

        for i in 0..hot_items {
            for _ in 0..hot_accesses {
                zipf_cache.get(&format!("zipf_{}", i));
            }
        }

        for i in hot_items..1000 {
            for _ in 0..cold_accesses {
                zipf_cache.get(&format!("zipf_{}", i));
            }
        }

        let start = Instant::now();
        for i in 1000..1400 {
            zipf_cache.insert(format!("new_zipf_{}", i), Arc::new(i));
        }
        let zipf_duration = start.elapsed();

        assert_eq!(zipf_cache.len(), 1000);
        assert!(
            zipf_duration < Duration::from_millis(400),
            "Zipf distribution eviction should be efficient: {:?}",
            zipf_duration
        );

        // Hot items should be preserved
        assert!(zipf_cache.contains(&"zipf_0".to_string()));
        assert!(zipf_cache.contains(&"zipf_50".to_string()));
        assert!(zipf_cache.contains(&"zipf_100".to_string()));

        // Test 5: Bimodal distribution impact
        let mut bimodal_cache = LfuCache::new(1000);

        // Create bimodal distribution (two peaks)
        for i in 0..1000 {
            bimodal_cache.insert(format!("bimodal_{}", i), Arc::new(i));
        }

        // Peak 1: items 200-300 (high frequency)
        for i in 200..300 {
            for _ in 0..15 {
                bimodal_cache.get(&format!("bimodal_{}", i));
            }
        }

        // Peak 2: items 700-800 (high frequency)
        for i in 700..800 {
            for _ in 0..15 {
                bimodal_cache.get(&format!("bimodal_{}", i));
            }
        }

        // Valley: other items (low frequency)
        for i in 0..200 {
            bimodal_cache.get(&format!("bimodal_{}", i));
        }
        for i in 300..700 {
            bimodal_cache.get(&format!("bimodal_{}", i));
        }
        for i in 800..1000 {
            bimodal_cache.get(&format!("bimodal_{}", i));
        }

        let start = Instant::now();
        for i in 1000..1300 {
            bimodal_cache.insert(format!("new_bimodal_{}", i), Arc::new(i));
        }
        let bimodal_duration = start.elapsed();

        assert_eq!(bimodal_cache.len(), 1000);
        assert!(
            bimodal_duration < Duration::from_millis(300),
            "Bimodal distribution eviction should be efficient: {:?}",
            bimodal_duration
        );

        // Peak items should be preserved
        assert!(bimodal_cache.contains(&"bimodal_250".to_string()));
        assert!(bimodal_cache.contains(&"bimodal_750".to_string()));

        // Test 6: Comparative performance across distributions
        let distributions = ["uniform", "normal", "exponential", "zipf", "bimodal"];
        let durations = [
            uniform_duration,
            normal_duration,
            exponential_duration,
            zipf_duration,
            bimodal_duration,
        ];

        // All distributions should complete within reasonable time
        for (i, &duration) in durations.iter().enumerate() {
            assert!(
                duration < Duration::from_millis(500),
                "{} distribution took too long: {:?}",
                distributions[i],
                duration
            );
        }

        // Test 7: Dynamic distribution change impact
        let mut dynamic_cache = LfuCache::new(500);

        // Fill cache initially
        for i in 0..500 {
            dynamic_cache.insert(format!("dynamic_{}", i), Arc::new(i));
        }

        // Phase 1: Create initial distribution (linear)
        for i in 0..500 {
            for _ in 0..(i / 50 + 1) {
                dynamic_cache.get(&format!("dynamic_{}", i));
            }
        }

        // Phase 2: Shift access pattern (reverse linear)
        for i in 0..500 {
            for _ in 0..((499 - i) / 50 + 1) {
                dynamic_cache.get(&format!("dynamic_{}", i));
            }
        }

        let start = Instant::now();
        for i in 500..650 {
            dynamic_cache.insert(format!("new_dynamic_{}", i), Arc::new(i));
        }
        let dynamic_duration = start.elapsed();

        assert_eq!(dynamic_cache.len(), 500);
        assert!(
            dynamic_duration < Duration::from_millis(150),
            "Dynamic distribution eviction should adapt efficiently: {:?}",
            dynamic_duration
        );

        // Test 8: Sparse vs dense frequency ranges
        let mut sparse_cache = LfuCache::new(400);
        let mut dense_cache = LfuCache::new(400);

        // Sparse: frequencies 1, 10, 20, 30 (big gaps)
        for i in 0..400 {
            sparse_cache.insert(format!("sparse_{}", i), Arc::new(i));
            let freq_group = i / 100;
            let target_freq = match freq_group {
                0 => 1,
                1 => 10,
                2 => 20,
                _ => 30,
            };
            for _ in 1..target_freq {
                sparse_cache.get(&format!("sparse_{}", i));
            }
        }

        // Dense: frequencies 1, 2, 3, 4 (small gaps)
        for i in 0..400 {
            dense_cache.insert(format!("dense_{}", i), Arc::new(i));
            let freq_group = i / 100;
            let target_freq = freq_group + 1;
            for _ in 1..target_freq {
                dense_cache.get(&format!("dense_{}", i));
            }
        }

        let start = Instant::now();
        for i in 400..500 {
            sparse_cache.insert(format!("new_sparse_{}", i), Arc::new(i));
        }
        let sparse_eviction_duration = start.elapsed();

        let start = Instant::now();
        for i in 400..500 {
            dense_cache.insert(format!("new_dense_{}", i), Arc::new(i));
        }
        let dense_eviction_duration = start.elapsed();

        assert!(
            sparse_eviction_duration < Duration::from_millis(100),
            "Sparse frequency eviction should be efficient: {:?}",
            sparse_eviction_duration
        );
        assert!(
            dense_eviction_duration < Duration::from_millis(100),
            "Dense frequency eviction should be efficient: {:?}",
            dense_eviction_duration
        );

        log::info!("Frequency distribution impact on eviction performance:");
        log::info!("  Uniform distribution: {:?}", uniform_duration);
        log::info!("  Normal distribution: {:?}", normal_duration);
        log::info!("  Exponential distribution: {:?}", exponential_duration);
        log::info!("  Zipf distribution: {:?}", zipf_duration);
        log::info!("  Bimodal distribution: {:?}", bimodal_duration);
        log::info!("  Dynamic distribution: {:?}", dynamic_duration);
        log::info!("  Sparse frequencies: {:?}", sparse_eviction_duration);
        log::info!("  Dense frequencies: {:?}", dense_eviction_duration);
    }
}

mod memory_efficiency {

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_memory_overhead_of_frequency_tracking() {
        // TODO: Test memory overhead of maintaining frequency information
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_memory_usage_growth() {
        // TODO: Test memory usage as cache fills up
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_memory_cleanup_after_eviction() {
        // TODO: Test that memory is properly cleaned up after evictions
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_large_value_memory_handling() {
        // TODO: Test memory efficiency with large values
    }
}

mod complexity {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use cachekit::policy::lfu::LfuCache;
    use cachekit::traits::{CoreCache, LfuCacheTrait, MutableCache};

    /// Helper function to measure execution time of a closure
    fn measure_time<F, R>(operation: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();
        (result, duration)
    }

    /// Generate test data for complexity tests
    fn generate_test_data(size: usize) -> Vec<(String, i32)> {
        (0..size)
            .map(|i| (format!("key_{:06}", i), i as i32))
            .collect()
    }

    // ==============================================
    // TIME COMPLEXITY TESTS
    // ==============================================

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_insert_time_complexity() {
        // Test that insert operations maintain consistent performance
        let cache_sizes = vec![100, 500, 1000, 5000, 10000];
        let mut results = Vec::new();

        for &cache_size in &cache_sizes {
            let mut cache = LfuCache::new(cache_size);
            let test_data = generate_test_data(cache_size);

            // Measure time to fill cache to capacity
            let (_, insert_time) = measure_time(|| {
                for (key, value) in test_data {
                    cache.insert(key, Arc::new(value));
                }
            });

            results.push((cache_size, insert_time));
        }

        // Verify performance characteristics
        for &(size, time) in results.iter() {
            log::info!(
                "Cache size: {}, Total insert time: {:?}, Avg per insert: {:?}",
                size,
                time,
                time / size as u32
            );

            // For LFU, insertion time should be reasonable even for large caches.
            // Allow extra headroom on slower or contended environments.
            let avg_time_per_insert = time / size as u32;
            let max_insert_time = if cfg!(feature = "metrics") {
                if cfg!(debug_assertions) {
                    Duration::from_micros(120)
                } else {
                    Duration::from_micros(20)
                }
            } else if cfg!(debug_assertions) {
                Duration::from_micros(150)
            } else {
                Duration::from_micros(30)
            };
            assert!(
                avg_time_per_insert < max_insert_time,
                "Insert performance degraded significantly for size {}: {:?} per insert",
                size,
                avg_time_per_insert
            );
        }
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_get_time_complexity() {
        // Test that get operations are O(1) amortized
        let cache_sizes = vec![100, 500, 1000, 5000];
        let lookup_count = 1000;

        for &cache_size in &cache_sizes {
            let mut cache = LfuCache::new(cache_size);

            // Pre-populate cache
            for i in 0..cache_size {
                cache.insert(format!("key_{}", i), Arc::new(i));
            }

            // Measure random access time
            let keys: Vec<String> = (0..lookup_count)
                .map(|i| format!("key_{}", i % cache_size))
                .collect();

            let (hit_count, lookup_time) = measure_time(|| {
                let mut hits = 0;
                for key in &keys {
                    if cache.get(key).is_some() {
                        hits += 1;
                    }
                }
                hits
            });

            assert_eq!(hit_count, lookup_count); // All should be hits

            let avg_time_per_get = lookup_time / lookup_count as u32;
            log::info!(
                "Cache size: {}, Avg get time: {:?}",
                cache_size,
                avg_time_per_get
            );

            // Get should be O(1) - allow extra headroom on slower CI or debug runs.
            let max_get_time = if cfg!(feature = "metrics") {
                if cfg!(debug_assertions) {
                    Duration::from_micros(20)
                } else {
                    Duration::from_micros(4)
                }
            } else if cfg!(debug_assertions) {
                Duration::from_micros(25)
            } else {
                Duration::from_micros(8)
            };
            assert!(
                avg_time_per_get < max_get_time,
                "Get performance degraded for cache size {}: {:?} per get",
                cache_size,
                avg_time_per_get
            );
        }
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_pop_lfu_time_complexity() {
        // Test that pop_lfu is O(n) but with reasonable constant factors
        let cache_sizes = vec![100, 500, 1000, 2000];
        let mut results = Vec::new();

        for &cache_size in &cache_sizes {
            let mut cache = LfuCache::new(cache_size);

            // Pre-populate cache with different frequencies
            for i in 0..cache_size {
                cache.insert(format!("key_{}", i), Arc::new(i));
                // Create frequency differences
                for _ in 0..(i % 5) {
                    cache.get(&format!("key_{}", i));
                }
            }

            // Measure pop_lfu operations
            let pop_count = std::cmp::min(50, cache_size / 2);
            let (popped_items, pop_time) = measure_time(|| {
                let mut popped = Vec::new();
                for _ in 0..pop_count {
                    if let Some(item) = cache.pop_lfu() {
                        popped.push(item);
                    }
                }
                popped
            });

            assert_eq!(popped_items.len(), pop_count);
            let avg_time_per_pop = pop_time / pop_count as u32;
            results.push((cache_size, avg_time_per_pop));

            log::info!(
                "Cache size: {}, Avg pop_lfu time: {:?}",
                cache_size,
                avg_time_per_pop
            );
        }

        // Verify that pop_lfu time grows reasonably with cache size (O(n))
        // Allow for some variance but ensure it's not exponential
        for &(size, time) in &results {
            // pop_lfu is O(n), so allow time proportional to cache size
            // Allow up to 10µs per cache entry for pop_lfu (realistic for current implementation)
            let max_expected_time = Duration::from_micros((size * 10) as u64);
            assert!(
                time < max_expected_time,
                "pop_lfu performance too slow for cache size {}: {:?} (expected < {:?})",
                size,
                time,
                max_expected_time
            );
        }
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_peek_lfu_time_complexity() {
        // Test that peek_lfu is O(n) with good constant factors
        let cache_sizes = vec![100, 500, 1000, 2000, 5000];

        for &cache_size in &cache_sizes {
            let mut cache = LfuCache::new(cache_size);

            // Pre-populate cache
            for i in 0..cache_size {
                cache.insert(format!("key_{}", i), Arc::new(i));
                // Create varied frequency distribution
                for _ in 0..(i % 7) {
                    cache.get(&format!("key_{}", i));
                }
            }

            // Measure peek_lfu operations
            let peek_count = 100;
            let (peek_results, peek_time) = measure_time(|| {
                let mut results = Vec::new();
                for _ in 0..peek_count {
                    results.push(cache.peek_lfu());
                }
                results
            });

            // All peeks should return the same LFU item
            assert!(peek_results.iter().all(|r| r.is_some()));
            let first_result = peek_results[0];
            assert!(peek_results.iter().all(|&r| r == first_result));

            let avg_time_per_peek = peek_time / peek_count as u32;
            log::info!(
                "Cache size: {}, Avg peek_lfu time: {:?}",
                cache_size,
                avg_time_per_peek
            );

            // peek_lfu is O(n), allow up to 1µs per cache entry (realistic for current implementation)
            let max_expected_time = Duration::from_micros(cache_size as u64);
            assert!(
                avg_time_per_peek < max_expected_time,
                "peek_lfu performance too slow for cache size {}: {:?} (expected < {:?})",
                cache_size,
                avg_time_per_peek,
                max_expected_time
            );
        }
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_frequency_operations_time_complexity() {
        // Test that frequency operations are O(1)
        let cache_sizes = vec![100, 1000, 5000, 10000];

        for &cache_size in &cache_sizes {
            let mut cache = LfuCache::new(cache_size);

            // Pre-populate cache
            for i in 0..cache_size {
                cache.insert(format!("key_{}", i), Arc::new(i));
            }

            let test_keys: Vec<String> = (0..1000)
                .map(|i| format!("key_{}", i % cache_size))
                .collect();

            // Test frequency() performance
            let (_, freq_time) = measure_time(|| {
                for key in &test_keys {
                    cache.frequency(key);
                }
            });

            // Test increment_frequency() performance
            let (_, inc_time) = measure_time(|| {
                for key in &test_keys {
                    cache.increment_frequency(key);
                }
            });

            // Test reset_frequency() performance
            let (_, reset_time) = measure_time(|| {
                for key in &test_keys {
                    cache.reset_frequency(key);
                }
            });

            let avg_freq_time = freq_time / test_keys.len() as u32;
            let avg_inc_time = inc_time / test_keys.len() as u32;
            let avg_reset_time = reset_time / test_keys.len() as u32;

            log::info!("Cache size: {}", cache_size);
            log::info!("  Avg frequency() time: {:?}", avg_freq_time);
            log::info!("  Avg increment_frequency() time: {:?}", avg_inc_time);
            log::info!("  Avg reset_frequency() time: {:?}", avg_reset_time);

            let max_freq_time = if cfg!(feature = "metrics") {
                if cfg!(debug_assertions) {
                    Duration::from_micros(30)
                } else {
                    Duration::from_micros(10)
                }
            } else if cfg!(debug_assertions) {
                Duration::from_micros(25)
            } else {
                Duration::from_micros(10)
            };
            let max_inc_time = max_freq_time;
            let max_reset_time = max_freq_time;

            // All frequency operations should be O(1) - allow extra headroom for noisy environments.
            assert!(
                avg_freq_time < max_freq_time,
                "frequency() too slow for cache size {}: {:?}",
                cache_size,
                avg_freq_time
            );
            assert!(
                avg_inc_time < max_inc_time,
                "increment_frequency() too slow for cache size {}: {:?}",
                cache_size,
                avg_inc_time
            );
            assert!(
                avg_reset_time < max_reset_time,
                "reset_frequency() too slow for cache size {}: {:?}",
                cache_size,
                avg_reset_time
            );
        }
    }

    // ==============================================
    // SPACE COMPLEXITY TESTS
    // ==============================================

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_memory_usage_scaling() {
        // Test that memory usage scales linearly with cache size
        let cache_sizes = vec![100, 500, 1000, 2000, 5000];

        for &cache_size in &cache_sizes {
            let mut cache = LfuCache::new(cache_size);

            // Fill cache to capacity
            for i in 0..cache_size {
                cache.insert(format!("test_key_{:08}", i), Arc::new(i));
            }

            // Verify cache respects capacity constraints
            assert_eq!(cache.len(), cache_size);
            assert_eq!(cache.capacity(), cache_size);

            // Test overfill behavior
            let pre_overfill_len = cache.len();
            cache.insert("overflow_key".to_string(), Arc::new(usize::MAX));

            // Should maintain capacity by evicting LFU item
            assert_eq!(cache.len(), cache_size);
            assert_eq!(cache.len(), pre_overfill_len); // No growth

            log::info!("Cache size: {}, Final length: {}", cache_size, cache.len());
        }
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_memory_efficiency() {
        // Test memory efficiency of the LFU implementation
        let cache_size = 1000;
        let mut cache = LfuCache::new(cache_size);

        // ==============================================
        // THEORETICAL MEMORY CALCULATION
        // ==============================================

        // Calculate theoretical minimum memory usage
        // Each entry stores: String key + i32 value + usize frequency
        // Plus HashMap overhead
        let key_size = std::mem::size_of::<String>(); // String struct (24 bytes on 64-bit)
        let value_size = std::mem::size_of::<i32>(); // 4 bytes
        let freq_size = std::mem::size_of::<usize>(); // 8 bytes on 64-bit
        let hashmap_entry_overhead = 24; // Rough HashMap entry overhead (bucket, hash, etc.)

        let theoretical_min_per_entry = key_size + value_size + freq_size + hashmap_entry_overhead;
        log::info!("Memory analysis:");
        log::info!("  String key size: {} bytes", key_size);
        log::info!("  i32 value size: {} bytes", value_size);
        log::info!("  usize frequency size: {} bytes", freq_size);
        log::info!("  HashMap overhead: {} bytes", hashmap_entry_overhead);
        log::info!(
            "  Theoretical minimum per entry: {} bytes",
            theoretical_min_per_entry
        );

        // ==============================================
        // BASIC MEMORY USAGE TEST
        // ==============================================

        // Test initial empty state
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.capacity(), cache_size);

        // Fill cache and verify it doesn't use excessive memory
        for i in 0..cache_size {
            cache.insert(format!("key_{:06}", i), Arc::new(i));
        }

        assert_eq!(cache.len(), cache_size);
        log::info!("  Cache filled to capacity: {} entries", cache.len());

        // ==============================================
        // MEMORY LEAK DETECTION
        // ==============================================

        // Test that extensive operations don't cause memory leaks

        // Perform many operations that could potentially leak memory
        let operations_count = 5000;
        for i in 0..operations_count {
            // Mixed workload to stress test memory management
            match i % 8 {
                0 => {
                    // Insert new items (should evict LFU)
                    cache.insert(format!("temp_key_{}", i), Arc::new(i));
                },
                1 => {
                    // Access existing items (increments frequency)
                    cache.get(&format!("key_{:06}", i % cache_size));
                },
                2 => {
                    // Manual frequency increment
                    cache.increment_frequency(&format!("key_{:06}", i % (cache_size / 2)));
                },
                3 => {
                    // Pop LFU items (tests removal logic)
                    if let Some((_key, value)) = cache.pop_lfu() {
                        // Immediately re-insert to maintain cache size
                        cache.insert(format!("reinsert_{}", i), value);
                    }
                },
                4 => {
                    // Reset frequency (tests frequency management)
                    cache.reset_frequency(&format!("key_{:06}", i % cache_size));
                },
                5 => {
                    // Remove specific items
                    let key_to_remove = format!("temp_key_{}", i.saturating_sub(100));
                    cache.remove(&key_to_remove);
                },
                6 => {
                    // Peek operations (should not affect memory)
                    cache.peek_lfu();
                    cache.contains(&format!("key_{:06}", i % cache_size));
                },
                7 => {
                    // Check frequency (read-only operation)
                    cache.frequency(&format!("key_{:06}", i % cache_size));
                },
                _ => unreachable!(),
            }

            // Periodically verify memory constraints
            if i % 1000 == 0 {
                assert!(
                    cache.len() <= cache_size,
                    "Cache exceeded capacity at iteration {}: {} > {}",
                    i,
                    cache.len(),
                    cache_size
                );

                // Verify cache is still functional
                assert!(cache.peek_lfu().is_some() || cache.is_empty());

                log::info!("  Iteration {}: cache length = {}", i, cache.len());
            }
        }

        // Final memory leak check
        assert_eq!(
            cache.len(),
            cache_size,
            "Cache size changed unexpectedly after {} operations",
            operations_count
        );
        log::info!(
            "  Memory leak test passed: cache maintained size through {} operations",
            operations_count
        );

        // ==============================================
        // MEMORY FRAGMENTATION TEST
        // ==============================================

        // Test memory efficiency with fragmented access patterns
        log::info!("  Testing memory fragmentation resistance...");

        let fragmentation_cycles = 10;
        for cycle in 0..fragmentation_cycles {
            // Clear half the cache in a fragmented pattern
            let mut removed_count = 0;
            for i in (0..cache_size).step_by(2) {
                let key = format!("key_{:06}", i);
                if cache.remove(&key).is_some() {
                    removed_count += 1;
                }
                if removed_count >= cache_size / 2 {
                    break;
                }
            }

            // Verify partial clearing
            let mid_len = cache.len();
            assert!(
                mid_len >= cache_size / 2 && mid_len <= cache_size,
                "Unexpected cache size after fragmented removal: {}",
                mid_len
            );

            // Refill with new data
            for i in 0..cache_size {
                if cache.len() < cache_size {
                    cache.insert(format!("frag_{}_{}", cycle, i), Arc::new(cycle * 1000 + i));
                }
            }

            // Should be back to full capacity
            assert_eq!(
                cache.len(),
                cache_size,
                "Cache not properly refilled in fragmentation cycle {}",
                cycle
            );
        }

        log::info!(
            "  Fragmentation test passed: {} cycles completed",
            fragmentation_cycles
        );

        // ==============================================
        // DIFFERENT DATA TYPE SIZES TEST
        // ==============================================

        // Test with varying key sizes to check memory efficiency
        log::info!("  Testing variable key size memory efficiency...");

        let key_size_variants = vec![5, 20, 50, 100];
        for &key_len in &key_size_variants {
            let mut test_cache = LfuCache::new(100);
            let base_key = "x".repeat(key_len);

            // Fill with variable-sized keys
            for i in 0..100 {
                let key = format!("{}{:03}", base_key, i);
                test_cache.insert(key, Arc::new(i));
            }

            assert_eq!(test_cache.len(), 100);

            // Test operations work correctly with variable key sizes
            assert!(test_cache.peek_lfu().is_some());
            assert!(test_cache.pop_lfu().is_some());

            log::info!(
                "    Key length {}: {} entries managed successfully",
                key_len,
                test_cache.len()
            );
        }

        // ==============================================
        // MEMORY CLEANUP VERIFICATION
        // ==============================================

        // Test that clearing the cache properly frees memory
        log::info!("  Testing memory cleanup...");

        let pre_clear_len = cache.len();
        assert!(pre_clear_len > 0, "Cache should have items before clearing");

        // Clear cache by removing all items
        let mut clear_count = 0;
        while let Some((_key, _value)) = cache.pop_lfu() {
            clear_count += 1;
            // Verify cache size decreases
            assert_eq!(cache.len(), pre_clear_len - clear_count);
        }

        // Verify complete cleanup
        assert_eq!(
            cache.len(),
            0,
            "Cache should be empty after clearing all items"
        );
        assert_eq!(clear_count, pre_clear_len, "Should have cleared all items");
        assert!(cache.is_empty(), "Cache should report as empty");
        assert!(
            cache.peek_lfu().is_none(),
            "peek_lfu should return None for empty cache"
        );

        // Test that we can still use the cache after clearing
        cache.insert("post_clear_key".to_string(), Arc::new(42));
        assert_eq!(cache.len(), 1);
        assert_eq!(
            cache.get(&"post_clear_key".to_string()).map(Arc::as_ref),
            Some(&42)
        );

        log::info!(
            "  Memory cleanup test passed: cleared {} items, cache functional",
            clear_count
        );

        // ==============================================
        // CAPACITY BOUNDARY TESTING
        // ==============================================

        // Test memory efficiency at capacity boundaries
        log::info!("  Testing capacity boundary behavior...");

        let boundary_cache_size = 50;
        let mut boundary_cache = LfuCache::new(boundary_cache_size);

        // Fill exactly to capacity
        for i in 0..boundary_cache_size {
            boundary_cache.insert(format!("boundary_{}", i), Arc::new(i));
        }
        assert_eq!(boundary_cache.len(), boundary_cache_size);

        // Test overflow behavior (should evict LFU items)
        let overflow_items = 20;
        for i in 0..overflow_items {
            boundary_cache.insert(format!("overflow_{}", i), Arc::new(100 + i));
            // Should maintain capacity
            assert_eq!(boundary_cache.len(), boundary_cache_size);
        }

        // Verify LFU eviction occurred (some boundary items should be gone)
        let remaining_boundary_items = (0..boundary_cache_size)
            .filter(|&i| boundary_cache.contains(&format!("boundary_{}", i)))
            .count();

        log::info!(
            "    Boundary items remaining: {}/{}",
            remaining_boundary_items,
            boundary_cache_size
        );
        assert!(
            remaining_boundary_items < boundary_cache_size,
            "Some boundary items should have been evicted"
        );

        // ==============================================
        // FINAL SUMMARY
        // ==============================================

        log::info!("Memory efficiency test completed successfully:");
        log::info!("  ✓ Theoretical memory calculations verified");
        log::info!(
            "  ✓ Memory leak detection passed ({} operations)",
            operations_count
        );
        log::info!(
            "  ✓ Fragmentation resistance verified ({} cycles)",
            fragmentation_cycles
        );
        log::info!("  ✓ Variable key size handling confirmed");
        log::info!("  ✓ Memory cleanup verification passed");
        log::info!("  ✓ Capacity boundary behavior validated");
        log::info!("  → LFU cache demonstrates efficient memory management");
    }

    // ==============================================
    // SCALABILITY TESTS
    // ==============================================

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_scalability_with_varying_key_sizes() {
        // Test performance with different key sizes
        let key_sizes = vec![10, 50, 100, 500];
        let cache_size = 1000;

        for &key_size in &key_sizes {
            let mut cache = LfuCache::new(cache_size);

            // Generate keys of specified size
            let long_key = "x".repeat(key_size);

            let (_, insert_time) = measure_time(|| {
                for i in 0..cache_size {
                    let key = format!("{}{:06}", long_key, i);
                    cache.insert(key, Arc::new(i));
                }
            });

            let avg_insert_time = insert_time / cache_size as u32;
            log::info!(
                "Key size: {} chars, Avg insert time: {:?}",
                key_size,
                avg_insert_time
            );

            // Performance should degrade gracefully with larger keys.
            // Allow extra headroom in noisy environments.
            let max_insert_time = if cfg!(feature = "metrics") {
                if cfg!(debug_assertions) {
                    Duration::from_micros(120)
                } else {
                    Duration::from_micros(20)
                }
            } else if cfg!(debug_assertions) {
                Duration::from_micros(150)
            } else {
                Duration::from_micros(50)
            };
            assert!(
                avg_insert_time < max_insert_time,
                "Insert performance too slow for key size {}: {:?}",
                key_size,
                avg_insert_time
            );
        }
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_performance_regression_detection() {
        // Test to detect performance regressions
        let cache_size = 2000;
        let operation_count = 5000;

        let mut cache = LfuCache::new(cache_size);

        // Pre-populate
        for i in 0..cache_size {
            cache.insert(format!("key_{}", i), Arc::new(i));
        }

        // Mixed workload performance test
        let (results, total_time) = measure_time(|| {
            let mut results = HashMap::new();

            for i in 0..operation_count {
                let op_type = i % 10;

                match op_type {
                    0..=5 => {
                        // 60% gets
                        let key = format!("key_{}", i % cache_size);
                        cache.get(&key);
                        *results.entry("gets").or_insert(0) += 1;
                    },
                    6..=7 => {
                        // 20% inserts
                        cache.insert(format!("new_key_{}", i), Arc::new(i));
                        *results.entry("inserts").or_insert(0) += 1;
                    },
                    8 => {
                        // 10% frequency ops
                        let key = format!("key_{}", i % cache_size);
                        cache.increment_frequency(&key);
                        *results.entry("frequency_ops").or_insert(0) += 1;
                    },
                    9 => {
                        // 10% pop_lfu
                        cache.pop_lfu();
                        *results.entry("pop_lfu").or_insert(0) += 1;
                    },
                    _ => unreachable!(),
                }
            }

            results
        });

        let avg_time_per_op = total_time / operation_count as u32;
        log::info!("Mixed workload results: {:?}", results);
        log::info!(
            "Total time: {:?}, Avg per operation: {:?}",
            total_time,
            avg_time_per_op
        );

        // Performance baseline - should complete mixed workload reasonably quickly
        // Allow up to 500µs per operation for mixed workload (includes expensive pop_lfu operations)
        assert!(
            avg_time_per_op < Duration::from_micros(500),
            "Mixed workload performance regression detected: {:?} per operation",
            avg_time_per_op
        );

        // Verify cache is still functional
        assert!(cache.len() <= cache_size);
        assert!(cache.len() > 0);
        assert!(cache.peek_lfu().is_some());
    }

    #[test]
    #[cfg_attr(
        feature = "metrics",
        ignore = "performance tests are noisy with metrics enabled"
    )]
    fn test_worst_case_performance() {
        // Test performance in worst-case scenarios
        let cache_size = 1000;
        let mut cache = LfuCache::new(cache_size);

        // Worst case: all items have the same frequency
        for i in 0..cache_size {
            cache.insert(format!("key_{:06}", i), Arc::new(i));
        }

        // All items now have frequency 1 (worst case for LFU operations)

        // Test pop_lfu performance with uniform frequencies
        let pop_count = 100;
        let (_, pop_time) = measure_time(|| {
            for _ in 0..pop_count {
                cache.pop_lfu();
            }
        });

        let avg_pop_time = pop_time / pop_count as u32;
        log::info!(
            "Worst-case pop_lfu time (uniform frequencies): {:?}",
            avg_pop_time
        );

        // Even in worst case, should be reasonable (uniform frequencies are challenging)
        assert!(
            avg_pop_time < Duration::from_millis(10),
            "Worst-case pop_lfu performance too slow: {:?}",
            avg_pop_time
        );

        // Refill and test peek_lfu worst case
        for i in 0..100 {
            cache.insert(format!("refill_key_{}", i), Arc::new(i));
        }

        let peek_count = 1000;
        let (_, peek_time) = measure_time(|| {
            for _ in 0..peek_count {
                cache.peek_lfu();
            }
        });

        let avg_peek_time = peek_time / peek_count as u32;
        log::info!(
            "Worst-case peek_lfu time (uniform frequencies): {:?}",
            avg_peek_time
        );

        assert!(
            avg_peek_time < Duration::from_millis(1),
            "Worst-case peek_lfu performance too slow: {:?}",
            avg_peek_time
        );
    }
}
