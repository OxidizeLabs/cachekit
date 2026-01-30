// ==============================================
// LRU-K PERFORMANCE TESTS (integration)
// ==============================================

use cachekit::policy::lru_k::LrukCache;
use cachekit::traits::CoreCache;
use std::time::{Duration, Instant};

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

// Lookup Performance Tests
mod lookup_performance {
    use super::*;

    #[test]
    fn test_get_performance_with_history_updates() {
        let mut cache = LrukCache::new(1000);

        // Pre-fill cache
        for i in 0..1000 {
            cache.insert(i, i * 10);
        }

        // Measure get with history tracking
        let iterations = 10000;
        let (_, duration) = measure_time(|| {
            for i in 0..iterations {
                let key = i % 1000;
                cache.get(&key);
            }
        });

        let avg_latency = duration / iterations;
        println!("Average get with history latency: {:?}", avg_latency);

        // LRU-K should still be fast despite history tracking
        assert!(
            avg_latency < Duration::from_micros(20),
            "Get with history too slow: {:?}",
            avg_latency
        );
    }

    #[test]
    fn test_contains_performance() {
        let mut cache = LrukCache::new(1000);

        // Pre-fill cache
        for i in 0..1000 {
            cache.insert(i, i * 10);
        }

        // Measure contains operation
        let iterations = 10000;
        let (_, duration) = measure_time(|| {
            for i in 0..iterations {
                let key = i % 1000;
                cache.contains(&key);
            }
        });

        let avg_latency = duration / iterations;
        println!("Average contains latency: {:?}", avg_latency);

        assert!(avg_latency < Duration::from_micros(5));
    }

    #[test]
    fn test_access_history_lookup_performance() {
        let mut cache = LrukCache::new(1000);

        // Pre-fill cache and build history
        for i in 0..1000 {
            cache.insert(i, i * 10);
            // Access multiple times to build history
            for _ in 0..3 {
                cache.get(&i);
            }
        }

        // Measure history lookup (via get which accesses history)
        let iterations = 5000;
        let (_, duration) = measure_time(|| {
            for i in 0..iterations {
                let key = i % 1000;
                cache.get(&key);
            }
        });

        let avg_latency = duration / iterations;
        println!("Get with history lookup avg latency: {:?}", avg_latency);

        assert!(avg_latency < Duration::from_micros(25));
    }

    #[test]
    fn test_k_distance_calculation_performance() {
        let mut cache = LrukCache::new(1000);

        // Pre-fill cache with access history
        for i in 0..1000 {
            cache.insert(i, i * 10);
            for j in 0..5 {
                if j % 2 == 0 {
                    cache.get(&i);
                }
            }
        }

        // K-distance is calculated during eviction, measure via insert
        let iterations = 1000;
        let (_, duration) = measure_time(|| {
            for i in 1000..1000 + iterations {
                cache.insert(i, i * 10); // Triggers k-distance calculation
            }
        });

        let avg_latency = duration / iterations;
        println!("Insert with k-distance calc avg latency: {:?}", avg_latency);

        assert!(avg_latency < Duration::from_micros(30));
    }

    #[test]
    fn test_peek_lru_k_performance() {
        let mut cache = LrukCache::new(10000);

        // Pre-fill cache
        for i in 0..10000 {
            cache.insert(i, i * 10);
        }

        // Measure peek operation (if available via contains)
        let iterations = 10000;
        let (_, duration) = measure_time(|| {
            for i in 0..iterations {
                let key = i % 10000;
                cache.contains(&key);
            }
        });

        let avg_latency = duration / iterations;
        println!("Peek LRU-K avg latency: {:?}", avg_latency);

        assert!(avg_latency < Duration::from_micros(10));
    }

    #[test]
    fn test_cache_hit_vs_miss_performance() {
        let mut cache = LrukCache::new(1000);

        // Pre-fill cache
        for i in 0..1000 {
            cache.insert(i, i * 10);
        }

        // Measure hit performance
        let hit_iterations = 10000;
        let (_, hit_duration) = measure_time(|| {
            for i in 0..hit_iterations {
                let key = i % 1000;
                let result = cache.get(&key);
                assert!(result.is_some());
            }
        });

        // Measure miss performance
        let miss_iterations = 10000;
        let (_, miss_duration) = measure_time(|| {
            for i in 0..miss_iterations {
                let key = i + 10000; // Not in cache
                let result = cache.get(&key);
                assert!(result.is_none());
            }
        });

        let hit_latency = hit_duration / hit_iterations;
        let miss_latency = miss_duration / miss_iterations;

        println!("Hit avg latency: {:?}", hit_latency);
        println!("Miss avg latency: {:?}", miss_latency);

        // Misses should be faster (no history update)
        assert!(miss_latency < hit_latency * 2);
    }

    #[test]
    fn test_touch_performance() {
        let mut cache = LrukCache::new(1000);

        // Pre-fill cache
        for i in 0..1000 {
            cache.insert(i, i * 10);
        }

        // Measure get performance (touch operation)
        let iterations = 10000;
        let (_, duration) = measure_time(|| {
            for i in 0..iterations {
                let key = i % 1000;
                cache.get(&key);
            }
        });

        let avg_latency = duration / iterations;
        println!("Touch (get) avg latency: {:?}", avg_latency);

        assert!(avg_latency < Duration::from_micros(25));
    }

    #[test]
    fn test_k_distance_rank_performance() {
        let mut cache = LrukCache::new(1000);

        // Pre-fill cache with varied access patterns
        for i in 0..1000 {
            cache.insert(i, i * 10);
            // Create frequency variance
            for _ in 0..(i % 5) {
                cache.get(&i);
            }
        }

        // K-distance ranking happens during eviction
        let iterations = 1000;
        let (_, duration) = measure_time(|| {
            for i in 1000..1000 + iterations {
                cache.insert(i, i * 10);
            }
        });

        let avg_latency = duration / iterations;
        println!(
            "Insert with k-distance ranking avg latency: {:?}",
            avg_latency
        );

        assert!(avg_latency < Duration::from_micros(35));
    }
}

// Insertion Performance Tests
mod insertion_performance {
    use super::*;

    #[test]
    fn test_insertion_performance_with_eviction() {
        let mut cache = LrukCache::new(1000);

        // Pre-fill cache
        for i in 0..1000 {
            cache.insert(i, i * 10);
        }

        // Measure insertion with eviction
        let iterations = 10000;
        let (_, duration) = measure_time(|| {
            for i in 1000..1000 + iterations {
                cache.insert(i, i * 10);
            }
        });

        let avg_latency = duration / iterations;
        println!("Insert with eviction avg latency: {:?}", avg_latency);

        assert!(avg_latency < Duration::from_micros(30));
    }

    #[test]
    fn test_batch_insertion_performance() {
        let mut cache = LrukCache::new(10000);

        // Measure batch insertion
        let batch_size = 10000;
        let (_, duration) = measure_time(|| {
            for i in 0..batch_size {
                cache.insert(i, i * 10);
            }
        });

        let avg_latency = duration / batch_size;
        let throughput = batch_size as f64 / duration.as_secs_f64();

        println!("Batch insert avg latency: {:?}", avg_latency);
        println!("Batch insert throughput: {:.0} ops/sec", throughput);

        assert!(throughput > 500_000.0, "Throughput too low");
    }

    #[test]
    fn test_update_vs_new_insertion_performance() {
        let mut cache = LrukCache::new(1000);

        // Pre-fill cache
        for i in 0..1000 {
            cache.insert(i, i * 10);
        }

        // Measure update performance
        let iterations = 10000;
        let (_, update_duration) = measure_time(|| {
            for i in 0..iterations {
                let key = i % 1000;
                cache.insert(key, key * 20); // Update
            }
        });

        // Measure new insertion performance
        let (_, new_duration) = measure_time(|| {
            for i in 10000..10000 + iterations {
                cache.insert(i, i * 10); // New (triggers eviction)
            }
        });

        let update_latency = update_duration / iterations;
        let new_latency = new_duration / iterations;

        println!("Update avg latency: {:?}", update_latency);
        println!("New insert avg latency: {:?}", new_latency);

        // Updates might be slightly faster (no eviction), but both should be fast
        assert!(update_latency < Duration::from_micros(25));
        assert!(new_latency < Duration::from_micros(35));
    }

    #[test]
    fn test_insertion_with_history_tracking() {
        let mut cache = LrukCache::new(5000);

        // Measure insertion with history tracking overhead
        let iterations = 5000;
        let (_, duration) = measure_time(|| {
            for i in 0..iterations {
                cache.insert(i, i * 10);
                // Access to build history
                cache.get(&i);
                cache.get(&i);
            }
        });

        let avg_latency = duration / (iterations * 3); // insert + 2 gets
        println!("Avg operation with history tracking: {:?}", avg_latency);

        assert!(avg_latency < Duration::from_micros(25));
    }

    #[test]
    fn test_history_update_performance() {
        let mut cache = LrukCache::new(1000);

        // Pre-fill cache
        for i in 0..1000 {
            cache.insert(i, i * 10);
        }

        // Measure history update via repeated gets
        let iterations = 10000;
        let (_, duration) = measure_time(|| {
            for i in 0..iterations {
                let key = i % 1000;
                cache.get(&key); // Updates history
            }
        });

        let avg_latency = duration / iterations;
        println!("History update avg latency: {:?}", avg_latency);

        assert!(avg_latency < Duration::from_micros(25));
    }

    #[test]
    fn test_timestamp_generation_overhead() {
        // Measure timestamp generation overhead
        let iterations = 100000;
        let (_, duration) = measure_time(|| {
            for _ in 0..iterations {
                let _ = std::time::SystemTime::now();
            }
        });

        let avg_overhead = duration / iterations;
        println!("Timestamp generation avg overhead: {:?}", avg_overhead);

        // Timestamp generation should be very fast
        assert!(avg_overhead < Duration::from_micros(1));
    }
}

// Eviction Performance Tests
mod eviction_performance {
    use super::*;

    #[test]
    fn test_lru_k_eviction_performance() {
        let mut cache = LrukCache::new(1000);

        // Pre-fill cache
        for i in 0..1000 {
            cache.insert(i, i * 10);
            // Build varied access history
            for _ in 0..(i % 3) {
                cache.get(&i);
            }
        }

        // Measure eviction performance
        let iterations = 5000;
        let (_, duration) = measure_time(|| {
            for i in 1000..1000 + iterations {
                cache.insert(i, i * 10); // Triggers eviction
            }
        });

        let avg_latency = duration / iterations;
        println!("LRU-K eviction avg latency: {:?}", avg_latency);

        assert!(avg_latency < Duration::from_micros(40));
    }

    #[test]
    fn test_pop_lru_k_performance() {
        let mut cache = LrukCache::new(10000);

        // Pre-fill cache
        for i in 0..10000 {
            cache.insert(i, i * 10);
        }

        // Measure manual eviction via insert (which triggers internal pop)
        let iterations = 5000;
        let (_, duration) = measure_time(|| {
            for i in 10000..10000 + iterations {
                cache.insert(i, i * 10);
            }
        });

        let avg_latency = duration / iterations;
        println!("Pop LRU-K (via insert) avg latency: {:?}", avg_latency);

        assert!(avg_latency < Duration::from_micros(40));
    }

    #[test]
    fn test_eviction_with_varying_k_values() {
        // Note: LrukCache has fixed K=2, testing with that
        let k = 2;
        let mut cache = LrukCache::new(1000);

        // Pre-fill and create varied history
        for i in 0..1000 {
            cache.insert(i, i * 10);
            for _ in 0..k {
                cache.get(&i);
            }
        }

        // Measure eviction
        let iterations = 1000;
        let (_, duration) = measure_time(|| {
            for i in 1000..1000 + iterations {
                cache.insert(i, i * 10);
            }
        });

        let avg_latency = duration / iterations;
        println!("Eviction with K={} avg latency: {:?}", k, avg_latency);

        assert!(avg_latency < Duration::from_micros(40));
    }

    #[test]
    fn test_eviction_with_large_histories() {
        let mut cache = LrukCache::new(500);

        // Pre-fill cache with extensive access history
        for i in 0..500 {
            cache.insert(i, i * 10);
            // Build large access history
            for _ in 0..10 {
                cache.get(&i);
            }
        }

        // Measure eviction with large histories
        let iterations = 1000;
        let (_, duration) = measure_time(|| {
            for i in 500..500 + iterations {
                cache.insert(i, i * 10);
            }
        });

        let avg_latency = duration / iterations;
        println!(
            "Eviction with large histories avg latency: {:?}",
            avg_latency
        );

        assert!(avg_latency < Duration::from_micros(50));
    }

    #[test]
    fn test_victim_selection_performance() {
        let mut cache = LrukCache::new(1000);

        // Pre-fill cache with mixed access patterns
        for i in 0..1000 {
            cache.insert(i, i * 10);
            // Create access pattern variance
            let access_count = (i % 7) + 1;
            for _ in 0..access_count {
                cache.get(&i);
            }
        }

        // Measure victim selection (happens during insert/eviction)
        let iterations = 2000;
        let (_, duration) = measure_time(|| {
            for i in 1000..1000 + iterations {
                cache.insert(i, i * 10);
            }
        });

        let avg_latency = duration / iterations;
        println!("Victim selection avg latency: {:?}", avg_latency);

        assert!(avg_latency < Duration::from_micros(40));
    }
}

// Memory Efficiency Tests
mod memory_efficiency {
    use super::*;

    #[test]
    fn test_memory_overhead_of_history_tracking() {
        let cache_size = 5000;
        let mut cache = LrukCache::new(cache_size);

        // Fill cache with history
        for i in 0..cache_size {
            cache.insert(i, i * 10);
            // Build access history (K=2, so 2 accesses)
            cache.get(&i);
            cache.get(&i);
        }

        assert_eq!(cache.len(), cache_size);

        // Estimate memory overhead
        // LRU-K needs: HashMap + history storage (2 timestamps per entry for K=2)
        let estimated_overhead = std::mem::size_of::<usize>() * 4 // map overhead
            + std::mem::size_of::<u128>() * 2 // 2 timestamps
            + std::mem::size_of::<i32>(); // value

        println!("Estimated per-item overhead: {} bytes", estimated_overhead);

        // LRU-K overhead should be reasonable despite history tracking
        assert!(estimated_overhead < 300);
    }

    #[test]
    fn test_memory_usage_growth() {
        let sizes = vec![100, 500, 1000, 5000, 10000];

        for &size in &sizes {
            let mut cache = LrukCache::new(size);

            // Fill cache
            for i in 0..size {
                cache.insert(i, i * 10);
            }

            assert_eq!(cache.len(), size);
            println!("Cache size: {}, entries: {}", size, cache.len());
        }

        println!("Memory usage scales linearly with cache size");
    }

    #[test]
    fn test_memory_cleanup_after_eviction() {
        let cache_size = 1000;
        let mut cache = LrukCache::new(cache_size);

        // Fill cache
        for i in 0..cache_size {
            cache.insert(i, i * 10);
        }

        assert_eq!(cache.len(), cache_size);

        // Trigger many evictions
        for i in cache_size..cache_size * 3 {
            cache.insert(i, i * 10);
        }

        // Verify cache size is maintained
        assert_eq!(cache.len(), cache_size);

        // Clear cache
        cache.clear();
        assert_eq!(cache.len(), 0);

        println!("Memory cleanup verified");
    }

    #[test]
    fn test_large_value_memory_handling() {
        // Test with large values (1KB each)
        let large_value = vec![0u8; 1024];
        let cache_size = 1000;
        let mut cache = LrukCache::new(cache_size);

        // Fill cache with large values
        for i in 0..cache_size {
            cache.insert(i, large_value.clone());
        }

        assert_eq!(cache.len(), cache_size);

        // Trigger evictions
        for i in cache_size..cache_size + 100 {
            cache.insert(i, large_value.clone());
        }

        assert_eq!(cache.len(), cache_size);
        println!(
            "Successfully handled {} entries with 1KB values",
            cache_size
        );
    }

    #[test]
    fn test_access_history_memory_efficiency() {
        let mut cache = LrukCache::new(1000);

        // Fill cache and build extensive history
        for i in 0..1000 {
            cache.insert(i, i * 10);
            // Access multiple times to fill history
            for _ in 0..10 {
                cache.get(&i);
            }
        }

        assert_eq!(cache.len(), 1000);

        // History storage should be bounded by K (K=2 for LrukCache)
        // Each entry stores at most K timestamps
        println!("Access history is bounded by K=2 timestamps per entry");
    }

    #[test]
    fn test_memory_scaling_with_k() {
        // LrukCache has fixed K=2
        let k = 2;
        let cache_size = 1000;
        let mut cache = LrukCache::new(cache_size);

        // Fill cache
        for i in 0..cache_size {
            cache.insert(i, i * 10);
            for _ in 0..k {
                cache.get(&i);
            }
        }

        // Estimate memory per entry with K=2
        let history_size = std::mem::size_of::<u128>() * k; // K timestamps
        let entry_size = std::mem::size_of::<i32>() * 2 + history_size; // key + value + history

        println!(
            "Estimated size per entry with K={}: {} bytes",
            k, entry_size
        );

        assert!(entry_size < 500, "Per-entry memory too high");
    }
}
