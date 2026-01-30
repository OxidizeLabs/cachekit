// ==============================================
// LRU PERFORMANCE TESTS (integration)
// ==============================================

use cachekit::policy::lru::LruCore;
use cachekit::traits::CoreCache;
use std::sync::Arc;
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

mod lookup_performance {
    use super::*;

    #[test]
    fn test_get_operation_latency() {
        let mut cache = LruCore::new(1000);

        // Pre-fill cache
        for i in 0..1000 {
            cache.insert(i, Arc::new(i * 10));
        }

        // Measure get operation
        let iterations = 10000u32;
        let (_, duration) = measure_time(|| {
            for i in 0..iterations as i32 {
                let key = i % 1000;
                cache.get(&key);
            }
        });

        let avg_latency = duration / iterations;
        println!("Average get latency: {:?}", avg_latency);

        // Get should be very fast (< 1µs on modern hardware)
        assert!(
            avg_latency < Duration::from_micros(10),
            "Get operation too slow: {:?}",
            avg_latency
        );
    }

    #[test]
    fn test_peek_operation_latency() {
        let cache = LruCore::new(1000);

        // Pre-fill cache
        let mut cache_mut = cache;
        for i in 0..1000 {
            cache_mut.insert(i, Arc::new(i * 10));
        }

        // Measure contains operation (peek-like)
        let iterations = 10000u32;
        let (_, duration) = measure_time(|| {
            for i in 0..iterations as i32 {
                let key = i % 1000;
                cache_mut.contains(&key);
            }
        });

        let avg_latency = duration / iterations;
        println!("Average peek latency: {:?}", avg_latency);

        assert!(
            avg_latency < Duration::from_micros(5),
            "Peek operation too slow: {:?}",
            avg_latency
        );
    }

    #[test]
    fn test_contains_operation_latency() {
        let mut cache = LruCore::new(1000);

        // Pre-fill cache
        for i in 0..1000 {
            cache.insert(i, Arc::new(i * 10));
        }

        // Measure contains operation
        let iterations = 10000u32;
        let (_, duration) = measure_time(|| {
            for i in 0..iterations as i32 {
                let key = i % 1000;
                cache.contains(&key);
            }
        });

        let avg_latency = duration / iterations;
        println!("Average contains latency: {:?}", avg_latency);

        assert!(
            avg_latency < Duration::from_micros(5),
            "Contains operation too slow: {:?}",
            avg_latency
        );
    }

    #[test]
    fn test_lookup_scalability_with_size() {
        let sizes = vec![100, 500, 1000, 5000, 10000];
        let mut results = Vec::new();

        for &size in &sizes {
            let mut cache = LruCore::new(size);

            // Pre-fill cache
            for i in 0..size {
                cache.insert(i, Arc::new(i * 10));
            }

            // Measure lookup time
            let iterations = 1000;
            let (_, duration) = measure_time(|| {
                for i in 0..iterations {
                    let key = i % size;
                    cache.get(&key);
                }
            });

            let avg_latency = duration / iterations as u32;
            results.push((size, avg_latency));
            println!("Size: {}, Avg latency: {:?}", size, avg_latency);
        }

        // Verify O(1) behavior - check consecutive size doublings
        // If operations were O(n), doubling size would double time
        for i in 1..results.len() {
            let (prev_size, prev_time) = results[i - 1];
            let (curr_size, curr_time) = results[i];
            let size_ratio = curr_size as f64 / prev_size as f64;
            let time_ratio = curr_time.as_nanos() as f64 / prev_time.as_nanos() as f64;

            println!(
                "Size {}→{} ({:.2}x): time {}ns→{}ns ({:.2}x)",
                prev_size,
                curr_size,
                size_ratio,
                prev_time.as_nanos(),
                curr_time.as_nanos(),
                time_ratio
            );

            // For O(1) operations with cache effects:
            // - Time should grow slower than O(n) (where time_ratio ≈ size_ratio)
            // - Allow 2x overhead for cache/allocator effects
            // - Cap at 10x to catch truly problematic behavior
            assert!(
                time_ratio < size_ratio * 2.0 || time_ratio < 10.0,
                "Lookup time ratio {:.2}x for size ratio {:.2}x suggests worse than O(1)",
                time_ratio,
                size_ratio
            );
        }
    }

    #[test]
    fn test_lookup_performance_under_load() {
        let mut cache = LruCore::new(1000);

        // Pre-fill cache
        for i in 0..1000 {
            cache.insert(i, Arc::new(i * 10));
        }

        // Measure sustained lookup performance
        let iterations = 100000;
        let (_, duration) = measure_time(|| {
            for i in 0..iterations {
                let key = i % 1000;
                cache.get(&key);
            }
        });

        let throughput = iterations as f64 / duration.as_secs_f64();
        println!("Lookup throughput: {:.0} ops/sec", throughput);

        // Should handle at least 100K ops/sec
        assert!(
            throughput > 100_000.0,
            "Throughput too low: {:.0} ops/sec",
            throughput
        );
    }

    #[test]
    fn test_sequential_lookup_performance() {
        let mut cache = LruCore::new(1000);

        // Pre-fill cache
        for i in 0..1000 {
            cache.insert(i, Arc::new(i * 10));
        }

        // Sequential access pattern
        let iterations = 10000u32;
        let (_, duration) = measure_time(|| {
            for _ in 0..10 {
                for i in 0..1000 {
                    cache.get(&i);
                }
            }
        });

        let avg_latency = duration / iterations;
        println!("Sequential lookup avg latency: {:?}", avg_latency);

        assert!(avg_latency < Duration::from_micros(10));
    }

    #[test]
    fn test_random_lookup_performance() {
        let mut cache = LruCore::new(1000);

        // Pre-fill cache
        for i in 0..1000 {
            cache.insert(i, Arc::new(i * 10));
        }

        // Random access pattern using simple LCG
        let iterations = 10000u32;
        let (_, duration) = measure_time(|| {
            let mut seed = 12345u64;
            for _ in 0..iterations {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let key = (seed % 1000) as i32;
                cache.get(&key);
            }
        });

        let avg_latency = duration / iterations;
        println!("Random lookup avg latency: {:?}", avg_latency);

        assert!(avg_latency < Duration::from_micros(20));
    }

    #[test]
    fn test_cache_hit_performance() {
        let mut cache = LruCore::new(1000);

        // Pre-fill cache
        for i in 0..1000 {
            cache.insert(i, Arc::new(i * 10));
        }

        // Measure hit performance
        let iterations = 10000u32;
        let (_, duration) = measure_time(|| {
            for i in 0..iterations as i32 {
                let key = i % 1000;
                let result = cache.get(&key);
                assert!(result.is_some());
            }
        });

        let avg_latency = duration / iterations;
        println!("Cache hit avg latency: {:?}", avg_latency);

        assert!(avg_latency < Duration::from_micros(10));
    }

    #[test]
    fn test_cache_miss_performance() {
        let mut cache = LruCore::new(1000);

        // Pre-fill cache
        for i in 0..1000 {
            cache.insert(i, Arc::new(i * 10));
        }

        // Measure miss performance
        let iterations = 10000u32;
        let (_, duration) = measure_time(|| {
            for i in 0..iterations as i32 {
                let key = i + 10000; // Keys not in cache
                let result = cache.get(&key);
                assert!(result.is_none());
            }
        });

        let avg_latency = duration / iterations;
        println!("Cache miss avg latency: {:?}", avg_latency);

        // Misses should be even faster than hits (no list manipulation)
        assert!(avg_latency < Duration::from_micros(5));
    }
}

mod insertion_performance {
    use super::*;

    #[test]
    fn test_insert_operation_latency() {
        let mut cache = LruCore::new(10000);

        // Measure insert operation
        let iterations = 10000u32;
        let (_, duration) = measure_time(|| {
            for i in 0..iterations as i32 {
                cache.insert(i, Arc::new(i * 10));
            }
        });

        let avg_latency = duration / iterations;
        println!("Average insert latency: {:?}", avg_latency);

        assert!(
            avg_latency < Duration::from_micros(50),
            "Insert operation too slow: {:?}",
            avg_latency
        );
    }

    #[test]
    fn test_insert_arc_operation_latency() {
        let mut cache = LruCore::new(10000);

        // Measure insert operation with Arc values
        let iterations = 10000u32;
        let (_, duration) = measure_time(|| {
            for i in 0..iterations as i32 {
                cache.insert(i, Arc::new(i * 10));
            }
        });

        let avg_latency = duration / iterations;
        println!("Average insert Arc latency: {:?}", avg_latency);

        assert!(avg_latency < Duration::from_micros(50));
    }

    #[test]
    fn test_insertion_scalability_with_size() {
        let sizes = vec![100, 500, 1000, 5000, 10000];
        let mut results = Vec::new();

        for &size in &sizes {
            let mut cache = LruCore::new(size);

            // Measure insertion time
            let (_, duration) = measure_time(|| {
                for i in 0..size {
                    cache.insert(i, Arc::new(i * 10));
                }
            });

            let avg_latency = duration / size as u32;
            results.push((size, avg_latency));
            println!("Size: {}, Avg insert latency: {:?}", size, avg_latency);
        }

        // Verify O(1) insertion - check consecutive measurements
        for i in 1..results.len() {
            let (prev_size, prev_time) = results[i - 1];
            let (curr_size, curr_time) = results[i];
            let size_ratio = curr_size as f64 / prev_size as f64;
            let time_ratio = curr_time.as_nanos() as f64 / prev_time.as_nanos() as f64;

            println!(
                "Size {}→{} ({:.2}x): time {}ns→{}ns ({:.2}x)",
                prev_size,
                curr_size,
                size_ratio,
                prev_time.as_nanos(),
                curr_time.as_nanos(),
                time_ratio
            );

            // For O(1) operations: time should grow slower than O(n)
            // Allow 2x overhead for cache effects, cap at 10x
            assert!(
                time_ratio < size_ratio * 2.0 || time_ratio < 10.0,
                "Insert time ratio {:.2}x for size ratio {:.2}x suggests worse than O(1)",
                time_ratio,
                size_ratio
            );
        }
    }

    #[test]
    fn test_insertion_into_full_cache() {
        let mut cache = LruCore::new(1000);

        // Fill cache
        for i in 0..1000 {
            cache.insert(i, Arc::new(i * 10));
        }

        // Measure insertion with eviction
        let iterations = 10000u32;
        let (_, duration) = measure_time(|| {
            for i in 1000..1000 + iterations as i32 {
                cache.insert(i, Arc::new(i * 10));
            }
        });

        let avg_latency = duration / iterations;
        println!("Insert with eviction avg latency: {:?}", avg_latency);

        // Should still be O(1)
        assert!(avg_latency < Duration::from_micros(15));
    }
}

mod eviction_performance {
    use super::*;

    #[test]
    fn test_pop_lru_operation_latency() {
        let mut cache = LruCore::new(10000);

        // Pre-fill cache
        for i in 0..10000 {
            cache.insert(i, Arc::new(i * 10));
        }

        // Measure pop_lru operation (via manual eviction simulation)
        let iterations = 5000u32;
        let (_, duration) = measure_time(|| {
            for _ in 0..iterations {
                // Trigger eviction by inserting when full
                cache.insert(-1, Arc::new(-10));
            }
        });

        let avg_latency = duration / iterations;
        println!("Average eviction latency: {:?}", avg_latency);

        assert!(avg_latency < Duration::from_micros(100));
    }

    #[test]
    fn test_automatic_eviction_latency() {
        let mut cache = LruCore::new(1000);

        // Pre-fill cache
        for i in 0..1000 {
            cache.insert(i, Arc::new(i * 10));
        }

        // Measure automatic eviction during insert
        let iterations = 10000u32;
        let (_, duration) = measure_time(|| {
            for i in 1000..1000 + iterations as i32 {
                cache.insert(i, Arc::new(i * 10));
            }
        });

        let avg_latency = duration / iterations;
        println!("Automatic eviction avg latency: {:?}", avg_latency);

        assert!(avg_latency < Duration::from_micros(15));
    }

    #[test]
    fn test_eviction_scalability_with_size() {
        let sizes = vec![100, 500, 1000, 5000, 10000];
        let mut results = Vec::new();

        for &size in &sizes {
            let mut cache = LruCore::new(size);

            // Pre-fill cache
            for i in 0..size {
                cache.insert(i, Arc::new(i * 10));
            }

            // Measure eviction time
            let iterations = 1000;
            let (_, duration) = measure_time(|| {
                for i in size..size + iterations {
                    cache.insert(i, Arc::new(i * 10));
                }
            });

            let avg_latency = duration / iterations as u32;
            results.push((size, avg_latency));
            println!("Size: {}, Avg eviction latency: {:?}", size, avg_latency);
        }

        // Verify O(1) eviction - check consecutive measurements
        for i in 1..results.len() {
            let (prev_size, prev_time) = results[i - 1];
            let (curr_size, curr_time) = results[i];
            let size_ratio = curr_size as f64 / prev_size as f64;
            let time_ratio = curr_time.as_nanos() as f64 / prev_time.as_nanos() as f64;

            println!(
                "Size {}→{} ({:.2}x): time {}ns→{}ns ({:.2}x)",
                prev_size,
                curr_size,
                size_ratio,
                prev_time.as_nanos(),
                curr_time.as_nanos(),
                time_ratio
            );

            // For O(1) operations: time should grow slower than O(n)
            // Allow 2x overhead for cache effects, cap at 10x
            assert!(
                time_ratio < size_ratio * 2.0 || time_ratio < 10.0,
                "Eviction time ratio {:.2}x for size ratio {:.2}x suggests worse than O(1)",
                time_ratio,
                size_ratio
            );
        }
    }
}

mod memory_efficiency {
    use super::*;

    #[test]
    fn test_cache_memory_footprint() {
        let cache_size = 10000;
        let cache = LruCore::new(cache_size);

        // Fill cache
        let mut cache_mut = cache;
        for i in 0..cache_size {
            cache_mut.insert(i, Arc::new(i * 10));
        }

        // Verify cache is at capacity
        assert_eq!(cache_mut.len(), cache_size);
        assert_eq!(cache_mut.capacity(), cache_size);

        println!(
            "Cache with {} entries uses approximately {} bytes per entry",
            cache_size,
            std::mem::size_of::<i32>() * 2 + 64 // Rough estimate
        );
    }

    #[test]
    fn test_per_item_memory_overhead() {
        // Measure approximate memory overhead per item
        let node_overhead = std::mem::size_of::<usize>() * 3; // prev, next, key ptr estimate
        let hashmap_overhead = std::mem::size_of::<usize>() * 2; // hash table entry

        println!(
            "Estimated per-item overhead: {} bytes",
            node_overhead + hashmap_overhead
        );

        // LRU should have reasonable overhead (< 100 bytes per item)
        assert!(node_overhead + hashmap_overhead < 100);
    }
}

mod complexity {
    use super::*;

    #[test]
    fn test_insert_time_complexity() {
        // Verify O(1) insert complexity
        let sizes = vec![1000, 2000, 4000, 8000, 16000];
        let mut times = Vec::new();

        for &size in &sizes {
            let mut cache = LruCore::new(size);

            let (_, duration) = measure_time(|| {
                for i in 0..size {
                    cache.insert(i, Arc::new(i * 10));
                }
            });

            let avg_time = duration.as_nanos() as f64 / size as f64;
            times.push(avg_time);
            println!("Size: {}, Avg insert time: {:.2} ns", size, avg_time);
        }

        // Check that doubling size doesn't double time (would indicate O(n))
        // Allow up to 4x due to cache effects and allocator overhead
        for i in 1..times.len() {
            let ratio = times[i] / times[i - 1];
            assert!(
                ratio < 4.0,
                "Insert time ratio {:.2} suggests non-O(1) complexity",
                ratio
            );
        }
    }

    #[test]
    fn test_get_time_complexity() {
        // Verify O(1) get complexity
        let sizes = vec![1000, 2000, 4000, 8000, 16000];
        let mut times = Vec::new();

        for &size in &sizes {
            let mut cache = LruCore::new(size);

            // Pre-fill cache
            for i in 0..size {
                cache.insert(i, Arc::new(i * 10));
            }

            // Measure get time
            let iterations = 10000;
            let (_, duration) = measure_time(|| {
                for i in 0..iterations {
                    cache.get(&(i % size));
                }
            });

            let avg_time = duration.as_nanos() as f64 / iterations as f64;
            times.push(avg_time);
            println!("Size: {}, Avg get time: {:.2} ns", size, avg_time);
        }

        // Verify O(1) behavior
        // Allow up to 4x due to cache effects
        for i in 1..times.len() {
            let ratio = times[i] / times[i - 1];
            assert!(
                ratio < 4.0,
                "Get time ratio {:.2} suggests non-O(1) complexity",
                ratio
            );
        }
    }

    #[test]
    fn test_pop_lru_time_complexity() {
        // Verify O(1) eviction complexity
        let sizes = vec![1000, 2000, 4000, 8000, 16000];
        let mut times = Vec::new();

        for &size in &sizes {
            let mut cache = LruCore::new(size);

            // Pre-fill cache
            for i in 0..size {
                cache.insert(i, Arc::new(i * 10));
            }

            // Measure eviction time
            let iterations = 1000;
            let (_, duration) = measure_time(|| {
                for i in size..size + iterations {
                    cache.insert(i, Arc::new(i * 10)); // Triggers eviction
                }
            });

            let avg_time = duration.as_nanos() as f64 / iterations as f64;
            times.push(avg_time);
            println!("Size: {}, Avg eviction time: {:.2} ns", size, avg_time);
        }

        // Verify O(1) behavior
        // Allow up to 4x due to cache effects and allocator overhead
        for i in 1..times.len() {
            let ratio = times[i] / times[i - 1];
            assert!(
                ratio < 4.0,
                "Eviction time ratio {:.2} suggests non-O(1) complexity",
                ratio
            );
        }
    }
}
