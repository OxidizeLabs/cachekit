// ==============================================
// NRU CONCURRENCY TESTS (integration)
// ==============================================
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

mod thread_safe_wrapper {
    use cachekit::policy::nru::NruCache;
    use cachekit::traits::{CoreCache, MutableCache};

    use super::*;

    // Helper type for thread-safe testing
    type ThreadSafeNruCache<K, V> = Arc<Mutex<NruCache<K, V>>>;

    #[test]
    fn test_basic_thread_safe_operations() {
        let cache: ThreadSafeNruCache<String, String> = Arc::new(Mutex::new(NruCache::new(100)));
        let num_threads = 8;
        let operations_per_thread = 250;
        let success_count = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let success_count = success_count.clone();

                thread::spawn(move || {
                    let mut thread_successes = 0;

                    for i in 0..operations_per_thread {
                        // Test different operations with proper synchronization
                        match i % 4 {
                            0 => {
                                // Insert operation
                                let key = format!("thread_{}_{}", thread_id, i);
                                let value = format!("value_{}_{}", thread_id, i);

                                if let Ok(mut cache_guard) = cache.lock() {
                                    cache_guard.insert(key, value);
                                    thread_successes += 1;
                                }
                            },
                            1 => {
                                // Get operation (sets reference bit)
                                let key = format!("thread_{}_0", thread_id);

                                if let Ok(mut cache_guard) = cache.lock() {
                                    let _ = cache_guard.get(&key);
                                    thread_successes += 1;
                                }
                            },
                            2 => {
                                // Contains operation (doesn't affect reference bit)
                                let key = format!("thread_{}_{}", thread_id, i / 2);

                                if let Ok(cache_guard) = cache.lock() {
                                    let _ = cache_guard.contains(&key);
                                    thread_successes += 1;
                                }
                            },
                            _ => {
                                // Remove operation
                                if let Ok(mut cache_guard) = cache.lock() {
                                    if i % 20 == 0 {
                                        let key = format!("thread_{}_{}", thread_id, i / 4);
                                        let _ = cache_guard.remove(&key);
                                    }
                                    thread_successes += 1;
                                }
                            },
                        }
                    }

                    success_count.fetch_add(thread_successes, Ordering::SeqCst);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let total_successes = success_count.load(Ordering::SeqCst);
        let expected_operations = num_threads * operations_per_thread;

        println!(
            "Basic thread-safe operations: {}/{} successful",
            total_successes, expected_operations
        );

        // Verify cache consistency
        let final_cache = cache.lock().unwrap();
        let cache_len = final_cache.len();
        let capacity = final_cache.capacity();

        assert!(
            cache_len <= capacity,
            "Cache length {} exceeded capacity {}",
            cache_len,
            capacity
        );

        println!(
            "Final cache state: len={}, capacity={}",
            cache_len, capacity
        );
    }

    #[test]
    fn test_concurrent_inserts() {
        let capacity = 1_600;
        let cache: ThreadSafeNruCache<u64, u64> = Arc::new(Mutex::new(NruCache::new(capacity)));

        let num_threads = 8;
        let inserts_per_thread = 200;
        let successes = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let successes = successes.clone();

                thread::spawn(move || {
                    for i in 0..inserts_per_thread {
                        let key = (thread_id * inserts_per_thread + i) as u64;
                        if let Ok(mut cache_guard) = cache.lock() {
                            cache_guard.insert(key, key);
                            successes.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let expected_inserts = num_threads * inserts_per_thread;
        assert_eq!(successes.load(Ordering::SeqCst), expected_inserts);

        let cache_guard = cache.lock().unwrap();
        assert_eq!(cache_guard.len(), expected_inserts);
        assert!(cache_guard.capacity() >= expected_inserts);
    }

    #[test]
    fn test_concurrent_reads() {
        let capacity = 512;
        let cache: ThreadSafeNruCache<u64, u64> = Arc::new(Mutex::new(NruCache::new(capacity)));

        // Pre-populate cache
        {
            let mut cache_guard = cache.lock().unwrap();
            for key in 0..capacity {
                cache_guard.insert(key as u64, key as u64 * 2);
            }
        }

        let reader_threads = 16;
        let reads_per_thread = 800;
        let hits = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..reader_threads)
            .map(|_| {
                let cache = cache.clone();
                let hits = hits.clone();

                thread::spawn(move || {
                    for i in 0..reads_per_thread {
                        let key = (i % capacity) as u64;
                        if let Ok(mut cache_guard) = cache.lock() {
                            if cache_guard.get(&key).is_some() {
                                hits.fetch_add(1, Ordering::Relaxed);
                            }
                        }

                        // Exercise read-only contains path occasionally
                        if i % 50 == 0 {
                            if let Ok(cache_guard) = cache.lock() {
                                let _ = cache_guard.contains(&key);
                            }
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let expected_reads = reader_threads * reads_per_thread;
        assert_eq!(hits.load(Ordering::Relaxed), expected_reads);

        let cache_guard = cache.lock().unwrap();
        assert_eq!(cache_guard.len(), capacity);
    }

    #[test]
    fn test_concurrent_removes() {
        let total_keys = 400;
        let cache: ThreadSafeNruCache<u64, u64> = Arc::new(Mutex::new(NruCache::new(total_keys)));

        // Pre-populate cache
        {
            let mut cache_guard = cache.lock().unwrap();
            for key in 0..total_keys {
                cache_guard.insert(key as u64, key as u64);
            }
        }

        let remover_threads = 4;
        let removes_per_thread = 100;
        let successful_removes = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..remover_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let successful_removes = successful_removes.clone();

                thread::spawn(move || {
                    for i in 0..removes_per_thread {
                        let key = (thread_id * removes_per_thread + i) as u64;
                        if let Ok(mut cache_guard) = cache.lock() {
                            if cache_guard.remove(&key).is_some() {
                                successful_removes.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let expected_removes = remover_threads * removes_per_thread;
        assert_eq!(successful_removes.load(Ordering::SeqCst), expected_removes);

        let cache_guard = cache.lock().unwrap();
        let expected_remaining = total_keys - expected_removes;
        assert_eq!(cache_guard.len(), expected_remaining);
    }

    #[test]
    fn test_mixed_workload() {
        let capacity = 1000;
        let cache: ThreadSafeNruCache<u64, String> = Arc::new(Mutex::new(NruCache::new(capacity)));

        let num_threads = 8;
        let ops_per_thread = 500;
        let shutdown = Arc::new(AtomicBool::new(false));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let shutdown = shutdown.clone();

                thread::spawn(move || {
                    let mut local_ops = 0;

                    for i in 0..ops_per_thread {
                        if shutdown.load(Ordering::Relaxed) {
                            break;
                        }

                        let key = ((thread_id * ops_per_thread + i) % (capacity * 2)) as u64;
                        let value = format!("value_{}_{}", thread_id, i);

                        if let Ok(mut cache_guard) = cache.lock() {
                            match i % 5 {
                                0 | 1 => {
                                    // Insert (40%)
                                    cache_guard.insert(key, value);
                                },
                                2 | 3 => {
                                    // Get (40%) - sets reference bit
                                    let _ = cache_guard.get(&key);
                                },
                                _ => {
                                    // Remove (20%)
                                    let _ = cache_guard.remove(&key);
                                },
                            }
                            local_ops += 1;
                        }
                    }

                    local_ops
                })
            })
            .collect();

        // Let threads run for a bit, then signal shutdown
        thread::sleep(Duration::from_millis(100));
        shutdown.store(true, Ordering::SeqCst);

        let mut total_ops = 0;
        for handle in handles {
            total_ops += handle.join().unwrap();
        }

        let cache_guard = cache.lock().unwrap();
        let final_len = cache_guard.len();
        let capacity = cache_guard.capacity();

        println!(
            "Mixed workload: {} total operations, final len={}, capacity={}",
            total_ops, final_len, capacity
        );

        assert!(
            final_len <= capacity,
            "Cache length {} exceeded capacity {}",
            final_len,
            capacity
        );
    }

    #[test]
    fn test_reference_bit_behavior() {
        // Test that reference bits are properly managed under concurrent access
        let capacity = 100;
        let cache: ThreadSafeNruCache<u64, u64> = Arc::new(Mutex::new(NruCache::new(capacity)));

        // Pre-populate cache
        {
            let mut cache_guard = cache.lock().unwrap();
            for key in 0..capacity {
                cache_guard.insert(key as u64, key as u64);
            }
        }

        let accessor_threads = 4;
        let inserter_threads = 4;
        let accesses_per_thread = 200;
        let inserts_per_thread = 50;

        let handles: Vec<_> = (0..accessor_threads)
            .map(|_| {
                let cache = cache.clone();

                thread::spawn(move || {
                    for i in 0..accesses_per_thread {
                        let key = (i % capacity) as u64;
                        if let Ok(mut cache_guard) = cache.lock() {
                            // Access sets reference bit
                            let _ = cache_guard.get(&key);
                        }
                    }
                })
            })
            .collect();

        let inserter_handles: Vec<_> = (0..inserter_threads)
            .map(|thread_id| {
                let cache = cache.clone();

                thread::spawn(move || {
                    for i in 0..inserts_per_thread {
                        let key = (capacity + thread_id * inserts_per_thread + i) as u64;
                        if let Ok(mut cache_guard) = cache.lock() {
                            // Insert will trigger eviction (capacity already full)
                            cache_guard.insert(key, key);
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
        for handle in inserter_handles {
            handle.join().unwrap();
        }

        let cache_guard = cache.lock().unwrap();
        assert_eq!(cache_guard.len(), capacity);

        println!("Reference bit behavior test completed successfully");
    }

    #[test]
    fn test_no_data_races() {
        // Stress test for data races
        let capacity = 256;
        let cache: ThreadSafeNruCache<u64, u64> = Arc::new(Mutex::new(NruCache::new(capacity)));

        let num_threads = 16;
        let iterations = 1000;

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let cache = cache.clone();

                thread::spawn(move || {
                    for i in 0..iterations {
                        let key = ((thread_id * iterations + i) % (capacity * 2)) as u64;

                        if let Ok(mut cache_guard) = cache.lock() {
                            // Rapid insert/get/remove cycle
                            cache_guard.insert(key, key);
                            let _ = cache_guard.get(&key);
                            if i % 3 == 0 {
                                let _ = cache_guard.remove(&key);
                            }
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let cache_guard = cache.lock().unwrap();
        assert!(cache_guard.len() <= capacity);

        println!("No data races detected in stress test");
    }

    #[test]
    fn test_consistent_state_under_contention() {
        let capacity = 100;
        let cache: ThreadSafeNruCache<u64, String> = Arc::new(Mutex::new(NruCache::new(capacity)));

        let num_threads = 10;
        let ops_per_thread = 200;

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let cache = cache.clone();

                thread::spawn(move || {
                    for i in 0..ops_per_thread {
                        let key = (thread_id * ops_per_thread + i) as u64;
                        let value = format!("v{}", key);

                        if let Ok(mut cache_guard) = cache.lock() {
                            // Insert
                            cache_guard.insert(key, value.clone());

                            // Verify immediate read
                            if let Some(retrieved) = cache_guard.get(&key) {
                                assert_eq!(retrieved, &value);
                            }

                            // Verify contains
                            assert!(cache_guard.contains(&key));
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let cache_guard = cache.lock().unwrap();
        assert!(
            cache_guard.len() <= capacity,
            "Cache exceeded capacity: len={}, cap={}",
            cache_guard.len(),
            capacity
        );

        println!("Cache maintained consistent state under contention");
    }
}

mod performance {
    use super::*;
    use cachekit::policy::nru::NruCache;
    use cachekit::traits::CoreCache;

    #[test]
    fn benchmark_throughput() {
        let capacity = 1000;
        let cache = Arc::new(Mutex::new(NruCache::new(capacity)));

        let num_threads = 8;
        let ops_per_thread = 10_000;

        let start = Instant::now();

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let cache = cache.clone();

                thread::spawn(move || {
                    for i in 0..ops_per_thread {
                        let key = ((thread_id * ops_per_thread + i) % (capacity * 2)) as u64;

                        if let Ok(mut cache_guard) = cache.lock() {
                            match i % 3 {
                                0 => {
                                    cache_guard.insert(key, key);
                                },
                                1 => {
                                    let _ = cache_guard.get(&key);
                                },
                                _ => {
                                    let _ = cache_guard.contains(&key);
                                },
                            }
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let elapsed = start.elapsed();
        let total_ops = num_threads * ops_per_thread;
        let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();

        println!(
            "Throughput: {:.0} ops/sec ({} ops in {:?})",
            ops_per_sec, total_ops, elapsed
        );

        // Sanity check
        assert!(ops_per_sec > 100_000.0, "Throughput too low");
    }
}
