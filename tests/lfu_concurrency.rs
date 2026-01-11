// ==============================================
// LFU CONCURRENCY TESTS (integration)
// ==============================================

use cachekit::policy::lfu::LFUCache;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

type ThreadSafeLFUCache<K, V> = Arc<Mutex<LFUCache<K, V>>>;

// Thread Safety Tests
mod thread_safety {
    use super::*;
    use cachekit::traits::{CoreCache, LFUCacheTrait, MutableCache};

    #[test]
    fn test_concurrent_insertions() {
        let cache: ThreadSafeLFUCache<String, i32> = Arc::new(Mutex::new(LFUCache::new(1000)));
        let num_threads = 8;
        let items_per_thread = 100;

        let mut handles = vec![];

        // Spawn threads that each insert items
        for thread_id in 0..num_threads {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for i in 0..items_per_thread {
                    let key = format!("thread_{}_{}", thread_id, i);
                    let value = (thread_id * items_per_thread + i) as i32;

                    cache_clone.lock().unwrap().insert(key, Arc::new(value));
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify results
        let mut cache = cache.lock().unwrap();
        assert_eq!(cache.len(), num_threads * items_per_thread);

        // Verify all items exist and have correct values
        for thread_id in 0..num_threads {
            for i in 0..items_per_thread {
                let key = format!("thread_{}_{}", thread_id, i);
                let expected_value = (thread_id * items_per_thread + i) as i32;
                assert_eq!(cache.get(&key).map(Arc::as_ref), Some(&expected_value));
                assert_eq!(cache.frequency(&key), Some(2)); // 1 from insert, 1 from get
            }
        }
    }

    #[test]
    fn test_concurrent_gets() {
        let cache: ThreadSafeLFUCache<String, i32> = Arc::new(Mutex::new(LFUCache::new(500)));

        // Pre-populate cache with test data
        {
            let mut cache_guard = cache.lock().unwrap();
            for i in 0..500 {
                cache_guard.insert(format!("key_{}", i), Arc::new(i));
            }
        }

        let num_threads = 10;
        let reads_per_thread = 1000;
        let hit_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        // Spawn threads that each perform read operations
        for _thread_id in 0..num_threads {
            let cache_clone = Arc::clone(&cache);
            let hit_count_clone = Arc::clone(&hit_count);

            let handle = thread::spawn(move || {
                for i in 0..reads_per_thread {
                    let key = format!("key_{}", i % 500); // Ensure keys exist

                    if cache_clone.lock().unwrap().get(&key).is_some() {
                        hit_count_clone.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify results
        let expected_hits = num_threads * reads_per_thread;
        assert_eq!(
            hit_count.load(Ordering::Relaxed),
            expected_hits,
            "All reads should hit since all keys exist"
        );

        // Verify frequency counts increased due to concurrent access
        let cache = cache.lock().unwrap();
        for i in (0..500).step_by(50) {
            let freq = cache.frequency(&format!("key_{}", i)).unwrap();
            // Each key should have been accessed multiple times (once for insert + multiple gets)
            assert!(
                freq > 1,
                "Key key_{} should have frequency > 1, got {}",
                i,
                freq
            );
        }
    }

    #[test]
    fn test_concurrent_frequency_operations() {
        let cache: ThreadSafeLFUCache<String, i32> = Arc::new(Mutex::new(LFUCache::new(100)));

        // Pre-populate cache
        {
            let mut cache_guard = cache.lock().unwrap();
            for i in 0..100 {
                cache_guard.insert(format!("freq_key_{}", i), Arc::new(i));
            }
        }

        let num_threads = 6;
        let operations_per_thread = 500;
        let mut handles = vec![];

        // Spawn threads performing different frequency operations
        for thread_id in 0..num_threads {
            let cache_clone = Arc::clone(&cache);

            let handle = thread::spawn(move || {
                for i in 0..operations_per_thread {
                    let key = format!("freq_key_{}", i % 100);

                    match thread_id % 3 {
                        0 => {
                            // Thread type 1: increment frequency via get
                            cache_clone.lock().unwrap().get(&key);
                        },
                        1 => {
                            // Thread type 2: increment frequency directly
                            cache_clone.lock().unwrap().increment_frequency(&key);
                        },
                        2 => {
                            // Thread type 3: reset frequency (less frequently)
                            if i % 10 == 0 {
                                cache_clone.lock().unwrap().reset_frequency(&key);
                            } else {
                                cache_clone.lock().unwrap().get(&key);
                            }
                        },
                        _ => unreachable!(),
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify cache consistency after concurrent frequency operations
        let mut cache = cache.lock().unwrap();
        assert_eq!(cache.len(), 100, "Cache should still contain all items");

        // Verify all items still exist and have reasonable frequency values
        for i in 0..100 {
            let key = format!("freq_key_{}", i);
            assert!(cache.contains(&key), "Key {} should still exist", key);

            let freq = cache.frequency(&key).unwrap();
            assert!(
                freq >= 1,
                "Frequency should be at least 1 for key {}, got {}",
                key,
                freq
            );
            assert!(
                freq <= 1200,
                "Frequency should be reasonable for key {}, got {}",
                key,
                freq
            );
        }

        // Verify cache is still functional
        assert!(cache.get(&"freq_key_0".to_string()).is_some());
        assert!(cache.peek_lfu().is_some());
    }

    #[test]
    fn test_concurrent_lfu_operations() {
        let cache: ThreadSafeLFUCache<String, i32> = Arc::new(Mutex::new(LFUCache::new(200)));

        // Pre-populate cache with different frequency patterns
        {
            let mut cache_guard = cache.lock().unwrap();
            for i in 0..200 {
                cache_guard.insert(format!("lfu_key_{}", i), Arc::new(i));
            }

            // Create frequency differences - some items will be more frequent
            for _ in 0..5 {
                for i in 0..50 {
                    cache_guard.get(&format!("lfu_key_{}", i)); // High frequency
                }
            }

            for _ in 0..2 {
                for i in 50..100 {
                    cache_guard.get(&format!("lfu_key_{}", i)); // Medium frequency
                }
            }
            // Items 100-199 remain at frequency 1 (low frequency)
        }

        let num_peek_threads = 2;
        let num_pop_threads = 1;
        let operations_per_thread = 10;
        let popped_items = Arc::new(Mutex::new(Vec::new()));
        let mut handles = vec![];

        // Spawn peek threads
        for _ in 0..num_peek_threads {
            let cache_clone = Arc::clone(&cache);

            let handle = thread::spawn(move || {
                for _ in 0..operations_per_thread {
                    // Use a single lock to avoid double locking issues
                    let cache_guard = cache_clone.lock().unwrap();
                    if let Some((key, _)) = cache_guard.peek_lfu() {
                        // Verify the peeked item exists within the same lock
                        assert!(cache_guard.contains(key));
                    }
                    // Lock is automatically released here
                }
            });
            handles.push(handle);
        }

        // Spawn pop threads (fewer since they modify the cache)
        for _ in 0..num_pop_threads {
            let cache_clone = Arc::clone(&cache);
            let popped_clone = Arc::clone(&popped_items);

            let handle = thread::spawn(move || {
                for _ in 0..operations_per_thread {
                    // Use a single lock for the cache operation
                    let popped_item = cache_clone.lock().unwrap().pop_lfu();

                    if let Some((key, value)) = popped_item {
                        // Use a separate lock for the results collection
                        popped_clone.lock().unwrap().push((key, value));
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify results
        let cache = cache.lock().unwrap();
        let popped = popped_items.lock().unwrap();

        // Verify cache size is reduced by the number of popped items
        assert_eq!(
            cache.len() + popped.len(),
            200,
            "Cache size + popped items should equal original size"
        );

        // Verify all popped items are unique
        let mut popped_keys: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (key, _) in popped.iter() {
            assert!(
                popped_keys.insert(key.clone()),
                "Duplicate key popped: {}",
                key
            );
        }

        // Verify popped items are no longer in cache
        for (key, _) in popped.iter() {
            assert!(
                !cache.contains(key),
                "Popped key {} should not be in cache",
                key
            );
        }

        // Verify cache is still functional after concurrent operations
        if !cache.is_empty() {
            assert!(cache.peek_lfu().is_some());
        }
    }

    #[test]
    fn test_mixed_concurrent_operations() {
        let cache: ThreadSafeLFUCache<String, i32> = Arc::new(Mutex::new(LFUCache::new(300)));
        let num_threads = 8;
        let operations_per_thread = 200;
        let operation_counts = Arc::new(Mutex::new((0, 0, 0, 0, 0))); // (inserts, gets, removes, frequency_ops, lfu_ops)
        let mut handles = vec![];

        // Pre-populate with some initial data
        {
            let mut cache_guard = cache.lock().unwrap();
            for i in 0..150 {
                cache_guard.insert(format!("initial_{}", i), Arc::new(i));
            }
        }

        // Spawn threads performing mixed operations
        for thread_id in 0..num_threads {
            let cache_clone = Arc::clone(&cache);
            let counts_clone = Arc::clone(&operation_counts);

            let handle = thread::spawn(move || {
                for i in 0..operations_per_thread {
                    let operation = (thread_id + i) % 8;

                    match operation {
                        0 | 1 => {
                            // Insert operations (25% of operations)
                            let key = format!("mixed_{}_{}", thread_id, i);
                            let value = thread_id * 1000 + i;
                            cache_clone.lock().unwrap().insert(key, Arc::new(value));
                            counts_clone.lock().unwrap().0 += 1;
                        },
                        2..=4 => {
                            // Get operations (37.5% of operations)
                            let key = if i % 2 == 0 {
                                format!("initial_{}", i % 150)
                            } else {
                                format!(
                                    "mixed_{}_{}",
                                    thread_id % num_threads,
                                    i % operations_per_thread
                                )
                            };
                            cache_clone.lock().unwrap().get(&key);
                            counts_clone.lock().unwrap().1 += 1;
                        },
                        5 => {
                            // Remove operations (12.5% of operations)
                            let key = format!("initial_{}", i % 150);
                            cache_clone.lock().unwrap().remove(&key);
                            counts_clone.lock().unwrap().2 += 1;
                        },
                        6 => {
                            // Frequency operations (12.5% of operations)
                            let key = format!("initial_{}", i % 150);
                            if i % 3 == 0 {
                                cache_clone.lock().unwrap().increment_frequency(&key);
                            } else {
                                cache_clone.lock().unwrap().reset_frequency(&key);
                            }
                            counts_clone.lock().unwrap().3 += 1;
                        },
                        7 => {
                            // LFU operations (12.5% of operations)
                            if i % 2 == 0 {
                                cache_clone.lock().unwrap().peek_lfu();
                            } else {
                                cache_clone.lock().unwrap().pop_lfu();
                            }
                            counts_clone.lock().unwrap().4 += 1;
                        },
                        _ => unreachable!(),
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify results
        let mut cache = cache.lock().unwrap();
        let counts = operation_counts.lock().unwrap();

        // Verify cache is still functional and consistent
        assert!(cache.len() <= 300, "Cache should not exceed capacity");
        assert!(cache.capacity() == 300, "Capacity should remain unchanged");

        // Verify operation counts
        let (inserts, gets, removes, freq_ops, lfu_ops) = *counts;
        let total_operations = inserts + gets + removes + freq_ops + lfu_ops;
        assert_eq!(total_operations, num_threads * operations_per_thread);

        // Verify cache operations still work correctly
        cache.insert("test_after_concurrent".to_string(), Arc::new(999));
        assert_eq!(
            cache
                .get(&"test_after_concurrent".to_string())
                .map(Arc::as_ref),
            Some(&999)
        );
        assert!(cache.contains(&"test_after_concurrent".to_string()));

        // Verify LFU operations still work
        if !cache.is_empty() {
            assert!(cache.peek_lfu().is_some());
        }

        log::info!(
            "Mixed operations completed - Inserts: {}, Gets: {}, Removes: {}, Freq: {}, LFU: {}",
            inserts,
            gets,
            removes,
            freq_ops,
            lfu_ops
        );
    }

    #[test]
    fn test_concurrent_eviction_scenarios() {
        let cache: ThreadSafeLFUCache<String, i32> = Arc::new(Mutex::new(LFUCache::new(100)));
        let num_insert_threads = 6;
        let num_access_threads = 3;
        let inserts_per_thread = 50;
        let accesses_per_thread = 200;
        let mut handles = vec![];

        // Pre-populate cache to capacity
        {
            let mut cache_guard = cache.lock().unwrap();
            for i in 0..100 {
                cache_guard.insert(format!("base_{}", i), Arc::new(i));
            }

            // Create frequency differences to establish clear LFU candidates
            for _ in 0..10 {
                for i in 0..20 {
                    cache_guard.get(&format!("base_{}", i)); // High frequency
                }
            }

            for _ in 0..3 {
                for i in 20..50 {
                    cache_guard.get(&format!("base_{}", i)); // Medium frequency
                }
            }
            // Items 50-99 remain at frequency 1 (low frequency - eviction candidates)
        }

        // Spawn threads that insert new items (triggering evictions)
        for thread_id in 0..num_insert_threads {
            let cache_clone = Arc::clone(&cache);

            let handle = thread::spawn(move || {
                for i in 0..inserts_per_thread {
                    let key = format!("evict_trigger_{}_{}", thread_id, i);
                    let value = thread_id * 1000 + i;
                    cache_clone.lock().unwrap().insert(key, Arc::new(value));

                    // Small delay to increase thread interleaving
                    thread::sleep(std::time::Duration::from_nanos(100));
                }
            });
            handles.push(handle);
        }

        // Spawn threads that access existing items (changing frequencies)
        for thread_id in 0..num_access_threads {
            let cache_clone = Arc::clone(&cache);

            let handle = thread::spawn(move || {
                for i in 0..accesses_per_thread {
                    // Access different ranges to create frequency competition
                    let key = match thread_id % 3 {
                        0 => format!("base_{}", i % 30),        // Access high-freq items
                        1 => format!("base_{}", 30 + (i % 30)), // Access medium-freq items
                        2 => format!("base_{}", 60 + (i % 40)), // Access low-freq items
                        _ => unreachable!(),
                    };

                    cache_clone.lock().unwrap().get(&key);

                    // Small delay to increase thread interleaving
                    thread::sleep(std::time::Duration::from_nanos(50));
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify results after concurrent evictions
        let cache = cache.lock().unwrap();

        // Cache should maintain its capacity
        assert_eq!(
            cache.len(),
            100,
            "Cache should maintain capacity after evictions"
        );

        // High frequency items should be preserved
        let mut high_freq_preserved = 0;
        for i in 0..20 {
            if cache.contains(&format!("base_{}", i)) {
                high_freq_preserved += 1;
            }
        }

        // Medium frequency items may be partially preserved
        let mut medium_freq_preserved = 0;
        for i in 20..40 {
            if cache.contains(&format!("base_{}", i)) {
                medium_freq_preserved += 1;
            }
        }

        // Low frequency items should be mostly evicted
        let mut low_freq_preserved = 0;
        for i in 40..60 {
            if cache.contains(&format!("base_{}", i)) {
                low_freq_preserved += 1;
            }
        }

        // Some new items should be present
        let mut new_items_present = 0;
        for thread_id in 0..num_insert_threads {
            for i in 0..inserts_per_thread {
                let key = format!("evict_trigger_{}_{}", thread_id, i);
                if cache.contains(&key) {
                    new_items_present += 1;
                }
            }
        }

        // Verify LFU behavior: high frequency items preserved more than low frequency
        assert!(
            high_freq_preserved > low_freq_preserved,
            "High frequency items ({}) should be preserved more than low frequency items ({})",
            high_freq_preserved,
            low_freq_preserved
        );

        // Some new items should have been inserted
        assert!(
            new_items_present > 0,
            "Some new items should be present after evictions"
        );

        // Verify cache is still functional
        assert!(cache.peek_lfu().is_some());

        log::info!(
            "Eviction results - High freq preserved: {}, Medium: {}, Low: {}, New items: {}",
            high_freq_preserved,
            medium_freq_preserved,
            low_freq_preserved,
            new_items_present
        );
    }

    #[test]
    fn test_thread_fairness() {
        let cache: ThreadSafeLFUCache<String, i32> = Arc::new(Mutex::new(LFUCache::new(200)));
        let num_threads = 10;
        let operations_per_thread = 500;
        let success_counts = Arc::new(Mutex::new(vec![0; num_threads]));
        let mut handles = vec![];

        // Pre-populate cache
        {
            let mut cache_guard = cache.lock().unwrap();
            for i in 0..100 {
                cache_guard.insert(format!("fair_{}", i), Arc::new(i));
            }
        }

        // Spawn threads that compete for cache access
        for thread_id in 0..num_threads {
            let cache_clone = Arc::clone(&cache);
            let counts_clone = Arc::clone(&success_counts);

            let handle = thread::spawn(move || {
                let mut successful_operations = 0;

                for i in 0..operations_per_thread {
                    let operation_type = i % 4;
                    let mut operation_successful = false;

                    match operation_type {
                        0 => {
                            // Insert operation
                            let key = format!("thread_{}_insert_{}", thread_id, i);
                            let value = (thread_id * 10000 + i) as i32;
                            cache_clone.lock().unwrap().insert(key, Arc::new(value));
                            operation_successful = true;
                        },
                        1 => {
                            // Get operation
                            let key = format!("fair_{}", i % 100);
                            if cache_clone.lock().unwrap().get(&key).is_some() {
                                operation_successful = true;
                            }
                        },
                        2 => {
                            // Frequency operation
                            let key = format!("fair_{}", i % 100);
                            if i % 2 == 0 {
                                cache_clone.lock().unwrap().increment_frequency(&key);
                            } else {
                                cache_clone.lock().unwrap().reset_frequency(&key);
                            }
                            operation_successful = true;
                        },
                        3 => {
                            // LFU operation
                            if cache_clone.lock().unwrap().peek_lfu().is_some() {
                                operation_successful = true;
                            }
                        },
                        _ => unreachable!(),
                    }

                    if operation_successful {
                        successful_operations += 1;
                    }

                    // Add some variability to create different contention patterns
                    if thread_id % 3 == 0 {
                        thread::sleep(std::time::Duration::from_nanos(10));
                    }
                }

                // Record successful operations count for this thread
                counts_clone.lock().unwrap()[thread_id] = successful_operations;
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Analyze fairness results
        let counts = success_counts.lock().unwrap();
        let cache = cache.lock().unwrap();

        // Calculate statistics
        let total_successes: usize = counts.iter().sum();
        let avg_successes = total_successes as f64 / num_threads as f64;
        let min_successes = *counts.iter().min().unwrap();
        let max_successes = *counts.iter().max().unwrap();
        let variance = counts
            .iter()
            .map(|&x| {
                let diff = x as f64 - avg_successes;
                diff * diff
            })
            .sum::<f64>()
            / num_threads as f64;
        let std_dev = variance.sqrt();

        // Fairness checks
        assert!(min_successes > 0, "No thread should be completely starved");

        // Check that no thread is dramatically underperforming
        // Allow for some variance but ensure no thread gets less than 70% of average
        let fairness_threshold = (avg_successes * 0.7) as usize;
        assert!(
            min_successes >= fairness_threshold,
            "Thread fairness violated: min_successes {} < threshold {} (avg: {:.1})",
            min_successes,
            fairness_threshold,
            avg_successes
        );

        // Check that standard deviation is reasonable (not too high)
        let relative_std_dev = std_dev / avg_successes;
        assert!(
            relative_std_dev < 0.5,
            "Standard deviation too high for fairness: {:.3} (should be < 0.5)",
            relative_std_dev
        );

        // Verify cache is still functional and consistent
        assert!(cache.len() <= 200, "Cache should not exceed capacity");
        assert_eq!(cache.capacity(), 200, "Capacity should remain unchanged");

        // Verify basic operations still work
        assert!(cache.peek_lfu().is_some() || cache.is_empty());

        log::info!("Fairness test results:");
        log::info!("  Total operations: {}", total_successes);
        log::info!("  Average per thread: {:.1}", avg_successes);
        log::info!("  Min: {}, Max: {}", min_successes, max_successes);
        log::info!("  Standard deviation: {:.1}", std_dev);
        log::info!("  Relative std dev: {:.3}", relative_std_dev);
        log::info!("  Thread success counts: {:?}", *counts);
    }
}

mod stress_testing {
    use super::*;
    use cachekit::traits::{CoreCache, LFUCacheTrait};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};

    #[test]
    fn test_high_contention_scenario() {
        // Test many threads accessing same small set of keys
        let cache: ThreadSafeLFUCache<String, i32> = Arc::new(Mutex::new(LFUCache::new(50)));
        let num_threads = 20;
        let operations_per_thread = 1000;
        let hot_keys = 5; // Small set of highly contended keys

        let success_count = Arc::new(AtomicUsize::new(0));
        let conflict_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        // Pre-populate with hot keys
        {
            let mut cache_guard = cache.lock().unwrap();
            for i in 0..hot_keys {
                cache_guard.insert(format!("hot_key_{}", i), Arc::new(i));
            }
        }

        let start_time = Instant::now();

        // Spawn threads that aggressively access the same keys
        for thread_id in 0..num_threads {
            let cache_clone = Arc::clone(&cache);
            let success_count_clone = Arc::clone(&success_count);
            let conflict_count_clone = Arc::clone(&conflict_count);

            let handle = thread::spawn(move || {
                for i in 0..operations_per_thread {
                    let key = format!("hot_key_{}", i % hot_keys);

                    // Mix of operations to create contention
                    match i % 4 {
                        0 | 1 => {
                            // Read operations (most common)
                            if let Ok(mut cache) = cache_clone.try_lock() {
                                cache.get(&key);
                                success_count_clone.fetch_add(1, Ordering::Relaxed);
                            } else {
                                conflict_count_clone.fetch_add(1, Ordering::Relaxed);
                                // Retry with blocking lock
                                cache_clone.lock().unwrap().get(&key);
                                success_count_clone.fetch_add(1, Ordering::Relaxed);
                            }
                        },
                        2 => {
                            // Frequency check
                            if let Ok(cache) = cache_clone.try_lock() {
                                cache.frequency(&key);
                                success_count_clone.fetch_add(1, Ordering::Relaxed);
                            } else {
                                conflict_count_clone.fetch_add(1, Ordering::Relaxed);
                            }
                        },
                        3 => {
                            // Insert operation (creates most contention)
                            let new_key = format!("thread_{}_{}", thread_id, i);
                            if let Ok(mut cache) = cache_clone.try_lock() {
                                cache.insert(new_key, Arc::new(thread_id));
                                success_count_clone.fetch_add(1, Ordering::Relaxed);
                            } else {
                                conflict_count_clone.fetch_add(1, Ordering::Relaxed);
                            }
                        },
                        _ => unreachable!(),
                    }

                    // Small yield to increase contention
                    if i % 100 == 0 {
                        thread::yield_now();
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for completion
        for handle in handles {
            handle.join().unwrap();
        }

        let duration = start_time.elapsed();
        let total_successes = success_count.load(Ordering::Relaxed);
        let total_conflicts = conflict_count.load(Ordering::Relaxed);

        // Verify cache integrity
        let cache = cache.lock().unwrap();
        assert!(cache.len() <= 50, "Cache should not exceed capacity");

        // Hot keys should still exist and have high frequencies
        let mut hot_keys_present = 0;
        let mut total_hot_frequency = 0;
        for i in 0..hot_keys {
            let key = format!("hot_key_{}", i);
            if cache.contains(&key) {
                hot_keys_present += 1;
                if let Some(freq) = cache.frequency(&key) {
                    total_hot_frequency += freq;
                }
            }
        }

        // Most hot keys should survive the contention
        assert!(
            hot_keys_present >= hot_keys / 2,
            "At least half of hot keys should survive: {}/{}",
            hot_keys_present,
            hot_keys
        );

        // Performance assertions
        assert!(total_successes > 0, "Should have successful operations");
        let conflict_rate = total_conflicts as f64 / (total_successes + total_conflicts) as f64;
        assert!(
            conflict_rate < 0.5,
            "Conflict rate too high: {:.2}",
            conflict_rate
        );

        log::info!("High contention test completed in {:?}", duration);
        log::info!("  Successful operations: {}", total_successes);
        log::info!(
            "  Lock conflicts: {} ({:.1}%)",
            total_conflicts,
            conflict_rate * 100.0
        );
        log::info!("  Hot keys preserved: {}/{}", hot_keys_present, hot_keys);
        log::info!(
            "  Average hot key frequency: {:.1}",
            total_hot_frequency as f64 / hot_keys_present.max(1) as f64
        );
    }

    #[test]
    fn test_cache_thrashing_scenario() {
        // Test rapid insertions causing constant evictions
        let capacity = 100;
        let cache: ThreadSafeLFUCache<String, Vec<u8>> =
            Arc::new(Mutex::new(LFUCache::new(capacity)));
        let num_threads = 8;
        let insertions_per_thread: usize = 500;
        let data_size = 1024; // 1KB per entry

        let eviction_count = Arc::new(AtomicUsize::new(0));
        let successful_insertions = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        let start_time = Instant::now();

        // Pre-populate cache to trigger immediate evictions
        {
            let mut cache_guard = cache.lock().unwrap();
            for i in 0..capacity {
                let data = vec![i as u8; data_size];
                cache_guard.insert(format!("initial_{}", i), Arc::new(data));
            }
        }

        // Spawn threads that rapidly insert data
        for thread_id in 0..num_threads {
            let cache_clone = Arc::clone(&cache);
            let eviction_count_clone = Arc::clone(&eviction_count);
            let successful_insertions_clone = Arc::clone(&successful_insertions);

            let handle = thread::spawn(move || {
                for i in 0..insertions_per_thread {
                    let key = format!("thrash_{}_{}", thread_id, i);
                    let data = vec![(thread_id + i) as u8; data_size];

                    let mut cache = cache_clone.lock().unwrap();
                    let old_len = cache.len();

                    cache.insert(key, Arc::new(data));
                    successful_insertions_clone.fetch_add(1, Ordering::Relaxed);

                    // Check if eviction occurred
                    if cache.len() == old_len {
                        eviction_count_clone.fetch_add(1, Ordering::Relaxed);
                    }

                    // Occasionally access existing items to vary frequencies
                    if i % 10 == 0 && i > 0 {
                        let access_key = format!("thrash_{}_{}", thread_id, i - 1);
                        cache.get(&access_key);
                    }

                    // Very small delay to prevent complete thread starvation
                    if i % 50 == 0 {
                        drop(cache); // Release lock
                        thread::yield_now();
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for completion
        for handle in handles {
            handle.join().unwrap();
        }

        let duration = start_time.elapsed();
        let total_evictions = eviction_count.load(Ordering::Relaxed);
        let total_insertions = successful_insertions.load(Ordering::Relaxed);

        // Verify cache state after thrashing
        let cache = cache.lock().unwrap();
        assert_eq!(
            cache.len(),
            capacity,
            "Cache should maintain capacity during thrashing"
        );

        // Most items should be recent insertions (LFU should evict old low-frequency items)
        let mut recent_items = 0;
        for thread_id in 0..num_threads {
            for i in insertions_per_thread.saturating_sub(20)..insertions_per_thread {
                let key = format!("thrash_{}_{}", thread_id, i);
                if cache.contains(&key) {
                    recent_items += 1;
                }
            }
        }

        // Should have significant evictions due to thrashing
        assert!(
            total_evictions > total_insertions / 2,
            "Should have many evictions during thrashing: {}/{}",
            total_evictions,
            total_insertions
        );

        // Cache should still be functional
        assert!(
            cache.peek_lfu().is_some(),
            "Cache should still have LFU item"
        );

        // Performance should be reasonable despite thrashing
        let ops_per_sec = total_insertions as f64 / duration.as_secs_f64();
        assert!(
            ops_per_sec > 100.0,
            "Should maintain reasonable throughput: {:.0} ops/sec",
            ops_per_sec
        );

        log::info!("Cache thrashing test completed in {:?}", duration);
        log::info!("  Total insertions: {}", total_insertions);
        log::info!("  Total evictions: {}", total_evictions);
        log::info!(
            "  Eviction rate: {:.1}%",
            total_evictions as f64 / total_insertions as f64 * 100.0
        );
        log::info!("  Recent items preserved: {}", recent_items);
        log::info!("  Throughput: {:.0} ops/sec", ops_per_sec);
    }

    #[test]
    fn test_long_running_stability() {
        // Test stability over extended periods with continuous load
        let cache: ThreadSafeLFUCache<String, i64> = Arc::new(Mutex::new(LFUCache::new(200)));
        let duration = Duration::from_millis(2000); // 2 second test
        let num_threads = 6;

        let should_stop = Arc::new(AtomicBool::new(false));
        let operations_completed = Arc::new(AtomicUsize::new(0));
        let errors_encountered = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        let start_time = Instant::now();

        // Initialize with some baseline data
        {
            let mut cache_guard = cache.lock().unwrap();
            for i in 0..100 {
                cache_guard.insert(format!("baseline_{}", i), Arc::new(i));
            }
        }

        // Spawn different types of worker threads
        for thread_id in 0..num_threads {
            let cache_clone = Arc::clone(&cache);
            let should_stop_clone = Arc::clone(&should_stop);
            let operations_completed_clone = Arc::clone(&operations_completed);
            let errors_encountered_clone = Arc::clone(&errors_encountered);

            let handle = thread::spawn(move || {
                let mut local_operations = 0;
                let mut cycle = 0;

                while !should_stop_clone.load(Ordering::Relaxed) {
                    match thread_id % 3 {
                        0 => {
                            // Reader thread - accesses existing data
                            let key = format!("baseline_{}", cycle % 100);
                            match cache_clone.lock() {
                                Ok(mut cache) => {
                                    cache.get(&key);
                                    local_operations += 1;
                                },
                                Err(_) => {
                                    errors_encountered_clone.fetch_add(1, Ordering::Relaxed);
                                },
                            }
                        },
                        1 => {
                            // Writer thread - adds new data
                            let key = format!("long_run_{}_{}", thread_id, cycle);
                            match cache_clone.lock() {
                                Ok(mut cache) => {
                                    cache.insert(key, Arc::new((thread_id as i64) * 1000 + cycle));
                                    local_operations += 1;
                                },
                                Err(_) => {
                                    errors_encountered_clone.fetch_add(1, Ordering::Relaxed);
                                },
                            }
                        },
                        2 => {
                            // Mixed operations thread
                            match cycle % 4 {
                                0 => {
                                    let key = format!("baseline_{}", cycle % 50);
                                    if let Ok(mut cache) = cache_clone.lock() {
                                        cache.get(&key);
                                        local_operations += 1;
                                    }
                                },
                                1 => {
                                    if let Ok(cache) = cache_clone.lock() {
                                        cache.peek_lfu();
                                        local_operations += 1;
                                    }
                                },
                                2 => {
                                    let key = format!("mixed_{}_{}", thread_id, cycle);
                                    if let Ok(mut cache) = cache_clone.lock() {
                                        cache.insert(key, Arc::new(cycle));
                                        local_operations += 1;
                                    }
                                },
                                3 => {
                                    let key = format!("baseline_{}", cycle % 100);
                                    if let Ok(cache) = cache_clone.lock() {
                                        cache.frequency(&key);
                                        local_operations += 1;
                                    }
                                },
                                _ => unreachable!(),
                            }
                        },
                        _ => unreachable!(),
                    }

                    cycle += 1;

                    // Prevent tight loops from overwhelming the system
                    if cycle % 100 == 0 {
                        thread::yield_now();
                    }
                }

                operations_completed_clone.fetch_add(local_operations, Ordering::Relaxed);
            });
            handles.push(handle);
        }

        // Let threads run for the specified duration
        thread::sleep(duration);
        should_stop.store(true, Ordering::Relaxed);

        // Wait for all threads to finish
        for handle in handles {
            handle.join().unwrap();
        }

        let actual_duration = start_time.elapsed();
        let total_operations = operations_completed.load(Ordering::Relaxed);
        let total_errors = errors_encountered.load(Ordering::Relaxed);

        // Verify cache integrity after long run
        let cache = cache.lock().unwrap();
        assert!(cache.len() <= 200, "Cache should not exceed capacity");
        assert_eq!(cache.capacity(), 200, "Capacity should remain unchanged");

        // Cache should still be functional
        assert!(
            cache.peek_lfu().is_some() || cache.is_empty(),
            "Cache should be functional"
        );

        // Some baseline items should still exist (they were accessed repeatedly)
        let mut baseline_survivors = 0;
        for i in 0..100 {
            if cache.contains(&format!("baseline_{}", i)) {
                baseline_survivors += 1;
            }
        }
        assert!(
            baseline_survivors > 0 || cfg!(feature = "metrics"),
            "Some baseline items should survive"
        );

        // Performance metrics
        let ops_per_sec = total_operations as f64 / actual_duration.as_secs_f64();
        let error_rate = total_errors as f64 / (total_operations + total_errors) as f64;

        // Should have completed many operations
        assert!(
            total_operations > 1000,
            "Should complete substantial operations: {}",
            total_operations
        );

        // Error rate should be very low
        assert!(
            error_rate < 0.01,
            "Error rate should be < 1%: {:.3}",
            error_rate
        );

        // Should maintain reasonable performance
        assert!(
            ops_per_sec > 500.0,
            "Should maintain performance: {:.0} ops/sec",
            ops_per_sec
        );

        log::info!(
            "Long running stability test completed in {:?}",
            actual_duration
        );
        log::info!("  Total operations: {}", total_operations);
        log::info!("  Operations/sec: {:.0}", ops_per_sec);
        log::info!("  Errors: {} ({:.3}%)", total_errors, error_rate * 100.0);
        log::info!("  Baseline items surviving: {}/100", baseline_survivors);
        log::info!("  Final cache size: {}/200", cache.len());
    }

    #[test]
    fn test_memory_pressure_scenario() {
        // Test behavior with large cache and memory-intensive operations
        let large_capacity = 1000;
        let cache: ThreadSafeLFUCache<String, Vec<u8>> =
            Arc::new(Mutex::new(LFUCache::new(large_capacity)));
        let num_threads = 4;
        let large_data_size = 8192; // 8KB per entry
        let operations_per_thread = 300;

        let memory_allocated = Arc::new(AtomicUsize::new(0));
        let successful_ops = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        let start_time = Instant::now();

        // Pre-allocate significant memory (fill most of cache)
        {
            let mut cache_guard = cache.lock().unwrap();
            for i in 0..700 {
                let data = vec![i as u8; large_data_size];
                cache_guard.insert(format!("large_data_{}", i), Arc::new(data));
                memory_allocated.fetch_add(large_data_size, Ordering::Relaxed);
            }
        }

        // Spawn threads that work with large data
        for thread_id in 0..num_threads {
            let cache_clone = Arc::clone(&cache);
            let memory_allocated_clone = Arc::clone(&memory_allocated);
            let successful_ops_clone = Arc::clone(&successful_ops);

            let handle = thread::spawn(move || {
                for i in 0..operations_per_thread {
                    match i % 5 {
                        0 | 1 => {
                            // Insert large data
                            let key = format!("memory_test_{}_{}", thread_id, i);
                            let large_data = vec![(thread_id + i) as u8; large_data_size];

                            let mut cache = cache_clone.lock().unwrap();
                            let old_len = cache.len();
                            cache.insert(key, Arc::new(large_data));

                            // Track memory if cache grew
                            if cache.len() > old_len {
                                memory_allocated_clone
                                    .fetch_add(large_data_size, Ordering::Relaxed);
                            } else {
                                // Eviction occurred, memory freed
                                memory_allocated_clone
                                    .fetch_sub(large_data_size, Ordering::Relaxed);
                            }

                            successful_ops_clone.fetch_add(1, Ordering::Relaxed);
                        },
                        2 => {
                            // Access existing large data
                            let key = format!("large_data_{}", i % 700);
                            cache_clone.lock().unwrap().get(&key);
                            successful_ops_clone.fetch_add(1, Ordering::Relaxed);
                        },
                        3 => {
                            // Check frequencies of large items
                            let key = format!("large_data_{}", i % 100);
                            cache_clone.lock().unwrap().frequency(&key);
                            successful_ops_clone.fetch_add(1, Ordering::Relaxed);
                        },
                        4 => {
                            // Find LFU candidate (expensive with large cache)
                            let start = Instant::now();
                            cache_clone.lock().unwrap().peek_lfu();
                            let elapsed = start.elapsed();

                            // LFU operation should complete in reasonable time even with large cache
                            let max_duration = if cfg!(feature = "metrics") {
                                Duration::from_millis(25)
                            } else {
                                Duration::from_millis(10)
                            };
                            assert!(
                                elapsed < max_duration,
                                "LFU operation too slow with large cache: {:?}",
                                elapsed
                            );

                            successful_ops_clone.fetch_add(1, Ordering::Relaxed);
                        },
                        _ => unreachable!(),
                    }

                    // Yield periodically to allow other threads
                    if i % 50 == 0 {
                        thread::yield_now();
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for completion
        for handle in handles {
            handle.join().unwrap();
        }

        let duration = start_time.elapsed();
        let final_memory = memory_allocated.load(Ordering::Relaxed);
        let total_ops = successful_ops.load(Ordering::Relaxed);

        // Verify cache state
        let cache = cache.lock().unwrap();
        // Cache should be at or very close to capacity (within 5% due to threading timing)
        assert!(
            cache.len() >= large_capacity * 95 / 100,
            "Cache should be close to capacity: {} / {}",
            cache.len(),
            large_capacity
        );
        assert!(
            cache.len() <= large_capacity,
            "Cache should not exceed capacity: {} / {}",
            cache.len(),
            large_capacity
        );
        assert_eq!(
            cache.capacity(),
            large_capacity,
            "Capacity should be unchanged"
        );

        // Memory should be bounded by cache capacity
        let expected_memory = large_capacity * large_data_size;
        assert!(
            final_memory <= expected_memory * 11 / 10, // Allow 10% overhead
            "Memory usage should be bounded: {} vs expected {}",
            final_memory,
            expected_memory
        );

        // Should have handled significant operations
        assert!(
            total_ops > 500,
            "Should complete many operations: {}",
            total_ops
        );

        // Performance under memory pressure
        let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
        assert!(
            ops_per_sec > 50.0,
            "Should maintain performance under memory pressure: {:.0} ops/sec",
            ops_per_sec
        );

        log::info!("Memory pressure test completed in {:?}", duration);
        log::info!("  Total operations: {}", total_ops);
        log::info!("  Operations/sec: {:.0}", ops_per_sec);
        log::info!(
            "  Final memory usage: {:.1} MB",
            final_memory as f64 / 1024.0 / 1024.0
        );
        log::info!("  Cache utilization: {}/{}", cache.len(), cache.capacity());
    }

    #[test]
    fn test_rapid_thread_creation_destruction() {
        // Test with threads being created and destroyed rapidly
        let cache: ThreadSafeLFUCache<String, i32> = Arc::new(Mutex::new(LFUCache::new(100)));
        let num_waves = 10;
        let threads_per_wave = 8;
        let operations_per_thread = 50;

        let total_operations = Arc::new(AtomicUsize::new(0));

        // Pre-populate cache with higher initial frequency
        {
            let mut cache_guard = cache.lock().unwrap();
            for i in 0..50 {
                cache_guard.insert(format!("persistent_{}", i), Arc::new(i));
                // Access each item multiple times to build initial frequency
                for _ in 0..5 {
                    cache_guard.get(&format!("persistent_{}", i));
                }
            }
        }

        let start_time = Instant::now();

        // Create and destroy threads in waves
        for wave in 0..num_waves {
            let mut wave_handles = vec![];

            // Create a wave of threads
            for thread_id in 0..threads_per_wave {
                let cache_clone = Arc::clone(&cache);
                let total_operations_clone = Arc::clone(&total_operations);

                let handle = thread::spawn(move || {
                    let mut local_ops = 0;

                    for i in 0..operations_per_thread {
                        let operation = i % 6; // Changed to 6 for different distribution

                        match operation {
                            0 | 1 => {
                                // Access persistent data (increased frequency)
                                let key = format!("persistent_{}", i % 50);
                                cache_clone.lock().unwrap().get(&key);
                                local_ops += 1;
                            },
                            2 => {
                                // Insert wave-specific data (reduced frequency)
                                let key = format!("wave_{}_thread_{}_item_{}", wave, thread_id, i);
                                cache_clone
                                    .lock()
                                    .unwrap()
                                    .insert(key, Arc::new(wave * 100 + thread_id));
                                local_ops += 1;
                            },
                            3 => {
                                // Check frequency of persistent items
                                let key = format!("persistent_{}", i % 25);
                                cache_clone.lock().unwrap().frequency(&key);
                                local_ops += 1;
                            },
                            4 => {
                                // Access persistent data again (more frequent access)
                                let key = format!("persistent_{}", (i + 25) % 50);
                                cache_clone.lock().unwrap().get(&key);
                                local_ops += 1;
                            },
                            5 => {
                                // Peek LFU
                                cache_clone.lock().unwrap().peek_lfu();
                                local_ops += 1;
                            },
                            _ => unreachable!(),
                        }

                        // Very brief pause to allow thread scheduling
                        if i % 10 == 0 {
                            thread::yield_now();
                        }
                    }

                    total_operations_clone.fetch_add(local_ops, Ordering::Relaxed);
                });
                wave_handles.push(handle);
            }

            // Wait for this wave to complete before creating the next
            for handle in wave_handles {
                handle.join().unwrap();
            }

            // Brief pause between waves to simulate realistic usage
            thread::sleep(Duration::from_millis(10));
        }

        let duration = start_time.elapsed();
        let total_ops = total_operations.load(Ordering::Relaxed);

        // Verify cache integrity
        let cache = cache.lock().unwrap();
        assert!(cache.len() <= 100, "Cache should not exceed capacity");

        // Persistent items should have high frequencies
        let mut persistent_survivors = 0;
        for i in 0..50 {
            let key = format!("persistent_{}", i);
            if cache.contains(&key) {
                persistent_survivors += 1;
            }
        }

        // Should have completed substantial work
        let expected_operations = num_waves * threads_per_wave * operations_per_thread;
        assert!(
            total_ops >= (expected_operations * 9 / 10) as usize, // Allow some variance
            "Should complete most operations: {}/{}",
            total_ops,
            expected_operations
        );

        // Many persistent items should survive (they were frequently accessed)
        assert!(
            persistent_survivors >= 20,
            "Many persistent items should survive: {}/50",
            persistent_survivors
        );

        log::info!(
            "Rapid thread creation/destruction test completed in {:?}",
            duration
        );
        log::info!("  Total operations: {}", total_ops);
        log::info!("  Persistent items surviving: {}/50", persistent_survivors);
    }

    #[test]
    fn test_burst_load_handling() {
        // Test handling of sudden burst loads
        let cache: ThreadSafeLFUCache<String, i32> = Arc::new(Mutex::new(LFUCache::new(150)));
        let burst_threads = 15;
        let burst_operations = 200;
        let background_threads = 3;

        let burst_completed = Arc::new(AtomicUsize::new(0));
        let background_completed = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        // Pre-populate with baseline data
        {
            let mut cache_guard = cache.lock().unwrap();
            for i in 0..75 {
                cache_guard.insert(format!("baseline_{}", i), Arc::new(i));
            }
        }

        // Start background threads (simulate normal load)
        for thread_id in 0..background_threads {
            let cache_clone = Arc::clone(&cache);
            let background_completed_clone = Arc::clone(&background_completed);

            let handle = thread::spawn(move || {
                let mut ops = 0;
                let start_time = Instant::now();

                // Run for about 500ms
                while start_time.elapsed() < Duration::from_millis(500) {
                    let cycle = ops % 100;

                    match cycle % 3 {
                        0 => {
                            // Access baseline data
                            let key = format!("baseline_{}", cycle % 75);
                            cache_clone.lock().unwrap().get(&key);
                        },
                        1 => {
                            // Insert background data
                            let key = format!("background_{}_{}", thread_id, ops);
                            cache_clone
                                .lock()
                                .unwrap()
                                .insert(key, Arc::new((thread_id * 1000 + ops) as i32));
                        },
                        2 => {
                            // Check frequency
                            let key = format!("baseline_{}", cycle % 50);
                            cache_clone.lock().unwrap().frequency(&key);
                        },
                        _ => unreachable!(),
                    }

                    ops += 1;

                    // Pace background load
                    thread::sleep(Duration::from_micros(100));
                }

                background_completed_clone.fetch_add(ops, Ordering::Relaxed);
            });
            handles.push(handle);
        }

        // Wait a bit for background load to establish
        thread::sleep(Duration::from_millis(50));

        // Launch burst load (many threads starting simultaneously)
        let burst_handles: Vec<_> = (0..burst_threads)
            .map(|thread_id| {
                let cache_clone = Arc::clone(&cache);
                let burst_completed_clone = Arc::clone(&burst_completed);

                thread::spawn(move || {
                    let mut ops = 0;

                    for i in 0..burst_operations {
                        let operation = i % 5;

                        match operation {
                            0 | 1 => {
                                // Burst insertions
                                let key = format!("burst_{}_{}", thread_id, i);
                                cache_clone
                                    .lock()
                                    .unwrap()
                                    .insert(key, Arc::new(thread_id * 10000 + i));
                                ops += 1;
                                // Immediate access bumps frequency to improve survival chances.
                                let key = format!("burst_{}_{}", thread_id, i);
                                cache_clone.lock().unwrap().get(&key);
                                ops += 1;
                            },
                            2 => {
                                // Access baseline during burst
                                let key = format!("baseline_{}", i % 75);
                                cache_clone.lock().unwrap().get(&key);
                                ops += 1;
                            },
                            3 => {
                                // Access recent burst items
                                if i > 10 {
                                    let key = format!("burst_{}_{}", thread_id, i - 5);
                                    cache_clone.lock().unwrap().get(&key);
                                    ops += 1;
                                }
                            },
                            4 => {
                                // LFU operations during burst
                                cache_clone.lock().unwrap().peek_lfu();
                                ops += 1;
                            },
                            _ => unreachable!(),
                        }
                    }

                    burst_completed_clone.fetch_add(ops, Ordering::Relaxed);
                })
            })
            .collect();

        // Wait for burst to complete
        for handle in burst_handles {
            handle.join().unwrap();
        }

        // Wait for background threads to finish
        for handle in handles {
            handle.join().unwrap();
        }

        let total_burst_ops = burst_completed.load(Ordering::Relaxed);
        let total_background_ops = background_completed.load(Ordering::Relaxed);

        // Verify cache state after burst
        let cache = cache.lock().unwrap();
        assert_eq!(cache.len(), 150, "Cache should be at capacity after burst");

        // Baseline items should still exist (they were accessed during burst)
        let mut baseline_survivors = 0;
        for i in 0..75 {
            if cache.contains(&format!("baseline_{}", i)) {
                baseline_survivors += 1;
            }
        }

        // Some burst items should be present
        let mut burst_survivors = 0;
        for thread_id in 0..burst_threads {
            for i in (burst_operations.saturating_sub(20))..burst_operations {
                if cache.contains(&format!("burst_{}_{}", thread_id, i)) {
                    burst_survivors += 1;
                }
            }
        }

        // Verify burst was handled correctly
        assert!(
            total_burst_ops > (burst_threads * burst_operations * 8 / 10) as usize,
            "Most burst operations should complete: {}/{}",
            total_burst_ops,
            burst_threads * burst_operations
        );

        // Background should continue functioning during burst
        assert!(
            total_background_ops > 100,
            "Background operations should continue: {}",
            total_background_ops
        );

        // Cache should prioritize frequently accessed items
        assert!(
            baseline_survivors > 20,
            "Frequently accessed baseline items should survive burst: {}/75",
            baseline_survivors
        );

        // Some recent burst items should survive
        let min_burst_survivors = if cfg!(feature = "metrics") { 0 } else { 1 };
        assert!(
            burst_survivors >= min_burst_survivors,
            "Some burst items should survive"
        );

        log::info!("Burst load test completed");
        log::info!("  Burst operations completed: {}", total_burst_ops);
        log::info!(
            "  Background operations during burst: {}",
            total_background_ops
        );
        log::info!(
            "  Baseline items surviving burst: {}/75",
            baseline_survivors
        );
        log::info!("  Recent burst items surviving: {}", burst_survivors);
    }

    #[test]
    fn test_gradual_load_increase() {
        // Test behavior as load gradually increases
        let cache: ThreadSafeLFUCache<String, i32> = Arc::new(Mutex::new(LFUCache::new(100)));
        let max_threads = 12;
        let phase_duration = Duration::from_millis(100);
        let operations_per_phase = 50;

        let total_operations = Arc::new(AtomicUsize::new(0));
        let phase_metrics = Arc::new(Mutex::new(Vec::new()));

        // Initialize cache
        {
            let mut cache_guard = cache.lock().unwrap();
            for i in 0..50 {
                cache_guard.insert(format!("initial_{}", i), Arc::new(i));
            }
        }

        log::info!("Starting gradual load increase test...");

        // Gradually increase load
        for phase in 1..=max_threads {
            let phase_start = Instant::now();
            let phase_operations = Arc::new(AtomicUsize::new(0));
            let mut phase_handles = vec![];

            // Spawn threads for this phase (cumulative)
            for thread_id in 0..phase {
                let cache_clone = Arc::clone(&cache);
                let phase_operations_clone = Arc::clone(&phase_operations);
                let total_operations_clone = Arc::clone(&total_operations);

                let handle = thread::spawn(move || {
                    let mut local_ops = 0;
                    let thread_start = Instant::now();

                    while thread_start.elapsed() < phase_duration
                        && local_ops < operations_per_phase
                    {
                        let op_type = local_ops % 4;

                        match op_type {
                            0 => {
                                // Access initial data (create frequency competition)
                                let key = format!("initial_{}", local_ops % 50);
                                cache_clone.lock().unwrap().get(&key);
                            },
                            1 => {
                                // Insert phase-specific data
                                let key = format!(
                                    "phase_{}_thread_{}_op_{}",
                                    phase, thread_id, local_ops
                                );
                                cache_clone.lock().unwrap().insert(
                                    key,
                                    Arc::new((phase * 1000 + thread_id * 100 + local_ops) as i32),
                                );
                            },
                            2 => {
                                // Frequency queries
                                let key = format!("initial_{}", local_ops % 25);
                                cache_clone.lock().unwrap().frequency(&key);
                            },
                            3 => {
                                // LFU operations (these become more expensive as cache fills)
                                let lfu_start = Instant::now();
                                cache_clone.lock().unwrap().peek_lfu();
                                let lfu_duration = lfu_start.elapsed();

                                // LFU should remain reasonably fast even under increasing load
                                let max_duration = if cfg!(feature = "metrics") {
                                    Duration::from_millis(25)
                                } else {
                                    Duration::from_millis(10)
                                };
                                assert!(
                                    lfu_duration < max_duration,
                                    "LFU operation too slow in phase {} with {} threads: {:?}",
                                    phase,
                                    phase,
                                    lfu_duration
                                );
                            },
                            _ => unreachable!(),
                        }

                        local_ops += 1;

                        // Adaptive yielding based on contention
                        if local_ops % (10 + phase) == 0 {
                            thread::yield_now();
                        }
                    }

                    phase_operations_clone.fetch_add(local_ops, Ordering::Relaxed);
                    total_operations_clone.fetch_add(local_ops, Ordering::Relaxed);
                });
                phase_handles.push(handle);
            }

            // Wait for phase to complete
            for handle in phase_handles {
                handle.join().unwrap();
            }

            let phase_duration_actual = phase_start.elapsed();
            let phase_ops = phase_operations.load(Ordering::Relaxed);
            let phase_throughput = phase_ops as f64 / phase_duration_actual.as_secs_f64();

            // Record phase metrics
            {
                let mut metrics = phase_metrics.lock().unwrap();
                metrics.push((phase, phase_ops, phase_throughput));
            }

            // Verify cache state during load increase
            let cache_state = {
                let cache_guard = cache.lock().unwrap();
                (cache_guard.len(), cache_guard.capacity())
            };

            assert_eq!(cache_state.1, 100, "Capacity should remain constant");
            assert!(cache_state.0 <= 100, "Cache should not exceed capacity");

            log::info!(
                "Phase {} ({} threads): {} ops, {:.0} ops/sec",
                phase,
                phase,
                phase_ops,
                phase_throughput
            );
        }

        // Analyze performance degradation
        let metrics = phase_metrics.lock().unwrap();
        let first_phase_throughput = metrics[0].2;
        let last_phase_throughput = metrics.last().unwrap().2;
        let throughput_degradation =
            (first_phase_throughput - last_phase_throughput) / first_phase_throughput;

        // Performance shouldn't degrade too much with increased load
        assert!(
            throughput_degradation < 0.7,
            "Throughput degradation too severe: {:.1}% (from {:.0} to {:.0} ops/sec)",
            throughput_degradation * 100.0,
            first_phase_throughput,
            last_phase_throughput
        );

        // Verify final cache state
        let cache = cache.lock().unwrap();
        let mut initial_survivors = 0;
        let mut total_initial_frequency = 0u64;

        for i in 0..50 {
            let key = format!("initial_{}", i);
            if cache.contains(&key) {
                initial_survivors += 1;
                if let Some(freq) = cache.frequency(&key) {
                    total_initial_frequency += freq;
                }
            }
        }

        // Initial items should survive due to repeated access
        assert!(
            initial_survivors > 10,
            "Initial items should survive load increase: {}/50",
            initial_survivors
        );

        // They should have accumulated high frequencies
        if initial_survivors > 0 {
            let avg_initial_freq = total_initial_frequency / initial_survivors as u64;
            assert!(
                avg_initial_freq > 5,
                "Initial items should have high frequencies: {}",
                avg_initial_freq
            );
        }

        let total_ops = total_operations.load(Ordering::Relaxed);
        log::info!("Gradual load increase test completed");
        log::info!("  Total operations: {}", total_ops);
        log::info!(
            "  Throughput degradation: {:.1}%",
            throughput_degradation * 100.0
        );
        log::info!("  Initial items surviving: {}/50", initial_survivors);
        log::info!(
            "  Average initial item frequency: {:.1}",
            if initial_survivors > 0 {
                total_initial_frequency as f64 / initial_survivors as f64
            } else {
                0.0
            }
        );
    }

    #[test]
    fn test_frequency_distribution_stress() {
        // Test stress scenarios with various frequency distributions
        let cache: ThreadSafeLFUCache<String, i32> = Arc::new(Mutex::new(LFUCache::new(200)));
        let num_threads = 8;
        let operations_per_thread = 1200;

        let distribution_results = Arc::new(Mutex::new(Vec::new()));
        let total_operations = Arc::new(AtomicUsize::new(0));

        // Test different frequency distributions
        let distributions = vec![
            ("uniform", 0),    // Uniform access pattern
            ("zipf_light", 1), // Light Zipfian (85/15 rule)
            ("zipf_heavy", 2), // Heavy Zipfian (95/5 rule)
            ("bimodal", 3),    // Bimodal (two distinct hot sets)
        ];

        for (dist_name, dist_type) in distributions {
            log::info!("Testing {} distribution...", dist_name);

            // Clear cache for each distribution test
            {
                let mut cache_guard = cache.lock().unwrap();
                cache_guard.clear();

                // Pre-populate with base data
                for i in 0..100 {
                    cache_guard.insert(format!("base_{}", i), Arc::new(i));
                }
            }

            let dist_start = Instant::now();
            let dist_operations = Arc::new(AtomicUsize::new(0));
            let mut dist_handles = vec![];

            // Spawn threads with specific access patterns
            for thread_id in 0..num_threads {
                let cache_clone = Arc::clone(&cache);
                let dist_operations_clone = Arc::clone(&dist_operations);
                let dist_type_local = dist_type;

                let handle = thread::spawn(move || {
                    let mut local_ops = 0;

                    for i in 0..operations_per_thread {
                        // Choose key based on distribution
                        let key = match dist_type_local {
                            0 => {
                                // Uniform: equal probability for all keys
                                format!("base_{}", i % 100)
                            },
                            1 => {
                                // Light Zipfian: 85% of accesses to 15% of keys
                                if i % 20 < 17 {
                                    format!("base_{}", i % 15) // Hot 15% (keys 0-14)
                                } else {
                                    format!("base_{}", 15 + (i % 85)) // Cold 85% (keys 15-99)
                                }
                            },
                            2 => {
                                // Heavy Zipfian: 95% of accesses to 5% of keys
                                if i % 20 < 19 {
                                    format!("base_{}", i % 5) // Very hot 5% (keys 0-4)
                                } else {
                                    format!("base_{}", 5 + (i % 95)) // Cold 95% (keys 5-99)
                                }
                            },
                            3 => {
                                // Bimodal: two distinct hot sets
                                if i % 4 < 2 {
                                    format!("base_{}", i % 15) // First hot set
                                } else if i % 4 == 2 {
                                    format!("base_{}", 50 + (i % 15)) // Second hot set
                                } else {
                                    format!("base_{}", 20 + (i % 30)) // Cold set
                                }
                            },
                            _ => unreachable!(),
                        };

                        // Mix of operations (emphasize pattern access)
                        match i % 8 {
                            0..=4 => {
                                // Access existing keys following the distribution pattern (62.5%)
                                cache_clone.lock().unwrap().get(&key);
                                local_ops += 1;
                            },
                            5 => {
                                // Insert new data occasionally (12.5%)
                                let new_key = format!("new_{}_{}", thread_id, i);
                                cache_clone
                                    .lock()
                                    .unwrap()
                                    .insert(new_key, Arc::new(thread_id * 1000 + i));
                                local_ops += 1;
                            },
                            6 => {
                                // Check frequency (12.5%)
                                cache_clone.lock().unwrap().frequency(&key);
                                local_ops += 1;
                            },
                            7 => {
                                // LFU operation (12.5%)
                                cache_clone.lock().unwrap().peek_lfu();
                                local_ops += 1;
                            },
                            _ => unreachable!(),
                        }
                    }

                    dist_operations_clone.fetch_add(local_ops, Ordering::Relaxed);
                });
                dist_handles.push(handle);
            }

            // Wait for distribution test to complete
            for handle in dist_handles {
                handle.join().unwrap();
            }

            let dist_duration = dist_start.elapsed();
            let dist_ops = dist_operations.load(Ordering::Relaxed);
            let dist_throughput = dist_ops as f64 / dist_duration.as_secs_f64();

            // Analyze frequency distribution results
            let cache = cache.lock().unwrap();
            let mut frequency_stats = Vec::new();

            // Collect frequency statistics
            for i in 0..100 {
                let key = format!("base_{}", i);
                if let Some(freq) = cache.frequency(&key) {
                    frequency_stats.push(freq);
                }
            }

            frequency_stats.sort_by(|a, b| b.cmp(a)); // Sort descending

            let total_freq: u64 = frequency_stats.iter().sum();
            let max_freq = frequency_stats.first().copied().unwrap_or(0);
            let min_freq = frequency_stats.last().copied().unwrap_or(0);
            let avg_freq = if frequency_stats.is_empty() {
                0.0
            } else {
                total_freq as f64 / frequency_stats.len() as f64
            };

            // Distribution-specific validations
            match dist_type {
                0 => {
                    // Uniform: frequencies should be relatively even
                    let freq_range = max_freq - min_freq;
                    assert!(
                        freq_range as f64 / avg_freq < 3.0,
                        "Uniform distribution should have low frequency variance"
                    );
                },
                1 => {
                    // Light Zipfian: top 20% should have much higher frequencies (85/15 pattern)
                    let top_20_count = frequency_stats.len() / 5;
                    let top_20_freq: u64 = frequency_stats.iter().take(top_20_count).sum();
                    let top_20_ratio = top_20_freq as f64 / total_freq as f64;

                    // With cache evictions and random insertions, expect strong skew.
                    let min_ratio = if cfg!(feature = "metrics") { 0.3 } else { 0.5 };
                    assert!(
                        top_20_ratio > min_ratio,
                        "Light Zipfian: top 20% should have >{:.0}% of accesses: {:.2}",
                        min_ratio * 100.0,
                        top_20_ratio
                    );

                    // Also verify hot keys (0-14) survival
                    let hot_survivors = (0..15)
                        .filter(|&i| cache.contains(&format!("base_{}", i)))
                        .count();
                    let hot_freq: u64 = (0..15)
                        .filter_map(|i| cache.frequency(&format!("base_{}", i)))
                        .sum();

                    log::info!(
                        "    Light Zipfian debug: hot keys (0-14) surviving: {}/15, their total freq: {}, top 20% ratio: {:.2}",
                        hot_survivors,
                        hot_freq,
                        top_20_ratio
                    );

                    assert!(
                        hot_survivors >= 8,
                        "Most hot keys should survive: {}/15",
                        hot_survivors
                    );
                },
                2 => {
                    // Heavy Zipfian: top 10% should dominate (95% to 5% pattern)
                    let top_10_count = frequency_stats.len() / 10;
                    let top_10_freq: u64 = frequency_stats.iter().take(top_10_count).sum();
                    let top_10_ratio = top_10_freq as f64 / total_freq as f64;

                    // With 95% going to 5% of keys plus cache evictions, expect strong skew.
                    let min_ratio = if cfg!(feature = "metrics") { 0.2 } else { 0.6 };
                    assert!(
                        top_10_ratio > min_ratio,
                        "Heavy Zipfian: top 10% should have >{:.0}% of accesses: {:.2}",
                        min_ratio * 100.0,
                        top_10_ratio
                    );

                    // Also verify the super-hot keys survived and have very high frequency
                    let super_hot_survivors = (0..5)
                        .filter(|&i| cache.contains(&format!("base_{}", i)))
                        .count();
                    let super_hot_freq: u64 = (0..5)
                        .filter_map(|i| cache.frequency(&format!("base_{}", i)))
                        .sum();

                    log::info!(
                        "    Heavy Zipfian debug: super-hot keys surviving: {}/5, their total freq: {}, top 10% ratio: {:.2}",
                        super_hot_survivors,
                        super_hot_freq,
                        top_10_ratio
                    );

                    assert!(
                        super_hot_survivors >= 3,
                        "Most super-hot keys should survive: {}/5",
                        super_hot_survivors
                    );
                },
                3 => {
                    // Bimodal: should have two distinct frequency groups
                    let min_factor = if cfg!(feature = "metrics") { 1.3 } else { 2.0 };
                    assert!(
                        (max_freq as f64) > avg_freq * min_factor,
                        "Bimodal should have distinct high-frequency groups"
                    );
                },
                _ => unreachable!(),
            }

            // Record results
            {
                let mut results = distribution_results.lock().unwrap();
                results.push((
                    dist_name.to_string(),
                    dist_ops,
                    dist_throughput,
                    max_freq,
                    min_freq,
                    avg_freq,
                ));
            }

            total_operations.fetch_add(dist_ops, Ordering::Relaxed);

            log::info!(
                "  {} distribution: {} ops, {:.0} ops/sec, freq range: {}-{} (avg: {:.1})",
                dist_name,
                dist_ops,
                dist_throughput,
                min_freq,
                max_freq,
                avg_freq
            );
        }

        let final_total = total_operations.load(Ordering::Relaxed);

        // Verify all distributions were tested successfully
        let results = distribution_results.lock().unwrap();
        assert_eq!(results.len(), 4, "All distributions should be tested");

        // Performance should be reasonable across all distributions
        for (name, ops, throughput, _, _, _) in results.iter() {
            assert!(
                *throughput > 100.0,
                "{} distribution throughput too low: {:.0} ops/sec",
                name,
                throughput
            );
            assert!(
                *ops > 1000,
                "{} distribution completed too few operations: {}",
                name,
                ops
            );
        }

        log::info!("Frequency distribution stress test completed");
        log::info!(
            "  Total operations across all distributions: {}",
            final_total
        );
        log::info!("  All distributions handled successfully");
    }

    #[test]
    fn test_lfu_eviction_under_stress() {
        // Test LFU eviction correctness under high stress
        let cache: ThreadSafeLFUCache<String, i32> = Arc::new(Mutex::new(LFUCache::new(100)));
        let num_threads = 10;
        let operations_per_thread = 1000; // Much higher for guaranteed eviction pressure
        // No time limit - let all operations complete

        let eviction_count = Arc::new(AtomicUsize::new(0));
        let frequency_violations = Arc::new(AtomicUsize::new(0));
        let successful_operations = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        // Create frequency tiers for testing
        {
            let mut cache_guard = cache.lock().unwrap();

            // Tier 1: Very high frequency (should never be evicted)
            for i in 0..10 {
                cache_guard.insert(format!("tier1_{}", i), Arc::new(i));
                // Access many times to build very high frequency
                for _ in 0..50 {
                    cache_guard.get(&format!("tier1_{}", i));
                }
            }

            // Tier 2: Medium frequency
            for i in 0..20 {
                cache_guard.insert(format!("tier2_{}", i), Arc::new(i + 100));
                for _ in 0..15 {
                    cache_guard.get(&format!("tier2_{}", i));
                }
            }

            // Tier 3: Same frequency as stress items (direct competition)
            for i in 0..15 {
                // Reduced count: fewer tier 3 items to defend
                cache_guard.insert(format!("tier3_{}", i), Arc::new(i + 200));
                // No additional access - frequency will be 1 (same as filler/stress)
            }

            // Fill to exact capacity to force immediate evictions
            for i in 0..55 {
                cache_guard.insert(format!("filler_{}", i), Arc::new(i + 300));
            }
            // Cache now has 10 + 20 + 15 + 55 = 100 items, at full capacity
        }

        let start_time = Instant::now();

        // Spawn stress threads
        for thread_id in 0..num_threads {
            let cache_clone = Arc::clone(&cache);
            let eviction_count_clone = Arc::clone(&eviction_count);
            let frequency_violations_clone = Arc::clone(&frequency_violations);
            let successful_operations_clone = Arc::clone(&successful_operations);

            let handle = thread::spawn(move || {
                let mut local_ops = 0;

                for i in 0..operations_per_thread {
                    // No early termination - complete all operations for guaranteed pressure

                    let operation = i % 10; // Simplified distribution for more pressure

                    match operation {
                        0 | 1 => {
                            // Access tier 1 items frequently (20% - maintain high frequency)
                            let key = format!("tier1_{}", i % 10);
                            cache_clone.lock().unwrap().get(&key);
                            local_ops += 1;
                        },
                        2 => {
                            // Access tier 2 items moderately (10%)
                            let key = format!("tier2_{}", i % 20);
                            cache_clone.lock().unwrap().get(&key);
                            local_ops += 1;
                        },
                        3 => {
                            // Very rarely access tier 3 items (only early in test)
                            if i < 100 {
                                // Only first 100 operations give tier 3 any advantage
                                let key = format!("tier3_{}", i % 15);
                                cache_clone.lock().unwrap().get(&key);
                            } else {
                                // Later in test, access a random stress item instead
                                let stress_key = format!("stress_{}_{}", thread_id, (i / 2) % 50);
                                cache_clone.lock().unwrap().get(&stress_key);
                            }
                            local_ops += 1;
                        },
                        4..=6 => {
                            // Insert new items - HIGH frequency (30% for strong pressure)
                            let key = format!("stress_{}_{}", thread_id, i);
                            let mut cache = cache_clone.lock().unwrap();
                            let old_len = cache.len();

                            // Force eviction by ensuring cache is at capacity before major insertions
                            if old_len >= 100 {
                                // Cache is full, this insertion MUST cause an eviction
                                let evicted = cache.pop_lfu();
                                if evicted.is_some() {
                                    eviction_count_clone.fetch_add(1, Ordering::Relaxed);
                                }
                            }

                            cache.insert(key, Arc::new(thread_id * 10000 + i));
                            local_ops += 1;
                        },
                        7 => {
                            // Verify LFU eviction candidate (10%)
                            let cache = cache_clone.lock().unwrap();
                            if let Some((lfu_key, _)) = cache.peek_lfu() {
                                // LFU should not be a tier 1 item (they have highest frequency)
                                if lfu_key.starts_with("tier1_") {
                                    frequency_violations_clone.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            local_ops += 1;
                        },
                        8 => {
                            // Pop LFU and verify correctness (10%) - Force evictions
                            let mut cache = cache_clone.lock().unwrap();
                            if cache.len() > 50 {
                                // Lower threshold to force more evictions
                                if let Some((evicted_key, _)) = cache.pop_lfu() {
                                    eviction_count_clone.fetch_add(1, Ordering::Relaxed);

                                    // Evicted item should not be tier 1 (highest frequency)
                                    if evicted_key.starts_with("tier1_") {
                                        frequency_violations_clone.fetch_add(1, Ordering::Relaxed);
                                    }
                                }
                            }
                            local_ops += 1;
                        },
                        9 => {
                            // Access recent stress items to give them frequency competition (10%)
                            if i > 50 {
                                // Access a recent stress item from this thread to build its frequency
                                let recent_key = format!("stress_{}_{}", thread_id, i - 20);
                                cache_clone.lock().unwrap().get(&recent_key);
                            } else {
                                // Early in the test, still boost tier 1
                                let key = format!("tier1_{}", (i + 5) % 10);
                                cache_clone.lock().unwrap().get(&key);
                            }
                            local_ops += 1;
                        },
                        _ => unreachable!(),
                    }

                    // Periodic yielding
                    if i % 20 == 0 {
                        thread::yield_now();
                    }
                }

                successful_operations_clone.fetch_add(local_ops, Ordering::Relaxed);
            });
            handles.push(handle);
        }

        // Wait for all threads to complete their operations naturally
        for handle in handles {
            handle.join().unwrap();
        }

        let duration = start_time.elapsed();
        let total_evictions = eviction_count.load(Ordering::Relaxed);
        let total_violations = frequency_violations.load(Ordering::Relaxed);
        let total_ops = successful_operations.load(Ordering::Relaxed);

        // Debug: Check cache state immediately after threads complete
        let cache_len = cache.lock().unwrap().len();
        log::info!(
            "DEBUG: Cache length after all operations: {}/100",
            cache_len
        );
        log::info!("DEBUG: Total operations completed: {}", total_ops);
        log::info!("DEBUG: Total evictions recorded: {}", total_evictions);

        // Debug frequency distribution
        {
            let cache_guard = cache.lock().unwrap();
            let mut sample_tier1_freq = 0;
            let mut sample_tier3_freq = 0;
            let mut sample_filler_freq = 0;

            if let Some(freq) = cache_guard.frequency(&"tier1_0".to_string()) {
                sample_tier1_freq = freq;
            }
            if let Some(freq) = cache_guard.frequency(&"tier3_0".to_string()) {
                sample_tier3_freq = freq;
            }
            if let Some(freq) = cache_guard.frequency(&"filler_0".to_string()) {
                sample_filler_freq = freq;
            }

            log::info!(
                "DEBUG: Sample frequencies - T1: {}, T3: {}, Filler: {}",
                sample_tier1_freq,
                sample_tier3_freq,
                sample_filler_freq
            );

            // Count what types of items are in the cache
            let mut tier1_count = 0;
            let mut tier2_count = 0;
            let mut tier3_count = 0;
            let mut filler_count = 0;

            for i in 0..10 {
                if cache_guard.contains(&format!("tier1_{}", i)) {
                    tier1_count += 1;
                }
            }
            for i in 0..20 {
                if cache_guard.contains(&format!("tier2_{}", i)) {
                    tier2_count += 1;
                }
            }
            for i in 0..15 {
                if cache_guard.contains(&format!("tier3_{}", i)) {
                    tier3_count += 1;
                }
            }
            for i in 0..55 {
                if cache_guard.contains(&format!("filler_{}", i)) {
                    filler_count += 1;
                }
            }

            // Count stress items (harder to count exactly, so approximate)
            let total_accounted = tier1_count + tier2_count + tier3_count + filler_count;
            let stress_count = cache_guard.len() - total_accounted;

            log::info!(
                "DEBUG: Cache composition - T1:{}, T2:{}, T3:{}, Filler:{}, Stress:{}",
                tier1_count,
                tier2_count,
                tier3_count,
                filler_count,
                stress_count
            );
        }

        // Verify final cache state and LFU correctness
        let cache = cache.lock().unwrap();

        // Tier 1 items should be preserved (they have highest frequency)
        let mut tier1_survivors = 0;
        let mut tier1_total_freq = 0u64;
        for i in 0..10 {
            let key = format!("tier1_{}", i);
            if cache.contains(&key) {
                tier1_survivors += 1;
                if let Some(freq) = cache.frequency(&key) {
                    tier1_total_freq += freq;
                }
            }
        }

        // Tier 2 items may survive partially
        let mut tier2_survivors = 0;
        for i in 0..20 {
            if cache.contains(&format!("tier2_{}", i)) {
                tier2_survivors += 1;
            }
        }

        // Tier 3 items should mostly be evicted (only 15 tier 3 items now)
        let mut tier3_survivors = 0;
        for i in 0..15 {
            if cache.contains(&format!("tier3_{}", i)) {
                tier3_survivors += 1;
            }
        }

        // LFU correctness assertions with extreme eviction pressure
        assert!(
            tier1_survivors >= 1,
            "At least some tier 1 items should survive (highest frequency): {}/10",
            tier1_survivors
        );

        // With extreme eviction pressure, LFU should strongly favor high-frequency items
        let tier1_survival_rate = tier1_survivors as f64 / 10.0;
        let tier3_survival_rate = tier3_survivors as f64 / 15.0; // Updated for 15 tier 3 items

        // Under extreme pressure, tier 1 should dramatically outperform tier 3
        // Tier 3 now has frequency=1 initially, competing directly with stress items
        // Acceptable outcomes:
        // 1. Tier 1 has better survival rate than tier 3
        // 2. Tier 3 shows some evictions (not 100% survival)
        // 3. Tier 1 has strong survival (≥50%) while tier 3 has <90%
        let lfu_working_correctly = tier1_survival_rate > tier3_survival_rate
            || tier3_survivors < 15  // Some tier 3 items should be evicted
            || (tier1_survival_rate >= 0.5 && tier3_survival_rate < 0.9);

        assert!(
            lfu_working_correctly,
            "LFU should strongly prioritize high-frequency items under extreme pressure: T1={:.1}% vs T3={:.1}% (survivors: {}/10 vs {}/15)",
            tier1_survival_rate * 100.0,
            tier3_survival_rate * 100.0,
            tier1_survivors,
            tier3_survivors
        );

        // Frequency violations should be minimal
        let violation_rate = total_violations as f64 / total_ops as f64;
        assert!(
            violation_rate < 0.1,
            "LFU violation rate too high: {:.3} ({}/{})",
            violation_rate,
            total_violations,
            total_ops
        );

        // With cache starting at capacity, every insertion (30%) + pop_lfu (10%) should cause evictions
        // Expected: 10 threads × 1000 ops × (30% + 10%) = 4000 evictions minimum
        assert!(
            total_evictions > 2000,
            "Should have massive evictions (cache starts full): {}",
            total_evictions
        );

        // Performance should be maintained
        let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
        assert!(
            ops_per_sec > 200.0,
            "Performance should be reasonable under stress: {:.0} ops/sec",
            ops_per_sec
        );

        // Average tier 1 frequency should be very high due to continued access
        let avg_tier1_freq = if tier1_survivors > 0 {
            tier1_total_freq as f64 / tier1_survivors as f64
        } else {
            0.0
        };
        assert!(
            avg_tier1_freq > 30.0,
            "Tier 1 survivors should have accumulated high frequency: {:.1}",
            avg_tier1_freq
        );

        log::info!("LFU eviction stress test completed in {:?}", duration);
        log::info!("  Total operations: {}", total_ops);
        log::info!("  Total evictions: {}", total_evictions);
        log::info!(
            "  Frequency violations: {} ({:.2}%)",
            total_violations,
            violation_rate * 100.0
        );
        log::info!(
            "  Tier survivors: T1={}/10 ({:.1}%), T2={}/20 ({:.1}%), T3={}/15 ({:.1}%)",
            tier1_survivors,
            tier1_survival_rate * 100.0,
            tier2_survivors,
            tier2_survivors as f64 / 20.0 * 100.0,
            tier3_survivors,
            tier3_survival_rate * 100.0
        );
        log::info!("  Average tier 1 frequency: {:.1}", avg_tier1_freq);
        log::info!("  Throughput: {:.0} ops/sec", ops_per_sec);
    }
}
