// ==============================================
// FIFO CONCURRENCY TESTS (integration)
// ==============================================
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

mod thread_safe_wrapper {
    use cachekit::prelude::FifoCache;
    use cachekit::traits::{CoreCache, FifoCacheTrait};

    use super::*;

    // Helper type for thread-safe testing
    type ThreadSafeFifoCache<K, V> = Arc<Mutex<FifoCache<K, V>>>;

    #[test]
    fn test_basic_thread_safe_operations() {
        let cache: ThreadSafeFifoCache<String, String> = Arc::new(Mutex::new(FifoCache::new(100)));
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
                                // Get operation
                                let key = format!("thread_{}_0", thread_id);

                                if let Ok(mut cache_guard) = cache.lock() {
                                    let _ = cache_guard.get(&key);
                                    thread_successes += 1;
                                }
                            },
                            2 => {
                                // Contains operation
                                let key = format!("thread_{}_{}", thread_id, i / 2);

                                if let Ok(cache_guard) = cache.lock() {
                                    let _ = cache_guard.contains(&key);
                                    thread_successes += 1;
                                }
                            },
                            _ => {
                                // FIFO-specific operations
                                if let Ok(mut cache_guard) = cache.lock() {
                                    let _ = cache_guard.peek_oldest();
                                    if i % 20 == 0 {
                                        let _ = cache_guard.pop_oldest();
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
            "Cache length should not exceed capacity"
        );
        assert!(
            total_successes > expected_operations / 2,
            "Most operations should succeed"
        );

        println!(
            "Final cache state: {} items, capacity {}",
            cache_len, capacity
        );
    }

    #[test]
    fn test_read_heavy_workload() {
        let cache: ThreadSafeFifoCache<String, String> = Arc::new(Mutex::new(FifoCache::new(200)));
        let num_reader_threads = 12;
        let num_writer_threads = 2;
        let reads_per_thread = 500;
        let writes_per_thread = 100;

        let read_successes = Arc::new(AtomicUsize::new(0));
        let write_successes = Arc::new(AtomicUsize::new(0));

        // Pre-populate cache
        {
            let mut cache_guard = cache.lock().unwrap();
            for i in 0..150 {
                cache_guard.insert(format!("initial_{}", i), format!("value_{}", i));
            }
        }

        // Spawn reader threads
        let reader_handles: Vec<_> = (0..num_reader_threads)
            .map(|_thread_id| {
                let cache = cache.clone();

                thread::spawn({
                    let value = read_successes.clone();
                    move || {
                        let mut successful_reads = 0;

                        for i in 0..reads_per_thread {
                            let key = format!("initial_{}", i % 150);

                            if let Ok(mut cache_guard) = cache.lock() {
                                if cache_guard.get(&key).is_some() {
                                    successful_reads += 1;
                                }

                                // Occasionally use FIFO read operations
                                if i % 10 == 0 {
                                    let _ = cache_guard.peek_oldest();
                                    let _ = cache_guard.age_rank(&key);
                                }
                            }
                        }

                        value.fetch_add(successful_reads, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        // Spawn writer threads (fewer writers, occasional writes)
        let writer_handles: Vec<_> = (0..num_writer_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let write_successes = write_successes.clone();

                thread::spawn(move || {
                    let mut successful_writes = 0;

                    for i in 0..writes_per_thread {
                        let key = format!("writer_{}_{}", thread_id, i);
                        let value = format!("writer_value_{}_{}", thread_id, i);

                        if let Ok(mut cache_guard) = cache.lock() {
                            cache_guard.insert(key, value);
                            successful_writes += 1;

                            // Occasionally trigger eviction
                            if i % 50 == 0 {
                                let _ = cache_guard.pop_oldest();
                            }
                        }

                        // Writers sleep slightly to allow more reader access
                        thread::sleep(Duration::from_millis(1));
                    }

                    write_successes.fetch_add(successful_writes, Ordering::SeqCst);
                })
            })
            .collect();

        // Wait for all threads
        for handle in reader_handles {
            handle.join().unwrap();
        }
        for handle in writer_handles {
            handle.join().unwrap();
        }

        let total_reads = read_successes.load(Ordering::SeqCst);
        let total_writes = write_successes.load(Ordering::SeqCst);
        let expected_reads = num_reader_threads * reads_per_thread;
        let expected_writes = num_writer_threads * writes_per_thread;

        println!(
            "Read-heavy workload: {} successful reads, {} successful writes",
            total_reads, total_writes
        );

        // Most reads should succeed since we pre-populated
        assert!(
            total_reads > expected_reads / 3,
            "Should have many successful reads"
        );
        assert_eq!(total_writes, expected_writes, "All writes should succeed");

        // Verify final state
        let final_cache = cache.lock().unwrap();
        assert!(final_cache.len() <= final_cache.capacity());
    }

    #[test]
    fn test_write_heavy_workload() {
        let cache: ThreadSafeFifoCache<String, String> = Arc::new(Mutex::new(FifoCache::new(150)));
        let num_threads = 10;
        let writes_per_thread = 200;
        let total_writes = Arc::new(AtomicUsize::new(0));
        let evictions_triggered = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let total_writes = total_writes.clone();
                let evictions_triggered = evictions_triggered.clone();

                thread::spawn(move || {
                    let mut writes_count = 0;
                    let mut evictions_count = 0;

                    for i in 0..writes_per_thread {
                        if let Ok(mut cache_guard) = cache.lock() {
                            let len_before = cache_guard.len();

                            // Heavy insertion workload
                            let key = format!("heavy_{}_{}", thread_id, i);
                            let value = format!("data_{}_{}", thread_id, i);
                            cache_guard.insert(key, value);
                            writes_count += 1;

                            // Check if eviction occurred
                            let len_after = cache_guard.len();
                            let capacity = cache_guard.capacity();

                            if len_before == capacity && len_after == capacity {
                                evictions_count += 1;
                            }

                            // Occasionally use FIFO-specific operations
                            if i % 25 == 0 {
                                let _ = cache_guard.pop_oldest();
                                evictions_count += 1;
                            }

                            if i % 30 == 0 {
                                let _ = cache_guard.pop_oldest_batch(3);
                                evictions_count += 3;
                            }
                        }
                    }

                    total_writes.fetch_add(writes_count, Ordering::SeqCst);
                    evictions_triggered.fetch_add(evictions_count, Ordering::SeqCst);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let writes = total_writes.load(Ordering::SeqCst);
        let evictions = evictions_triggered.load(Ordering::SeqCst);
        let expected_writes = num_threads * writes_per_thread;

        println!(
            "Write-heavy workload: {} writes, {} evictions",
            writes, evictions
        );

        assert_eq!(writes, expected_writes, "All writes should succeed");
        assert!(
            evictions > 0,
            "Should have triggered evictions under heavy write load"
        );

        // Verify final state
        let final_cache = cache.lock().unwrap();
        assert_eq!(
            final_cache.len(),
            final_cache.capacity(),
            "Cache should be at capacity after heavy writes"
        );
    }

    #[test]
    fn test_mixed_operations_concurrency() {
        let cache: ThreadSafeFifoCache<String, String> = Arc::new(Mutex::new(FifoCache::new(100)));
        let num_threads = 16;
        let operations_per_thread = 150;
        let operation_counts = Arc::new(AtomicUsize::new(0));

        // Pre-populate with some data
        {
            let mut cache_guard = cache.lock().unwrap();
            for i in 0..50 {
                cache_guard.insert(format!("base_{}", i), format!("base_value_{}", i));
            }
        }

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let operation_counts = operation_counts.clone();

                thread::spawn(move || {
                    let mut ops_completed = 0;

                    for i in 0..operations_per_thread {
                        if let Ok(mut cache_guard) = cache.lock() {
                            // Randomized operation mix
                            let operation = (thread_id + i) % 7;

                            match operation {
                                0 | 1 => {
                                    // Insert (high frequency)
                                    let key = format!("mixed_{}_{}", thread_id, i);
                                    let value = format!("mixed_value_{}_{}", thread_id, i);
                                    cache_guard.insert(key, value);
                                },
                                2 => {
                                    // Get
                                    let key = format!("base_{}", i % 50);
                                    let _ = cache_guard.get(&key);
                                },
                                3 => {
                                    // Contains
                                    let key = format!("mixed_{}_{}", thread_id, i / 2);
                                    let _ = cache_guard.contains(&key);
                                },
                                4 => {
                                    // FIFO peek
                                    let _ = cache_guard.peek_oldest();
                                },
                                5 => {
                                    // FIFO pop
                                    let _ = cache_guard.pop_oldest();
                                },
                                _ => {
                                    // FIFO batch operations
                                    let _ = cache_guard.pop_oldest_batch(2);
                                    let key = format!("base_{}", i % 30);
                                    let _ = cache_guard.age_rank(&key);
                                },
                            }

                            ops_completed += 1;
                        }
                    }

                    operation_counts.fetch_add(ops_completed, Ordering::SeqCst);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let total_operations = operation_counts.load(Ordering::SeqCst);
        let expected_operations = num_threads * operations_per_thread;

        println!(
            "Mixed operations concurrency: {}/{} operations completed",
            total_operations, expected_operations
        );

        assert_eq!(
            total_operations, expected_operations,
            "All operations should complete"
        );

        // Verify cache consistency
        let final_cache = cache.lock().unwrap();
        let cache_len = final_cache.len();
        let capacity = final_cache.capacity();
        let insertion_order_len = final_cache.insertion_order_len();

        assert!(cache_len <= capacity, "Cache should not exceed capacity");
        assert!(
            insertion_order_len >= cache_len,
            "Insertion order should track at least current entries"
        );

        println!(
            "Final state: {} items, {} in insertion order, capacity {}",
            cache_len, insertion_order_len, capacity
        );
    }

    #[test]
    fn test_deadlock_prevention() {
        let cache: ThreadSafeFifoCache<String, String> = Arc::new(Mutex::new(FifoCache::new(50)));
        let num_threads = 20;
        let timeout_duration = Duration::from_secs(10);
        let start_time = Instant::now();
        let completed_operations = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let completed_operations = completed_operations.clone();

                thread::spawn(move || {
                    let mut operations = 0;
                    let thread_start = Instant::now();

                    while thread_start.elapsed() < timeout_duration {
                        // Try to acquire lock with timeout to detect deadlocks
                        if let Ok(mut cache_guard) = cache.try_lock() {
                            // Perform operation that might cause deadlock in poorly designed code
                            let key = format!("deadlock_test_{}_{}", thread_id, operations);
                            let value = format!("value_{}", operations);

                            cache_guard.insert(key.clone(), value);
                            let _ = cache_guard.get(&key);
                            let _ = cache_guard.peek_oldest();

                            operations += 1;

                            // Don't hold lock too long
                            if operations % 10 == 0 {
                                drop(cache_guard);
                                thread::sleep(Duration::from_millis(1));
                            }
                        } else {
                            // If we can't acquire lock, wait briefly and try again
                            thread::sleep(Duration::from_millis(1));
                        }
                    }

                    completed_operations.fetch_add(operations, Ordering::SeqCst);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let elapsed = start_time.elapsed();
        let total_operations = completed_operations.load(Ordering::SeqCst);

        println!(
            "Deadlock prevention test: {} operations in {:?}",
            total_operations, elapsed
        );

        // Test should complete within timeout (no deadlocks)
        assert!(
            elapsed < timeout_duration + Duration::from_secs(1),
            "Test should complete without deadlocks"
        );
        assert!(total_operations > 0, "Should complete some operations");

        // Verify cache is still functional
        let final_cache = cache.lock().unwrap();
        assert!(final_cache.len() <= final_cache.capacity());
    }

    #[test]
    fn test_fairness_across_threads() {
        let cache: ThreadSafeFifoCache<String, String> = Arc::new(Mutex::new(FifoCache::new(80)));
        let num_threads = 8;
        let target_operations = 200;
        let test_duration = Duration::from_secs(5);

        let thread_operation_counts = Arc::new(Mutex::new(vec![0; num_threads]));
        let start_time = Instant::now();

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let thread_operation_counts = thread_operation_counts.clone();

                thread::spawn(move || {
                    let mut operations = 0;

                    while start_time.elapsed() < test_duration && operations < target_operations {
                        if let Ok(mut cache_guard) = cache.lock() {
                            // Perform operation
                            let key = format!("fair_{}_{}", thread_id, operations);
                            let value = format!("value_{}_{}", thread_id, operations);
                            cache_guard.insert(key, value);

                            operations += 1;

                            // Release lock periodically to allow other threads
                            if operations % 5 == 0 {
                                drop(cache_guard);
                                thread::yield_now();
                            }
                        } else {
                            thread::yield_now();
                        }
                    }

                    // Record this thread's operation count
                    if let Ok(mut counts) = thread_operation_counts.lock() {
                        counts[thread_id] = operations;
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let operation_counts = thread_operation_counts.lock().unwrap();
        let total_operations: usize = operation_counts.iter().sum();
        let min_operations = *operation_counts.iter().min().unwrap();
        let max_operations = *operation_counts.iter().max().unwrap();
        let avg_operations = total_operations as f64 / num_threads as f64;

        println!("Fairness test results:");
        println!("  Total operations: {}", total_operations);
        println!("  Average per thread: {:.1}", avg_operations);
        println!("  Min operations: {}", min_operations);
        println!("  Max operations: {}", max_operations);
        println!("  Operation counts: {:?}", *operation_counts);

        // Check fairness - no thread should be starved
        assert!(min_operations > 0, "No thread should be completely starved");

        // Check that the difference between min and max isn't too extreme
        let fairness_ratio = max_operations as f64 / min_operations.max(1) as f64;
        assert!(
            fairness_ratio < 10.0,
            "Fairness ratio should be reasonable, got {:.2}",
            fairness_ratio
        );

        // Verify cache final state
        let final_cache = cache.lock().unwrap();
        assert_eq!(
            final_cache.len(),
            final_cache.capacity(),
            "Cache should be at capacity with high thread contention"
        );
    }
}

// Stress Testing
mod stress_testing {
    use cachekit::prelude::FifoCache;
    use cachekit::traits::{CoreCache, FifoCacheTrait};

    use super::*;

    // Helper type for thread-safe testing
    type ThreadSafeFifoCache<K, V> = Arc<Mutex<FifoCache<K, V>>>;

    #[test]
    fn test_high_contention_scenario() {
        // Many threads accessing same small set of keys
        let cache: ThreadSafeFifoCache<String, String> = Arc::new(Mutex::new(FifoCache::new(50)));
        let num_threads = 20;
        let operations_per_thread = 500;
        let hot_keys = 10; // Small set of hotly contested keys

        let successful_ops = Arc::new(AtomicUsize::new(0));
        let contention_detected = Arc::new(AtomicBool::new(false));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let successful_ops = successful_ops.clone();
                let contention_detected = contention_detected.clone();

                thread::spawn(move || {
                    let mut ops = 0;
                    let mut lock_failures = 0;

                    for i in 0..operations_per_thread {
                        // All threads compete for same hot keys
                        let hot_key = format!("hot_key_{}", i % hot_keys);
                        let value = format!("thread_{}_value_{}", thread_id, i);

                        // Try with timeout to detect high contention
                        let start = Instant::now();
                        match cache.try_lock() {
                            Ok(mut cache_guard) => {
                                cache_guard.insert(hot_key.clone(), value);
                                let _ = cache_guard.get(&hot_key);
                                ops += 1;
                            },
                            Err(_) => {
                                lock_failures += 1;
                                thread::sleep(Duration::from_millis(1));
                            },
                        }

                        // Detect high contention
                        if start.elapsed() > Duration::from_millis(10) {
                            contention_detected.store(true, Ordering::SeqCst);
                        }
                    }

                    successful_ops.fetch_add(ops, Ordering::SeqCst);

                    println!(
                        "Thread {}: {} successful ops, {} lock failures",
                        thread_id, ops, lock_failures
                    );
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let total_ops = successful_ops.load(Ordering::SeqCst);
        let had_contention = contention_detected.load(Ordering::SeqCst);

        println!(
            "High contention test: {} operations completed, contention detected: {}",
            total_ops, had_contention
        );

        // Verify system remained functional under high contention
        assert!(
            total_ops > 0,
            "Should complete some operations despite contention"
        );

        // Verify final cache state is consistent
        let final_cache = cache.lock().unwrap();
        assert!(final_cache.len() <= final_cache.capacity());

        println!("Final cache state: {} items", final_cache.len());
    }

    #[test]
    fn test_cache_thrashing_scenario() {
        // Rapid insertions causing constant evictions (cache thrashing)
        let cache: ThreadSafeFifoCache<String, String> = Arc::new(Mutex::new(FifoCache::new(100)));
        let num_threads = 15;
        let operations_per_thread = 300;
        let key_space_multiplier = 10; // 10x more keys than capacity

        let evictions_detected = Arc::new(AtomicUsize::new(0));
        let total_insertions = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let evictions_detected = evictions_detected.clone();
                let total_insertions = total_insertions.clone();

                thread::spawn(move || {
                    let mut insertions = 0;
                    let mut evictions = 0;

                    for i in 0..operations_per_thread {
                        if let Ok(mut cache_guard) = cache.lock() {
                            let len_before = cache_guard.len();

                            // Insert with large key space to force thrashing
                            let key_idx = (thread_id * operations_per_thread + i)
                                % (cache_guard.capacity() * key_space_multiplier);
                            let key = format!("thrash_key_{}", key_idx);
                            let value = format!("thrash_value_{}_{}", thread_id, i);

                            cache_guard.insert(key, value);
                            insertions += 1;

                            let len_after = cache_guard.len();
                            let capacity = cache_guard.capacity();

                            // Detect if eviction occurred
                            if len_before == capacity && len_after == capacity {
                                evictions += 1;
                            }

                            // Occasionally force more evictions
                            if i % 20 == 0 {
                                let _ = cache_guard.pop_oldest_batch(5);
                                evictions += 5;
                            }
                        }
                    }

                    total_insertions.fetch_add(insertions, Ordering::SeqCst);
                    evictions_detected.fetch_add(evictions, Ordering::SeqCst);

                    println!(
                        "Thread {}: {} insertions, {} evictions",
                        thread_id, insertions, evictions
                    );
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let insertions = total_insertions.load(Ordering::SeqCst);
        let evictions = evictions_detected.load(Ordering::SeqCst);
        let expected_insertions = num_threads * operations_per_thread;

        println!(
            "Cache thrashing test: {} insertions, {} evictions",
            insertions, evictions
        );

        assert_eq!(
            insertions, expected_insertions,
            "All insertions should complete"
        );
        assert!(
            evictions > insertions / 2,
            "Should have high eviction rate due to thrashing"
        );

        // Verify cache remained stable despite thrashing
        let final_cache = cache.lock().unwrap();
        assert!(
            final_cache.len() <= final_cache.capacity(),
            "Cache should not exceed capacity after thrashing"
        );
    }

    #[test]
    fn test_long_running_stability() {
        // Verify stability over extended periods with continuous load
        let cache: ThreadSafeFifoCache<String, String> = Arc::new(Mutex::new(FifoCache::new(200)));
        let num_threads = 8;
        let test_duration = Duration::from_secs(15); // Extended test
        let stability_check_interval = Duration::from_secs(3);

        let operations_completed = Arc::new(AtomicUsize::new(0));
        let consistency_violations = Arc::new(AtomicUsize::new(0));
        let stop_signal = Arc::new(AtomicBool::new(false));

        // Stability checker thread
        let cache_checker = cache.clone();
        let consistency_violations_checker = consistency_violations.clone();
        let stop_signal_checker = stop_signal.clone();

        let checker_handle = thread::spawn(move || {
            let mut check_count = 0;

            while !stop_signal_checker.load(Ordering::SeqCst) {
                if let Ok(cache_guard) = cache_checker.try_lock() {
                    let len = cache_guard.len();
                    let capacity = cache_guard.capacity();
                    let insertion_order_len = cache_guard.insertion_order_len();

                    // Check consistency invariants
                    if len > capacity
                        || (len > 0 && insertion_order_len == 0)
                        || insertion_order_len > capacity * 3
                    {
                        // Allow for stale entries
                        consistency_violations_checker.fetch_add(1, Ordering::SeqCst);
                    }

                    check_count += 1;
                }

                thread::sleep(stability_check_interval);
            }

            println!("Stability checker completed {} checks", check_count);
        });

        // Worker threads
        let start_time = Instant::now();
        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let operations_completed = operations_completed.clone();

                thread::spawn(move || {
                    let mut ops = 0;
                    let thread_start = Instant::now();

                    while thread_start.elapsed() < test_duration {
                        if let Ok(mut cache_guard) = cache.lock() {
                            // Varied workload over time
                            let phase = (thread_start.elapsed().as_secs() / 5) % 3;

                            match phase {
                                0 => {
                                    // Insert phase
                                    let key = format!("stable_{}_{}", thread_id, ops);
                                    let value = format!("stable_value_{}_{}", thread_id, ops);
                                    cache_guard.insert(key, value);
                                },
                                1 => {
                                    // Mixed operations phase
                                    if ops % 3 == 0 {
                                        let _ = cache_guard.pop_oldest();
                                    } else {
                                        let key = format!("stable_{}_{}", thread_id, ops / 2);
                                        let _ = cache_guard.get(&key);
                                    }
                                },
                                _ => {
                                    // FIFO operations phase
                                    let _ = cache_guard.peek_oldest();
                                    if ops % 10 == 0 {
                                        let _ = cache_guard.pop_oldest_batch(3);
                                    }
                                },
                            }

                            ops += 1;
                        }

                        // Brief pause to prevent overwhelming
                        if ops % 100 == 0 {
                            thread::sleep(Duration::from_millis(10));
                        }
                    }

                    operations_completed.fetch_add(ops, Ordering::SeqCst);
                    println!("Thread {} completed {} operations", thread_id, ops);
                })
            })
            .collect();

        // Wait for all worker threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Stop checker and wait
        stop_signal.store(true, Ordering::SeqCst);
        checker_handle.join().unwrap();

        let total_ops = operations_completed.load(Ordering::SeqCst);
        let violations = consistency_violations.load(Ordering::SeqCst);
        let elapsed = start_time.elapsed();

        println!(
            "Long-running stability test: {} operations in {:?}, {} violations",
            total_ops, elapsed, violations
        );

        assert!(
            total_ops > 1000,
            "Should complete substantial number of operations"
        );
        assert_eq!(violations, 0, "Should maintain consistency throughout test");
        assert!(elapsed >= test_duration, "Should run for full duration");

        // Final consistency check
        let final_cache = cache.lock().unwrap();
        assert!(final_cache.len() <= final_cache.capacity());
    }

    #[test]
    fn test_memory_pressure_scenario() {
        // Test behavior with large cache and memory-intensive operations
        let large_capacity = 5000;
        let cache: ThreadSafeFifoCache<String, String> =
            Arc::new(Mutex::new(FifoCache::new(large_capacity)));
        let num_threads = 12;
        let operations_per_thread = 500;

        let memory_operations = Arc::new(AtomicUsize::new(0));
        let large_value_size = 1024; // 1KB values to increase memory pressure

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let memory_operations = memory_operations.clone();

                thread::spawn(move || {
                    let mut ops = 0;

                    for i in 0..operations_per_thread {
                        if let Ok(mut cache_guard) = cache.lock() {
                            // Create large values to increase memory pressure
                            let large_value = "x".repeat(large_value_size);
                            let key = format!("memory_{}_{}", thread_id, i);
                            let value = format!("{}_{}", large_value, i);

                            cache_guard.insert(key.clone(), value);

                            // Occasionally read back to test memory access
                            if i % 10 == 0 {
                                let _ = cache_guard.get(&key);
                            }

                            // Trigger evictions to test memory cleanup
                            if i % 50 == 0 {
                                let _ = cache_guard.pop_oldest_batch(10);
                            }

                            ops += 1;
                        }
                    }

                    memory_operations.fetch_add(ops, Ordering::SeqCst);
                    println!("Thread {} completed {} memory operations", thread_id, ops);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let total_memory_ops = memory_operations.load(Ordering::SeqCst);
        let expected_ops = num_threads * operations_per_thread;

        println!(
            "Memory pressure test: {} operations with large values",
            total_memory_ops
        );

        assert_eq!(
            total_memory_ops, expected_ops,
            "All memory operations should complete"
        );

        // Verify cache handles memory pressure correctly
        let final_cache = cache.lock().unwrap();
        let final_len = final_cache.len();

        assert!(
            final_len <= large_capacity,
            "Cache should not exceed capacity under memory pressure"
        );
        println!(
            "Final cache state: {} items (capacity {})",
            final_len, large_capacity
        );

        // Estimate memory usage
        let estimated_memory = final_len * large_value_size;
        println!("Estimated memory usage: ~{} KB", estimated_memory / 1024);
    }

    #[test]
    fn test_rapid_thread_creation_destruction() {
        // Test with threads being created and destroyed rapidly
        let cache: ThreadSafeFifoCache<String, String> = Arc::new(Mutex::new(FifoCache::new(150)));
        let num_thread_waves = 20;
        let threads_per_wave = 10;
        let operations_per_thread = 50;

        let total_operations = Arc::new(AtomicUsize::new(0));
        let thread_creation_count = Arc::new(AtomicUsize::new(0));

        for wave in 0..num_thread_waves {
            let wave_handles: Vec<_> = (0..threads_per_wave)
                .map(|thread_id| {
                    let cache = cache.clone();
                    let total_operations = total_operations.clone();
                    let thread_creation_count = thread_creation_count.clone();

                    thread_creation_count.fetch_add(1, Ordering::SeqCst);

                    thread::spawn(move || {
                        let mut ops = 0;

                        for i in 0..operations_per_thread {
                            if let Ok(mut cache_guard) = cache.lock() {
                                let key = format!("rapid_{}_{}_{}", wave, thread_id, i);
                                let value = format!("rapid_value_{}", i);
                                cache_guard.insert(key, value);

                                // Mix in some reads and FIFO operations
                                if i % 5 == 0 {
                                    let _ = cache_guard.peek_oldest();
                                }
                                if i % 15 == 0 {
                                    let _ = cache_guard.pop_oldest();
                                }

                                ops += 1;
                            }
                        }

                        total_operations.fetch_add(ops, Ordering::SeqCst);
                    })
                })
                .collect();

            // Wait for this wave to complete before starting next
            for handle in wave_handles {
                handle.join().unwrap();
            }

            // Brief pause between waves
            thread::sleep(Duration::from_millis(50));
        }

        let total_ops = total_operations.load(Ordering::SeqCst);
        let thread_count = thread_creation_count.load(Ordering::SeqCst);
        let expected_ops = num_thread_waves * threads_per_wave * operations_per_thread;
        let expected_threads = num_thread_waves * threads_per_wave;

        println!(
            "Rapid thread creation test: {} threads created, {} operations",
            thread_count, total_ops
        );

        assert_eq!(
            thread_count, expected_threads,
            "Should create expected number of threads"
        );
        assert_eq!(total_ops, expected_ops, "All operations should complete");

        // Verify cache consistency after rapid thread churn
        let final_cache = cache.lock().unwrap();
        assert!(final_cache.len() <= final_cache.capacity());

        println!(
            "Final cache state after rapid thread churn: {} items",
            final_cache.len()
        );
    }

    #[test]
    fn test_burst_load_handling() {
        // Test handling of sudden burst loads
        let cache: ThreadSafeFifoCache<String, String> = Arc::new(Mutex::new(FifoCache::new(300)));
        let burst_threads = 25;
        let operations_per_burst_thread = 100;
        let background_threads = 5;
        let background_operations = 200;

        let burst_operations = Arc::new(AtomicUsize::new(0));
        let background_operations_count = Arc::new(AtomicUsize::new(0));
        let burst_start_signal = Arc::new(AtomicBool::new(false));

        // Start background threads first (steady load)
        let background_handles: Vec<_> = (0..background_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let background_operations_count = background_operations_count.clone();
                let burst_start_signal = burst_start_signal.clone();

                thread::spawn(move || {
                    let mut ops = 0;

                    for i in 0..background_operations {
                        if let Ok(mut cache_guard) = cache.lock() {
                            let key = format!("background_{}_{}", thread_id, i);
                            let value = format!("bg_value_{}", i);
                            cache_guard.insert(key, value);
                            ops += 1;
                        }

                        // Signal burst to start midway through background load
                        if i == background_operations / 2 && thread_id == 0 {
                            burst_start_signal.store(true, Ordering::SeqCst);
                        }

                        thread::sleep(Duration::from_millis(10)); // Steady pace
                    }

                    background_operations_count.fetch_add(ops, Ordering::SeqCst);
                })
            })
            .collect();

        // Wait for burst signal
        while !burst_start_signal.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(10));
        }

        // Create sudden burst of threads
        println!("Starting burst load...");
        let burst_start_time = Instant::now();

        let burst_handles: Vec<_> = (0..burst_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let burst_operations = burst_operations.clone();

                thread::spawn(move || {
                    let mut ops = 0;

                    for i in 0..operations_per_burst_thread {
                        if let Ok(mut cache_guard) = cache.lock() {
                            let key = format!("burst_{}_{}", thread_id, i);
                            let value = format!("burst_value_{}_{}", thread_id, i);
                            cache_guard.insert(key, value);

                            // High-frequency operations during burst
                            if i % 3 == 0 {
                                let _ = cache_guard.pop_oldest();
                            }

                            ops += 1;
                        }
                    }

                    burst_operations.fetch_add(ops, Ordering::SeqCst);
                })
            })
            .collect();

        // Wait for burst to complete
        for handle in burst_handles {
            handle.join().unwrap();
        }

        let burst_duration = burst_start_time.elapsed();
        println!("Burst completed in {:?}", burst_duration);

        // Wait for background threads to complete
        for handle in background_handles {
            handle.join().unwrap();
        }

        let total_burst_ops = burst_operations.load(Ordering::SeqCst);
        let total_bg_ops = background_operations_count.load(Ordering::SeqCst);
        let expected_burst_ops = burst_threads * operations_per_burst_thread;
        let expected_bg_ops = background_threads * background_operations;

        println!(
            "Burst load test: {} burst operations, {} background operations",
            total_burst_ops, total_bg_ops
        );

        assert_eq!(
            total_burst_ops, expected_burst_ops,
            "All burst operations should complete"
        );
        assert_eq!(
            total_bg_ops, expected_bg_ops,
            "Background operations should not be disrupted"
        );

        // Verify system handled burst gracefully
        assert!(
            burst_duration < Duration::from_secs(30),
            "Burst should complete in reasonable time"
        );

        let final_cache = cache.lock().unwrap();
        assert!(
            final_cache.len() <= final_cache.capacity(),
            "Cache should remain consistent after burst"
        );
    }

    #[test]
    fn test_gradual_load_increase() {
        // Test behavior as load gradually increases
        let cache: ThreadSafeFifoCache<String, String> = Arc::new(Mutex::new(FifoCache::new(200)));
        let max_threads = 20;
        let operations_per_thread = 100;
        let ramp_up_steps = 10;

        let mut total_operations = 0;
        let mut performance_metrics = Vec::new();

        for step in 1..=ramp_up_steps {
            let num_threads = (max_threads * step) / ramp_up_steps;
            let step_operations = Arc::new(AtomicUsize::new(0));

            println!(
                "Load step {}/{}: {} threads",
                step, ramp_up_steps, num_threads
            );

            let step_start = Instant::now();

            let handles: Vec<_> = (0..num_threads)
                .map(|thread_id| {
                    let cache = cache.clone();
                    let step_operations = step_operations.clone();

                    thread::spawn(move || {
                        let mut ops = 0;

                        for i in 0..operations_per_thread {
                            if let Ok(mut cache_guard) = cache.lock() {
                                let key = format!("gradual_{}_{}_{}", step, thread_id, i);
                                let value = format!("gradual_value_{}", i);
                                cache_guard.insert(key, value);

                                // Mix operations based on load level
                                if step > 5 {
                                    // Higher load levels
                                    if i % 7 == 0 {
                                        let _ = cache_guard.pop_oldest();
                                    }
                                }

                                ops += 1;
                            }
                        }

                        step_operations.fetch_add(ops, Ordering::SeqCst);
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }

            let step_duration = step_start.elapsed();
            let step_ops = step_operations.load(Ordering::SeqCst);
            let ops_per_sec = step_ops as f64 / step_duration.as_secs_f64();

            performance_metrics.push((step, num_threads, step_ops, ops_per_sec));
            total_operations += step_ops;

            println!(
                "  Completed {} operations in {:?} ({:.1} ops/sec)",
                step_ops, step_duration, ops_per_sec
            );

            // Brief pause between load increases
            thread::sleep(Duration::from_millis(200));
        }

        println!("\nGradual load increase results:");
        for (step, threads, ops, ops_per_sec) in &performance_metrics {
            println!(
                "  Step {}: {} threads, {} ops, {:.1} ops/sec",
                step, threads, ops, ops_per_sec
            );
        }

        let expected_total_ops = (1..=ramp_up_steps)
            .map(|step| ((max_threads * step) / ramp_up_steps) * operations_per_thread)
            .sum::<usize>();

        assert_eq!(
            total_operations, expected_total_ops,
            "All operations should complete"
        );

        // Verify performance doesn't degrade catastrophically with load
        let first_step_perf = performance_metrics[0].3;
        let last_step_perf = performance_metrics.last().unwrap().3;
        let performance_ratio = last_step_perf / first_step_perf;

        println!("Performance ratio (last/first): {:.2}", performance_ratio);
        assert!(
            performance_ratio > 0.1,
            "Performance shouldn't degrade too severely with load"
        );

        // Verify final cache state
        let final_cache = cache.lock().unwrap();
        assert_eq!(
            final_cache.len(),
            final_cache.capacity(),
            "Cache should be at capacity after gradual load increase"
        );
    }
}
