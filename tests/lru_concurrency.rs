use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

// ==============================================
// LRU CONCURRENCY TESTS (integration)
// ==============================================

mod thread_safety {
    use cachekit::policy::lru::ConcurrentLRUCache;

    use super::*;

    #[test]
    fn test_concurrent_insert_operations() {
        let capacity = 1_600;
        let cache = Arc::new(ConcurrentLRUCache::new(capacity));

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
                        cache.insert(key, key);
                        successes.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let expected_inserts = num_threads * inserts_per_thread;
        assert_eq!(successes.load(Ordering::SeqCst), expected_inserts);
        assert_eq!(cache.len(), expected_inserts);
        assert!(cache.capacity() >= expected_inserts);
        assert!(cache.contains(&(expected_inserts as u64 - 1)));
    }

    #[test]
    fn test_concurrent_get_operations() {
        let capacity = 512;
        let cache = Arc::new(ConcurrentLRUCache::new(capacity));

        for key in 0..capacity {
            cache.insert(key as u64, key as u64 * 2);
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
                        if cache.get(&key).is_some() {
                            hits.fetch_add(1, Ordering::Relaxed);
                        }

                        // Exercise read-only path occasionally
                        if i % 50 == 0 {
                            let _ = cache.peek(&key);
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
        assert_eq!(cache.len(), capacity);
        assert!(cache.peek_lru().is_some());
    }

    #[test]
    fn test_concurrent_remove_operations() {
        let total_keys = 400;
        let cache = Arc::new(ConcurrentLRUCache::new(total_keys));

        for key in 0..total_keys {
            cache.insert(key as u64, key as u64);
        }

        let num_threads = 8;
        let keys_per_thread = total_keys / num_threads;
        let removed = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let removed = removed.clone();

                thread::spawn(move || {
                    let start = thread_id * keys_per_thread;
                    let end = start + keys_per_thread;

                    for key in start..end {
                        if cache.remove(&(key as u64)).is_some() {
                            removed.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(removed.load(Ordering::SeqCst), total_keys);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_concurrent_mixed_operations() {
        let capacity = 300;
        let cache = Arc::new(ConcurrentLRUCache::new(capacity));

        // Seed cache with some entries
        for key in 0..100 {
            cache.insert(key as u64, key as u64);
        }

        let num_threads = 10;
        let operations_per_thread = 200;
        let op_count = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let op_count = op_count.clone();

                thread::spawn(move || {
                    for i in 0..operations_per_thread {
                        let op_selector = (thread_id + i) % 6;
                        match op_selector {
                            0 => {
                                // Insert new item
                                let key = 1_000 + (thread_id * operations_per_thread + i) as u64;
                                cache.insert(key, key);
                            },
                            1 => {
                                // Read hot keys
                                let key = (i % 120) as u64;
                                let _ = cache.get(&key);
                            },
                            2 => {
                                // Touch recently used keys
                                let key = (i % 80) as u64;
                                let _ = cache.touch(&key);
                            },
                            3 => {
                                // Occasional removals
                                if i % 10 == 0 {
                                    let key = (i % 100) as u64;
                                    let _ = cache.remove(&key);
                                }
                            },
                            4 => {
                                // Peek without updating LRU
                                let key = (i % 150) as u64;
                                let _ = cache.peek(&key);
                            },
                            _ => {
                                // Evict oldest occasionally
                                if i % 20 == 0 {
                                    let _ = cache.pop_lru();
                                }
                            },
                        }

                        op_count.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let expected_ops = num_threads * operations_per_thread;
        assert_eq!(op_count.load(Ordering::SeqCst), expected_ops);
        assert_eq!(cache.capacity(), capacity);
        assert!(cache.len() <= cache.capacity());
        assert!(!cache.is_empty());
    }
}

mod stress_testing {
    use cachekit::policy::lru::ConcurrentLRUCache;

    use super::*;

    #[test]
    fn test_high_throughput_operations() {
        let cache = Arc::new(ConcurrentLRUCache::new(256));
        let num_threads = 16;
        let operations_per_thread = 1_000;
        let completed = Arc::new(AtomicUsize::new(0));

        let start = Instant::now();

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let completed = completed.clone();

                thread::spawn(move || {
                    for i in 0..operations_per_thread {
                        if i % 2 == 0 {
                            let key = ((thread_id * operations_per_thread + i) % 1_024) as u64;
                            cache.insert(key, key);
                        } else {
                            let key = (i % 256) as u64;
                            let _ = cache.get(&key);
                        }
                        completed.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let elapsed = start.elapsed();
        let expected = num_threads * operations_per_thread;

        assert_eq!(completed.load(Ordering::Relaxed), expected);
        assert!(cache.len() <= cache.capacity());
        assert!(elapsed < Duration::from_secs(10));
    }

    #[test]
    fn test_massive_concurrent_threads() {
        let cache = Arc::new(ConcurrentLRUCache::new(512));
        let num_threads = 64;
        let operations_per_thread = 40;
        let operations_done = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let operations_done = operations_done.clone();

                thread::spawn(move || {
                    for i in 0..operations_per_thread {
                        let key = (thread_id * operations_per_thread + i) as u64;
                        cache.insert(key, key);

                        if i % 3 == 0 {
                            let _ = cache.get(&key);
                        }

                        if i % 10 == 0 {
                            let _ = cache.pop_lru();
                        }

                        operations_done.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let expected_ops = num_threads * operations_per_thread;
        assert_eq!(operations_done.load(Ordering::SeqCst), expected_ops);
        assert!(cache.len() <= cache.capacity());
    }

    #[test]
    fn test_burst_load_handling() {
        let cache = Arc::new(ConcurrentLRUCache::new(400));

        let background_threads = 4;
        let background_ops = 200;
        let burst_threads = 20;
        let burst_ops = 50;

        let bg_completed = Arc::new(AtomicUsize::new(0));
        let burst_completed = Arc::new(AtomicUsize::new(0));
        let burst_signal = Arc::new(AtomicBool::new(false));

        // Background workload (steady pace)
        let bg_handles: Vec<_> = (0..background_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let bg_completed = bg_completed.clone();
                let burst_signal = burst_signal.clone();

                thread::spawn(move || {
                    for i in 0..background_ops {
                        let key = (thread_id * 10_000 + i) as u64;
                        cache.insert(key, key);

                        if i % 25 == 0 {
                            let _ = cache.peek(&((i % 150) as u64));
                        }

                        if i == background_ops / 2 && thread_id == 0 {
                            burst_signal.store(true, Ordering::SeqCst);
                        }

                        bg_completed.fetch_add(1, Ordering::SeqCst);
                        thread::sleep(Duration::from_millis(2));
                    }
                })
            })
            .collect();

        // Wait for burst trigger
        while !burst_signal.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(2));
        }

        // Burst workload (sudden spike)
        let burst_handles: Vec<_> = (0..burst_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                let burst_completed = burst_completed.clone();

                thread::spawn(move || {
                    for i in 0..burst_ops {
                        let key = 50_000 + (thread_id * burst_ops + i) as u64;
                        cache.insert(key, key);

                        if i % 3 == 0 {
                            let _ = cache.pop_lru();
                        }

                        burst_completed.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        for handle in burst_handles {
            handle.join().unwrap();
        }

        for handle in bg_handles {
            handle.join().unwrap();
        }

        let expected_bg = background_threads * background_ops;
        let expected_burst = burst_threads * burst_ops;

        assert_eq!(bg_completed.load(Ordering::SeqCst), expected_bg);
        assert_eq!(burst_completed.load(Ordering::SeqCst), expected_burst);
        assert!(cache.len() <= cache.capacity());
        assert!(cache.len() > cache.capacity() / 4);
    }
}
