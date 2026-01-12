use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

// ==============================================
// LRU-K CONCURRENCY TESTS (integration)
// ==============================================

// Thread Safety Tests
mod thread_safety {
    use cachekit::policy::lru_k::LRUKCache;
    use cachekit::traits::{CoreCache, LRUKCacheTrait};

    use super::*;

    type ThreadSafeLRUK<K, V> = Arc<Mutex<LRUKCache<K, V>>>;

    #[test]
    fn test_concurrent_insertions() {
        let cache: ThreadSafeLRUK<u64, u64> = Arc::new(Mutex::new(LRUKCache::with_k(256, 2)));
        let threads = 8;
        let inserts_per_thread = 120;
        let inserted = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let cache = cache.clone();
                let inserted = inserted.clone();
                thread::spawn(move || {
                    for i in 0..inserts_per_thread {
                        let key = t as u64 * 1_000 + i as u64;
                        if let Ok(mut guard) = cache.lock() {
                            guard.insert(key, key * 2);
                            inserted.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(
            inserted.load(Ordering::SeqCst),
            threads * inserts_per_thread
        );
        let guard = cache.lock().unwrap();
        assert!(guard.len() <= guard.capacity());
    }

    #[test]
    fn test_concurrent_gets() {
        let cache: ThreadSafeLRUK<u64, u64> = Arc::new(Mutex::new(LRUKCache::with_k(128, 3)));

        {
            let mut guard = cache.lock().unwrap();
            for i in 0..128 {
                guard.insert(i, i * 10);
            }
        }

        let threads = 12;
        let reads_per_thread = 300;
        let hits = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let cache = cache.clone();
                let hits = hits.clone();
                thread::spawn(move || {
                    for i in 0..reads_per_thread {
                        if let Ok(mut guard) = cache.lock() {
                            let key = (i % 128) as u64;
                            if guard.get(&key).is_some() {
                                hits.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(hits.load(Ordering::Relaxed), threads * reads_per_thread);
        let guard = cache.lock().unwrap();
        assert_eq!(guard.len(), 128);
    }

    #[test]
    fn test_concurrent_history_operations() {
        let cache: ThreadSafeLRUK<u64, u64> = Arc::new(Mutex::new(LRUKCache::with_k(64, 2)));

        {
            let mut guard = cache.lock().unwrap();
            for i in 0..64 {
                guard.insert(i, i);
            }
        }

        let threads = 6;
        let ops_per_thread = 200;

        let handles: Vec<_> = (0..threads)
            .map(|tid| {
                let cache = cache.clone();
                thread::spawn(move || {
                    for i in 0..ops_per_thread {
                        let key = (i % 64) as u64;
                        if let Ok(mut guard) = cache.lock() {
                            match (tid + i) % 3 {
                                0 => {
                                    let _ = guard.touch(&key);
                                },
                                1 => {
                                    let _ = guard.access_history(&key);
                                },
                                _ => {
                                    let _ = guard.k_distance(&key);
                                },
                            }
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let guard = cache.lock().unwrap();
        assert_eq!(guard.len(), 64);
    }

    #[test]
    fn test_concurrent_lru_k_operations() {
        let cache: ThreadSafeLRUK<u64, u64> = Arc::new(Mutex::new(LRUKCache::with_k(50, 2)));

        {
            let mut guard = cache.lock().unwrap();
            for i in 0..50 {
                guard.insert(i, i);
            }
        }

        let pop_results = Arc::new(Mutex::new(Vec::new()));

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let cache = cache.clone();
                let pop_results = pop_results.clone();
                thread::spawn(move || {
                    for j in 0..25 {
                        if let Ok(mut guard) = cache.lock() {
                            if (i + j) % 3 == 0 {
                                let _ = guard.peek_lru_k();
                            } else if let Some((k, v)) = guard.pop_lru_k() {
                                pop_results.lock().unwrap().push((k, v));
                            }
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let popped = pop_results.lock().unwrap();
        let guard = cache.lock().unwrap();
        assert!(guard.len() + popped.len() <= 50);
    }

    #[test]
    fn test_mixed_concurrent_operations() {
        let cache: ThreadSafeLRUK<u64, u64> = Arc::new(Mutex::new(LRUKCache::with_k(120, 2)));
        {
            let mut guard = cache.lock().unwrap();
            for i in 0..80 {
                guard.insert(i, i * 3);
            }
        }

        let threads = 10;
        let ops_per_thread = 150;
        let op_counter = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..threads)
            .map(|tid| {
                let cache = cache.clone();
                let op_counter = op_counter.clone();
                thread::spawn(move || {
                    for i in 0..ops_per_thread {
                        if let Ok(mut guard) = cache.lock() {
                            match (tid + i) % 6 {
                                0 => {
                                    let key = 1_000 + (tid * ops_per_thread + i) as u64;
                                    guard.insert(key, key);
                                },
                                1 | 2 => {
                                    let key = (i % 100) as u64;
                                    let _ = guard.get(&key);
                                },
                                3 => {
                                    let key = (i % 60) as u64;
                                    let _ = guard.touch(&key);
                                },
                                4 => {
                                    let _ = guard.peek_lru_k();
                                },
                                _ => {
                                    let _ = guard.pop_lru_k();
                                },
                            }
                        }
                        op_counter.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(op_counter.load(Ordering::SeqCst), threads * ops_per_thread);
        let guard = cache.lock().unwrap();
        assert!(guard.len() <= guard.capacity());
    }
}

// Stress Testing (kept lightweight to avoid long runtimes)
mod stress_testing {
    use cachekit::policy::lru_k::LRUKCache;
    use cachekit::traits::{CoreCache, LRUKCacheTrait};

    use super::*;

    type ThreadSafeLRUK<K, V> = Arc<Mutex<LRUKCache<K, V>>>;

    #[test]
    fn test_high_contention_scenario() {
        let cache: ThreadSafeLRUK<u64, u64> = Arc::new(Mutex::new(LRUKCache::with_k(64, 2)));
        let threads = 16;
        let ops = 200;
        let hits = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let cache = cache.clone();
                let hits = hits.clone();
                thread::spawn(move || {
                    for i in 0..ops {
                        let key = (i % 10) as u64; // hot set
                        if let Ok(mut guard) = cache.lock() {
                            if (t + i) % 2 == 0 {
                                guard.insert(key, key + 1);
                            } else if guard.get(&key).is_some() {
                                hits.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let guard = cache.lock().unwrap();
        assert!(guard.len() <= guard.capacity());
        assert!(hits.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_cache_thrashing_scenario() {
        let cache: ThreadSafeLRUK<u64, u64> = Arc::new(Mutex::new(LRUKCache::with_k(50, 2)));
        let threads = 12;
        let ops = 150;

        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let cache = cache.clone();
                thread::spawn(move || {
                    for i in 0..ops {
                        let key = (t * ops + i) as u64;
                        if let Ok(mut guard) = cache.lock() {
                            guard.insert(key, key);
                            if i % 5 == 0 {
                                let _ = guard.pop_lru_k();
                            }
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let guard = cache.lock().unwrap();
        assert!(guard.len() <= guard.capacity());
    }

    #[test]
    fn test_long_running_stability() {
        let cache: ThreadSafeLRUK<u64, u64> = Arc::new(Mutex::new(LRUKCache::with_k(80, 2)));
        let threads = 6;
        let ops = 400;

        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let cache = cache.clone();
                thread::spawn(move || {
                    for i in 0..ops {
                        if let Ok(mut guard) = cache.lock() {
                            let phase = (t + i) % 3;
                            let key = (i % 120) as u64;
                            match phase {
                                0 => {
                                    guard.insert(key, key + 2);
                                },
                                1 => {
                                    let _ = guard.get(&key);
                                },
                                _ => {
                                    let _ = guard.touch(&key);
                                },
                            }
                        }

                        if i % 80 == 0 {
                            thread::sleep(Duration::from_millis(2));
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let guard = cache.lock().unwrap();
        assert!(guard.len() <= guard.capacity());
    }

    #[test]
    fn test_burst_load_handling() {
        let cache: ThreadSafeLRUK<u64, u64> = Arc::new(Mutex::new(LRUKCache::with_k(120, 2)));
        let steady_threads = 4;
        let burst_threads = 10;
        let steady_ops = 150;
        let burst_ops = 60;

        let steady_done = Arc::new(AtomicUsize::new(0));
        let burst_done = Arc::new(AtomicUsize::new(0));
        let start_burst = Arc::new(AtomicUsize::new(0));

        let steady_handles: Vec<_> = (0..steady_threads)
            .map(|t| {
                let cache = cache.clone();
                let steady_done = steady_done.clone();
                let start_burst = start_burst.clone();
                thread::spawn(move || {
                    for i in 0..steady_ops {
                        if let Ok(mut guard) = cache.lock() {
                            let key = (t * 10 + i) as u64;
                            guard.insert(key, key);
                            if i % 20 == 0 {
                                let _ = guard.peek_lru_k();
                            }
                        }
                        if i == steady_ops / 2 && t == 0 {
                            start_burst.store(1, Ordering::SeqCst);
                        }
                        steady_done.fetch_add(1, Ordering::SeqCst);
                        thread::sleep(Duration::from_millis(1));
                    }
                })
            })
            .collect();

        // Wait for burst trigger
        while start_burst.load(Ordering::SeqCst) == 0 {
            thread::sleep(Duration::from_millis(2));
        }

        let burst_handles: Vec<_> = (0..burst_threads)
            .map(|t| {
                let cache = cache.clone();
                let burst_done = burst_done.clone();
                thread::spawn(move || {
                    for i in 0..burst_ops {
                        if let Ok(mut guard) = cache.lock() {
                            let key = 10_000 + (t * burst_ops + i) as u64;
                            guard.insert(key, key);
                            if i % 3 == 0 {
                                let _ = guard.pop_lru_k();
                            }
                        }
                        burst_done.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        for h in burst_handles {
            h.join().unwrap();
        }
        for h in steady_handles {
            h.join().unwrap();
        }

        let guard = cache.lock().unwrap();
        assert!(guard.len() <= guard.capacity());
        assert!(steady_done.load(Ordering::SeqCst) > 0);
        assert_eq!(burst_done.load(Ordering::SeqCst), burst_threads * burst_ops);
    }
}
