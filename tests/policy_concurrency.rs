// ==============================================
// NATIVE CONCURRENT POLICY TESTS (integration)
// ==============================================

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

mod concurrent_lru {
    use cachekit::policy::lru::ConcurrentLruCache;

    use super::*;

    #[test]
    fn concurrent_insert_operations() {
        let capacity = 400;
        let cache = Arc::new(ConcurrentLruCache::new(capacity));
        let num_threads = 8;
        let inserts_per_thread = 100;
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

        assert_eq!(
            successes.load(Ordering::SeqCst),
            num_threads * inserts_per_thread
        );
        assert!(cache.len() <= cache.capacity());
    }

    #[test]
    fn concurrent_get_operations() {
        let capacity = 256;
        let cache = Arc::new(ConcurrentLruCache::new(capacity));
        for key in 0..capacity {
            cache.insert(key as u64, key as u64);
        }

        let hits = Arc::new(AtomicUsize::new(0));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let cache = cache.clone();
                let hits = hits.clone();
                thread::spawn(move || {
                    for i in 0..500 {
                        let key = (i % capacity) as u64;
                        if cache.get(&key).is_some() {
                            hits.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(hits.load(Ordering::Relaxed), 8 * 500);
        assert_eq!(cache.len(), capacity);
    }

    #[test]
    fn concurrent_mixed_operations() {
        let capacity = 200;
        let cache = Arc::new(ConcurrentLruCache::new(capacity));
        for key in 0..50 {
            cache.insert(key, key);
        }

        let handles: Vec<_> = (0..8)
            .map(|thread_id| {
                let cache = cache.clone();
                thread::spawn(move || {
                    for i in 0..150 {
                        match (thread_id + i) % 4 {
                            0 => {
                                cache.insert(1000 + (thread_id * 150 + i) as u64, i as u64);
                            },
                            1 => {
                                let _ = cache.get(&(i as u64 % 60));
                            },
                            2 => {
                                let _ = cache.peek(&(i as u64 % 60));
                            },
                            _ => {
                                let _ = cache.remove(&(i as u64 % 40));
                            },
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(cache.len() <= capacity);
    }

    #[test]
    fn burst_load_handling() {
        let cache = Arc::new(ConcurrentLruCache::new(128));
        let burst_signal = Arc::new(AtomicBool::new(false));
        let burst_done = Arc::new(AtomicUsize::new(0));

        let bg = cache.clone();
        let signal = burst_signal.clone();
        let bg_handle = thread::spawn(move || {
            for i in 0..200 {
                bg.insert(i, i);
                if i == 100 {
                    signal.store(true, Ordering::SeqCst);
                }
            }
        });

        while !burst_signal.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(1));
        }

        let burst_handles: Vec<_> = (0..12)
            .map(|tid| {
                let cache = cache.clone();
                let burst_done = burst_done.clone();
                thread::spawn(move || {
                    for i in 0..40 {
                        cache.insert(10_000 + (tid * 40 + i) as u64, i as u64);
                    }
                    burst_done.fetch_add(40, Ordering::SeqCst);
                })
            })
            .collect();

        for h in burst_handles {
            h.join().unwrap();
        }
        bg_handle.join().unwrap();
        assert_eq!(burst_done.load(Ordering::SeqCst), 12 * 40);
        assert!(cache.len() <= cache.capacity());
    }
}

#[cfg(feature = "policy-fifo")]
mod concurrent_fifo {
    use cachekit::policy::fifo::ConcurrentFifoCache;

    use super::*;

    #[test]
    fn parallel_inserts_respect_capacity() {
        let capacity = 64;
        let cache = Arc::new(ConcurrentFifoCache::new(capacity));
        let barrier = Arc::new(Barrier::new(8));

        let handles: Vec<_> = (0..8)
            .map(|tid| {
                let cache = cache.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    for i in 0..40 {
                        cache.insert((tid * 40 + i) as u64, i as u64);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(cache.len() <= capacity);
    }

    #[test]
    fn parallel_get_and_peek() {
        let cache = Arc::new(ConcurrentFifoCache::new(128));
        for i in 0..128u64 {
            cache.insert(i, i);
        }

        let hits = Arc::new(AtomicUsize::new(0));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let cache = cache.clone();
                let hits = hits.clone();
                thread::spawn(move || {
                    for i in 0..400 {
                        let key = (i % 128) as u64;
                        if cache.get(&key).is_some() {
                            hits.fetch_add(1, Ordering::Relaxed);
                        }
                        if i % 10 == 0 {
                            let _ = cache.peek(&key);
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(hits.load(Ordering::Relaxed), 8 * 400);
    }

    #[test]
    fn mixed_ops_under_load() {
        let cache = Arc::new(ConcurrentFifoCache::new(50));
        let start = Instant::now();
        let handles: Vec<_> = (0..10)
            .map(|tid| {
                let cache = cache.clone();
                thread::spawn(move || {
                    for i in 0..100 {
                        cache.insert((tid * 100 + i) as u64, i as u64);
                        if i % 7 == 0 {
                            let _ = cache.pop_oldest();
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(start.elapsed() < Duration::from_secs(5));
        assert!(cache.len() <= cache.capacity());
    }
}

#[cfg(feature = "policy-s3-fifo")]
mod concurrent_s3_fifo {
    use cachekit::policy::s3_fifo::ConcurrentS3FifoCache;

    use super::*;

    #[test]
    fn parallel_inserts_respect_capacity() {
        let capacity = 32;
        let cache = Arc::new(ConcurrentS3FifoCache::new(capacity));
        let barrier = Arc::new(Barrier::new(8));

        let handles: Vec<_> = (0..8)
            .map(|tid| {
                let cache = cache.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    for i in 0..20 {
                        cache.insert((tid * 20 + i) as u64, i as u64);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(cache.len() <= capacity);
        assert!(cache.small_len() + cache.main_len() <= capacity);
    }

    #[test]
    fn queue_lens_invariant_under_mixed_load() {
        let cache = Arc::new(ConcurrentS3FifoCache::new(40));
        let handles: Vec<_> = (0..6)
            .map(|tid| {
                let cache = cache.clone();
                thread::spawn(move || {
                    for i in 0..80 {
                        cache.insert((tid * 80 + i) as u64, i as u64);
                        if i % 15 == 0 {
                            let _ = cache.get(&((i % 20) as u64));
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(cache.small_len() + cache.main_len() <= cache.capacity());
    }
}
