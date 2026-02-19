// ==============================================
// SLAB STORE CONCURRENCY TESTS (integration)
// ==============================================
//
// Tests for race conditions and atomicity issues in ConcurrentSlabStore.
// These require multi-threaded execution and cannot live inline.

#![cfg(feature = "concurrency")]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use cachekit::store::slab::ConcurrentSlabStore;
use cachekit::store::traits::{ConcurrentStore, ConcurrentStoreRead};

// ==============================================
// TOCTOU Race: Update Path Data Corruption
// ==============================================
//
// try_insert's update path drops the index read lock before acquiring the
// entries write lock. A concurrent remove + reinsert can recycle the slot,
// causing the original thread to overwrite an unrelated key's value.

mod toctou_update {
    use super::*;

    #[test]
    fn concurrent_update_after_remove_preserves_invariants() {
        let iterations = 500;

        for _ in 0..iterations {
            let store: Arc<ConcurrentSlabStore<u64, String>> =
                Arc::new(ConcurrentSlabStore::new(100));

            store
                .try_insert(1, Arc::new("key1_original".into()))
                .unwrap();
            store
                .try_insert(2, Arc::new("key2_original".into()))
                .unwrap();

            let barrier = Arc::new(Barrier::new(3));

            let store_a = store.clone();
            let barrier_a = barrier.clone();
            let t_a = thread::spawn(move || {
                barrier_a.wait();
                let _ = store_a.try_insert(1, Arc::new("key1_updated".into()));
            });

            let store_b = store.clone();
            let barrier_b = barrier.clone();
            let t_b = thread::spawn(move || {
                barrier_b.wait();
                let _ = store_b.remove(&1);
            });

            let store_c = store.clone();
            let barrier_c = barrier.clone();
            let t_c = thread::spawn(move || {
                barrier_c.wait();
                let _ = store_c.try_insert(3, Arc::new("key3_value".into()));
            });

            t_a.join().unwrap();
            t_b.join().unwrap();
            t_c.join().unwrap();

            if let Some(val) = store.get(&2) {
                assert_eq!(
                    *val, "key2_original",
                    "key 2 was corrupted by a concurrent update to a recycled slot"
                );
            }

            if let Some(val) = store.get(&3) {
                assert_eq!(
                    *val, "key3_value",
                    "key 3 was corrupted by a concurrent update to a recycled slot"
                );
            }
        }
    }
}

// ==============================================
// TOCTOU Race: Capacity Overshoot
// ==============================================
//
// The capacity check and actual insert use separate locks, allowing multiple
// threads to pass the check simultaneously and exceed capacity.

mod capacity_overshoot {
    use super::*;

    #[test]
    fn concurrent_inserts_respect_capacity() {
        let capacity = 10;
        let num_threads = 20;
        let inserts_per_thread = 5;

        for _ in 0..200 {
            let store: Arc<ConcurrentSlabStore<u64, u64>> =
                Arc::new(ConcurrentSlabStore::new(capacity));
            let barrier = Arc::new(Barrier::new(num_threads));

            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let store = store.clone();
                    let barrier = barrier.clone();
                    thread::spawn(move || {
                        barrier.wait();
                        for i in 0..inserts_per_thread {
                            let key = (tid * inserts_per_thread + i) as u64;
                            let _ = store.try_insert(key, Arc::new(key));
                        }
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }

            assert!(
                store.len() <= capacity,
                "ConcurrentSlabStore len ({}) exceeds capacity ({})",
                store.len(),
                capacity,
            );
        }
    }
}

// ==============================================
// Atomic clear()
// ==============================================
//
// Validates that clear() is atomic: concurrent get() calls never observe
// a half-cleared state where the index has a key but the entry is missing.
// With the single-lock design, get() reads both index and entries under
// one read lock, so its result is always self-consistent.

mod atomic_clear {
    use super::*;

    #[test]
    fn clear_concurrent_with_get_is_consistent() {
        let store: Arc<ConcurrentSlabStore<u64, u64>> = Arc::new(ConcurrentSlabStore::new(1000));
        let stop = Arc::new(AtomicBool::new(false));
        let inconsistencies = Arc::new(AtomicUsize::new(0));

        for i in 0..1000u64 {
            store.try_insert(i, Arc::new(i)).unwrap();
        }

        let store_r = store.clone();
        let stop_r = stop.clone();
        let inconsistencies_r = inconsistencies.clone();
        let reader = thread::spawn(move || {
            while !stop_r.load(Ordering::Relaxed) {
                for i in 0..100u64 {
                    // get() should be internally consistent: if it finds
                    // the key in the index, the entry must also exist.
                    // A Some result with a wrong value would indicate corruption.
                    if let Some(val) = store_r.get(&i) {
                        if *val != i {
                            inconsistencies_r.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }
        });

        let store_w = store.clone();
        let stop_w = stop.clone();
        let writer = thread::spawn(move || {
            for _ in 0..500 {
                store_w.clear();
                for i in 0..100u64 {
                    let _ = store_w.try_insert(i, Arc::new(i));
                }
            }
            stop_w.store(true, Ordering::Relaxed);
        });

        reader.join().unwrap();
        writer.join().unwrap();

        assert_eq!(
            inconsistencies.load(Ordering::Relaxed),
            0,
            "get() returned an inconsistent value during concurrent clear()"
        );
    }
}
