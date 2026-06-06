// ==============================================
// CONCURRENT STORE TESTS (integration)
// ==============================================

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use cachekit::store::handle::ConcurrentHandleStore;
use cachekit::store::hashmap::{ConcurrentHashMapStore, ShardedHashMapStore};
use cachekit::store::traits::{ConcurrentStore, ConcurrentStoreRead};
use cachekit::store::weight::ConcurrentWeightStore;

mod hashmap_store {
    use super::*;

    #[test]
    fn concurrent_inserts_respect_capacity() {
        let capacity = 10;
        let num_threads = 20;
        let inserts_per_thread = 5;

        for _ in 0..50 {
            let store: Arc<ConcurrentHashMapStore<u64, u64>> =
                Arc::new(ConcurrentHashMapStore::try_new(capacity).unwrap());
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
                "len {} exceeds capacity {}",
                store.len(),
                capacity
            );
        }
    }

    #[test]
    fn clear_concurrent_with_get_is_consistent() {
        let store: Arc<ConcurrentHashMapStore<u64, u64>> =
            Arc::new(ConcurrentHashMapStore::try_new(256).unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let inconsistencies = Arc::new(AtomicUsize::new(0));

        for i in 0..256u64 {
            store.try_insert(i, Arc::new(i)).unwrap();
        }

        let store_r = store.clone();
        let stop_r = stop.clone();
        let inconsistencies_r = inconsistencies.clone();
        let reader = thread::spawn(move || {
            while !stop_r.load(Ordering::Relaxed) {
                for i in 0..64u64 {
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
            for _ in 0..100 {
                store_w.clear();
                for i in 0..64u64 {
                    let _ = store_w.try_insert(i, Arc::new(i));
                }
            }
            stop_w.store(true, Ordering::Relaxed);
        });

        reader.join().unwrap();
        writer.join().unwrap();

        assert_eq!(inconsistencies.load(Ordering::Relaxed), 0);
    }
}

mod sharded_store {
    use super::*;

    #[test]
    fn concurrent_inserts_respect_capacity() {
        let capacity = 16;
        let shards = 4;
        let store: Arc<ShardedHashMapStore<u64, u64>> =
            Arc::new(ShardedHashMapStore::try_new(capacity, shards).unwrap());
        assert_eq!(store.shard_count(), shards);

        let num_threads = 16;
        let barrier = Arc::new(Barrier::new(num_threads));
        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let store = store.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    for i in 0..8 {
                        let key = (tid * 8 + i) as u64;
                        let _ = store.try_insert(key, Arc::new(key));
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(store.len() <= capacity);
    }
}

mod handle_store {
    use super::*;

    #[test]
    fn concurrent_inserts_respect_capacity() {
        let capacity = 8;
        let store: Arc<ConcurrentHandleStore<u64, u64>> =
            Arc::new(ConcurrentHandleStore::try_new(capacity).unwrap());
        let num_threads = 12;
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let store = store.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    for i in 0..4 {
                        let key = (tid * 4 + i) as u64;
                        let _ = store.try_insert(key, Arc::new(key));
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(store.len() <= capacity);
    }

    #[test]
    fn concurrent_update_after_remove_preserves_other_keys() {
        for _ in 0..100 {
            let store: Arc<ConcurrentHandleStore<u64, String>> =
                Arc::new(ConcurrentHandleStore::try_new(32).unwrap());
            store.try_insert(1, Arc::new("one".into())).unwrap();
            store.try_insert(2, Arc::new("two".into())).unwrap();

            let barrier = Arc::new(Barrier::new(3));
            let s1 = store.clone();
            let s2 = store.clone();
            let s3 = store.clone();
            let b1 = barrier.clone();
            let b2 = barrier.clone();
            let b3 = barrier.clone();

            let t1 = thread::spawn(move || {
                b1.wait();
                let _ = s1.try_insert(1, Arc::new("one_updated".into()));
            });
            let t2 = thread::spawn(move || {
                b2.wait();
                let _ = s2.remove(&1);
            });
            let t3 = thread::spawn(move || {
                b3.wait();
                let _ = s3.try_insert(3, Arc::new("three".into()));
            });
            t1.join().unwrap();
            t2.join().unwrap();
            t3.join().unwrap();

            if let Some(v) = store.get(&2) {
                assert_eq!(v.as_str(), "two");
            }
        }
    }
}

mod weight_store {
    use super::*;

    fn unit_weight(_: &Vec<u8>) -> usize {
        1
    }

    #[test]
    fn concurrent_inserts_respect_entry_and_weight_capacity() {
        let store: Arc<ConcurrentWeightStore<u64, Vec<u8>, _>> =
            Arc::new(ConcurrentWeightStore::try_with_capacity(8, 8, unit_weight).unwrap());
        let num_threads = 10;
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let store = store.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    for i in 0..4 {
                        let key = (tid * 4 + i) as u64;
                        let _ = store.try_insert(key, Arc::new(vec![key as u8; 64]));
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        assert!(store.len() <= 8);
        assert!(store.total_weight() <= store.capacity_weight());
    }

    #[cfg(feature = "metrics")]
    mod metrics_tests {
        use super::*;

        #[test]
        fn clear_records_bulk_removes() {
            let store = ConcurrentWeightStore::try_with_capacity(4, 4, unit_weight).unwrap();
            for i in 0..4u64 {
                store.try_insert(i, Arc::new(vec![1u8; 8])).unwrap();
            }
            let before = store.metrics().removes;
            store.clear();
            assert!(store.metrics().removes >= before + 4);
        }
    }
}
