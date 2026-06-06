//! Dual-implementation equivalence tests.

use std::collections::HashSet;
use std::sync::Arc;

#[cfg(feature = "policy-clock")]
use cachekit::ds::ClockRing;
#[cfg(feature = "policy-clock")]
use cachekit::policy::clock::ClockCache;
#[cfg(feature = "policy-fast-lru")]
use cachekit::policy::fast_lru::FastLru;
#[cfg(feature = "policy-lru")]
use cachekit::policy::lru::LruCore;
use cachekit::traits::{Cache, EvictingCache, VictimInspectable};

use crate::abstract_models::{Op, op_strategy};

#[cfg(all(feature = "policy-lru", feature = "policy-fast-lru"))]
#[test]
fn dual_lru_core_vs_fast_lru() {
    let ops = [
        Op::Insert(1),
        Op::Insert(2),
        Op::Get(1),
        Op::Peek(2),
        Op::Touch(1),
        Op::Insert(3),
        Op::Insert(4),
        Op::EvictOne,
        Op::Remove(2),
    ];
    let mut lru = LruCore::<u8, u8>::new(3);
    let mut fast = FastLru::new(3);

    for op in &ops {
        match op {
            Op::Insert(k) => {
                lru.insert(*k, Arc::new(*k));
                fast.insert(*k, *k);
            },
            Op::Get(k) => {
                lru.get(k);
                fast.get(k);
            },
            Op::Peek(k) => {
                lru.peek(k);
                fast.peek(k);
            },
            Op::Touch(k) => {
                lru.touch(k);
                fast.touch(k);
            },
            Op::Remove(k) => {
                lru.remove(k);
                fast.remove(k);
            },
            Op::EvictOne => {
                let _ = lru.evict_one();
                let _ = fast.evict_one();
            },
            Op::GetMut(_) => {},
        }

        for k in 0..=255u8 {
            assert_eq!(
                lru.contains(&k),
                fast.contains(&k),
                "contains {k} after {op:?}"
            );
        }
        assert_eq!(
            lru.peek_victim().map(|(k, _)| *k),
            fast.peek_victim().map(|(k, _)| *k)
        );
        if let (Some(lru_rank), Some(fast_rank)) = (lru.recency_rank(&1), fast.recency_rank(&1)) {
            if lru.contains(&1) {
                assert_eq!(lru_rank, fast_rank);
            }
        }
    }
}

#[cfg(feature = "policy-clock")]
#[test]
fn dual_clock_cache_vs_clock_ring() {
    let ops = [
        Op::Insert(1),
        Op::Get(1),
        Op::Insert(2),
        Op::Insert(3),
        Op::Insert(4),
    ];
    let mut cache = ClockCache::<u8, ()>::new(3);
    let mut ring = ClockRing::new(3);

    for op in &ops {
        match op {
            Op::Insert(k) => {
                cache.insert(*k, ());
                ring.insert(*k, ());
            },
            Op::Get(k) => {
                cache.get(k);
                let _ = ring.get(k);
            },
            Op::Peek(k) => {
                cache.peek(k);
                ring.peek(k);
            },
            Op::Remove(k) => {
                cache.remove(k);
                ring.remove(k);
            },
            _ => {},
        }
        let cache_r: HashSet<u8> = (0..=255u8).filter(|k| cache.contains(k)).collect();
        let ring_r: HashSet<u8> = ring.keys().copied().collect();
        assert_eq!(cache_r, ring_r, "after {op:?}");
    }
}

#[cfg(all(feature = "policy-lru", feature = "policy-fast-lru"))]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_dual_lru_observables(ops in prop::collection::vec(op_strategy(), 0..60)) {
            let mut lru = LruCore::<u8, u8>::new(8);
            let mut fast = FastLru::new(8);
            for op in ops {
                match op {
                    Op::Insert(k) => {
                        lru.insert(k, Arc::new(k));
                        fast.insert(k, k);
                    },
                    Op::Get(k) => { lru.get(&k); fast.get(&k); },
                    Op::Peek(k) => { lru.peek(&k); fast.peek(&k); },
                    Op::Touch(k) => { lru.touch(&k); fast.touch(&k); },
                    Op::Remove(k) => { lru.remove(&k); fast.remove(&k); },
                    Op::EvictOne => {
                        let _ = lru.evict_one();
                        let _ = fast.evict_one();
                    },
                    Op::GetMut(k) => { fast.get_mut(&k); },
                }
                for key in 0..=255u8 {
                    assert_eq!(lru.contains(&key), fast.contains(&key));
                }
            }
        }
    }
}
