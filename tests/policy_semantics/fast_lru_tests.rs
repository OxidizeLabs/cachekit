//! Fast-LRU dual-run semantic oracle tests.
//!
//! **Model:** `LruOccupancyModel` · **Op strategy:** `op_strategy_with_get_mut` (0..120)
//! **Asserted:** residency, `peek_victim`, recency rank

use cachekit::policy::fast_lru::FastLru;
use cachekit::traits::{EvictingCache, VictimInspectable};
use proptest::prelude::*;

use crate::abstract_models::driver::{assert_peek_victim, assert_recency_rank, probe_resident};
use crate::abstract_models::exact::lru::LruOccupancyModel;
use crate::abstract_models::{Op, PolicyModel, op_strategy_with_get_mut, standard_capacity};

fn run_ops(cache: &mut FastLru<u8, u8>, model: &mut LruOccupancyModel<u8>, ops: &[Op<u8>]) {
    for op in ops {
        let rank_before = match op {
            Op::Peek(k) => cache.recency_rank(k),
            _ => None,
        };

        let step = model.apply(op.clone());

        match op {
            Op::Insert(k) => {
                cache.insert(*k, *k);
            },
            Op::Get(k) => {
                let _ = cache.get(k);
            },
            Op::Peek(k) => {
                let _ = cache.peek(k);
            },
            Op::GetMut(k) => {
                let _ = cache.get_mut(k);
            },
            Op::Touch(k) => {
                cache.touch(k);
            },
            Op::Remove(k) => {
                cache.remove(k);
            },
            Op::EvictOne => {
                if cache.peek_victim().is_some() {
                    let _ = cache.evict_one();
                }
            },
        }

        let resident = probe_resident(|k| cache.contains(k));
        assert_eq!(resident, step.resident, "after {op:?}");
        assert!(cache.len() <= cache.capacity());

        if let Op::Peek(k) = op {
            if let Some(rank) = rank_before {
                assert_eq!(cache.recency_rank(k), Some(rank));
            }
        }

        if matches!(op, Op::Get(_) | Op::GetMut(_) | Op::Touch(_)) {
            let key = match op {
                Op::Get(k) | Op::GetMut(k) | Op::Touch(k) => *k,
                _ => unreachable!(),
            };
            assert_recency_rank(cache, model.model_recency_rank(&key), &key);
        }

        if let Some(evicted) = step.evicted_on_insert {
            assert!(!cache.contains(&evicted));
        }

        assert_peek_victim(cache, model);
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_fast_lru_matches_model(
        capacity in standard_capacity(),
        ops in prop::collection::vec(op_strategy_with_get_mut(), 0..120),
    ) {
        let mut cache = FastLru::new(capacity);
        let mut model = LruOccupancyModel::new(capacity);
        run_ops(&mut cache, &mut model, &ops);
    }
}

#[test]
fn smoke_fast_lru() {
    let ops = [
        Op::Insert(1),
        Op::Insert(2),
        Op::GetMut(1),
        Op::Peek(2),
        Op::Insert(3),
        Op::Insert(4),
        Op::EvictOne,
    ];
    let mut cache = FastLru::new(3);
    let mut model = LruOccupancyModel::new(3);
    run_ops(&mut cache, &mut model, &ops);
}
