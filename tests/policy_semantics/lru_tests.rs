//! LRU (`LruCore`) dual-run semantic oracle tests.
//!
//! **Model:** `LruOccupancyModel` · **Op strategy:** `standard_op_list`
//! **Asserted:** residency, `peek_victim`, recency rank (peek must not promote)

use std::sync::Arc;

use cachekit::policy::lru::LruCore;
use cachekit::traits::{Cache, EvictingCache, VictimInspectable};
use proptest::prelude::*;

use crate::abstract_models::driver::{assert_peek_victim, assert_recency_rank, probe_resident};
use crate::abstract_models::exact::lru::LruOccupancyModel;
use crate::abstract_models::{Op, PolicyModel, standard_capacity, standard_op_list};

fn run_ops(cache: &mut LruCore<u8, u8>, model: &mut LruOccupancyModel<u8>, ops: &[Op<u8>]) {
    for op in ops {
        let rank_before = match op {
            Op::Peek(k) => cache.recency_rank(k),
            _ => None,
        };

        let step = model.apply(op.clone());

        match op {
            Op::Insert(k) => {
                cache.insert(*k, Arc::new(*k));
            },
            Op::Get(k) => {
                let _ = cache.get(k);
            },
            Op::Peek(k) => {
                let _ = cache.peek(k);
            },
            Op::GetMut(_) => {},
            Op::Touch(k) => {
                cache.touch(k);
            },
            Op::Remove(k) => {
                cache.remove(k);
            },
            Op::EvictOne => {
                if let Some((expected, _)) = cache.peek_victim() {
                    let key = *expected;
                    let evicted = cache.evict_one();
                    assert_eq!(evicted.map(|(k, _)| k), Some(key));
                } else {
                    assert!(cache.evict_one().is_none());
                }
            },
        }

        let resident = probe_resident(|k| cache.contains(k));
        assert_eq!(resident, step.resident, "residency mismatch after {op:?}");
        assert!(cache.len() <= cache.capacity());

        if let Op::Peek(k) = op {
            if let Some(rank) = rank_before {
                assert_eq!(cache.recency_rank(k), Some(rank));
            }
        }

        if matches!(op, Op::Get(_) | Op::Touch(_)) {
            if let Some(k) = op_key(op) {
                assert_recency_rank(cache, model.model_recency_rank(k), k);
            }
        }

        if let Some(evicted) = step.evicted_on_insert {
            assert!(!cache.contains(&evicted));
        }

        assert_peek_victim(cache, model);
    }
}

fn op_key(op: &Op<u8>) -> Option<&u8> {
    match op {
        Op::Get(k) | Op::Peek(k) | Op::Touch(k) => Some(k),
        _ => None,
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_lru_core_matches_model(
        capacity in standard_capacity(),
        ops in standard_op_list(),
    ) {
        let mut cache = LruCore::new(capacity);
        let mut model = LruOccupancyModel::new(capacity);
        run_ops(&mut cache, &mut model, &ops);
    }
}

#[test]
fn smoke_lru_core() {
    let ops = [
        Op::Insert(1),
        Op::Insert(2),
        Op::Insert(3),
        Op::Get(1),
        Op::Peek(2),
        Op::Touch(3),
        Op::Insert(4),
        Op::EvictOne,
        Op::Remove(2),
    ];
    let mut cache = LruCore::new(3);
    let mut model = LruOccupancyModel::new(3);
    run_ops(&mut cache, &mut model, &ops);
}
