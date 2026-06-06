//! MRU semantic oracle tests.

use std::collections::HashSet;

use cachekit::policy::mru::MruCore;
use cachekit::traits::EvictingCache;
use proptest::prelude::*;

use crate::abstract_models::exact::mru::MruModel;
use crate::abstract_models::{Op, PolicyModel, standard_capacity, standard_op_list};

fn run_ops(cache: &mut MruCore<u8, u8>, model: &mut MruModel<u8>, ops: &[Op<u8>]) {
    for op in ops {
        let before: HashSet<u8> = (0..=255u8).filter(|k| cache.contains(k)).collect();
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
            Op::GetMut(_) | Op::Touch(_) => {},
            Op::EvictOne => {
                let _ = cache.evict_one();
            },
            Op::Remove(k) => {
                cache.remove(k);
            },
        }
        let after: HashSet<u8> = (0..=255u8).filter(|k| cache.contains(k)).collect();
        assert_eq!(after, step.resident, "after {op:?}");
        if let (Op::Insert(k), Some(evicted)) = (&op, &step.evicted_on_insert) {
            if !before.contains(k) {
                assert!(!after.contains(evicted));
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_mru_matches_model(capacity in standard_capacity(), ops in standard_op_list()) {
        let mut cache = MruCore::new(capacity);
        let mut model = MruModel::new(capacity);
        run_ops(&mut cache, &mut model, &ops);
    }
}

#[test]
fn smoke_mru() {
    let ops = [
        Op::Insert(1),
        Op::Get(1),
        Op::Insert(2),
        Op::Insert(3),
        Op::Insert(4),
    ];
    let mut cache = MruCore::new(3);
    let mut model = MruModel::new(3);
    run_ops(&mut cache, &mut model, &ops);
}
