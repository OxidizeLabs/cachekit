//! LIFO semantic oracle tests.

use std::collections::HashSet;

use cachekit::policy::lifo::LifoCore;
use cachekit::traits::{Cache, EvictingCache};
use proptest::prelude::*;

use crate::abstract_models::driver::assert_peek_victim;
use crate::abstract_models::exact::lifo::LifoModel;
use crate::abstract_models::{Op, PolicyModel, standard_capacity, standard_op_list};

fn run_ops(cache: &mut LifoCore<u8, u8>, model: &mut LifoModel<u8>, ops: &[Op<u8>]) {
    for op in ops {
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
            Op::Remove(k) => {
                cache.remove(k);
            },
            Op::EvictOne => {
                let _ = cache.evict_one();
            },
        }
        let resident: HashSet<u8> = (0..=255u8).filter(|k| cache.contains(k)).collect();
        assert_eq!(resident, step.resident);
        assert_peek_victim(cache, model);
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_lifo_matches_model(capacity in standard_capacity(), ops in standard_op_list()) {
        let mut cache = LifoCore::new(capacity);
        let mut model = LifoModel::new(capacity);
        run_ops(&mut cache, &mut model, &ops);
    }
}

#[test]
fn smoke_lifo() {
    let ops = [
        Op::Insert(1),
        Op::Insert(2),
        Op::Insert(3),
        Op::Insert(4),
        Op::EvictOne,
    ];
    let mut cache = LifoCore::new(3);
    let mut model = LifoModel::new(3);
    run_ops(&mut cache, &mut model, &ops);
}
