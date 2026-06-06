//! SLRU semantic oracle tests.

use std::collections::HashSet;

use cachekit::policy::slru::SlruCore;
use proptest::prelude::*;

use crate::abstract_models::exact::slru::SlruModel;
use crate::abstract_models::{Op, PolicyModel, standard_capacity, standard_op_list_no_evict};

const PROBATIONARY_FRAC: f64 = 0.25;

fn run_ops(cache: &mut SlruCore<u8, u8>, model: &mut SlruModel<u8>, ops: &[Op<u8>]) {
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
            Op::GetMut(_) | Op::Touch(_) | Op::EvictOne => {},
            Op::Remove(k) => {
                cache.remove(k);
            },
        }
        let resident: HashSet<u8> = (0..=255u8).filter(|k| cache.contains(k)).collect();
        assert_eq!(resident, step.resident);
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_slru_matches_model(capacity in standard_capacity(), ops in standard_op_list_no_evict()) {
        let mut cache = SlruCore::new(capacity, PROBATIONARY_FRAC);
        let mut model = SlruModel::new(capacity, PROBATIONARY_FRAC);
        run_ops(&mut cache, &mut model, &ops);
    }
}

#[test]
fn smoke_slru() {
    let ops = [Op::Insert(1), Op::Get(1), Op::Insert(2), Op::Insert(3)];
    let mut cache = SlruCore::new(4, 0.25);
    let mut model = SlruModel::new(4, 0.25);
    run_ops(&mut cache, &mut model, &ops);
}
