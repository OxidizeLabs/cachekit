//! MFU semantic oracle tests.

use std::collections::HashSet;

use cachekit::policy::mfu::MfuCore;
use proptest::prelude::*;

use crate::abstract_models::exact::mfu::MfuModel;
use crate::abstract_models::{Op, PolicyModel, standard_capacity, standard_op_list_no_evict};

fn run_ops(cache: &mut MfuCore<u8, u8>, model: &mut MfuModel<u8>, ops: &[Op<u8>]) {
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
            Op::EvictOne => {},
        }
        let resident: HashSet<u8> = (0..=255u8).filter(|k| cache.contains(k)).collect();
        assert_eq!(resident, step.resident, "after {op:?}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_mfu_matches_model(capacity in standard_capacity(), ops in standard_op_list_no_evict()) {
        let mut cache = MfuCore::new(capacity);
        let mut model = MfuModel::new(capacity);
        run_ops(&mut cache, &mut model, &ops);
    }
}

#[test]
fn smoke_mfu_reinsert_after_update() {
    let ops = [Op::Insert(140), Op::Insert(140), Op::Insert(0)];
    let mut cache = MfuCore::new(1);
    let mut model = MfuModel::new(1);
    run_ops(&mut cache, &mut model, &ops);
}

#[test]
fn smoke_mfu() {
    let ops = [
        Op::Insert(1),
        Op::Get(1),
        Op::Get(1),
        Op::Insert(2),
        Op::Insert(3),
        Op::Insert(4),
    ];
    let mut cache = MfuCore::new(3);
    let mut model = MfuModel::new(3);
    run_ops(&mut cache, &mut model, &ops);
}
