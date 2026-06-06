//! 2Q dual-run semantic oracle tests.
//!
//! **Model:** `TwoQModel` · **Op strategy:** `standard_op_list_no_evict`
//! **Asserted:** residency only (no `EvictingCache`)

use cachekit::policy::two_q::TwoQCore;
use proptest::prelude::*;

use crate::abstract_models::driver::probe_resident;
use crate::abstract_models::exact::two_q::TwoQModel;
use crate::abstract_models::{Op, PolicyModel, standard_capacity, standard_op_list_no_evict};

const PROBATION_FRAC: f64 = 0.25;

fn run_ops(cache: &mut TwoQCore<u8, u8>, model: &mut TwoQModel<u8>, ops: &[Op<u8>]) {
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
        let resident = probe_resident(|k| cache.contains(k));
        assert_eq!(resident, step.resident, "after {op:?}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_two_q_matches_model(capacity in standard_capacity(), ops in standard_op_list_no_evict()) {
        let mut cache = TwoQCore::new(capacity, PROBATION_FRAC);
        let mut model = TwoQModel::new(capacity, PROBATION_FRAC);
        run_ops(&mut cache, &mut model, &ops);
    }
}

#[test]
fn smoke_two_q() {
    let ops = [
        Op::Insert(1),
        Op::Get(1),
        Op::Insert(2),
        Op::Insert(3),
        Op::Insert(4),
    ];
    let mut cache = TwoQCore::new(5, 0.4);
    let mut model = TwoQModel::new(5, 0.4);
    run_ops(&mut cache, &mut model, &ops);
}
