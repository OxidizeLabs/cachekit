//! MRU dual-run semantic oracle tests.
//!
//! **Model:** `MruModel` · **Op strategy:** `standard_op_list`
//! **Asserted:** residency, insert eviction
//!
//! Cross-model: `NaiveMruModel` vs `MruModel`. naive ≠ exact → fix spec or model; naive = exact
//! but impl fails → fix implementation or adapter.

use cachekit::policy::mru::MruCore;
use cachekit::traits::EvictingCache;
use proptest::prelude::*;

use crate::abstract_models::driver::{
    assert_dual_run_step_no_victim, assert_models_agree, probe_resident,
};
use crate::abstract_models::exact::mru::MruModel;
use crate::abstract_models::reference::mru::NaiveMruModel;
use crate::abstract_models::{Op, PolicyModel, standard_capacity, standard_op_list};

fn run_ops(cache: &mut MruCore<u8, u8>, model: &mut MruModel<u8>, ops: &[Op<u8>]) {
    for op in ops {
        let before = probe_resident(|k| cache.contains(k));
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
        assert_dual_run_step_no_victim(
            cache,
            model,
            &step,
            |k| cache.contains(k),
            |cache, _, step| {
                if let (Op::Insert(k), Some(evicted)) = (op, &step.evicted_on_insert) {
                    if !before.contains(k) {
                        assert!(!cache.contains(evicted));
                    }
                }
            },
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_mru_naive_matches_current_model(
        capacity in standard_capacity(),
        ops in standard_op_list(),
    ) {
        let mut naive = NaiveMruModel::new(capacity);
        let mut current = MruModel::new(capacity);
        assert_models_agree(&mut naive, &mut current, &ops);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_mru_matches_model(capacity in standard_capacity(), ops in standard_op_list()) {
        let mut cache = MruCore::new(capacity);
        let mut model = MruModel::new(capacity);
        run_ops(&mut cache, &mut model, &ops);
    }
}

#[test]
fn smoke_mru_naive_agreement() {
    let ops = [
        Op::Insert(1),
        Op::Get(1),
        Op::Insert(2),
        Op::Insert(3),
        Op::Insert(4),
    ];
    let mut naive = NaiveMruModel::new(3);
    let mut current = MruModel::new(3);
    assert_models_agree(&mut naive, &mut current, &ops);
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
