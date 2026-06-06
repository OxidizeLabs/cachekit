//! LIFO dual-run semantic oracle tests.
//!
//! **Model:** `LifoModel` · **Op strategy:** `standard_op_list`
//! **Asserted:** residency, `peek_victim`
//!
//! Cross-model: `NaiveLifoModel` vs `LifoModel`. naive ≠ exact → fix spec or model; naive = exact
//! but impl fails → fix implementation or adapter.

use cachekit::policy::lifo::LifoCore;
use cachekit::traits::{Cache, EvictingCache};
use proptest::prelude::*;

use crate::abstract_models::driver::{assert_dual_run_step, assert_models_agree};
use crate::abstract_models::exact::lifo::LifoModel;
use crate::abstract_models::reference::lifo::NaiveLifoModel;
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
        assert_dual_run_step(cache, model, &step, |k| cache.contains(k), |_, _, _| {});
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_lifo_naive_matches_current_model(
        capacity in standard_capacity(),
        ops in standard_op_list(),
    ) {
        let mut naive = NaiveLifoModel::new(capacity);
        let mut current = LifoModel::new(capacity);
        assert_models_agree(&mut naive, &mut current, &ops);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_lifo_matches_model(capacity in standard_capacity(), ops in standard_op_list()) {
        let mut cache = LifoCore::new(capacity);
        let mut model = LifoModel::new(capacity);
        run_ops(&mut cache, &mut model, &ops);
    }
}

#[test]
fn smoke_lifo_naive_agreement() {
    let ops = [
        Op::Insert(1),
        Op::Insert(2),
        Op::Insert(3),
        Op::Insert(4),
        Op::EvictOne,
        Op::Get(2),
        Op::Remove(2),
    ];
    let mut naive = NaiveLifoModel::new(3);
    let mut current = LifoModel::new(3);
    assert_models_agree(&mut naive, &mut current, &ops);
}

#[test]
fn smoke_lifo() {
    let ops = [
        Op::Insert(1),
        Op::Insert(2),
        Op::Insert(3),
        Op::Insert(4),
        Op::EvictOne,
        Op::Get(2),
        Op::Remove(2),
    ];
    let mut cache = LifoCore::new(3);
    let mut model = LifoModel::new(3);
    run_ops(&mut cache, &mut model, &ops);
}
