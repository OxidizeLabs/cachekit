//! LFU dual-run semantic oracle tests.
//!
//! **Model:** `LfuModel` · **Op strategy:** `standard_op_list`
//! **Asserted:** residency, frequency, `peek_victim`
//!
//! Cross-model: `NaiveLfuModel` vs `LfuModel`. naive ≠ exact → fix spec or model; naive = exact
//! but impl fails → fix implementation or adapter.

use std::sync::Arc;

use cachekit::policy::lfu::LfuCache;
use cachekit::traits::{Cache, EvictingCache};
use proptest::prelude::*;

use crate::abstract_models::driver::{assert_dual_run_step, assert_models_agree};
use crate::abstract_models::exact::lfu::LfuModel;
use crate::abstract_models::reference::lfu::NaiveLfuModel;
use crate::abstract_models::{Op, PolicyModel, standard_capacity, standard_op_list};

fn run_ops(cache: &mut LfuCache<u8, u8>, model: &mut LfuModel<u8>, ops: &[Op<u8>]) {
    for op in ops {
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
                cache.increment_frequency(k);
            },
            Op::Remove(k) => {
                cache.remove(k);
            },
            Op::EvictOne => {
                let _ = cache.evict_one();
            },
        }
        assert_dual_run_step(
            cache,
            model,
            &step,
            |k| cache.contains(k),
            |cache, model, step| {
                for k in &step.resident {
                    assert_eq!(cache.frequency(k), model.frequency(k));
                }
            },
        );
    }
}

#[test]
fn hand_written_lfu_fifo_tie_break() {
    let mut cache = LfuCache::new(3);
    let mut model = LfuModel::new(3);
    let ops = [Op::Insert(1), Op::Insert(2), Op::Insert(3), Op::Insert(4)];
    run_ops(&mut cache, &mut model, &ops);
    assert!(!cache.contains(&1));
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_lfu_naive_matches_current_model(
        capacity in standard_capacity(),
        ops in standard_op_list(),
    ) {
        let mut naive = NaiveLfuModel::new(capacity);
        let mut current = LfuModel::new(capacity);
        assert_models_agree(&mut naive, &mut current, &ops);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_lfu_matches_model(capacity in standard_capacity(), ops in standard_op_list()) {
        let mut cache = LfuCache::new(capacity);
        let mut model = LfuModel::new(capacity);
        run_ops(&mut cache, &mut model, &ops);
    }
}

#[test]
fn smoke_lfu_naive_agreement() {
    let ops = [Op::Insert(1), Op::Insert(2), Op::Insert(3), Op::Insert(4)];
    let mut naive = NaiveLfuModel::new(3);
    let mut current = LfuModel::new(3);
    assert_models_agree(&mut naive, &mut current, &ops);
}

#[test]
fn lfu_naive_agreement_uses_bucket_arrival_for_ties() {
    let ops = [
        Op::Insert(1),
        Op::Insert(2),
        Op::Touch(2),
        Op::Touch(1),
        Op::EvictOne,
    ];
    let mut naive = NaiveLfuModel::new(2);
    let mut current = LfuModel::new(2);
    assert_models_agree(&mut naive, &mut current, &ops);
}

#[test]
fn smoke_lfu() {
    let ops = [
        Op::Insert(1),
        Op::Get(1),
        Op::Insert(2),
        Op::Insert(3),
        Op::Insert(4),
    ];
    let mut cache = LfuCache::new(3);
    let mut model = LfuModel::new(3);
    run_ops(&mut cache, &mut model, &ops);
}
