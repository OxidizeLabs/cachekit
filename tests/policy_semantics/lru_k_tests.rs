//! LRU-K dual-run semantic oracle tests.
//!
//! **Model:** `LruKModel` · **Op strategy:** `standard_op_list`
//! **Asserted:** residency, `peek_victim`, access count
//!
//! Cross-model: `NaiveLruKModel` vs `LruKModel`. naive ≠ exact → fix spec or model; naive = exact
//! but impl fails → fix implementation or adapter.

use cachekit::policy::lru_k::LrukCache;
use cachekit::traits::{Cache, EvictingCache};
use proptest::prelude::*;

use crate::abstract_models::driver::{assert_dual_run_step, assert_models_agree};
use crate::abstract_models::exact::lru_k::LruKModel;
use crate::abstract_models::reference::lru_k::NaiveLruKModel;
use crate::abstract_models::{Op, PolicyModel, standard_capacity, standard_op_list};

const K: usize = 2;

fn run_ops(cache: &mut LrukCache<u8, u8>, model: &mut LruKModel<u8>, ops: &[Op<u8>]) {
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
        assert_dual_run_step(
            cache,
            model,
            &step,
            |k| cache.contains(k),
            |cache, model, step| {
                for key in &step.resident {
                    assert_eq!(cache.access_count(key), model.access_count(key));
                }
            },
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_lru_k_naive_matches_current_model(
        capacity in standard_capacity(),
        ops in standard_op_list(),
    ) {
        let mut naive = NaiveLruKModel::new(capacity, K);
        let mut current = LruKModel::new(capacity, K);
        assert_models_agree(&mut naive, &mut current, &ops);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_lru_k_matches_model(capacity in standard_capacity(), ops in standard_op_list()) {
        let mut cache = LrukCache::with_k(capacity, K);
        let mut model = LruKModel::new(capacity, K);
        run_ops(&mut cache, &mut model, &ops);
    }
}

#[test]
fn smoke_lru_k_naive_agreement() {
    let ops = [
        Op::Insert(1),
        Op::Get(1),
        Op::Insert(2),
        Op::Insert(3),
        Op::Insert(4),
    ];
    let mut naive = NaiveLruKModel::new(3, K);
    let mut current = LruKModel::new(3, K);
    assert_models_agree(&mut naive, &mut current, &ops);
}

#[test]
fn smoke_lru_k() {
    let ops = [
        Op::Insert(1),
        Op::Get(1),
        Op::Insert(2),
        Op::Insert(3),
        Op::Insert(4),
    ];
    let mut cache = LrukCache::with_k(3, K);
    let mut model = LruKModel::new(3, K);
    run_ops(&mut cache, &mut model, &ops);
}
