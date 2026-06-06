//! Heap-LFU dual-run semantic oracle tests.
//!
//! **Model:** `HeapLfuModel` · **Op strategy:** `standard_op_list`
//! **Asserted:** residency only (heap rebuild handles staleness)
//!
//! Cross-model: `NaiveHeapLfuModel` vs `HeapLfuModel`. naive ≠ exact → fix spec or model; naive = exact
//! but impl fails → fix implementation or adapter.

use std::sync::Arc;

use cachekit::policy::heap_lfu::HeapLfuCache;
use cachekit::traits::{Cache, EvictingCache};
use proptest::prelude::*;

use crate::abstract_models::driver::{assert_dual_run_step_no_victim, assert_models_agree};
use crate::abstract_models::exact::heap_lfu::HeapLfuModel;
use crate::abstract_models::reference::heap_lfu::NaiveHeapLfuModel;
use crate::abstract_models::{Op, PolicyModel, standard_capacity, standard_op_list};

fn run_ops(cache: &mut HeapLfuCache<u8, u8>, model: &mut HeapLfuModel<u8>, ops: &[Op<u8>]) {
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
        assert_dual_run_step_no_victim(cache, model, &step, |k| cache.contains(k), |_, _, _| {});
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_heap_lfu_naive_matches_current_model(
        capacity in standard_capacity(),
        ops in standard_op_list(),
    ) {
        let mut naive = NaiveHeapLfuModel::new(capacity);
        let mut current = HeapLfuModel::new(capacity);
        assert_models_agree(&mut naive, &mut current, &ops);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_heap_lfu_matches_model(capacity in standard_capacity(), ops in standard_op_list()) {
        let mut cache = HeapLfuCache::new(capacity);
        let mut model = HeapLfuModel::new(capacity);
        run_ops(&mut cache, &mut model, &ops);
    }
}

#[test]
fn smoke_heap_lfu_naive_agreement() {
    let ops = [
        Op::Insert(1),
        Op::Insert(2),
        Op::Get(1),
        Op::Insert(3),
        Op::Insert(4),
    ];
    let mut naive = NaiveHeapLfuModel::new(3);
    let mut current = HeapLfuModel::new(3);
    assert_models_agree(&mut naive, &mut current, &ops);
}

#[test]
fn smoke_heap_lfu() {
    let ops = [
        Op::Insert(1),
        Op::Insert(2),
        Op::Get(1),
        Op::Insert(3),
        Op::Insert(4),
    ];
    let mut cache = HeapLfuCache::new(3);
    let mut model = HeapLfuModel::new(3);
    run_ops(&mut cache, &mut model, &ops);
}
