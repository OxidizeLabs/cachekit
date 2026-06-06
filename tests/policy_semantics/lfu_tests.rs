//! LFU dual-run semantic oracle tests.
//!
//! **Model:** `LfuModel` · **Op strategy:** `standard_op_list`
//! **Asserted:** residency, frequency, `peek_victim`

use std::sync::Arc;

use cachekit::policy::lfu::LfuCache;
use cachekit::traits::{Cache, EvictingCache};
use proptest::prelude::*;

use crate::abstract_models::driver::{assert_peek_victim, probe_resident};
use crate::abstract_models::exact::lfu::LfuModel;
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
        let resident = probe_resident(|k| cache.contains(k));
        assert_eq!(resident, step.resident);
        for k in &resident {
            assert_eq!(cache.frequency(k), model.frequency(k));
        }
        assert_peek_victim(cache, model);
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
    fn prop_lfu_matches_model(capacity in standard_capacity(), ops in standard_op_list()) {
        let mut cache = LfuCache::new(capacity);
        let mut model = LfuModel::new(capacity);
        run_ops(&mut cache, &mut model, &ops);
    }
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
