//! FIFO dual-run semantic oracle tests.
//!
//! **Model:** `FifoModel` · **Op strategy:** `standard_op_list`
//! **Asserted:** residency, `peek_victim`, insert eviction

use cachekit::policy::fifo::FifoCache;
use cachekit::traits::{Cache, EvictingCache};
use proptest::prelude::*;

use crate::abstract_models::driver::{assert_peek_victim, probe_resident};
use crate::abstract_models::exact::fifo::FifoModel;
use crate::abstract_models::{Op, PolicyModel, standard_capacity, standard_op_list};

fn run_ops(cache: &mut FifoCache<u8, u8>, model: &mut FifoModel<u8>, ops: &[Op<u8>]) {
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
        let resident = probe_resident(|k| cache.contains(k));
        assert_eq!(resident, step.resident, "after {op:?}");
        assert!(cache.len() <= cache.capacity());
        if let Some(e) = step.evicted_on_insert {
            assert!(!cache.contains(&e));
        }
        assert_peek_victim(cache, model);
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_fifo_matches_model(capacity in standard_capacity(), ops in standard_op_list()) {
        let mut cache = FifoCache::new(capacity);
        let mut model = FifoModel::new(capacity);
        run_ops(&mut cache, &mut model, &ops);
    }
}

#[test]
fn smoke_fifo() {
    let ops = [
        Op::Insert(1),
        Op::Insert(2),
        Op::Insert(3),
        Op::Insert(4),
        Op::Get(1),
        Op::EvictOne,
        Op::Remove(2),
    ];
    let mut cache = FifoCache::new(3);
    let mut model = FifoModel::new(3);
    run_ops(&mut cache, &mut model, &ops);
}
