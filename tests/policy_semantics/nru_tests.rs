//! NRU dual-run semantic oracle tests.
//!
//! **Model:** `NruModel` · **Op strategy:** `short_op_list_no_evict`
//! **Asserted:** residency only (O(n) eviction; no `EvictingCache`)

use cachekit::policy::nru::NruCache;
use cachekit::traits::Cache;
use proptest::prelude::*;

use crate::abstract_models::driver::probe_resident;
use crate::abstract_models::exact::nru::NruModel;
use crate::abstract_models::{Op, PolicyModel, short_op_list_no_evict, standard_capacity};

fn run_ops(cache: &mut NruCache<u8, u8>, model: &mut NruModel<u8>, ops: &[Op<u8>]) {
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
        assert_eq!(resident, step.resident);
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_nru_matches_model(capacity in standard_capacity(), ops in short_op_list_no_evict()) {
        let mut cache = NruCache::new(capacity);
        let mut model = NruModel::new(capacity);
        run_ops(&mut cache, &mut model, &ops);
    }
}

#[test]
fn smoke_nru() {
    let ops = [
        Op::Insert(1),
        Op::Insert(2),
        Op::Get(1),
        Op::Insert(3),
        Op::Insert(4),
    ];
    let mut cache = NruCache::new(3);
    let mut model = NruModel::new(3);
    run_ops(&mut cache, &mut model, &ops);
}
