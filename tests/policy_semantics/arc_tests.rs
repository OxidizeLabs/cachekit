//! ARC bounded semantic tests.

use std::collections::HashSet;

use cachekit::policy::arc::ArcCore;
use cachekit::traits::Cache;
use proptest::prelude::*;

use crate::abstract_models::{Op, standard_capacity, standard_op_list};

fn run_ops(cache: &mut ArcCore<u8, u8>, ops: &[Op<u8>]) {
    for op in ops {
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
            Op::Remove(k) => {
                cache.remove(k);
            },
            Op::GetMut(_) | Op::Touch(_) | Op::EvictOne => {},
        }
        assert!(cache.len() <= cache.capacity());
        cache.debug_validate_invariants();
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_arc_invariants(capacity in standard_capacity(), ops in standard_op_list()) {
        let mut cache = ArcCore::new(capacity);
        run_ops(&mut cache, &ops);
    }
}

#[test]
fn smoke_arc() {
    let ops = [Op::Insert(1), Op::Get(1), Op::Insert(2), Op::Insert(3)];
    let mut cache = ArcCore::new(4);
    run_ops(&mut cache, &ops);
    let _: HashSet<u8> = (0..=255u8).filter(|k| cache.contains(k)).collect();
}
