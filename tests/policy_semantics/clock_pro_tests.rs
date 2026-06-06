//! Clock-PRO bounded semantic tests.

use std::collections::HashSet;

use cachekit::policy::clock_pro::ClockProCache;
use cachekit::traits::Cache;
use proptest::prelude::*;

use crate::abstract_models::{Op, standard_capacity, standard_op_list};

fn run_ops(cache: &mut ClockProCache<u8, u8>, ops: &[Op<u8>]) {
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
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_clock_pro_residency(capacity in standard_capacity(), ops in standard_op_list()) {
        let mut cache = ClockProCache::new(capacity);
        run_ops(&mut cache, &ops);
    }
}

#[test]
fn smoke_clock_pro() {
    let ops = [
        Op::Insert(1),
        Op::Get(1),
        Op::Insert(2),
        Op::Insert(3),
        Op::Insert(4),
    ];
    let mut cache = ClockProCache::new(3);
    run_ops(&mut cache, &ops);
    let _: HashSet<u8> = (0..=255u8).filter(|k| cache.contains(k)).collect();
}
