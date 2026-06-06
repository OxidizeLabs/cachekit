//! S3-FIFO bounded semantic tests (residency + invariants).

use std::collections::HashSet;

use cachekit::policy::s3_fifo::S3FifoCache;
use proptest::prelude::*;

use crate::abstract_models::{Op, op_strategy_with_get_mut, standard_capacity};

fn run_ops(cache: &mut S3FifoCache<u8, u8>, ops: &[Op<u8>]) {
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
            Op::GetMut(k) => {
                let _ = cache.get_mut(k);
            },
            Op::Touch(_) | Op::EvictOne => {},
            Op::Remove(k) => {
                cache.remove(k);
            },
        }
        assert!(cache.len() <= cache.capacity());
        #[cfg(debug_assertions)]
        cache.check_invariants().expect("s3-fifo invariants");
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prop_s3_fifo_invariants(
        capacity in standard_capacity(),
        ops in prop::collection::vec(op_strategy_with_get_mut(), 0..80),
    ) {
        let mut cache = S3FifoCache::new(capacity);
        run_ops(&mut cache, &ops);
    }
}

#[test]
fn smoke_s3_fifo() {
    let ops = [
        Op::Insert(1),
        Op::Get(1),
        Op::Insert(2),
        Op::Insert(3),
        Op::Insert(4),
    ];
    let mut cache = S3FifoCache::new(3);
    run_ops(&mut cache, &ops);
    let resident: HashSet<u8> = (0..=255u8).filter(|k| cache.contains(k)).collect();
    assert!(resident.len() <= 3);
}
