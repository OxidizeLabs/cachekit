//! S3-FIFO invariant-only semantic tests (no `PolicyModel` dual-run).
//!
//! **Model:** none · **Op strategy:** `op_strategy_with_get_mut` (0..80)
//! **Asserted:** `len <= capacity`, `check_invariants`, smoke residency bound

use cachekit::policy::s3_fifo::S3FifoCache;
use proptest::prelude::*;

use crate::abstract_models::driver::{probe_resident, run_invariant_trace};
use crate::abstract_models::{Op, op_strategy_with_get_mut, standard_capacity};

fn apply_s3_fifo_op(cache: &mut S3FifoCache<u8, u8>, op: Op<u8>) {
    match op {
        Op::Insert(k) => {
            cache.insert(k, k);
        },
        Op::Get(k) => {
            let _ = cache.get(&k);
        },
        Op::Peek(k) => {
            let _ = cache.peek(&k);
        },
        Op::GetMut(k) => {
            let _ = cache.get_mut(&k);
        },
        Op::Touch(_) | Op::EvictOne => {},
        Op::Remove(k) => {
            cache.remove(&k);
        },
    }
}

fn run_ops(cache: &mut S3FifoCache<u8, u8>, ops: &[Op<u8>]) {
    run_invariant_trace(cache, ops, apply_s3_fifo_op, |cache| {
        #[cfg(debug_assertions)]
        cache.check_invariants().expect("s3-fifo invariants");
    });
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
        Op::GetMut(1),
        Op::Insert(2),
        Op::Insert(3),
        Op::Insert(4),
    ];
    let mut cache = S3FifoCache::new(3);
    run_ops(&mut cache, &ops);
    let resident = probe_resident(|k| cache.contains(k));
    assert!(resident.len() <= cache.capacity());
}
