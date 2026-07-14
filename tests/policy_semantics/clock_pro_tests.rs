//! Clock-PRO invariant-only semantic tests (no `PolicyModel` dual-run).
//!
//! **Model:** none · **Op strategy:** `standard_op_list` (`GetMut`/`Touch`/`EvictOne` no-op)
//! **Asserted:** `len <= capacity`

use cachekit::policy::clock_pro::ClockProCache;
use cachekit::traits::Cache;
use proptest::prelude::*;

use crate::abstract_models::driver::run_invariant_trace;
use crate::abstract_models::{Op, standard_capacity, standard_op_list};

fn apply_clock_pro_op(cache: &mut ClockProCache<u8, u8>, op: Op<u8>) {
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
        Op::Remove(k) => {
            cache.remove(&k);
        },
        Op::GetMut(_) | Op::Touch(_) | Op::EvictOne => {},
    }
}

fn run_ops(cache: &mut ClockProCache<u8, u8>, ops: &[Op<u8>]) {
    run_invariant_trace(cache, ops, apply_clock_pro_op, |_| {});
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
}
