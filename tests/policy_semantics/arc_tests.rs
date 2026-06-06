//! ARC invariant-only semantic tests (no `PolicyModel` dual-run).
//!
//! **Model:** none · **Op strategy:** `standard_op_list` (`GetMut`/`Touch`/`EvictOne` no-op)
//! **Asserted:** `len <= capacity`, `debug_validate_invariants`

use cachekit::policy::arc::ArcCore;
use cachekit::traits::Cache;
use proptest::prelude::*;

use crate::abstract_models::driver::run_invariant_trace;
use crate::abstract_models::{Op, standard_capacity, standard_op_list};

fn apply_arc_op(cache: &mut ArcCore<u8, u8>, op: Op<u8>) {
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

fn run_ops(cache: &mut ArcCore<u8, u8>, ops: &[Op<u8>]) {
    run_invariant_trace(cache, ops, apply_arc_op, |cache| {
        cache.debug_validate_invariants();
    });
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
    let ops = [Op::Insert(1), Op::Insert(2), Op::Get(1), Op::Insert(3)];
    let mut cache = ArcCore::new(2);
    run_ops(&mut cache, &ops);
}
