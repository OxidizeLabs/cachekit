//! CAR invariant-only semantic tests (no `PolicyModel` dual-run).
//!
//! **Model:** none · **Op strategy:** `standard_op_list` (`GetMut`/`Touch`/`EvictOne` no-op)
//! **Asserted:** `len <= capacity`, `CarCore::debug_validate_invariants`

use cachekit::policy::car::CarCore;
use cachekit::traits::Cache;
use proptest::prelude::*;

use crate::abstract_models::driver::probe_resident;
use crate::abstract_models::{Op, standard_capacity, standard_op_list};

fn run_ops(cache: &mut CarCore<u8, u8>, ops: &[Op<u8>]) {
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
    fn prop_car_invariants(capacity in standard_capacity(), ops in standard_op_list()) {
        let mut cache = CarCore::new(capacity);
        run_ops(&mut cache, &ops);
    }
}

#[test]
fn smoke_car() {
    let ops = [Op::Insert(1), Op::Get(1), Op::Insert(2), Op::Insert(3)];
    let mut cache = CarCore::new(4);
    run_ops(&mut cache, &ops);
    let _ = probe_resident(|k| cache.contains(k));
}
