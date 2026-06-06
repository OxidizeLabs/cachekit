//! Shared helpers for policy semantic test modules.

use std::collections::HashSet;

use cachekit::traits::{Cache, EvictingCache, VictimInspectable};

use crate::abstract_models::driver::assert_peek_victim;
use crate::abstract_models::{Op, PolicyModel};

pub fn assert_resident_u8<C: Cache<u8, ()>>(cache: &C, expected: &HashSet<u8>, op: &Op<u8>) {
    let resident: HashSet<u8> = (0..=255u8).filter(|k| cache.contains(k)).collect();
    assert_eq!(resident, *expected, "residency mismatch after {op:?}");
    assert!(cache.len() <= cache.capacity());
}

pub fn apply_evict_one<C>(cache: &mut C)
where
    C: EvictingCache<u8, ()> + VictimInspectable<u8, ()>,
{
    if cache.peek_victim().is_some() {
        let _ = cache.evict_one();
    }
}

pub fn finish_step<C, M>(cache: &C, model: &M, step: &crate::abstract_models::ModelStep<u8>, op: &Op<u8>)
where
    C: VictimInspectable<u8, ()>,
    M: PolicyModel<u8>,
{
    assert_resident_u8(cache, &step.resident, op);
    if let Some(evicted) = &step.evicted_on_insert {
        assert!(!cache.contains(evicted));
    }
    assert_peek_victim(cache, model);
}
