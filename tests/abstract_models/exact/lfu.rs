//! LFU reference model using [`FrequencyBuckets`](cachekit::ds::FrequencyBuckets).
//!
//! **Tier:** exact.
//! **Source:** [`docs/testing/specs/policies/exact/lfu.md`](../../../docs/testing/specs/policies/exact/lfu.md) ·
//! [matrix.md](../../../docs/testing/specs/matrix.md)
//! **Cross-model sibling:** [`reference/lfu.rs`](../reference/lfu.rs) (`NaiveLfuModel`).
//! **Victim:** minimum frequency; FIFO tie-break within the min bucket.
//! **Tests:** `policy_semantics/lfu_tests.rs` — `VictimInspectable`, `FrequencyTracking`,
//! `EvictingCache`.
//! **Op strategy:** [`op_strategy`](super::super::op_strategy).

use std::collections::HashSet;
use std::hash::Hash;

use cachekit::ds::FrequencyBuckets;

use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

#[derive(Debug)]
pub struct LfuModel<K>
where
    K: Eq + Hash + Clone,
{
    buckets: FrequencyBuckets<K>,
    capacity: usize,
}

impl<K> LfuModel<K>
where
    K: Eq + Hash + Clone,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            buckets: FrequencyBuckets::with_capacity(capacity),
            capacity,
        }
    }

    fn collect_resident(&self) -> HashSet<K> {
        self.buckets.iter().map(|(_, m)| m.key.clone()).collect()
    }

    pub fn frequency(&self, key: &K) -> Option<u64> {
        self.buckets.frequency(key)
    }
}

impl<K> PolicyModel<K> for LfuModel<K>
where
    K: Eq + Hash + Clone,
{
    fn capacity(&self) -> usize {
        self.capacity
    }

    fn resident_set(&self) -> HashSet<K> {
        self.collect_resident()
    }

    fn peek_victim_key(&self) -> Option<K> {
        self.buckets.peek_min_key().cloned()
    }

    fn apply(&mut self, op: Op<K>) -> ModelStep<K> {
        let mut step = ModelStep::new(self.collect_resident());

        match op {
            Op::Insert(key) => {
                if self.buckets.contains(&key) {
                    return step;
                }
                if self.capacity == 0 {
                    return step;
                }
                if self.buckets.len() >= self.capacity {
                    step.evicted_on_insert = self.buckets.pop_min().map(|(k, _)| k);
                }
                self.buckets.insert(key);
            },
            Op::Get(key) => {
                let hit = self.buckets.contains(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.buckets.touch(&key);
                }
            },
            Op::Peek(key) => {
                let hit = self.buckets.contains(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
            },
            Op::GetMut(key) => {
                let hit = self.buckets.contains(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.buckets.touch(&key);
                }
            },
            Op::Touch(key) => {
                if self.buckets.touch(&key).is_some() {
                    step.hit = Some(HitMiss::MustHit);
                } else {
                    step.hit = Some(HitMiss::MustMiss);
                }
            },
            Op::Remove(key) => {
                self.buckets.remove(&key);
            },
            Op::EvictOne => {
                if let Some((victim, _)) = self.buckets.pop_min() {
                    step.victim = OracleExpectation::Exact(victim);
                }
            },
        }

        step.resident = self.collect_resident();
        step
    }
}
