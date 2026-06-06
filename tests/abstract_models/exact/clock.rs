//! Clock reference model — mirrors `ClockRing` semantics.

use std::collections::HashSet;
use std::hash::Hash;

use cachekit::ds::ClockRing;

use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

#[derive(Debug)]
pub struct ClockModel<K, V>
where
    K: Clone + Eq + Hash,
{
    ring: ClockRing<K, V>,
}

impl<K, V> ClockModel<K, V>
where
    K: Clone + Eq + Hash,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            ring: ClockRing::new(capacity),
        }
    }

    fn collect_resident(&self) -> HashSet<K> {
        self.ring.keys().cloned().collect()
    }
}

impl<K, V> PolicyModel<K> for ClockModel<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone + Default,
{
    fn capacity(&self) -> usize {
        self.ring.capacity()
    }

    fn resident_set(&self) -> HashSet<K> {
        self.collect_resident()
    }

    fn peek_victim_key(&self) -> Option<K> {
        self.ring.peek_victim().map(|(k, _)| k.clone())
    }

    fn apply(&mut self, op: Op<K>) -> ModelStep<K> {
        let mut step = ModelStep::new(self.collect_resident());

        match op {
            Op::Insert(key) => {
                let before = self.collect_resident();
                let evicted = self.ring.insert(key.clone(), V::default());
                if let Some((k, _)) = evicted {
                    if !before.contains(&key) {
                        step.evicted_on_insert = Some(k);
                    }
                }
            },
            Op::Get(key) => {
                let hit = self.ring.get(&key).is_some();
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
            },
            Op::Peek(key) => {
                let hit = self.ring.peek(&key).is_some();
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
            },
            Op::GetMut(_) | Op::Touch(_) => {
                step.hit = Some(HitMiss::MayHitOrMiss);
            },
            Op::Remove(key) => {
                self.ring.remove(&key);
            },
            Op::EvictOne => {
                if let Some((victim, _)) = self.ring.pop_victim() {
                    step.victim = OracleExpectation::Exact(victim);
                }
            },
        }

        step.resident = self.collect_resident();
        step
    }
}
