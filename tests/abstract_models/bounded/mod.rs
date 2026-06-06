//! Bounded reference models (legal victim sets + structural checks).

#![allow(dead_code)]

pub mod arc;
pub mod car;
pub mod clock_pro;
pub mod s3_fifo;

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

/// Tracks residency; on insert at capacity, legal victims = all residents.
#[derive(Debug)]
pub struct ResidencyBoundedModel<K>
where
    K: Clone + Eq + Hash,
{
    resident: HashMap<K, ()>,
    capacity: usize,
}

impl<K> ResidencyBoundedModel<K>
where
    K: Clone + Eq + Hash,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            resident: HashMap::new(),
            capacity,
        }
    }

    fn collect_resident(&self) -> HashSet<K> {
        self.resident.keys().cloned().collect()
    }

    fn legal_victims(&self) -> HashSet<K> {
        self.collect_resident()
    }
}

impl<K> PolicyModel<K> for ResidencyBoundedModel<K>
where
    K: Clone + Eq + Hash,
{
    fn capacity(&self) -> usize {
        self.capacity
    }

    fn resident_set(&self) -> HashSet<K> {
        self.collect_resident()
    }

    fn peek_victim_key(&self) -> Option<K> {
        self.resident.keys().next().cloned()
    }

    fn apply(&mut self, op: Op<K>) -> ModelStep<K> {
        let mut step = ModelStep::new(self.collect_resident());

        match op {
            Op::Insert(key) => {
                if self.resident.contains_key(&key) {
                    return step;
                }
                if self.capacity == 0 {
                    return step;
                }
                if self.resident.len() >= self.capacity {
                    let legal = self.legal_victims();
                    step.victim = OracleExpectation::Legal(legal);
                    if let Some(victim) = self.resident.keys().next().cloned() {
                        self.resident.remove(&victim);
                        step.evicted_on_insert = Some(victim);
                    }
                }
                self.resident.insert(key, ());
            },
            Op::Get(key) | Op::Peek(key) | Op::GetMut(key) => {
                let hit = self.resident.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
            },
            Op::Touch(key) => {
                let hit = self.resident.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
            },
            Op::Remove(key) => {
                self.resident.remove(&key);
            },
            Op::EvictOne => {
                if let Some(victim) = self.resident.keys().next().cloned() {
                    step.victim = OracleExpectation::Legal(self.legal_victims());
                    self.resident.remove(&victim);
                }
            },
        }

        step.resident = self.collect_resident();
        step
    }
}
