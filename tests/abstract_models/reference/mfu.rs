//! Spec-derived MFU reference model.
//!
//! **Source:** [`docs/testing/specs/mfu.md`](../../../docs/testing/specs/mfu.md) ·
//! [matrix.md](../../../docs/testing/specs/matrix.md)
//! **Tier:** reference (spec-first oracle).
//! **Formulation:** `HashMap<K, u64>` frequencies + per-key sequence numbers for heap tie-break;
//! independent of `BinaryHeap` exact model.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

#[derive(Debug, Clone)]
pub struct NaiveMfuModel<K> {
    freq: HashMap<K, u64>,
    last_seq: HashMap<K, u64>,
    seq: u64,
    capacity: usize,
}

impl<K> NaiveMfuModel<K>
where
    K: Clone + Eq + Hash,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            freq: HashMap::new(),
            last_seq: HashMap::new(),
            seq: 0,
            capacity,
        }
    }

    fn collect_resident(&self) -> HashSet<K> {
        self.freq.keys().cloned().collect()
    }

    fn record_heap_push(&mut self, key: &K) {
        self.seq += 1;
        self.last_seq.insert(key.clone(), self.seq);
    }

    /// Max-frequency victim; at ties, highest sequence (newest heap entry) is evicted first.
    fn pick_victim(&self) -> Option<K> {
        let max_freq = *self.freq.values().max()?;
        self.freq
            .iter()
            .filter(|(_, f)| **f == max_freq)
            .max_by_key(|(k, _)| self.last_seq.get(*k).copied().unwrap_or(0))
            .map(|(k, _)| k.clone())
    }

    fn bump_freq(&mut self, key: K) {
        if let Some(f) = self.freq.get_mut(&key) {
            *f += 1;
            self.record_heap_push(&key);
        }
    }

    pub fn frequency(&self, key: &K) -> Option<u64> {
        self.freq.get(key).copied()
    }
}

impl<K> PolicyModel<K> for NaiveMfuModel<K>
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
        self.pick_victim()
    }

    fn apply(&mut self, op: Op<K>) -> ModelStep<K> {
        let mut step = ModelStep::new(self.collect_resident());

        match op {
            Op::Insert(key) => {
                if self.freq.contains_key(&key) {
                    self.bump_freq(key);
                } else if self.capacity == 0 {
                    return step;
                } else {
                    while self.freq.len() >= self.capacity {
                        if let Some(victim) = self.pick_victim() {
                            if step.evicted_on_insert.is_none() {
                                step.evicted_on_insert = Some(victim.clone());
                            }
                            self.freq.remove(&victim);
                            self.last_seq.remove(&victim);
                        } else {
                            break;
                        }
                    }
                    self.freq.insert(key.clone(), 1);
                    self.record_heap_push(&key);
                }
            },
            Op::Get(key) => {
                let hit = self.freq.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.bump_freq(key);
                }
            },
            Op::Peek(key) => {
                step.hit = Some(if self.freq.contains_key(&key) {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
            },
            Op::GetMut(key) => {
                let hit = self.freq.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.bump_freq(key);
                }
            },
            Op::Touch(_) => {
                step.hit = Some(HitMiss::MayHitOrMiss);
            },
            Op::Remove(key) => {
                self.freq.remove(&key);
                self.last_seq.remove(&key);
            },
            Op::EvictOne => {
                if let Some(victim) = self.pick_victim() {
                    step.victim = OracleExpectation::Exact(victim.clone());
                    self.freq.remove(&victim);
                    self.last_seq.remove(&victim);
                }
            },
        }

        step.resident = self.collect_resident();
        step
    }
}
