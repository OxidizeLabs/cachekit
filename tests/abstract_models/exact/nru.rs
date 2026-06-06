//! NRU reference model (swap-remove eviction, new inserts start unreferenced).
//!
//! **Tier:** mirror.
//! **Victim:** first unreferenced key in insertion order; swap-remove on eviction.
//! **Tests:** `policy_semantics/nru_tests.rs` — residency only (no `EvictingCache`).
//! **Op strategy:** [`short_op_list_no_evict`](super::super::short_op_list_no_evict) — O(n)
//! eviction scans; no explicit `EvictOne`.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

#[derive(Debug)]
pub struct NruModel<K>
where
    K: Clone + Eq + Hash,
{
    keys: Vec<K>,
    referenced: HashMap<K, bool>,
    capacity: usize,
}

impl<K> NruModel<K>
where
    K: Clone + Eq + Hash,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            keys: Vec::new(),
            referenced: HashMap::new(),
            capacity,
        }
    }

    fn collect_resident(&self) -> HashSet<K> {
        self.keys.iter().cloned().collect()
    }

    fn swap_remove_at(&mut self, idx: usize) -> K {
        let victim = self.keys.swap_remove(idx);
        self.referenced.remove(&victim);
        victim
    }

    fn evict_one_inner(&mut self) -> Option<K> {
        if self.keys.is_empty() {
            return None;
        }

        for idx in 0..self.keys.len() {
            let key = &self.keys[idx];
            if !self.referenced.get(key).copied().unwrap_or(false) {
                return Some(self.swap_remove_at(idx));
            }
        }

        for key in &self.keys {
            self.referenced.insert(key.clone(), false);
        }

        if self.keys.is_empty() {
            return None;
        }
        Some(self.swap_remove_at(0))
    }
}

impl<K> PolicyModel<K> for NruModel<K>
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
        for key in &self.keys {
            if !self.referenced.get(key).copied().unwrap_or(false) {
                return Some(key.clone());
            }
        }
        self.keys.first().cloned()
    }

    fn apply(&mut self, op: Op<K>) -> ModelStep<K> {
        let mut step = ModelStep::new(self.collect_resident());

        match op {
            Op::Insert(key) => {
                if let std::collections::hash_map::Entry::Occupied(mut e) =
                    self.referenced.entry(key.clone())
                {
                    e.insert(true);
                    return step;
                }
                if self.capacity == 0 {
                    return step;
                }
                if self.keys.len() >= self.capacity {
                    step.evicted_on_insert = self.evict_one_inner();
                }
                self.keys.push(key.clone());
                self.referenced.insert(key, false);
            },
            Op::Get(key) => {
                let hit = self.referenced.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.referenced.insert(key, true);
                }
            },
            Op::Peek(key) => {
                let hit = self.referenced.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
            },
            Op::GetMut(key) => {
                let hit = self.referenced.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.referenced.insert(key, true);
                }
            },
            Op::Touch(_) => {
                step.hit = Some(HitMiss::MayHitOrMiss);
            },
            Op::Remove(key) => {
                if let Some(pos) = self.keys.iter().position(|k| k == &key) {
                    self.swap_remove_at(pos);
                }
            },
            Op::EvictOne => {
                if let Some(victim) = self.evict_one_inner() {
                    step.victim = OracleExpectation::Exact(victim);
                }
            },
        }

        step.resident = self.collect_resident();
        step
    }
}
