//! Spec-derived LRU-K reference model.
//!
//! **Source:** [`docs/testing/specs/policies/exact/lru-k.md`](../../../docs/testing/specs/policies/exact/lru-k.md) ·
//! [matrix.md](../../../docs/testing/specs/matrix.md)
//! **Tier:** reference (spec-first oracle).
//! **Formulation:** `Vec<K>` cold/hot segments + `HashMap` step history; independent of
//! `VecDeque` exact model.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Segment {
    Cold,
    Hot,
}

#[derive(Debug, Clone)]
pub struct NaiveLruKModel<K> {
    tick: u64,
    k: usize,
    cold: Vec<K>,
    hot: Vec<K>,
    segment: HashMap<K, Segment>,
    history: HashMap<K, Vec<u64>>,
    capacity: usize,
}

impl<K> NaiveLruKModel<K>
where
    K: Clone + Eq + Hash,
{
    pub fn new(capacity: usize, k: usize) -> Self {
        Self {
            tick: 0,
            k,
            cold: Vec::new(),
            hot: Vec::new(),
            segment: HashMap::new(),
            history: HashMap::new(),
            capacity,
        }
    }

    fn collect_resident(&self) -> HashSet<K> {
        self.segment.keys().cloned().collect()
    }

    fn push_history(&mut self, key: &K, time: u64) {
        let h = self.history.entry(key.clone()).or_default();
        h.push(time);
        if h.len() > self.k {
            let excess = h.len() - self.k;
            h.drain(0..excess);
        }
    }

    fn record_access(&mut self, key: &K) {
        self.tick = self.tick.saturating_add(1);
        self.push_history(key, self.tick);
    }

    fn detach(&mut self, key: &K) {
        match self.segment.get(key) {
            Some(Segment::Cold) => self.cold.retain(|x| x != key),
            Some(Segment::Hot) => self.hot.retain(|x| x != key),
            None => {},
        }
    }

    fn promote_if_needed(&mut self, key: K) {
        let count = self.history.get(&key).map(|h| h.len()).unwrap_or(0);
        if count >= self.k {
            self.detach(&key);
            self.hot.insert(0, key.clone());
            self.segment.insert(key, Segment::Hot);
        }
    }

    fn move_hot_front(&mut self, key: &K) {
        if matches!(self.segment.get(key), Some(Segment::Hot)) {
            self.detach(key);
            self.hot.insert(0, key.clone());
        }
    }

    fn evict_inner(&mut self) -> Option<K> {
        if let Some(victim) = self.cold.pop() {
            self.segment.remove(&victim);
            self.history.remove(&victim);
            return Some(victim);
        }
        if let Some(victim) = self.hot.pop() {
            self.segment.remove(&victim);
            self.history.remove(&victim);
            return Some(victim);
        }
        None
    }

    fn evict_if_needed(&mut self) -> Option<K> {
        let mut evicted = None;
        while self.segment.len() >= self.capacity {
            evicted = self.evict_inner();
        }
        evicted
    }

    pub fn access_count(&self, key: &K) -> Option<usize> {
        self.history.get(key).map(|h| h.len())
    }
}

impl<K> PolicyModel<K> for NaiveLruKModel<K>
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
        self.cold
            .last()
            .cloned()
            .or_else(|| self.hot.last().cloned())
    }

    fn apply(&mut self, op: Op<K>) -> ModelStep<K> {
        let mut step = ModelStep::new(self.collect_resident());

        match op {
            Op::Insert(key) => {
                if self.segment.contains_key(&key) {
                    self.record_access(&key);
                    self.promote_if_needed(key.clone());
                    self.move_hot_front(&key);
                    return step;
                }
                if self.capacity == 0 {
                    return step;
                }
                step.evicted_on_insert = self.evict_if_needed();
                self.tick = self.tick.saturating_add(1);
                self.push_history(&key, self.tick);
                self.cold.insert(0, key.clone());
                self.segment.insert(key, Segment::Cold);
            },
            Op::Get(key) => {
                let hit = self.segment.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.record_access(&key);
                    self.promote_if_needed(key.clone());
                    self.move_hot_front(&key);
                }
            },
            Op::Peek(key) => {
                step.hit = Some(if self.segment.contains_key(&key) {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
            },
            Op::GetMut(key) => {
                let hit = self.segment.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.record_access(&key);
                    self.promote_if_needed(key.clone());
                    self.move_hot_front(&key);
                }
            },
            Op::Touch(_) => {
                step.hit = Some(HitMiss::MayHitOrMiss);
            },
            Op::Remove(key) => {
                self.detach(&key);
                self.segment.remove(&key);
                self.history.remove(&key);
            },
            Op::EvictOne => {
                if let Some(victim) = self.evict_inner() {
                    step.victim = OracleExpectation::Exact(victim);
                }
            },
        }

        step.resident = self.collect_resident();
        step
    }
}
