//! LRU-K reference model (step-counter time, cold/hot segments).

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Segment {
    Cold,
    Hot,
}

#[derive(Debug)]
pub struct LruKModel<K>
where
    K: Clone + Eq + Hash,
{
    tick: u64,
    k: usize,
    cold: VecDeque<K>,
    hot: VecDeque<K>,
    segment: HashMap<K, Segment>,
    history: HashMap<K, VecDeque<u64>>,
    capacity: usize,
}

impl<K> LruKModel<K>
where
    K: Clone + Eq + Hash,
{
    pub fn new(capacity: usize, k: usize) -> Self {
        Self {
            tick: 0,
            k,
            cold: VecDeque::new(),
            hot: VecDeque::new(),
            segment: HashMap::new(),
            history: HashMap::new(),
            capacity,
        }
    }

    fn collect_resident(&self) -> HashSet<K> {
        self.segment.keys().cloned().collect()
    }

    fn record_access(&mut self, key: &K) {
        self.tick = self.tick.saturating_add(1);
        let h = self.history.entry(key.clone()).or_default();
        h.push_back(self.tick);
        while h.len() > self.k {
            h.pop_front();
        }
    }

    fn detach(&mut self, key: &K) {
        match self.segment.get(key) {
            Some(Segment::Cold) => self.cold.retain(|k| k != key),
            Some(Segment::Hot) => self.hot.retain(|k| k != key),
            None => {},
        }
    }

    fn promote_if_needed(&mut self, key: K) {
        let count = self.history.get(&key).map(|h| h.len()).unwrap_or(0);
        if count >= self.k {
            self.detach(&key);
            self.hot.push_front(key.clone());
            self.segment.insert(key, Segment::Hot);
        }
    }

    fn move_hot_front(&mut self, key: &K) {
        if matches!(self.segment.get(key), Some(Segment::Hot)) {
            self.detach(key);
            self.hot.push_front(key.clone());
        }
    }

    fn evict_inner(&mut self) -> Option<K> {
        if let Some(k) = self.cold.pop_back() {
            self.segment.remove(&k);
            self.history.remove(&k);
            return Some(k);
        }
        if let Some(k) = self.hot.pop_back() {
            self.segment.remove(&k);
            self.history.remove(&k);
            return Some(k);
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

impl<K> PolicyModel<K> for LruKModel<K>
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
        self.cold.back().or_else(|| self.hot.back()).cloned()
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
                self.history
                    .entry(key.clone())
                    .or_default()
                    .push_back(self.tick);
                self.cold.push_front(key.clone());
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
                let hit = self.segment.contains_key(&key);
                step.hit = Some(if hit {
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
