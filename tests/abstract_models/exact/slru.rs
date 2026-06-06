//! SLRU reference model — mirrors `SlruCore` segment caps and LRU ordering.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Segment {
    Probationary,
    Protected,
}

#[derive(Debug)]
pub struct SlruModel<K>
where
    K: Clone + Eq + Hash,
{
    /// Head = MRU, tail = LRU.
    probationary: VecDeque<K>,
    protected: VecDeque<K>,
    segments: HashMap<K, Segment>,
    capacity: usize,
    probationary_cap: usize,
}

impl<K> SlruModel<K>
where
    K: Clone + Eq + Hash,
{
    pub fn new(capacity: usize, probationary_frac: f64) -> Self {
        let probationary_cap = (capacity as f64 * probationary_frac) as usize;
        Self {
            probationary: VecDeque::new(),
            protected: VecDeque::new(),
            segments: HashMap::new(),
            capacity,
            probationary_cap,
        }
    }

    fn len(&self) -> usize {
        self.segments.len()
    }

    fn collect_resident(&self) -> HashSet<K> {
        self.segments.keys().cloned().collect()
    }

    fn detach(&mut self, key: &K) {
        if let Some(seg) = self.segments.get(key).copied() {
            match seg {
                Segment::Probationary => self.probationary.retain(|k| k != key),
                Segment::Protected => self.protected.retain(|k| k != key),
            }
        }
    }

    fn promote(&mut self, key: K) {
        self.detach(&key);
        self.protected.push_front(key.clone());
        self.segments.insert(key, Segment::Protected);
    }

    fn evict_inner(&mut self) -> Option<K> {
        if self.probationary.len() > self.probationary_cap {
            if let Some(k) = self.probationary.pop_back() {
                self.segments.remove(&k);
                return Some(k);
            }
        }
        if let Some(k) = self.protected.pop_back() {
            self.segments.remove(&k);
            return Some(k);
        }
        if let Some(k) = self.probationary.pop_back() {
            self.segments.remove(&k);
            return Some(k);
        }
        None
    }

    fn evict_if_needed(&mut self) -> Option<K> {
        let mut last = None;
        while self.len() >= self.capacity {
            last = self.evict_inner();
        }
        last
    }
}

impl<K> PolicyModel<K> for SlruModel<K>
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
        if self.probationary.len() > self.probationary_cap {
            return self.probationary.back().cloned();
        }
        self.protected
            .back()
            .or_else(|| self.probationary.back())
            .cloned()
    }

    fn apply(&mut self, op: Op<K>) -> ModelStep<K> {
        let mut step = ModelStep::new(self.collect_resident());

        match op {
            Op::Insert(key) => {
                if self.segments.contains_key(&key) {
                    return step;
                }
                if self.capacity == 0 {
                    return step;
                }
                step.evicted_on_insert = self.evict_if_needed();
                self.probationary.push_front(key.clone());
                self.segments.insert(key, Segment::Probationary);
            },
            Op::Get(key) => {
                let hit = self.segments.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.promote(key);
                }
            },
            Op::Peek(key) => {
                let hit = self.segments.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
            },
            Op::GetMut(key) => {
                let hit = self.segments.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.promote(key);
                }
            },
            Op::Touch(_) => {
                step.hit = Some(HitMiss::MayHitOrMiss);
            },
            Op::Remove(key) => {
                self.detach(&key);
                self.segments.remove(&key);
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
