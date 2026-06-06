//! 2Q reference model — mirrors `TwoQCore` queue caps and eviction order.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Queue {
    Probation,
    Protected,
}

#[derive(Debug)]
pub struct TwoQModel<K>
where
    K: Clone + Eq + Hash,
{
    /// Head = newest (insert here), tail = oldest (evict here).
    probation: VecDeque<K>,
    /// Head = MRU, tail = LRU (evict here).
    protected: VecDeque<K>,
    queues: HashMap<K, Queue>,
    protected_cap: usize,
    probation_cap: usize,
}

impl<K> TwoQModel<K>
where
    K: Clone + Eq + Hash,
{
    pub fn new(protected_cap: usize, a1_frac: f64) -> Self {
        let probation_cap = (protected_cap as f64 * a1_frac) as usize;
        Self {
            probation: VecDeque::new(),
            protected: VecDeque::new(),
            queues: HashMap::new(),
            protected_cap,
            probation_cap,
        }
    }

    fn len(&self) -> usize {
        self.queues.len()
    }

    fn collect_resident(&self) -> HashSet<K> {
        self.queues.keys().cloned().collect()
    }

    fn detach(&mut self, key: &K) {
        if let Some(q) = self.queues.get(key).copied() {
            match q {
                Queue::Probation => self.probation.retain(|k| k != key),
                Queue::Protected => self.protected.retain(|k| k != key),
            }
        }
    }

    fn promote_to_protected(&mut self, key: K) {
        self.detach(&key);
        self.protected.push_front(key.clone());
        self.queues.insert(key, Queue::Protected);
    }

    fn evict_one_inner(&mut self) -> Option<K> {
        if self.probation.len() > self.probation_cap {
            if let Some(k) = self.probation.pop_back() {
                self.queues.remove(&k);
                return Some(k);
            }
        }
        if let Some(k) = self.protected.pop_back() {
            self.queues.remove(&k);
            return Some(k);
        }
        if let Some(k) = self.probation.pop_back() {
            self.queues.remove(&k);
            return Some(k);
        }
        None
    }

    fn evict_if_needed(&mut self) -> Option<K> {
        let mut last = None;
        while self.len() >= self.protected_cap {
            last = self.evict_one_inner();
        }
        last
    }
}

impl<K> PolicyModel<K> for TwoQModel<K>
where
    K: Clone + Eq + Hash,
{
    fn capacity(&self) -> usize {
        self.protected_cap
    }

    fn resident_set(&self) -> HashSet<K> {
        self.collect_resident()
    }

    fn peek_victim_key(&self) -> Option<K> {
        if self.probation.len() > self.probation_cap {
            return self.probation.back().cloned();
        }
        self.protected
            .back()
            .or_else(|| self.probation.back())
            .cloned()
    }

    fn apply(&mut self, op: Op<K>) -> ModelStep<K> {
        let mut step = ModelStep::new(self.collect_resident());

        match op {
            Op::Insert(key) => {
                if self.queues.contains_key(&key) {
                    return step;
                }
                if self.protected_cap == 0 {
                    return step;
                }
                step.evicted_on_insert = self.evict_if_needed();
                self.probation.push_front(key.clone());
                self.queues.insert(key, Queue::Probation);
            },
            Op::Get(key) => {
                let hit = self.queues.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.promote_to_protected(key);
                }
            },
            Op::Peek(key) => {
                let hit = self.queues.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
            },
            Op::GetMut(key) => {
                let hit = self.queues.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.promote_to_protected(key);
                }
            },
            Op::Touch(_) => {
                step.hit = Some(HitMiss::MayHitOrMiss);
            },
            Op::Remove(key) => {
                self.detach(&key);
                self.queues.remove(&key);
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
