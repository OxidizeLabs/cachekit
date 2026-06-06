//! Heap-LFU reference model — mirrors `HeapLfuCache` heap eviction semantics.
//!
//! **Tier:** exact.
//! **Source:** [`docs/testing/specs/heap-lfu.md`](../../../docs/testing/specs/heap-lfu.md) ·
//! [matrix.md](../../../docs/testing/specs/matrix.md)
//! **Cross-model sibling:** [`reference/heap_lfu.rs`](../reference/heap_lfu.rs) (`NaiveHeapLfuModel`).
//! **Victim:** lowest frequency; `Ord` tie-break on key when frequencies tie.
//! **Peek:** min `(freq, key)` over `freq` map (Ord tie-break; aligns with heap `pop_lfu`).
//! **Tests:** `policy_semantics/heap_lfu_tests.rs` — residency only (heap stale entries).
//! **Op strategy:** [`standard_op_list`](super::super::standard_op_list) (not `mfu_safe`; heap
//! rebuild handles staleness on insert/evict).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::Hash;

use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

#[derive(Debug)]
pub struct HeapLfuModel<K>
where
    K: Clone + Eq + Hash + Ord,
{
    freq: HashMap<K, u64>,
    heap: BinaryHeap<Reverse<(u64, K)>>,
    capacity: usize,
}

impl<K> HeapLfuModel<K>
where
    K: Clone + Eq + Hash + Ord,
{
    const MAX_HEAP_FACTOR: usize = 4;

    pub fn new(capacity: usize) -> Self {
        Self {
            freq: HashMap::new(),
            heap: BinaryHeap::new(),
            capacity,
        }
    }

    fn collect_resident(&self) -> HashSet<K> {
        self.freq.keys().cloned().collect()
    }

    fn add_to_heap(&mut self, key: &K, frequency: u64) {
        self.heap.push(Reverse((frequency, key.clone())));
        self.maybe_rebuild_heap();
    }

    fn maybe_rebuild_heap(&mut self) {
        let live = self.freq.len().max(1);
        if self.heap.len() <= live.saturating_mul(Self::MAX_HEAP_FACTOR) {
            return;
        }
        self.heap.clear();
        for (key, &f) in &self.freq {
            self.heap.push(Reverse((f, key.clone())));
        }
    }

    fn pop_lfu(&mut self) -> Option<K> {
        let mut stale = 0usize;
        while let Some(Reverse((heap_freq, key))) = self.heap.peek().cloned() {
            if let Some(&current) = self.freq.get(&key) {
                if heap_freq == current {
                    let Reverse((_, key)) = self.heap.pop().unwrap();
                    return Some(key);
                }
            }
            self.heap.pop();
            stale += 1;
            if stale >= self.freq.len().max(1) {
                self.maybe_rebuild_heap();
                stale = 0;
            }
        }
        None
    }

    fn peek_lfu_key(&self) -> Option<K> {
        self.freq
            .iter()
            .min_by(|(k1, f1), (k2, f2)| f1.cmp(f2).then(k1.cmp(k2)))
            .map(|(k, _)| k.clone())
    }

    fn bump_freq(&mut self, key: &K) {
        let new_f = {
            let f = self.freq.get_mut(key).unwrap();
            *f += 1;
            *f
        };
        self.add_to_heap(key, new_f);
    }
}

impl<K> PolicyModel<K> for HeapLfuModel<K>
where
    K: Clone + Eq + Hash + Ord,
{
    fn capacity(&self) -> usize {
        self.capacity
    }

    fn resident_set(&self) -> HashSet<K> {
        self.collect_resident()
    }

    fn peek_victim_key(&self) -> Option<K> {
        self.peek_lfu_key()
    }

    fn apply(&mut self, op: Op<K>) -> ModelStep<K> {
        let mut step = ModelStep::new(self.collect_resident());

        match op {
            Op::Insert(key) => {
                if self.freq.contains_key(&key) {
                    return step;
                }
                if self.capacity == 0 {
                    return step;
                }
                if self.freq.len() >= self.capacity {
                    if let Some(victim) = self.pop_lfu() {
                        step.evicted_on_insert = Some(victim.clone());
                        self.freq.remove(&victim);
                    }
                }
                self.freq.insert(key.clone(), 1);
                self.add_to_heap(&key, 1);
            },
            Op::Get(key) => {
                let hit = self.freq.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.bump_freq(&key);
                }
            },
            Op::GetMut(key) => {
                let hit = self.freq.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.bump_freq(&key);
                }
            },
            Op::Peek(key) => {
                let hit = self.freq.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
            },
            Op::Touch(key) => {
                if self.freq.contains_key(&key) {
                    self.bump_freq(&key);
                    step.hit = Some(HitMiss::MustHit);
                } else {
                    step.hit = Some(HitMiss::MustMiss);
                }
            },
            Op::Remove(key) => {
                self.freq.remove(&key);
            },
            Op::EvictOne => {
                if let Some(victim) = self.pop_lfu() {
                    step.victim = OracleExpectation::Exact(victim.clone());
                    self.freq.remove(&victim);
                }
            },
        }

        step.resident = self.collect_resident();
        step
    }
}
