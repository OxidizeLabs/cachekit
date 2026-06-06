//! MFU reference model — mirrors `MfuCore` heap eviction semantics.
//!
//! **Tier:** exact.
//! **Source:** [`docs/testing/specs/policies/exact/mfu.md`](../../../docs/testing/specs/policies/exact/mfu.md) ·
//! [matrix.md](../../../docs/testing/specs/matrix.md)
//! **Cross-model sibling:** [`reference/mfu.rs`](../reference/mfu.rs) (`NaiveMfuModel`).
//! **Victim:** highest frequency; sequence-number tie-break (newest heap entry evicted first).
//! **Peek:** max valid `HeapEntry` in heap (aligns with `pop_mfu` tie-break).
//! **Tests:** `policy_semantics/mfu_tests.rs` — residency only.
//! **Op strategy:** [`op_strategy_mfu_safe`](super::super::op_strategy_mfu_safe) — skips
//! `Remove`/`EvictOne` because stale heap entries break `debug_validate_invariants`.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::hash::Hash;

use rustc_hash::FxHashMap;

use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

#[derive(Clone)]
struct HeapEntry<K> {
    freq: u64,
    seq: u64,
    key: K,
}

impl<K> PartialEq for HeapEntry<K> {
    fn eq(&self, other: &Self) -> bool {
        self.freq == other.freq && self.seq == other.seq
    }
}

impl<K> Eq for HeapEntry<K> {}

impl<K> PartialOrd for HeapEntry<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K> Ord for HeapEntry<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.freq
            .cmp(&other.freq)
            .then_with(|| self.seq.cmp(&other.seq))
    }
}

pub struct MfuModel<K>
where
    K: Eq + Hash + Clone,
{
    freq: FxHashMap<K, u64>,
    heap: BinaryHeap<HeapEntry<K>>,
    capacity: usize,
    seq: u64,
}

impl<K> MfuModel<K>
where
    K: Eq + Hash + Clone,
{
    const HEAP_REBUILD_FACTOR: usize = 3;

    pub fn new(capacity: usize) -> Self {
        Self {
            freq: FxHashMap::default(),
            heap: BinaryHeap::new(),
            capacity,
            seq: 0,
        }
    }

    fn collect_resident(&self) -> HashSet<K> {
        self.freq.keys().cloned().collect()
    }

    fn push_heap(&mut self, key: K, frequency: u64) {
        self.seq += 1;
        self.heap.push(HeapEntry {
            freq: frequency,
            seq: self.seq,
            key,
        });
        if self.heap.len() > self.freq.len() * Self::HEAP_REBUILD_FACTOR {
            self.rebuild_heap();
        }
    }

    fn rebuild_heap(&mut self) {
        let entries: Vec<_> = self.freq.iter().map(|(k, &f)| (k.clone(), f)).collect();
        self.heap.clear();
        for (key, f) in entries {
            self.push_heap(key, f);
        }
    }

    fn pop_mfu(&mut self) -> Option<K> {
        while let Some(entry) = self.heap.pop() {
            if let Some(&current) = self.freq.get(&entry.key) {
                if current == entry.freq {
                    self.freq.remove(&entry.key);
                    return Some(entry.key);
                }
            }
        }
        if !self.freq.is_empty() {
            self.rebuild_heap();
            if let Some(entry) = self.heap.pop() {
                self.freq.remove(&entry.key);
                return Some(entry.key);
            }
        }
        None
    }

    fn peek_mfu_key(&self) -> Option<K> {
        self.heap
            .iter()
            .filter(|entry| self.freq.get(&entry.key) == Some(&entry.freq))
            .max()
            .map(|entry| entry.key.clone())
    }

    fn bump_freq(&mut self, key: K) {
        let new_f = {
            let f = self.freq.get_mut(&key).unwrap();
            *f += 1;
            *f
        };
        self.push_heap(key, new_f);
    }
}

impl<K> PolicyModel<K> for MfuModel<K>
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
        self.peek_mfu_key()
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
                        if let Some(victim) = self.pop_mfu() {
                            if step.evicted_on_insert.is_none() {
                                step.evicted_on_insert = Some(victim);
                            }
                        } else {
                            break;
                        }
                    }
                    self.freq.insert(key.clone(), 1);
                    self.push_heap(key, 1);
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
            Op::Peek(key) => {
                let hit = self.freq.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
            },
            Op::Touch(_) => {
                step.hit = Some(HitMiss::MayHitOrMiss);
            },
            Op::Remove(key) => {
                self.freq.remove(&key);
            },
            Op::EvictOne => {
                if let Some(victim) = self.pop_mfu() {
                    step.victim = OracleExpectation::Exact(victim);
                }
            },
        }

        step.resident = self.collect_resident();
        step
    }
}
