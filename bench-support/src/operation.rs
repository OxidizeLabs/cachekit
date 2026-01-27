//! Operation semantics for workload benchmarking.
//!
//! Separates key selection from cache API behavior so benchmarks can model
//! read-through, write-heavy, and transactional access patterns.

use std::sync::Arc;

use cachekit::traits::CoreCache;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::workload::WorkloadGenerator;

#[derive(Debug, Clone, Copy)]
pub enum Operation {
    Get { key: u64 },
    Insert { key: u64 },
    Update { key: u64 },
}

#[derive(Debug, Clone, Copy)]
pub enum OpOutcome {
    Hit,
    Miss,
}

pub trait OpModel {
    fn next_op(&mut self, key: u64) -> Operation;

    fn on_result(&mut self, _key: u64, _op: &Operation, _outcome: OpOutcome) -> Option<Operation> {
        None
    }
}

#[derive(Debug, Clone, Copy)]
pub enum OpWorkload {
    ReadOnly,
    WriteOnly,
    ReadThrough { admit_prob: f64 },
    ReadWriteMix { read_fraction: f64 },
    ReadModifyWrite { rmw_prob: f64, insert_on_miss: bool },
}

impl OpWorkload {
    pub fn build(self, seed: u64) -> Box<dyn OpModel> {
        match self {
            OpWorkload::ReadOnly => Box::new(ReadOnly),
            OpWorkload::WriteOnly => Box::new(WriteOnly),
            OpWorkload::ReadThrough { admit_prob } => Box::new(ReadThrough::new(admit_prob, seed)),
            OpWorkload::ReadWriteMix { read_fraction } => {
                Box::new(ReadWriteMix::new(read_fraction, seed))
            },
            OpWorkload::ReadModifyWrite {
                rmw_prob,
                insert_on_miss,
            } => Box::new(ReadModifyWrite::new(rmw_prob, insert_on_miss, seed)),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OpCounts {
    pub gets: u64,
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
    pub updates: u64,
}

impl OpCounts {
    pub fn hit_rate(self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Execute a workload generator with pluggable operation semantics.
pub fn run_operations<C, V, F, M>(
    cache: &mut C,
    generator: &mut WorkloadGenerator,
    operations: usize,
    op_model: &mut M,
    value_for_key: F,
) -> OpCounts
where
    C: CoreCache<u64, Arc<V>>,
    F: Fn(u64) -> Arc<V>,
    M: OpModel,
{
    let mut counts = OpCounts::default();

    for _ in 0..operations {
        let key = generator.next_key();
        let op = op_model.next_op(key);
        apply_op(cache, generator, op_model, &value_for_key, &mut counts, op);
    }

    counts
}

fn apply_op<C, V, F, M>(
    cache: &mut C,
    generator: &mut WorkloadGenerator,
    op_model: &mut M,
    value_for_key: &F,
    counts: &mut OpCounts,
    op: Operation,
) where
    C: CoreCache<u64, Arc<V>>,
    F: Fn(u64) -> Arc<V>,
    M: OpModel,
{
    match op {
        Operation::Get { key } => {
            counts.gets += 1;
            let outcome = if cache.get(&key).is_some() {
                counts.hits += 1;
                OpOutcome::Hit
            } else {
                counts.misses += 1;
                OpOutcome::Miss
            };

            if let Some(followup) = op_model.on_result(key, &op, outcome) {
                apply_op(cache, generator, op_model, value_for_key, counts, followup);
            }
        },
        Operation::Insert { key } => {
            counts.inserts += 1;
            let value = value_for_key(key);
            let _ = cache.insert(key, value);
            generator.record_insert();
        },
        Operation::Update { key } => {
            counts.updates += 1;
            let value = value_for_key(key);
            let _ = cache.insert(key, value);
            generator.record_insert();
        },
    }
}

/// Read-only lookups.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReadOnly;

impl OpModel for ReadOnly {
    fn next_op(&mut self, key: u64) -> Operation {
        Operation::Get { key }
    }
}

/// Insert-only workload.
#[derive(Debug, Clone, Copy, Default)]
pub struct WriteOnly;

impl OpModel for WriteOnly {
    fn next_op(&mut self, key: u64) -> Operation {
        Operation::Insert { key }
    }
}

/// Read-through workload: insert on miss with an optional admission rate.
#[derive(Debug, Clone)]
pub struct ReadThrough {
    admit_prob: f64,
    rng: SmallRng,
}

impl ReadThrough {
    pub fn new(admit_prob: f64, seed: u64) -> Self {
        Self {
            admit_prob: admit_prob.clamp(0.0, 1.0),
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl OpModel for ReadThrough {
    fn next_op(&mut self, key: u64) -> Operation {
        Operation::Get { key }
    }

    fn on_result(&mut self, key: u64, _op: &Operation, outcome: OpOutcome) -> Option<Operation> {
        match outcome {
            OpOutcome::Hit => None,
            OpOutcome::Miss => {
                if self.rng.random::<f64>() <= self.admit_prob {
                    Some(Operation::Insert { key })
                } else {
                    None
                }
            },
        }
    }
}

/// Mix of reads and inserts.
#[derive(Debug, Clone)]
pub struct ReadWriteMix {
    read_fraction: f64,
    rng: SmallRng,
}

impl ReadWriteMix {
    pub fn new(read_fraction: f64, seed: u64) -> Self {
        Self {
            read_fraction: read_fraction.clamp(0.0, 1.0),
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl OpModel for ReadWriteMix {
    fn next_op(&mut self, key: u64) -> Operation {
        if self.rng.random::<f64>() < self.read_fraction {
            Operation::Get { key }
        } else {
            Operation::Insert { key }
        }
    }
}

/// Read-modify-write: reads followed by optional updates, with optional insert on miss.
#[derive(Debug, Clone)]
pub struct ReadModifyWrite {
    rmw_prob: f64,
    insert_on_miss: bool,
    rng: SmallRng,
}

impl ReadModifyWrite {
    pub fn new(rmw_prob: f64, insert_on_miss: bool, seed: u64) -> Self {
        Self {
            rmw_prob: rmw_prob.clamp(0.0, 1.0),
            insert_on_miss,
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl OpModel for ReadModifyWrite {
    fn next_op(&mut self, key: u64) -> Operation {
        Operation::Get { key }
    }

    fn on_result(&mut self, key: u64, _op: &Operation, outcome: OpOutcome) -> Option<Operation> {
        match outcome {
            OpOutcome::Hit => {
                if self.rng.random::<f64>() < self.rmw_prob {
                    Some(Operation::Update { key })
                } else {
                    None
                }
            },
            OpOutcome::Miss => {
                if self.insert_on_miss {
                    Some(Operation::Insert { key })
                } else {
                    None
                }
            },
        }
    }
}
