# Fuzz Testing for CacheKit

This directory contains fuzz tests for CacheKit data structures using `cargo-fuzz` and `libfuzzer-sys`.

## Prerequisites

Install cargo-fuzz:

```bash
cargo install cargo-fuzz
```

**Note**: The fuzz targets require nightly Rust and use `libfuzzer-sys = "0.4"`.

## Available Fuzz Targets

### 1. `clock_ring_arbitrary_ops`

Tests arbitrary sequences of all ClockRing operations (insert, get, peek, touch, update, remove, pop_victim).

**Purpose**: Find edge cases in operation interleaving and state transitions.

**Run**:
```bash
cd fuzz
cargo fuzz run clock_ring_arbitrary_ops
```

### 2. `clock_ring_insert_stress`

Stress tests with many insert operations.

**Purpose**: Find capacity-related bugs and eviction issues under high insert load.

**Run**:
```bash
cd fuzz
cargo fuzz run clock_ring_insert_stress
```

### 3. `clock_ring_eviction_patterns`

Tests second-chance algorithm with varying reference bit patterns.

**Purpose**: Verify CLOCK algorithm correctness with different access patterns.

**Run**:
```bash
cd fuzz
cargo fuzz run clock_ring_eviction_patterns
```

## FixedHistory Fuzz Targets

### 4. `fixed_history_arbitrary_ops`

Tests arbitrary sequences of all FixedHistory operations (record, most_recent, kth_most_recent, to_vec_mru, clear).

**Purpose**: Find edge cases in ring buffer operation interleaving and state transitions.

**Run**:
```bash
cd fuzz
cargo fuzz run fixed_history_arbitrary_ops
```

### 5. `fixed_history_record_stress`

Stress tests with many record operations using reference implementation validation.

**Purpose**: Find wrapping bugs and ordering issues under heavy recording load. Validates against a reference implementation to ensure correctness.

**Run**:
```bash
cd fuzz
cargo fuzz run fixed_history_record_stress
```

### 6. `fixed_history_property_tests`

Property-based tests verifying specific invariants:
- Order preservation after wrapping
- kth_most_recent consistency
- Boundary conditions (k=0, k>len)
- Clear operation correctness

**Purpose**: Verify fundamental properties and invariants hold under all conditions.

**Run**:
```bash
cd fuzz
cargo fuzz run fixed_history_property_tests
```

## FrequencyBuckets Fuzz Targets

### 7. `frequency_buckets_arbitrary_ops`

Tests arbitrary sequences of all FrequencyBuckets operations (insert, touch, remove, pop_min, peek_min, clear).

**Purpose**: Find edge cases in LFU tracking operation interleaving and state transitions.

**Run**:
```bash
cd fuzz
cargo fuzz run frequency_buckets_arbitrary_ops
```

### 8. `frequency_buckets_stress`

Stress tests with heavy insert, touch, and pop_min operations using reference HashMap validation.

**Purpose**: Find frequency tracking bugs and eviction issues under high LFU load. Validates against a reference implementation to ensure correctness.

**Run**:
```bash
cd fuzz
cargo fuzz run frequency_buckets_stress
```

### 9. `frequency_buckets_property_tests`

Property-based tests verifying specific invariants:
- Frequency monotonicity (touch always increments)
- FIFO ordering within same frequency bucket
- min_freq accuracy
- peek/pop consistency
- Clear operation correctness

**Purpose**: Verify fundamental LFU properties and invariants hold under all conditions.

**Run**:
```bash
cd fuzz
cargo fuzz run frequency_buckets_property_tests
```

## GhostList Fuzz Targets

### 10. `ghost_list_arbitrary_ops`

Tests arbitrary sequences of all GhostList operations (record, remove, contains, clear).

**Purpose**: Find edge cases in ghost list operation interleaving and state transitions.

**Run**:
```bash
cd fuzz
cargo fuzz run ghost_list_arbitrary_ops
```

### 11. `ghost_list_lru_stress`

Stress tests with heavy record operations using reference VecDeque validation.

**Purpose**: Find LRU eviction bugs under high load. Validates against a VecDeque reference implementation to ensure correct MRU/LRU ordering.

**Run**:
```bash
cd fuzz
cargo fuzz run ghost_list_lru_stress
```

### 12. `ghost_list_property_tests`

Property-based tests verifying specific invariants:
- LRU eviction order (oldest evicted first)
- Promotion to MRU on re-record
- Capacity bounds enforcement
- Clear operation correctness
- Zero capacity no-op behavior

**Purpose**: Verify fundamental ghost list properties and invariants hold under all conditions.

**Run**:
```bash
cd fuzz
cargo fuzz run ghost_list_property_tests
```

## KeyInterner Fuzz Targets

### 13. `interner_arbitrary_ops`

Tests arbitrary sequences of all KeyInterner operations (intern, get_handle, resolve, clear_shrink).

**Purpose**: Find edge cases in key-to-handle mapping operation interleaving and state transitions.

**Run**:
```bash
cd fuzz
cargo fuzz run interner_arbitrary_ops
```

### 14. `interner_stress`

Stress tests with heavy intern operations using reference HashMap validation.

**Purpose**: Find handle assignment bugs under high load. Validates against HashMap and reverse mapping to ensure correct bidirectional mapping.

**Run**:
```bash
cd fuzz
cargo fuzz run interner_stress
```

### 15. `interner_property_tests`

Property-based tests verifying specific invariants:
- Monotonic handle assignment (sequential from 0)
- Idempotency of intern (same key → same handle)
- Bidirectional mapping correctness
- Handle-resolve consistency
- Clear operation correctness

**Purpose**: Verify fundamental interner properties and invariants hold under all conditions.

**Run**:
```bash
cd fuzz
cargo fuzz run interner_property_tests
```

## IntrusiveList Fuzz Targets

### 16. `intrusive_list_arbitrary_ops`

Tests arbitrary sequences of all IntrusiveList operations (push_front, push_back, pop_front, pop_back, move_to_front, move_to_back, remove, get, clear).

**Purpose**: Find edge cases in doubly linked list operation interleaving and state transitions.

**Run**:
```bash
cd fuzz
cargo fuzz run intrusive_list_arbitrary_ops
```

### 17. `intrusive_list_stress`

Stress tests with heavy push/pop operations using reference VecDeque validation.

**Purpose**: Find ordering bugs under high load. Validates against VecDeque to ensure FIFO/LIFO correctness and front/back consistency.

**Run**:
```bash
cd fuzz
cargo fuzz run intrusive_list_stress
```

### 18. `intrusive_list_property_tests`

Property-based tests verifying specific invariants:
- FIFO ordering (push_back + pop_front)
- LIFO ordering (push_front + pop_front)
- LRU behavior (move_to_front)
- Remove consistency
- Clear operation correctness

**Purpose**: Verify fundamental doubly linked list properties and invariants hold under all conditions.

**Run**:
```bash
cd fuzz
cargo fuzz run intrusive_list_property_tests
```

## LazyMinHeap Fuzz Targets

### 19. `lazy_heap_arbitrary_ops`

Tests arbitrary sequences of all LazyMinHeap operations (update, remove, pop_best, score_of, rebuild, maybe_rebuild, clear).

**Purpose**: Find edge cases in lazy heap operation interleaving and stale entry handling.

**Run**:
```bash
cd fuzz
cargo fuzz run lazy_heap_arbitrary_ops
```

### 20. `lazy_heap_stress`

Stress tests with heavy update/pop operations using reference BinaryHeap validation.

**Purpose**: Find min-heap ordering bugs under high load. Validates against BinaryHeap to ensure correct priority ordering and stale entry skipping.

**Run**:
```bash
cd fuzz
cargo fuzz run lazy_heap_stress
```

### 21. `lazy_heap_property_tests`

Property-based tests verifying specific invariants:
- Min-heap ordering (pop returns smallest score)
- Update idempotency and overwriting
- Stale entry skipping during pop
- Rebuild correctness
- Clear operation correctness

**Purpose**: Verify fundamental lazy min-heap properties and invariants hold under all conditions.

**Run**:
```bash
cd fuzz
cargo fuzz run lazy_heap_property_tests
```

## ShardSelector Fuzz Targets

### 22. `shard_selector_arbitrary_ops`

Tests arbitrary shard selection operations with various shard counts and key types.

**Purpose**: Find edge cases in deterministic shard mapping and range validation.

**Run**:
```bash
cd fuzz
cargo fuzz run shard_selector_arbitrary_ops
```

### 23. `shard_selector_distribution`

Tests key distribution across shards and seed isolation properties.

**Purpose**: Verify keys distribute across shards and different seeds produce different mappings.

**Run**:
```bash
cd fuzz
cargo fuzz run shard_selector_distribution
```

### 24. `shard_selector_property_tests`

Property-based tests verifying specific invariants:
- Determinism (same key → same shard)
- Range validity (shard < shard_count)
- Zero shards clamped to 1
- Single shard always returns 0
- Seed isolation

**Purpose**: Verify fundamental shard selector properties and invariants hold under all conditions.

**Run**:
```bash
cd fuzz
cargo fuzz run shard_selector_property_tests
```

## Running Fuzz Tests

### Run a specific target
```bash
cd fuzz
cargo fuzz run clock_ring_arbitrary_ops
```

### Run with a time limit
```bash
cargo fuzz run clock_ring_arbitrary_ops -- -max_total_time=60
```

### Run with specific number of runs
```bash
cargo fuzz run clock_ring_arbitrary_ops -- -runs=1000000
```

### Run multiple jobs in parallel
```bash
cargo fuzz run clock_ring_arbitrary_ops -- -workers=4
```

## Reviewing Crashes

If a fuzz target finds a crash, the input is saved to `fuzz/artifacts/<target_name>/`.

To reproduce:
```bash
cargo fuzz run clock_ring_arbitrary_ops fuzz/artifacts/clock_ring_arbitrary_ops/crash-<hash>
```

## Corpus Management

The corpus (interesting inputs found during fuzzing) is stored in `fuzz/corpus/<target_name>/`.

### Minimize corpus
```bash
cargo fuzz cmin clock_ring_arbitrary_ops
```

### Minimize a single input
```bash
cargo fuzz tmin clock_ring_arbitrary_ops fuzz/artifacts/clock_ring_arbitrary_ops/crash-<hash>
```

## Integration with CI

Add to your CI pipeline:

```yaml
- name: Install cargo-fuzz
  run: cargo install cargo-fuzz

- name: Run fuzz tests (short)
  run: |
    cd fuzz
    cargo fuzz run clock_ring_arbitrary_ops -- -max_total_time=60 -seed=1
    cargo fuzz run clock_ring_insert_stress -- -max_total_time=60 -seed=2
    cargo fuzz run clock_ring_eviction_patterns -- -max_total_time=60 -seed=3
    cargo fuzz run fixed_history_arbitrary_ops -- -max_total_time=60 -seed=4
    cargo fuzz run fixed_history_record_stress -- -max_total_time=60 -seed=5
    cargo fuzz run fixed_history_property_tests -- -max_total_time=60 -seed=6
    cargo fuzz run frequency_buckets_arbitrary_ops -- -max_total_time=60 -seed=7
    cargo fuzz run frequency_buckets_stress -- -max_total_time=60 -seed=8
    cargo fuzz run frequency_buckets_property_tests -- -max_total_time=60 -seed=9
    cargo fuzz run ghost_list_arbitrary_ops -- -max_total_time=60 -seed=10
    cargo fuzz run ghost_list_lru_stress -- -max_total_time=60 -seed=11
    cargo fuzz run ghost_list_property_tests -- -max_total_time=60 -seed=12
    cargo fuzz run interner_arbitrary_ops -- -max_total_time=60 -seed=13
    cargo fuzz run interner_stress -- -max_total_time=60 -seed=14
    cargo fuzz run interner_property_tests -- -max_total_time=60 -seed=15
    cargo fuzz run intrusive_list_arbitrary_ops -- -max_total_time=60 -seed=16
    cargo fuzz run intrusive_list_stress -- -max_total_time=60 -seed=17
    cargo fuzz run intrusive_list_property_tests -- -max_total_time=60 -seed=18
    cargo fuzz run lazy_heap_arbitrary_ops -- -max_total_time=60 -seed=19
    cargo fuzz run lazy_heap_stress -- -max_total_time=60 -seed=20
    cargo fuzz run lazy_heap_property_tests -- -max_total_time=60 -seed=21
    cargo fuzz run shard_selector_arbitrary_ops -- -max_total_time=60 -seed=22
    cargo fuzz run shard_selector_distribution -- -max_total_time=60 -seed=23
    cargo fuzz run shard_selector_property_tests -- -max_total_time=60 -seed=24
```

## Coverage

Generate coverage reports:

```bash
cargo fuzz coverage clock_ring_arbitrary_ops
```

View coverage:
```bash
cargo cov -- show target/x86_64-unknown-linux-gnu/coverage/x86_64-unknown-linux-gnu/release/clock_ring_arbitrary_ops \
    --format=html -instr-profile=coverage/clock_ring_arbitrary_ops/coverage.profdata
```

## Best Practices

1. **Run continuously**: Fuzz targets find bugs over time; run for hours/days
2. **Save corpus**: Commit interesting corpus files to version control
3. **Minimize before committing**: Use `cargo fuzz cmin` to reduce corpus size
4. **Test locally**: Run fuzz targets on every major change before pushing
5. **Monitor coverage**: Ensure fuzz targets exercise all code paths

## Related Documentation

- [ClockRing Tests](../src/ds/clock_ring.rs) - Unit and property tests for ClockRing
- [FixedHistory Tests](../src/ds/fixed_history.rs) - Unit and property tests for FixedHistory
- [FrequencyBuckets Tests](../src/ds/frequency_buckets.rs) - Unit and property tests for FrequencyBuckets
- [GhostList Tests](../src/ds/ghost_list.rs) - Unit and property tests for GhostList
- [KeyInterner Tests](../src/ds/interner.rs) - Unit and property tests for KeyInterner
- [IntrusiveList Tests](../src/ds/intrusive_list.rs) - Unit and property tests for IntrusiveList
- [LazyMinHeap Tests](../src/ds/lazy_heap.rs) - Unit and property tests for LazyMinHeap
- [ShardSelector Tests](../src/ds/shard.rs) - Unit and property tests for ShardSelector
- [libFuzzer Documentation](https://llvm.org/docs/LibFuzzer.html)
- [cargo-fuzz Book](https://rust-fuzz.github.io/book/cargo-fuzz.html)
