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
- [libFuzzer Documentation](https://llvm.org/docs/LibFuzzer.html)
- [cargo-fuzz Book](https://rust-fuzz.github.io/book/cargo-fuzz.html)
