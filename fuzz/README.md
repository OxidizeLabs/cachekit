# Fuzz Testing for CacheKit

This directory contains fuzz tests for the ClockRing data structure using `cargo-fuzz` and `libfuzzer-sys`.

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

- [Property Tests](../src/ds/clock_ring.rs) - Property-based tests using proptest
- [Unit Tests](../src/ds/clock_ring.rs) - Traditional unit tests
- [libFuzzer Documentation](https://llvm.org/docs/LibFuzzer.html)
- [cargo-fuzz Book](https://rust-fuzz.github.io/book/cargo-fuzz.html)
