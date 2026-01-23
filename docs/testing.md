# Testing Strategy for CacheKit

CacheKit employs a multi-layered testing approach combining unit tests, property tests, and fuzz tests to ensure correctness and robustness.

## Testing Philosophy

Following the [workspace rules](.cursorrules), we:

1. **Test public APIs primarily** - Focus on the contract users depend on
2. **Test critical internal algorithms** - Property test complex logic like eviction policies
3. **Test invariants exhaustively** - Capacity bounds, index consistency, reference bit behavior
4. **Prefer property tests over manual cases** - Catch edge cases automatically
5. **Fuzz hot paths** - Public interfaces that handle arbitrary input

## Test Layers

### 1. Unit Tests

**Location**: `#[cfg(test)] mod tests` in each module

**Purpose**: Verify specific behaviors and edge cases

**Example**:
```rust
#[test]
fn clock_ring_eviction_prefers_unreferenced() {
    let mut ring = ClockRing::new(2);
    ring.insert("a", 1);
    ring.insert("b", 2);
    ring.touch(&"a");
    let evicted = ring.insert("c", 3);

    assert_eq!(evicted, Some(("b", 2)));
    assert!(ring.contains(&"a"));
}
```

**Run**:
```bash
cargo test
```

### 2. Property Tests

**Location**: `#[cfg(test)] mod property_tests` in each module

**Purpose**: Verify invariants hold across arbitrary inputs

**Dependencies**: `proptest = "1.5"`

**Key Properties for ClockRing**:
- Length never exceeds capacity
- Index and slot consistency
- Get after insert returns correct value
- Remove decreases length
- Update doesn't change length
- Referenced entries survive longer
- Hand position stays within bounds

**Example**:
```rust
proptest! {
    #[test]
    fn prop_len_within_capacity(
        capacity in 1usize..100,
        ops in prop::collection::vec((0u32..1000, 0u32..100), 0..200)
    ) {
        let mut ring = ClockRing::new(capacity);
        for (key, value) in ops {
            ring.insert(key, value);
            prop_assert!(ring.len() <= ring.capacity());
        }
    }
}
```

**Run**:
```bash
cargo test prop_
```

**Run with more cases**:
```bash
PROPTEST_CASES=10000 cargo test prop_len_within_capacity
```

### 3. Fuzz Tests

**Location**: `fuzz/fuzz_targets/`

**Purpose**: Find crashes and invariant violations through mutation-based testing

**Dependencies**: `cargo-fuzz`, `libfuzzer-sys`

**Targets**:
- `clock_ring_arbitrary_ops` - Random operation sequences
- `clock_ring_insert_stress` - Heavy insert load
- `clock_ring_eviction_patterns` - Reference bit patterns

**Run**:
```bash
cd fuzz
cargo fuzz run clock_ring_arbitrary_ops -- -max_total_time=60
```

See [fuzz/README.md](../fuzz/README.md) for detailed fuzzing instructions.

## Testing Private Methods

We expose complex private methods for testing using `#[cfg(test)] pub(crate)`:

```rust
impl<K, V> ClockRing<K, V> {
    #[cfg(any(test, debug_assertions))]
    pub fn debug_validate_invariants(&self) {
        // Check all invariants
        assert_eq!(self.len, self.slots.iter().filter(|s| s.is_some()).count());
        assert_eq!(self.len, self.index.len());
        // ...
    }

    #[cfg(any(test, debug_assertions))]
    pub fn debug_snapshot_slots(&self) -> Vec<Option<(&K, bool)>> {
        // Expose internal state for assertions
    }
}
```

This allows thorough testing without polluting the public API.

## Coverage Expectations

- **Public APIs**: 100% test coverage with both unit and property tests
- **Core algorithms**: Property tests covering all branches
- **Edge cases**: Zero capacity, capacity 1, full ring, empty ring
- **Concurrent wrappers**: Same invariants as single-threaded version

## Running All Tests

```bash
# Unit tests
cargo test

# Property tests with more cases
PROPTEST_CASES=10000 cargo test

# Fuzz tests (short run)
cd fuzz
cargo fuzz run clock_ring_arbitrary_ops -- -max_total_time=60

# With features enabled
cargo test --all-features

# Concurrency tests
cargo test --features concurrency
```

## CI Integration

Our CI runs:
1. Unit tests on all supported platforms
2. Property tests with default case count (256)
3. Short fuzz runs (60 seconds per target)
4. Tests with all feature combinations

Example CI configuration:
```yaml
- name: Run unit tests
  run: cargo test --all-features

- name: Run property tests
  run: PROPTEST_CASES=1000 cargo test prop_

- name: Run fuzz tests
  run: |
    cargo install cargo-fuzz
    cd fuzz
    for target in clock_ring_*; do
      cargo fuzz run $target -- -max_total_time=60
    done
```

## Debugging Test Failures

### Property Test Failures

When a property test fails, proptest generates a minimal failing case:

```
Test failed: prop_len_within_capacity
minimal failing input: capacity = 1, ops = [(5, 10)]
```

**Replay the failure**:
```rust
#[test]
fn reproduce_prop_failure() {
    let mut ring = ClockRing::new(1);
    ring.insert(5, 10);
    ring.debug_validate_invariants();
}
```

### Fuzz Test Crashes

Crashes are saved to `fuzz/artifacts/<target>/crash-<hash>`:

**Reproduce**:
```bash
cargo fuzz run clock_ring_arbitrary_ops fuzz/artifacts/clock_ring_arbitrary_ops/crash-abc123
```

**Debug in GDB**:
```bash
rust-gdb --args target/x86_64-unknown-linux-gnu/release/clock_ring_arbitrary_ops fuzz/artifacts/clock_ring_arbitrary_ops/crash-abc123
```

## Performance Tests

Performance-critical paths have separate benchmarks (see [benchmarks/](../benches/)):

```bash
cargo bench
```

Don't use `#[test]` for performance testing; use `criterion` benchmarks instead.

## Test Organization Guidelines

1. **Keep tests close to code** - Tests in same file as implementation
2. **Separate modules** - `tests`, `property_tests`, `fuzz_tests`
3. **Descriptive names** - `prop_len_within_capacity`, not `test1`
4. **Document test intent** - What invariant or behavior is being verified
5. **Use debug helpers** - `debug_validate_invariants()`, `debug_snapshot_slots()`

## Example: Adding Tests for a New Feature

```rust
// 1. Add unit test
#[test]
fn new_feature_basic_behavior() {
    // Test happy path
}

// 2. Add property test
proptest! {
    #[test]
    fn prop_new_feature_maintains_invariants(
        input in arbitrary_input_strategy()
    ) {
        // Verify invariants
    }
}

// 3. Add fuzz target (if public API)
// fuzz/fuzz_targets/new_feature.rs
fuzz_target!(|data: &[u8]| {
    // Decode and test
});
```

## Related Documentation

- [Contributing Guide](../CONTRIBUTING.md)
- [Fuzz Testing](../fuzz/README.md)
- [Benchmarking](../benches/README.md)
