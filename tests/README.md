# Test Organization

This directory contains all integration and regression tests for cachekit.

## Test Files

### Performance Tests

- **`performance_regression.rs`** - O(1) complexity verification and critical regression guards
  - Tests all cache policies (LRU, LRU-K, LFU, Clock, S3-FIFO)
  - Verifies operations remain O(1) as cache size increases
  - Fast execution (< 5 seconds) for CI

### Concurrency Tests

- **`fifo_concurrency.rs`** - FIFO concurrent access tests
- **`lfu_concurrency.rs`** - LFU concurrent access tests
- **`lru_concurrency.rs`** - LRU concurrent access tests
- **`lru_k_concurrency.rs`** - LRU-K concurrent access tests
- **`slab_concurrency.rs`** - ConcurrentSlabStore race condition and atomicity tests

### Invariant Tests

- **`policy_invariants.rs`** - Cross-policy behavioral consistency (e.g. capacity-0 semantics)

### Integration Tests

- **`lru_integration_test.rs`** - LRU policy integration tests

## Performance Testing Philosophy

We separate performance concerns into two distinct categories:

### 1. Regression Tests (`performance_regression.rs`)

**Purpose**: Catch critical performance regressions in CI

**What it tests**:
- O(1) complexity guarantees for core operations (get, insert, evict)
- Basic performance sanity checks (operations complete in reasonable time)
- Memory stability (cache doesn't grow unbounded)

**What it does NOT test**:
- Exact nanosecond timings (too brittle for CI)
- Cross-library comparisons (use benchmarks)
- Detailed throughput analysis (use benchmarks)

**Run with**: `cargo test --test performance_regression`

**Example output**:
```
[LRU] Size: 1000, Avg get time: 3159.17 ns
[LRU] Size: 2000, Avg get time: 7456.20 ns
[LRU] Size 1000→2000 (2.00x): time 3159.2ns→7456.2ns (2.36x) ✓
```

### 2. Detailed Benchmarks (`../benches/`)

**Purpose**: Measure and track performance over time

**What it provides**:
- Statistical analysis with Criterion
- Cross-policy comparisons (LRU vs LFU vs Clock)
- Cross-library comparisons (cachekit vs lru vs quick_cache)
- Detailed reports and charts
- Workload-specific measurements (Zipfian, scan patterns)

**Run with**: `cargo bench`

## Design Rationale

### Why Separate Tests from Benchmarks?

**Tests** should be:
- Fast enough to run in CI on every commit
- Deterministic (pass/fail, not measurements)
- Focused on correctness guarantees (like O(1) complexity)
- Resilient to minor performance variations

**Benchmarks** should be:
- Comprehensive and thorough
- Statistically rigorous
- Used for optimization decisions
- Run periodically, not on every commit

### Complexity Verification Strategy

The `performance_regression` tests verify algorithmic complexity by:
1. Running operations at multiple cache sizes (1K, 2K, 4K, 8K)
2. Computing average operation time at each size
3. Checking that time doesn't grow linearly with size

For O(1) operations:
- **Expected**: Doubling size causes minimal time increase (< 15x ratio)
- **Failure**: Time grows linearly with size (ratio ≈ size ratio)

**Why 15x threshold?**
The generous threshold accounts for:
- Cache effects (larger working sets cause more CPU cache misses)
- Hash table resizing
- Memory allocator behavior
- Debug build overhead
- Measurement noise

We want to catch O(n) bugs (where ratio ≈ size ratio), not minor performance variations.

**Example: O(1) behavior ✓**
```
[LRU] Size 1000→2000 (2.00x): time 3159.2ns→7456.2ns (2.36x) ✓
[LRU] Size 2000→4000 (2.00x): time 7456.2ns→11745.1ns (1.58x) ✓
```

**Example: O(n) bug detected ✗**
```
[BadImpl] Size 1000→2000 (2.00x): time 1000ns→2100ns (2.10x) ✗
[BadImpl] Size 2000→4000 (2.00x): time 2100ns→4200ns (2.00x) ✗
```

## Test History

**Previous approach** (removed in refactoring):
- `lru_performance.rs`, `lru_k_performance.rs`, `lfu_performance.rs` (~4,900 lines)
- Detailed micro-benchmarks with exact timing assertions
- Too brittle for CI (flaky due to environment variations)
- Mixed correctness checks with performance measurements

**Current approach**:
- `performance_regression.rs` (400 lines)
- Focused on algorithmic complexity guarantees
- Stable CI execution with loose thresholds
- Clear separation: tests verify correctness, benchmarks measure performance

## Adding New Tests

### Adding Complexity Tests for a New Policy

When implementing a new cache policy, add complexity tests:

```rust
// In performance_regression.rs
mod complexity_new_policy {
    use super::*;
    use cachekit::policy::new_policy::NewPolicy;

    #[test]
    fn test_get_is_o1() {
        verify_get_complexity::<i32, i32, _, _>(
            |size| NewPolicy::new(size),
            "NewPolicy"
        );
    }

    #[test]
    fn test_insert_is_o1() {
        verify_insert_complexity::<i32, i32, _, _>(
            |size| NewPolicy::new(size),
            "NewPolicy"
        );
    }

    #[test]
    fn test_eviction_is_o1() {
        verify_eviction_complexity::<i32, i32, _, _>(
            |size| NewPolicy::new(size),
            "NewPolicy"
        );
    }
}
```

**Note**: Adjust type parameters based on your cache's requirements:
- `<i32, i32, _, _>` for caches that store values directly
- `<i32, Arc<i32>, _, _>` for caches that require Arc-wrapped values (like LRU, LFU)

### Adding Detailed Benchmarks

For performance optimization work:

```rust
// In benches/ops.rs
group.bench_function("new_policy", |b| {
    b.iter_custom(|iters| {
        let mut cache = NewPolicy::new(CAPACITY);
        for i in 0..CAPACITY as u64 {
            cache.insert(i, i);
        }
        let start = Instant::now();
        for _ in 0..iters {
            for i in 0..OPS {
                let key = i % (CAPACITY as u64);
                black_box(cache.get(&key));
            }
        }
        start.elapsed()
    })
});
```

## Running Tests

```bash
# All tests (integration, unit, and performance)
cargo test

# Only performance regression tests
cargo test --test performance_regression

# Only complexity tests for specific policy
cargo test --test performance_regression complexity_lru
cargo test --test performance_regression complexity_lfu
cargo test --test performance_regression complexity_clock

# Only concurrency tests
cargo test lru_concurrency
cargo test lfu_concurrency

# Integration tests
cargo test --test lru_integration_test

# All benchmarks (detailed performance analysis)
cargo bench

# Specific benchmark suite
cargo bench --bench ops           # Micro-operation benchmarks
cargo bench --bench comparison    # Cross-library comparisons
cargo bench --bench workloads     # Workload simulations

# Skip performance tests (faster for development)
cargo test --lib --bins
```

## Test Output Examples

### Successful Complexity Test
```
test complexity_lru::test_get_is_o1 ... ok
```

### Failed Complexity Test (O(n) detected)
```
[BadCache] Size 1000→2000 (2.00x): time 1000ns→2100ns (2.10x)
[BadCache] Size 2000→4000 (2.00x): time 2100ns→4200ns (2.00x)
thread 'test_get_is_o1' panicked at:
[BadCache] Get operation appears to be O(n), not O(1):
Size increased by 2.00x but time increased by 2.10x
```

## CI/CD Integration

Performance regression tests are designed to run in CI:

- **Fast**: Complete in < 10 seconds
- **Stable**: Loose thresholds avoid flaky failures
- **Actionable**: Clear error messages guide debugging
- **Portable**: Work across different CI environments

Add to your CI pipeline:
```yaml
- name: Run tests
  run: cargo test

# Or explicitly:
- name: Performance regression check
  run: cargo test --test performance_regression
```
