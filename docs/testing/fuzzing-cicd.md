# Fuzzing Integration with CI/CD

This document describes how fuzz testing is integrated into the CacheKit CI/CD pipeline.

## Overview

CacheKit uses comprehensive fuzz testing with **27 fuzz targets** covering 9 core data structures. Fuzzing runs automatically in CI/CD to catch bugs, edge cases, and security issues.

## Fuzz Target Coverage

| Data Structure | Fuzz Targets | Coverage |
|----------------|--------------|----------|
| **ClockRing** | 3 | Arbitrary ops, insert stress, eviction patterns |
| **FixedHistory** | 3 | Arbitrary ops, record stress, property tests |
| **FrequencyBuckets** | 3 | Arbitrary ops, stress, property tests |
| **GhostList** | 3 | Arbitrary ops, LRU stress, property tests |
| **KeyInterner** | 3 | Arbitrary ops, stress, property tests |
| **IntrusiveList** | 3 | Arbitrary ops, stress, property tests |
| **LazyMinHeap** | 3 | Arbitrary ops, stress, property tests |
| **ShardSelector** | 3 | Arbitrary ops, distribution, property tests |
| **SlotArena** | 3 | Arbitrary ops, stress, property tests |
| **Total** | **27** | All core data structures |

## CI/CD Integration

### Dynamic Target Discovery

**Zero Configuration Required!** 🎉

The CI/CD pipeline automatically discovers all fuzz targets using `cargo fuzz list`. When you add a new fuzz target to `fuzz/fuzz_targets/` and register it in `fuzz/Cargo.toml`, the CI/CD pipeline will automatically:

- ✅ Include it in nightly continuous fuzzing
- ✅ Run it in PR smoke tests (if it matches naming convention)
- ✅ Generate coverage reports for it

**No workflow file updates needed!**

### Naming Convention

For best CI integration, follow this naming convention:

- `<module>_arbitrary_ops.rs` - Random operation sequences (used in PR smoke tests)
- `<module>_stress.rs` - Heavy load testing
- `<module>_property_tests.rs` - Specific invariant testing

Targets ending in `_arbitrary_ops` are automatically used for PR smoke tests as they provide the best general-purpose testing.

### 1. Pull Request Smoke Tests (`.github/workflows/ci.yml`)

**When**: Every pull request
**Duration**: ~60 seconds per `_arbitrary_ops` target
**Purpose**: Catch obvious bugs early before merging

Automatically discovers and runs all `_arbitrary_ops` targets:
```bash
# Automatic discovery
TARGETS=$(cargo fuzz list | grep '_arbitrary_ops$')

# Run each with different seed for reproducibility
for target in $TARGETS; do
  cargo fuzz run "$target" -- -max_total_time=60 -seed=$seed
done
```

**What it catches**:
- Basic crashes and panics
- Obvious logic errors
- Assertion failures
- Memory safety issues

### 2. Continuous Fuzzing (`.github/workflows/fuzz.yml`)

**When**:
- Nightly at 2 AM UTC
- On pushes to main branch
- Manual trigger with configurable duration

**Duration**: 1 hour per target by default (configurable)
**Purpose**: Deep fuzzing to find subtle bugs

**Strategy**:
- **Automatically discovers ALL fuzz targets** using `cargo fuzz list`
- Runs all targets in parallel (GitHub Actions matrix strategy)
- Each target runs independently with its own corpus
- Failures don't block other targets (`fail-fast: false`)

**How Discovery Works**:
```yaml
discover-targets:
  steps:
    - name: List fuzz targets
      run: |
        cd fuzz
        TARGETS=$(cargo fuzz list | jq -R -s -c 'split("\n") | map(select(length > 0))')
        echo "targets=$TARGETS" >> $GITHUB_OUTPUT

fuzz-continuous:
  needs: discover-targets
  strategy:
    matrix:
      target: ${{ fromJson(needs.discover-targets.outputs.targets) }}
```

**Features**:
- **Corpus Management**: Corpora are cached and restored between runs
- **Corpus Minimization**: Automatically minimizes corpus after each run
- **Crash Detection**: Uploads crash artifacts for reproduction
- **Issue Creation**: Automatically creates GitHub issues for crashes found during nightly runs
- **Coverage Reporting**: Generates coverage reports for representative targets

### 3. Corpus Management

Fuzzing corpora (interesting test inputs) are preserved across runs:

```yaml
- name: Restore corpus
  uses: actions/cache@v4
  with:
    path: fuzz/corpus/${{ matrix.target }}
    key: fuzz-corpus-${{ matrix.target }}-${{ github.sha }}
    restore-keys: |
      fuzz-corpus-${{ matrix.target }}-
```

This ensures:
- Accumulated fuzzing progress is preserved
- Previously discovered interesting inputs are retained
- Each run builds on previous fuzzing efforts

### 4. Crash Handling

When crashes are detected:

1. **Artifacts Upload**: Crash inputs are uploaded for 90 days
   ```yaml
   name: fuzz-crashes-${{ matrix.target }}-${{ github.sha }}
   path: fuzz/artifacts/${{ matrix.target }}/
   retention-days: 90
   ```

2. **GitHub Issue Creation** (nightly runs only):
   - Creates issue with `bug`, `fuzzing`, `security` labels
   - Includes reproduction instructions
   - Links to artifact downloads
   - Provides debugging guidance

3. **Workflow Failure**: Pipeline fails to alert maintainers

### 5. Coverage Reporting

For nightly runs, coverage is generated for representative targets:
- One arbitrary_ops target per data structure
- Generates JSON coverage reports
- Uploaded as artifacts for analysis

## Running Fuzzing Locally

### Quick Smoke Test

Run all smoke tests (60 seconds each):
```bash
cd fuzz
./run_smoke_tests.sh
```

Or manually:
```bash
cd fuzz
cargo fuzz run clock_ring_arbitrary_ops -- -max_total_time=60 -seed=1
cargo fuzz run fixed_history_arbitrary_ops -- -max_total_time=60 -seed=2
# ... etc
```

### Deep Fuzzing

Run a single target for extended duration:
```bash
cd fuzz
cargo fuzz run clock_ring_arbitrary_ops -- -max_total_time=3600
```

Run all targets (from fuzz/README.md):
```bash
cd fuzz
for target in $(cargo fuzz list); do
  echo "Fuzzing $target..."
  cargo fuzz run $target -- -max_total_time=300 -jobs=4
done
```

### Reproducing Crashes

If CI finds a crash:

1. Download the crash artifact from the GitHub Actions run
2. Reproduce locally:
   ```bash
   cd fuzz
   cargo fuzz run <target> fuzz/artifacts/<target>/<crash-file>
   ```
3. Debug with full backtrace:
   ```bash
   RUST_BACKTRACE=full cargo fuzz run <target> <crash-file>
   ```

## Monitoring and Maintenance

### Monitoring Fuzzing Health

Check the following regularly:

1. **GitHub Issues**: Look for `[FUZZ]` prefix and `fuzzing` label
2. **Actions Tab**: Review nightly fuzzing workflow results
3. **Corpus Growth**: Ensure corpora are growing over time
4. **Coverage Reports**: Check coverage artifacts for code coverage

### Corpus Maintenance

Periodically minimize corpora to remove redundant inputs:
```bash
cd fuzz
for target in $(cargo fuzz list); do
  cargo fuzz cmin $target
done
```

This is automatically done in CI after each run.

### Adding New Fuzz Targets

When adding new data structures or major features:

1. **Create fuzz target** in `fuzz/fuzz_targets/`:
   ```rust
   // fuzz/fuzz_targets/my_module_arbitrary_ops.rs
   #![no_main]
   use libfuzzer_sys::fuzz_target;
   // ... implementation
   ```

2. **Register in `fuzz/Cargo.toml`**:
   ```toml
   [[bin]]
   name = "my_module_arbitrary_ops"
   path = "fuzz_targets/my_module_arbitrary_ops.rs"
   test = false
   doc = false
   ```

3. **Document in `fuzz/README.md`** (optional but recommended)

**That's it!** ✅ The CI/CD pipeline will automatically discover and run your new target.

**Naming Tips**:
- End with `_arbitrary_ops` to include in PR smoke tests
- End with `_stress` or `_property_tests` for specialized testing (runs in nightly only)
- Use descriptive names like `<module>_<test_type>` for clarity

## Best Practices

### For Contributors

1. **Run smoke tests before pushing**:
   ```bash
   cd fuzz && ./run_smoke_tests.sh
   ```

2. **Fix fuzzing failures immediately**: Fuzzing bugs often indicate real issues

3. **Don't disable fuzz tests**: If a fuzz test is flaky, fix the test or the code

### For Maintainers

1. **Review fuzzing issues promptly**: They often reveal security issues
2. **Keep corpora**: Don't delete cached corpora without good reason
3. **Monitor nightly runs**: Check for patterns in failures
4. **Update fuzz targets**: When APIs change, update fuzz targets accordingly

## Configuration

### Workflow Configuration

Edit `.github/workflows/fuzz.yml` to adjust:
- `duration`: Default 3600 seconds (1 hour), configurable via manual trigger
- `timeout-minutes`: Maximum 120 minutes per target
- `rss_limit_mb`: Memory limit 4096 MB
- `schedule`: Currently runs at 2 AM UTC daily

### Resource Limits

Current limits:
- **Memory**: 4 GB per target
- **Time**: 2 hours maximum (including setup)
- **Parallelism**: All 27 targets run in parallel (GitHub Actions manages)

## Troubleshooting

### Fuzzing Timeouts

If targets timeout:
- Check if the fuzz target is too slow
- Consider reducing complexity or using faster operations
- Adjust `timeout-minutes` in workflow

### Corpus Cache Misses

If corpora aren't being restored:
- Check cache key format in workflow
- Verify artifact upload/download steps
- Check GitHub Actions cache limits (10 GB per repo)

### False Positives

If fuzz tests find "issues" that aren't real bugs:
- Review the fuzz target logic
- Add proper preconditions (prop_assume, input validation)
- Consider if the behavior is actually a bug

## Resources

- [Fuzz Target Documentation](../../fuzz/README.md)
- [libFuzzer Documentation](https://llvm.org/docs/LibFuzzer.html)
- [cargo-fuzz Book](https://rust-fuzz.github.io/book/cargo-fuzz.html)
- [GitHub Actions Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
