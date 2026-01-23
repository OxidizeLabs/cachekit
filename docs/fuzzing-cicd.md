# Fuzzing in CI/CD

This document explains how fuzzing is integrated into the CacheKit CI/CD pipeline.

## Overview

CacheKit uses a multi-layered fuzzing approach in CI:

1. **Quick fuzz tests on PRs** - Fast smoke tests (60s per target)
2. **Long-running continuous fuzzing** - Nightly 1-hour runs
3. **Manual fuzzing** - On-demand with custom duration
4. **Corpus management** - Automatic corpus preservation and minimization

## CI Workflows

### 1. Quick Fuzz Tests (ci.yml)

**Triggers**: Every pull request

**Duration**: ~3 minutes total

**Purpose**: Catch obvious bugs before merge

```yaml
fuzz-quick:
  - Runs 3 fuzz targets for 60 seconds each
  - Uses deterministic seeds for reproducibility
  - Runs in-tree fuzz smoke tests
  - Fast feedback on PRs
```

**What it catches**:
- Panics from new code
- Obvious assertion failures
- Simple invariant violations
- Regressions in existing fuzz cases

### 2. Continuous Fuzzing (fuzz.yml)

**Triggers**:
- Push to main branch
- Nightly at 2 AM UTC
- Manual dispatch

**Duration**: 1 hour per target (configurable)

**Purpose**: Deep bug hunting and corpus building

```yaml
fuzz-continuous:
  - Runs all targets in parallel
  - 1 hour per target by default
  - Corpus caching and minimization
  - Automatic issue creation on crashes
  - Coverage reporting
```

**What it catches**:
- Deep edge cases requiring many mutations
- Complex state machine bugs
- Race conditions (with longer runs)
- Subtle invariant violations

## Workflow Details

### Quick Fuzz on PRs

```yaml
# .github/workflows/ci.yml
fuzz-quick:
  name: Fuzz Tests (Quick)
  runs-on: ubuntu-latest
  if: github.event_name == 'pull_request'
  steps:
    - Install nightly Rust
    - Install cargo-fuzz
    - Run each target for 60s with fixed seed
    - Run in-tree smoke tests
```

**Execution**:
```bash
cargo fuzz run clock_ring_arbitrary_ops -- -max_total_time=60 -seed=1
cargo fuzz run clock_ring_insert_stress -- -max_total_time=60 -seed=2
cargo fuzz run clock_ring_eviction_patterns -- -max_total_time=60 -seed=3
```

**When it fails**: PR is blocked until fixed

### Continuous Fuzzing

```yaml
# .github/workflows/fuzz.yml
fuzz-continuous:
  name: Fuzz ${{ matrix.target }}
  runs-on: ubuntu-latest
  strategy:
    matrix:
      target: [arbitrary_ops, insert_stress, eviction_patterns]
  steps:
    - Restore corpus from cache
    - Run fuzz target for 1 hour
    - Minimize corpus
    - Save corpus to cache
    - Check for crashes
    - Create GitHub issue if crashes found
    - Upload crash artifacts
```

**Corpus Management**:
- Each target has its own corpus directory
- Corpus cached between runs (key: `fuzz-corpus-<target>-<sha>`)
- Automatically minimized after each run
- Interesting inputs preserved across runs

**Crash Handling**:
1. Crashes saved to `fuzz/artifacts/<target>/`
2. Uploaded as GitHub Actions artifacts (90-day retention)
3. GitHub issue automatically created with reproduction steps
4. Workflow marked as failed

### Coverage Reporting

```yaml
fuzz-coverage:
  name: Fuzz Coverage Report
  runs-on: ubuntu-latest
  needs: fuzz-continuous
  if: always() && github.event_name == 'schedule'
  steps:
    - Generate coverage for each target
    - Upload coverage reports
    - Post summary to workflow
```

**Output**: Coverage JSON showing which code paths were exercised

## Manual Fuzzing

You can manually trigger long-running fuzz sessions:

**Via GitHub UI**:
1. Go to Actions → Continuous Fuzzing
2. Click "Run workflow"
3. Set duration (default: 3600s = 1 hour)
4. Click "Run workflow"

**Custom duration example**:
```yaml
inputs:
  duration: 7200  # 2 hours
```

## Interpreting Results

### Successful Run

```
✅ All fuzz targets passed
No crashes found in any target
Corpus size: 234 inputs
```

### Crashes Found

```
⚠️ Some fuzz targets found issues

## Crashes found in clock_ring_arbitrary_ops
- crash-a1b2c3d4e5f6...
- crash-f6e5d4c3b2a1...
```

**GitHub Issue Created**:
- Title: `[FUZZ] Crashes found in clock_ring_arbitrary_ops`
- Labels: `bug`, `fuzzing`, `security`
- Body: Reproduction steps and artifact links

## Corpus Evolution

The corpus grows over time as fuzzing finds interesting inputs:

```
Day 1:  50 inputs, 60% coverage
Day 7:  150 inputs, 75% coverage
Day 30: 300 inputs, 85% coverage
```

**Corpus is**:
- Cached in GitHub Actions cache
- Minimized to remove redundant inputs
- Shared across runs on the same target
- Can be committed to git for regression testing

## Best Practices

### 1. Monitor Fuzzing Results

Check the Continuous Fuzzing workflow regularly:
- Look for new crashes
- Monitor coverage trends
- Review corpus growth

### 2. Fix Crashes Promptly

When fuzzing finds a crash:
1. Download the crash artifact
2. Reproduce locally: `cargo fuzz run <target> fuzz/artifacts/<target>/crash-...`
3. Debug and fix the issue
4. Add a regression test
5. Verify fix with local fuzzing

### 3. Commit Important Corpus Entries

After fixing a bug:
```bash
# Copy crash input to corpus for regression testing
cp fuzz/artifacts/<target>/crash-abc123 fuzz/corpus/<target>/
git add fuzz/corpus/<target>/crash-abc123
git commit -m "Add regression test for issue #123"
```

### 4. Extend Fuzzing for Major Changes

For significant refactors or new features:
```bash
# Run locally for extended period
cd fuzz
cargo fuzz run clock_ring_arbitrary_ops -- -max_total_time=7200
```

Or trigger manual GitHub Actions run with longer duration.

### 5. Keep Corpus Manageable

Regularly minimize corpus:
```bash
cd fuzz
cargo fuzz cmin clock_ring_arbitrary_ops
```

CI does this automatically, but local minimization helps too.

## Troubleshooting

### "cargo-fuzz not found"

CI installs it automatically. Locally:
```bash
cargo install cargo-fuzz
```

### "Fuzzing timeout"

Increase timeout in workflow:
```yaml
timeout-minutes: 120  # Increase if needed
```

### "Corpus cache too large"

GitHub Actions cache limit is 10GB per repo. If exceeded:
1. Minimize corpus more aggressively
2. Clear old cache entries manually
3. Reduce corpus size in workflow

### "False positive crashes"

If a crash is expected behavior:
1. Add filtering in fuzz target
2. Document why it's expected
3. Consider if it should be an error instead

## Performance Optimization

### Faster Fuzzing

```yaml
# Use more workers
cargo fuzz run <target> -- -workers=4

# Reduce memory limit for faster restarts
cargo fuzz run <target> -- -rss_limit_mb=2048
```

### Better Coverage

```yaml
# Use different dictionaries
cargo fuzz run <target> -- -dict=fuzz/dictionaries/clockring.dict

# Try different mutation strategies
cargo fuzz run <target> -- -mutate_depth=5
```

## Integration with Other Tools

### OSS-Fuzz (Optional)

For continuous fuzzing at scale:
1. Submit project to OSS-Fuzz
2. They run fuzz tests 24/7
3. Private bug reports for security issues
4. Free for open-source projects

### ClusterFuzz (Optional)

Google's fuzzing infrastructure:
- Massive parallel fuzzing
- Automatic bug filing
- Regression verification

## Metrics

Track these fuzzing metrics over time:

- **Corpus Size**: Number of unique inputs
- **Coverage**: Percentage of code exercised
- **Exec/sec**: Fuzzing throughput
- **Crashes Found**: Total bugs discovered
- **Time to First Crash**: How fast bugs are found

## Related Documentation

- [Testing Strategy](testing.md) - Overall testing approach
- [Fuzz README](../fuzz/README.md) - Local fuzzing guide
- [ClockRing Tests](../src/ds/clock_ring.rs) - Property and fuzz tests
- [Contributing](../CONTRIBUTING.md) - How to contribute

## References

- [libFuzzer Documentation](https://llvm.org/docs/LibFuzzer.html)
- [cargo-fuzz Book](https://rust-fuzz.github.io/book/cargo-fuzz.html)
- [Fuzzing in Rust](https://rust-fuzz.github.io/book/)
