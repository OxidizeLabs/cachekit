# Benchmark Quick Start

## 🚀 For Users

### View Latest Results

**Interactive Charts (Recommended)**
```
https://oxidizelabs.github.io/cachekit/benchmarks/latest/charts.html
```

**Markdown Tables**
```
https://oxidizelabs.github.io/cachekit/benchmarks/latest/index.md
```

**Raw JSON Data**
```
https://oxidizelabs.github.io/cachekit/benchmarks/latest/results.json
```

### Compare Versions

```
https://oxidizelabs.github.io/cachekit/benchmarks/v0.1.0/
https://oxidizelabs.github.io/cachekit/benchmarks/v0.2.0/
https://oxidizelabs.github.io/cachekit/benchmarks/latest/
```

---

## 🔧 For Contributors

### Run Benchmarks Locally

```bash
# Quick: One command for everything
./scripts/update_benchmark_docs.sh

# Manual: Step by step
cargo bench --bench runner
cargo run --package bench-support --bin render_docs -- \
    target/benchmarks/<run-id>/results.json \
    docs/benchmarks/latest

# View results
open docs/benchmarks/latest/charts.html
```

### Check CI Status

```bash
# View recent benchmark runs
gh run list --workflow=benchmarks.yml

# View logs from specific run
gh run view <run-id> --log

# Download artifacts
gh run download <run-id> -n benchmark-results-<sha>
```

---

## 🏗️ For Maintainers

### Weekly Benchmarks

**Automatic!** Runs every Monday at 00:00 UTC.

No action required. Check results at:
```
https://oxidizelabs.github.io/cachekit/benchmarks/latest/
```

### Pre-Release Validation

```bash
# Run benchmarks before release
gh workflow run benchmarks.yml

# Wait ~10 minutes
# View results: https://oxidizelabs.github.io/cachekit/benchmarks/latest/
```

### Create Release with Snapshot

```bash
# Tag release (auto-triggers benchmark workflow)
git tag v0.2.0
git push origin v0.2.0

# Workflow automatically:
# 1. Runs benchmarks
# 2. Creates snapshot at docs/benchmarks/v0.2.0/
# 3. Updates latest results
# 4. Deploys to GitHub Pages
```

### Manual Snapshot

```bash
# Create snapshot without tagging
gh workflow run benchmarks.yml -f create_snapshot=v0.2.0-rc1
```

---

## 📊 What Gets Measured

### Hit Rate (8 workloads)
- Uniform, HotSet 90/10, Scan
- Zipfian 1.0, Scrambled Zipfian
- Latest, Scan Resistance, Flash Crowd

### Performance (3 workloads)
- Throughput (Million ops/sec)
- Latency P99 (nanoseconds)

### Specialized Tests
- **Scan Resistance**: Recovery from scan pollution
- **Adaptation Speed**: Response to workload shifts

### Total: ~91 Benchmarks
- 7 policies × (8 hit rate + 3 comprehensive + 2 specialized)

---

## 🎯 Policy Selection

Quick recommendations:

| Use Case | Policy | Why |
|----------|--------|-----|
| General purpose | LRU, S3-FIFO | Best balance |
| Low latency | Clock, LRU | Fastest |
| Scan-heavy | S3-FIFO, Heap-LFU | Scan resistant |
| Frequency-aware | LFU, LRU-K | Track access counts |
| Shifting patterns | 2Q, S3-FIFO | Adapt quickly |

**See full guide in**: `benchmarks/latest/index.md`

---

## ⚙️ Configuration

### Change Benchmark Schedule

Edit `.github/workflows/benchmarks.yml`:

```yaml
schedule:
  - cron: '0 0 * * 1'  # Weekly Monday 00:00
  # Change to:
  # - cron: '0 0 * * *'     # Daily at midnight
  # - cron: '0 0 1 * *'     # Monthly on 1st
```

### Reduce Benchmark Time

Edit `benches/runner.rs`:

```rust
const OPS: usize = 200_000;  // Reduce to 100_000
```

### Use Self-Hosted Runner

Edit `.github/workflows/benchmarks.yml`:

```yaml
runs-on: self-hosted  # Instead of: ubuntu-latest
```

---

## 🐛 Troubleshooting

### Results Not Updating

**Check workflow ran:**
```bash
gh run list --workflow=benchmarks.yml
```

**Manual trigger:**
```bash
gh workflow run benchmarks.yml
```

**Check Jekyll deployment:**
```bash
gh run list --workflow=jekyll-gh-pages.yml
```

### Benchmarks Failing

**View logs:**
```bash
gh run view <run-id> --log
```

**Test locally:**
```bash
cargo bench --bench runner
```

### Old Results Cached

**Clear browser cache** or use:
```bash
curl -I https://oxidizelabs.github.io/cachekit/benchmarks/latest/
```

Check `Last-Modified` header.

---

## 📚 Documentation

- **Methodology**: [Benchmark docs](README.md)
- **CI/CD Details**: `CI_CD_SUMMARY.md`
- **Workflow Docs**: `.github/workflows/README.md`
- **Local Usage**: `benches/README.md`

---

## 🔗 Quick Links

| What | Where |
|------|-------|
| **View Charts** | https://oxidizelabs.github.io/cachekit/benchmarks/latest/charts.html |
| **View Tables** | https://oxidizelabs.github.io/cachekit/benchmarks/latest/index.md |
| **Raw Data** | https://oxidizelabs.github.io/cachekit/benchmarks/latest/results.json |
| **CI Status** | https://github.com/OxidizeLabs/cachekit/actions/workflows/benchmarks.yml |
| **Run Benchmarks** | `./scripts/update_benchmark_docs.sh` |
| **Manual Trigger** | `gh workflow run benchmarks.yml` |

---

**Questions?** Check `.github/workflows/README.md` or open an issue!
