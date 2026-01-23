# Benchmarks

This page links to the latest benchmark reports and release-tag snapshots.

## Quick Links

- **[📊 Interactive Charts](benchmarks/latest/charts.html)** - Visual comparison with Chart.js
- **[📄 Latest Results](benchmarks/latest/)** - Comprehensive Markdown tables
- **[📁 Raw JSON Data](benchmarks/latest/results.json)** - For tooling and custom analysis

## Automated Reports

Benchmark reports are **automatically generated weekly** via GitHub Actions and published to GitHub Pages.

**CI/CD Pipeline:**
- ⏰ **Schedule:** Weekly (Monday 00:00 UTC)
- 🏷️ **Releases:** Automatic on version tags (`v*`)
- 🔧 **Manual:** `gh workflow run benchmarks.yml`

**Run Locally:**
```bash
./scripts/update_benchmark_docs.sh
```

This runs the full benchmark suite and renders results to [docs/benchmarks/latest/](benchmarks/latest/).
