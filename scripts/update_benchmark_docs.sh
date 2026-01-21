#!/bin/bash
set -e

# Update benchmark documentation by running benchmarks and rendering to docs/
#
# Usage:
#   ./scripts/update_benchmark_docs.sh [--skip-bench]
#
# Options:
#   --skip-bench: Skip running benchmarks, just re-render existing results

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SKIP_BENCH=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --skip-bench)
            SKIP_BENCH=true
            shift
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--skip-bench]"
            exit 1
            ;;
    esac
done

echo "🚀 Updating benchmark documentation..."
echo

# Step 1: Run benchmarks (unless skipped)
if [ "$SKIP_BENCH" = false ]; then
    echo "📊 Running benchmarks..."
    cargo bench --bench runner --quiet
    echo "✅ Benchmarks complete"
    echo
fi

# Step 2: Find the latest results
LATEST_RESULTS=$(find target/benchmarks -name "results.json" -type f | sort -r | head -n 1)

if [ -z "$LATEST_RESULTS" ]; then
    echo "❌ No benchmark results found in target/benchmarks/"
    echo "   Run: cargo bench --bench runner"
    exit 1
fi

echo "📄 Using results: $LATEST_RESULTS"
echo

# Step 3: Render documentation
echo "📝 Rendering documentation..."
cargo run --quiet --package bench-support --bin render_docs -- \
    "$LATEST_RESULTS" \
    docs/benchmarks/latest

echo
echo "✅ Documentation updated:"
echo "   - docs/benchmarks/latest/index.md"
echo "   - docs/benchmarks/latest/results.json"
echo
echo "🌐 View locally:"
echo "   open docs/benchmarks/latest/index.md"
echo "   # Or use Jekyll: cd docs && bundle exec jekyll serve"
