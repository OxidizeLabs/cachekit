#!/usr/bin/env bash
# Quick smoke test for all fuzz targets
# Automatically discovers and runs representative targets (60 seconds each)

set -e

echo "🔥 Running fuzz smoke tests (60 seconds per target)..."
echo ""

# Discover targets (prefer _arbitrary_ops as they're most representative)
echo "Discovering fuzz targets..."
ARBITRARY_TARGETS=$(cargo fuzz list | grep '_arbitrary_ops$' || true)

if [ -n "$ARBITRARY_TARGETS" ]; then
    echo "Using arbitrary_ops targets (representative subset)"
    TARGETS=()
    while IFS= read -r line; do
        [ -n "$line" ] && TARGETS+=("$line")
    done <<< "$ARBITRARY_TARGETS"
else
    echo "No arbitrary_ops targets found, using all targets"
    TARGETS=()
    while IFS= read -r line; do
        [ -n "$line" ] && TARGETS+=("$line")
    done < <(cargo fuzz list)
fi

echo "Found ${#TARGETS[@]} target(s):"
printf '  - %s\n' "${TARGETS[@]}"
echo ""

FAILED=()
PASSED=()

for i in "${!TARGETS[@]}"; do
    target="${TARGETS[$i]}"
    seed=$((i + 1))

    echo "[$((i + 1))/${#TARGETS[@]}] Fuzzing $target..."

    if cargo fuzz run "$target" -- -max_total_time=60 -seed="$seed" -print_final_stats=1 2>&1 | tee "/tmp/fuzz_${target}.log"; then
        PASSED+=("$target")
        echo "✅ $target passed"
    else
        FAILED+=("$target")
        echo "❌ $target failed"
    fi

    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Summary:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Passed: ${#PASSED[@]}/${#TARGETS[@]}"
echo "❌ Failed: ${#FAILED[@]}/${#TARGETS[@]}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed targets:"
    for target in "${FAILED[@]}"; do
        echo "  - $target (see /tmp/fuzz_${target}.log)"
    done
    exit 1
fi

echo ""
echo "🎉 All smoke tests passed!"
