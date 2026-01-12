#!/usr/bin/env bash
set -euo pipefail

if ! command -v jq >/dev/null 2>&1; then
  echo "skipping: jq not found" >&2
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script="$repo_root/scripts/update_docs_benchmarks.sh"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

criterion_dir="$tmp/criterion"
docs_dir="$tmp/docs"
mkdir -p "$criterion_dir" "$docs_dir"

write_estimates() {
  local dir="$1"
  local mean="$2"
  mkdir -p "$dir/new"
  cat >"$dir/new/estimates.json" <<EOF
{ "mean": { "point_estimate": $mean } }
EOF
}

write_throughput() {
  local dir="$1"
  local elements="$2"
  mkdir -p "$dir/new"
  cat >"$dir/new/benchmark.json" <<EOF
{ "throughput": { "Elements": $elements } }
EOF
}

write_policy_bench() {
  local dir="$1"
  local elements="$2"
  local mean_ns="$3"
  write_throughput "$dir" "$elements"
  write_estimates "$dir" "$mean_ns"
}

# Micro-ops (ns)
write_estimates "$criterion_dir/lru_get_hit_ns" 30.9
write_estimates "$criterion_dir/lru_insert_full_ns" 138.2
write_estimates "$criterion_dir/lru_k_get_hit_ns" 43.1
write_estimates "$criterion_dir/lru_k_insert_full_ns" 189.9
write_estimates "$criterion_dir/lfu_get_hit_ns" 87.5
write_estimates "$criterion_dir/lfu_insert_full_ns" 195.0
write_estimates "$criterion_dir/lfu_policy_only_touch_ns" 66.7

# Policy throughput (Elements/op and mean ns/op -> Melem/s)
write_policy_bench "$criterion_dir/lru_policy/insert_get" 2048 192900.0
write_policy_bench "$criterion_dir/lru_policy/eviction_churn" 4096 630000.0
write_policy_bench "$criterion_dir/lru_policy/pop_lru" 1024 58800.0
write_policy_bench "$criterion_dir/lru_policy/touch_hotset" 1024 40800.0

write_policy_bench "$criterion_dir/lru_k_policy/insert_get" 2048 320000.0
write_policy_bench "$criterion_dir/lru_k_policy/eviction_churn" 4096 950000.0
write_policy_bench "$criterion_dir/lru_k_policy/pop_lru_k" 1024 90000.0
write_policy_bench "$criterion_dir/lru_k_policy/touch_hotset" 1024 70000.0

write_policy_bench "$criterion_dir/lfu_policy/insert_get" 2048 270000.0
write_policy_bench "$criterion_dir/lfu_policy/eviction_churn" 4096 980000.0

# Workload throughput
write_policy_bench "$criterion_dir/lru_workload_hit_rate/uniform" 200000 28700000.0
write_policy_bench "$criterion_dir/lru_workload_hit_rate/hotset_90_10" 200000 10000000.0
write_policy_bench "$criterion_dir/lru_workload_hit_rate/scan" 200000 27500000.0

write_policy_bench "$criterion_dir/lru_k_workload_hit_rate/uniform" 200000 36000000.0
write_policy_bench "$criterion_dir/lru_k_workload_hit_rate/hotset_90_10" 200000 12500000.0
write_policy_bench "$criterion_dir/lru_k_workload_hit_rate/scan" 200000 40000000.0

write_policy_bench "$criterion_dir/lfu_workload_hit_rate/uniform" 200000 42000000.0
write_policy_bench "$criterion_dir/lfu_workload_hit_rate/hotset_90_10" 200000 24000000.0
write_policy_bench "$criterion_dir/lfu_workload_hit_rate/scan" 200000 41000000.0

# Benchmarks doc with replacement markers.
benchmarks_md="$docs_dir/benchmarks.md"
cat >"$benchmarks_md" <<'EOF'
# Benchmarks

## Latest run (release)

<!-- LATEST_RUN_START -->
placeholder
<!-- LATEST_RUN_END -->

## Release summary
EOF

"$script" "$criterion_dir" "$benchmarks_md"

grep -q "| LRU | 30 | 138 | n/a |" "$benchmarks_md"
grep -q "| LRU-K | 43 | 189 | n/a |" "$benchmarks_md"
grep -q "| LFU | 87 | 195 | 66 |" "$benchmarks_md"

# Ensure missing LFU pop/hotset benches become TBD (we didn't create them).
grep -q "^| LFU | .* | .* | TBD | TBD |$" "$benchmarks_md"

# Ensure marker block still exists and placeholder is gone.
grep -q "<!-- LATEST_RUN_START -->" "$benchmarks_md"
grep -q "<!-- LATEST_RUN_END -->" "$benchmarks_md"
! grep -q "^placeholder$" "$benchmarks_md"

echo "ok: update_docs_benchmarks.sh"
