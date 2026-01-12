#!/usr/bin/env bash
set -euo pipefail

criterion_dir="${1:-target/criterion}"
benchmarks_md="${2:-docs/benchmarks.md}"

mean_ns() {
  local dir="$1"
  local est="$dir/new/estimates.json"
  if [[ ! -f "$est" ]]; then
    echo "TBD"
    return 0
  fi
  jq -r '.mean.point_estimate' "$est"
}

melems_from() {
  local dir="$1"
  local bench_json="$dir/new/benchmark.json"
  local est_json="$dir/new/estimates.json"
  local elements ns

  if [[ ! -f "$bench_json" || ! -f "$est_json" ]]; then
    echo "TBD"
    return 0
  fi

  elements="$(jq -r '.throughput.Elements // empty' "$bench_json")"
  ns="$(jq -r '.mean.point_estimate // empty' "$est_json")"
  if [[ -z "$elements" || -z "$ns" || "$elements" == "0" || "$ns" == "0" ]]; then
    echo "TBD"
    return 0
  fi

  awk -v e="$elements" -v ns="$ns" 'BEGIN { printf "%.2f", (e * 1e9 / ns) / 1e6 }'
}

melems_first_of() {
  local path
  for path in "$@"; do
    if [[ -d "$path" ]]; then
      melems_from "$path"
      return 0
    fi
  done
  echo "TBD"
}

ns_int() {
  local v="$1"
  if [[ "$v" == "TBD" ]]; then
    echo "TBD"
  else
    echo "${v%.*}"
  fi
}

if [[ ! -f "$benchmarks_md" ]]; then
  echo "benchmarks markdown not found: $benchmarks_md" >&2
  exit 1
fi

lru_get="$(mean_ns "$criterion_dir/lru_get_hit_ns")"
lru_insert="$(mean_ns "$criterion_dir/lru_insert_full_ns")"
lru_k_get="$(mean_ns "$criterion_dir/lru_k_get_hit_ns")"
lru_k_insert="$(mean_ns "$criterion_dir/lru_k_insert_full_ns")"
lfu_get="$(mean_ns "$criterion_dir/lfu_get_hit_ns")"
lfu_insert="$(mean_ns "$criterion_dir/lfu_insert_full_ns")"
lfu_touch="$(mean_ns "$criterion_dir/lfu_policy_only_touch_ns")"

lru_insert_get="$(melems_from "$criterion_dir/lru_policy/insert_get")"
lru_eviction="$(melems_from "$criterion_dir/lru_policy/eviction_churn")"
lru_pop="$(melems_from "$criterion_dir/lru_policy/pop_lru")"
lru_hot="$(melems_from "$criterion_dir/lru_policy/touch_hotset")"

lru_k_insert_get="$(melems_from "$criterion_dir/lru_k_policy/insert_get")"
lru_k_eviction="$(melems_from "$criterion_dir/lru_k_policy/eviction_churn")"
lru_k_pop="$(melems_from "$criterion_dir/lru_k_policy/pop_lru_k")"
lru_k_hot="$(melems_from "$criterion_dir/lru_k_policy/touch_hotset")"

lfu_insert_get="$(melems_from "$criterion_dir/lfu_policy/insert_get")"
lfu_eviction="$(melems_from "$criterion_dir/lfu_policy/eviction_churn")"
lfu_pop="$(melems_first_of "$criterion_dir/lfu_pop_lfu_policy" "$criterion_dir/lfu_policy/pop_lfu")"
lfu_hot="$(melems_first_of "$criterion_dir/lfu_get_hotset_policy" "$criterion_dir/lfu_policy/get_hotset")"

lru_uniform="$(melems_from "$criterion_dir/lru_workload_hit_rate/uniform")"
lru_hotset="$(melems_from "$criterion_dir/lru_workload_hit_rate/hotset_90_10")"
lru_scan="$(melems_from "$criterion_dir/lru_workload_hit_rate/scan")"

lru_k_uniform="$(melems_from "$criterion_dir/lru_k_workload_hit_rate/uniform")"
lru_k_hotset="$(melems_from "$criterion_dir/lru_k_workload_hit_rate/hotset_90_10")"
lru_k_scan="$(melems_from "$criterion_dir/lru_k_workload_hit_rate/scan")"

lfu_uniform="$(melems_from "$criterion_dir/lfu_workload_hit_rate/uniform")"
lfu_hotset="$(melems_from "$criterion_dir/lfu_workload_hit_rate/hotset_90_10")"
lfu_scan="$(melems_from "$criterion_dir/lfu_workload_hit_rate/scan")"

tmp_latest="$(mktemp)"
cat >"$tmp_latest" <<EOF
Micro-ops (ns/op):

| Cache | get_hit | insert_full | policy_only_touch |
| --- | --- | --- | --- |
| LRU | $(ns_int "$lru_get") | $(ns_int "$lru_insert") | n/a |
| LRU-K | $(ns_int "$lru_k_get") | $(ns_int "$lru_k_insert") | n/a |
| LFU | $(ns_int "$lfu_get") | $(ns_int "$lfu_insert") | $(ns_int "$lfu_touch") |

Policy throughput (Melem/s = million operations per second):

| Cache | insert_get | eviction_churn | pop | touch_hotset |
| --- | --- | --- | --- | --- |
| LRU | $lru_insert_get | $lru_eviction | $lru_pop | $lru_hot |
| LRU-K | $lru_k_insert_get | $lru_k_eviction | $lru_k_pop | $lru_k_hot |
| LFU | $lfu_insert_get | $lfu_eviction | $lfu_pop | $lfu_hot |

Workload throughput (Melem/s, 200k ops):

| Cache | uniform | hotset_90_10 | scan |
| --- | --- | --- | --- |
| LRU | $lru_uniform | $lru_hotset | $lru_scan |
| LRU-K | $lru_k_uniform | $lru_k_hotset | $lru_k_scan |
| LFU | $lfu_uniform | $lfu_hotset | $lfu_scan |
EOF

tmp_out="$(mktemp)"
awk -v latest="$tmp_latest" '
  /<!-- LATEST_RUN_START -->/ {
    print;
    while ((getline line < latest) > 0) print line;
    in_block=1;
    next
  }
  /<!-- LATEST_RUN_END -->/ {
    in_block=0;
    print;
    next
  }
  !in_block { print }
' "$benchmarks_md" >"$tmp_out"

mv "$tmp_out" "$benchmarks_md"
rm -f "$tmp_latest"
