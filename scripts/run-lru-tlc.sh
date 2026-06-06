#!/usr/bin/env bash
# Run TLC on the LRU TLA+ spec (manual check, not CI).
# See docs/testing/specs/lru-tlc.md

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "${ROOT}/scripts/run-tlc.sh" Lru lru.cfg
