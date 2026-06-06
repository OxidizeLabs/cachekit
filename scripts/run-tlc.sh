#!/usr/bin/env bash
# Run TLC on a TLA+ spec (manual check, not CI).
# Usage: ./scripts/run-tlc.sh [Module] [config]
# Default: Fifo fifo.cfg
# See docs/testing/specs/tla-guide.md

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPECS="${ROOT}/docs/testing/specs"

MODULE="${1:-Fifo}"
CONFIG="${2:-fifo.cfg}"

if ! command -v tlc >/dev/null 2>&1; then
    echo "error: tlc not found on PATH" >&2
    echo "Install TLA+ tools: https://github.com/tlaplus/tlaplus/releases" >&2
    exit 1
fi

cd "${SPECS}"
echo "Running TLC: ${CONFIG} + ${MODULE}.tla"
tlc -config "${CONFIG}" "${MODULE}.tla"
echo ""
echo "TLC: invariant check passed (${MODULE})."
echo "Guide: docs/testing/specs/tla-guide.md"
