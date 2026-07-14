#!/usr/bin/env bash
# Run TLC on a TLA+ spec (manual check, not CI).
# Usage: ./scripts/run-tlc.sh [policy]
#   policy: fifo (default) | lru
# See docs/testing/specs/tla-guide.md

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPECS="${ROOT}/docs/testing/specs"

POLICY="${1:-fifo}"

case "${POLICY}" in
    Fifo|fifo)
        SLUG="fifo"
        FORMAL_DIR="${SPECS}/formal/fifo"
        MODULE="Fifo"
        CONFIG="fifo.cfg"
        ;;
    Lru|lru)
        SLUG="lru"
        FORMAL_DIR="${SPECS}/formal/lru"
        MODULE="Lru"
        CONFIG="lru.cfg"
        ;;
    *)
        echo "error: unknown policy '${POLICY}' (expected fifo or lru)" >&2
        exit 1
        ;;
esac

if ! command -v tlc >/dev/null 2>&1; then
    echo "error: tlc not found on PATH" >&2
    echo "Install TLA+ tools: https://github.com/tlaplus/tlaplus/releases" >&2
    exit 1
fi

cd "${FORMAL_DIR}"
echo "Running TLC: ${CONFIG} + ${MODULE}.tla"
tlc -config "${CONFIG}" "${MODULE}.tla"
echo ""
echo "TLC: invariant check passed (${MODULE})."
echo "Guide: docs/testing/specs/tla-guide.md"
echo "Runbook: docs/testing/specs/formal/${SLUG}/tlc.md"
