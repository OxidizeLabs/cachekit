#!/usr/bin/env bash
# Run TLC on the FIFO TLA+ spec (manual check, not CI).
# Thin alias for scripts/run-tlc.sh — see docs/testing/specs/tla-guide.md

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "${ROOT}/scripts/run-tlc.sh" Fifo fifo.cfg
