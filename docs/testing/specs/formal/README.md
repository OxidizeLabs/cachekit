# Formal (TLA+) specs

Optional machine-readable specs for finite-state model checking with TLC. **Manual only** — not run in CI.

## Layout

Each policy with TLA+ support lives in its own subdirectory:

```text
formal/
├── fifo/
│   ├── Fifo.tla    # MODULE name must match filename
│   ├── fifo.cfg    # TLC constants and INVARIANT
│   └── tlc.md      # Runbook and alignment log
└── lru/
    ├── Lru.tla
    ├── lru.cfg
    └── tlc.md
```

Human operational specs remain under [`policies/`](../policies/README.md). Contributor guide: [tla-guide.md](../tla-guide.md).

## Run TLC

```bash
./scripts/run-tlc.sh fifo   # or: ./scripts/run-fifo-tlc.sh
./scripts/run-tlc.sh lru    # or: ./scripts/run-lru-tlc.sh
```

Success: no `SemanticOK` violation on the bundled config. See per-policy `tlc.md` runbooks for alignment checklists.
