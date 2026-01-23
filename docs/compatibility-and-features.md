# Compatibility and Features

## MSRV

CacheKit targets the Rust MSRV listed in `Cargo.toml` and reflected in the README badge.

## Feature Flags

- `metrics` — Enables hit/miss metrics and snapshots.
- `concurrency` — Enables concurrent wrappers (requires `parking_lot`).

## Optional Dependencies

- `parking_lot` — Used for concurrent wrappers behind the `concurrency` feature.
