# CI/CD release cycle (Rust crate)

This document describes a practical CI/CD software release cycle for a Rust library crate using GitHub Actions. It’s designed to keep PR feedback fast, keep `main` always releasable, and make releases repeatable and auditable.

## Goals

- Catch issues early (format, lint, tests) on every PR.
- Keep `main` green and always releasable (protected branch + required checks).
- Make releases reproducible (tagged source, locked toolchain, deterministic steps).
- Separate “fast PR checks” from “slow/deep validation” (Miri/bench).

## Branch + versioning model

- **Default branch:** `main`
- **Feature work:** branch from `main`, merge via PR.
- **Versioning:** SemVer in `Cargo.toml` and `CHANGELOG.md`.
- **Release signal:** annotated git tag `vX.Y.Z` created from a commit on `main`.

Recommended repo settings:

- Protect `main`:
  - Require PRs
  - Require status checks to pass (CI jobs)
  - Require up-to-date branches before merge
  - Require linear history (optional but helps)
- Use CODEOWNERS or required reviewers for release-sensitive areas (optional).

## Pipelines and required workflows

### 1) PR / main CI pipeline (`.github/workflows/ci.yml`)

Triggers:

- `pull_request` to `main`
- `push` to `main`

Required checks (typical):

- **Format:** `cargo fmt --check`
- **Lint:** `cargo clippy --all-targets --all-features -- -D warnings`
- **Test:** `cargo test --all-features --all-targets` (run on Linux, optionally also macOS/Windows)
- **Docs:** `cargo doc --no-deps --all-features` (treat warnings as errors)
- **MSRV:** `cargo check --all-features` on the project’s declared MSRV toolchain
- **Security audit:** `cargo audit` (or `rustsec/audit-check`)

Optional (usually *not* required to merge due to slowness/flakiness):

- **Miri:** run on Linux (and optionally macOS). Do not run on Windows.
- **Benchmarks:** run only on `main` or on demand (not required for PR merge).

Suggested commands:

- Tests (thorough): `cargo test --all-features --all-targets`
- Tests (fast PR): `cargo test`
- Docs: `RUSTDOCFLAGS='-Dwarnings' cargo doc --no-deps --all-features`

### 2) Release pipeline (`.github/workflows/release.yml`)

Purpose: produce a GitHub Release (notes + artifacts) and publish to crates.io (optional), only from a version tag.

Triggers:

- `push` tags: `v*.*.*`

Recommended release steps:

1. **Validate tag source:** ensure tag points to a commit on `main` (optional but recommended).
2. **Run full validation again** (or assert CI was green on the tagged commit):
   - `cargo fmt --check`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo test --all-features --all-targets`
   - `cargo doc --no-deps --all-features`
3. **Build artifacts** (optional, if you ship binaries or example builds):
   - `cargo build --release`
4. **Package verification** before publish:
   - `cargo package`
   - `cargo publish --dry-run`
5. **Create GitHub Release**:
   - Use generated notes or extract from `CHANGELOG.md`
6. **Publish to crates.io** (optional):
   - `cargo publish`
   - Requires `CARGO_REGISTRY_TOKEN` secret

If you ship binaries, add a matrix to build per OS and upload artifacts to the GitHub Release.

### 3) Scheduled security + drift checks (`.github/workflows/maintenance.yml`)

Purpose: catch dependency advisories and ecosystem drift even when there are no PRs.

Triggers:

- `schedule` (e.g., weekly)
- `workflow_dispatch` (manual)

Recommended jobs:

- `cargo audit` / `rustsec/audit-check`
- `cargo update -w --dry-run` (optional, informational)
- (Optional) run CI with the latest stable toolchain if you pin a toolchain, to detect upcoming breakages.

### 4) Documentation publishing (optional)

Only needed if you publish docs to GitHub Pages.

Options:

- Publish `cargo doc` output to `gh-pages` (for crate docs).
- Build and publish a separate `docs/` site (e.g., MkDocs / mdBook / Jekyll).

## Typical software release cycle (end-to-end)

1. **Development**
   - Work on a branch, run locally:
     - `cargo fmt`
     - `cargo clippy --all-targets --all-features -- -D warnings`
     - `cargo test --all-features --all-targets`
2. **Pull request**
   - CI runs required checks.
   - Review and iterate until green.
3. **Merge to `main`**
   - CI runs again on `main`.
   - `main` stays green (releasable).
4. **Release preparation**
   - Update `CHANGELOG.md` and bump `Cargo.toml` version.
   - Open PR “Release vX.Y.Z”.
5. **Tag and release**
   - After merge, create tag `vX.Y.Z` on `main`.
   - Release workflow runs full validation, builds artifacts, publishes release (and crates.io if configured).
6. **Post-release**
   - Verify published crate metadata and docs.
   - Triage any newly reported issues.

## What to require vs. what to keep optional

Recommended **required for merge**:

- fmt, clippy, tests (at least Linux), docs build, MSRV check, audit

Recommended **optional / non-blocking**:

- Miri (Linux/macOS only), benchmarks, extended multi-OS test matrices, stress/perf tests

## Commands cheat sheet (CI-friendly defaults)

- Format: `cargo fmt --check`
- Lint: `cargo clippy --all-targets --all-features -- -D warnings`
- Test (thorough): `cargo test --all-features --all-targets`
- Test (workspace): `cargo test --workspace --all-features --all-targets`
- Docs: `RUSTDOCFLAGS='-Dwarnings' cargo doc --no-deps --all-features`
- MSRV (example): `cargo +1.85.0 check --all-features`
- Audit: `cargo audit`
