# CI/CD release cycle

This document explains how CacheKit’s release cycle works in practice: what runs in CI,
what makes `main` “releasable”, and how a version becomes a GitHub Release and (optionally)
a crates.io publish.

For the hands-on, step-by-step release procedure, see [Releasing CacheKit](releasing.md).

## Mental model

- **All change goes through PRs into `main`.**
- **`main` should always be releasable.** CI is set up so that obvious breakages are caught
  before merge (format/lint/tests/docs/audit/MSRV).
- **A “release” is signaled by a git tag.**
  - Stable releases: `vX.Y.Z` (example: `v0.1.0`)
  - Pre-releases: `vX.Y.Z-alpha` / `vX.Y.Z-rc.1`, etc.
- **Crate versions do not use `v`.** In `Cargo.toml`, use `0.1.0-alpha`, not `v0.1.0-alpha`.

## What runs when

### Pull requests and pushes to `main` (`.github/workflows/ci.yml`)

CI runs on:

- `pull_request` targeting `main`
- `push` to `main`

What it does (project-specific):

- **Formatting:** `cargo fmt --check`
- **Lint:** `cargo clippy --all-targets --all-features -- -D warnings`
- **Tests:** `cargo test --all-features --all-targets` on Linux/macOS/Windows
- **Docs build:** `RUSTDOCFLAGS='-Dwarnings' cargo doc --no-deps --all-features`
- **Security audit:** `rustsec/audit-check`
- **Benchmarks:** `cargo bench --no-fail-fast` (only on `main`, not required for PRs)
- **MSRV check:** `cargo check --all-features` on Rust `1.85.0`
- **Miri:** curated subset on nightly for Linux/macOS (slower; not on Windows)

Local equivalents (useful before opening a PR):

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features --all-targets
RUSTDOCFLAGS='-Dwarnings' cargo doc --no-deps --all-features
```

### Release automation on tags (`.github/workflows/release.yml`)

The release workflow runs on:

- `push` tags matching `v*.*.*`

Important:

- This tag pattern matches stable tags like `v0.1.0`.
- It does **not** match pre-release tags like `v0.1.0-alpha` or `v0.2.0-rc.1` unless you
  expand the trigger pattern.

What the workflow does:

1. Verifies the tag commit is reachable from `origin/main`.
2. Re-runs full validation: fmt, clippy, tests, docs.
3. Verifies the crate packages cleanly: `cargo package` and `cargo publish --dry-run`.
4. Creates a GitHub Release (currently using auto-generated release notes).
5. Optionally publishes to crates.io.

crates.io publishing:

- The publish job only runs if the repository variable `CARGO_REGISTRY_TOKEN` is set.
- When enabled, the job runs `cargo publish` using that token.

### Scheduled maintenance (`.github/workflows/maintenance.yml`)

Maintenance runs on a schedule and manually (`workflow_dispatch`) to catch:

- New RustSec advisories and dependency issues
- Ecosystem drift when there isn’t active development

### Docs site publishing (`.github/workflows/jekyll-gh-pages.yml`)

On each push to `main` (or manually), GitHub Pages builds the site from `docs/` via Jekyll.
This is for the documentation site under the repo’s Pages URL (not `cargo doc` output).

## How a release happens (end-to-end)

### 1) Prepare a release PR

In a PR that targets `main`:

- Bump `Cargo.toml` `version = "X.Y.Z(...)"` (no `v`).
- Finalize `CHANGELOG.md` for that version/date.
- (Optional) Update [Benchmarks](benchmarks.md) after a local run:
  - `cargo bench`
  - `scripts/update_docs_benchmarks.sh target/criterion docs/benchmarks.md`

### 2) Merge to `main`

Merge once CI is green. This keeps `main` always in a releasable state.

### 3) Tag the release

Tag the exact commit on `main` you want to release.

- **Stable tags** (`vX.Y.Z`) will trigger the automated release workflow.
- **Pre-release tags** (`vX.Y.Z-alpha`, etc.) currently will not trigger the release
  workflow with the default tag pattern.

### 4) Watch the automation

For stable tags, GitHub Actions validates, creates the GitHub Release, and optionally
publishes to crates.io if configured.

### 5) After the release

- Verify the GitHub Release notes/tag/version.
- If published, verify the crates.io page and metadata.
- Keep `[Unreleased]` in `CHANGELOG.md` ready for the next cycle.

## Troubleshooting

- **Release workflow didn’t run:** the tag must match `v*.*.*` (stable) unless you change
  the workflow trigger.
- **Publish job skipped:** ensure `CARGO_REGISTRY_TOKEN` is configured as a repository
  variable (or update the workflow to use secrets).
- **Docs site didn’t update:** confirm the Pages workflow is enabled and the `docs/`
  folder builds successfully with Jekyll.
