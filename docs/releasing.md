# Releasing CacheKit

This document describes the end-to-end release process for the `cachekit` crate:
versioning, validation, tagging, publishing, and documentation updates.

If you only need a quick set of checks, see `docs/release-checklist.md`.

## Release types

### Stable releases

- **Tag format:** `vX.Y.Z` (example: `v0.1.0`)
- **Automation:** GitHub Actions runs `.github/workflows/release.yml` on tag push.

### Pre-releases (alpha/beta/rc)

- **Cargo versions:** `X.Y.Z-alpha`, `X.Y.Z-alpha.1`, `X.Y.Z-beta.1`, `X.Y.Z-rc.1`, etc.
- **Note:** the current release workflow trigger only matches `v*.*.*`, so tags like
  `v0.1.0-alpha` / `v0.2.0-alpha.1` will **not** run the automated release pipeline unless the
  workflow trigger is updated.

## Before you start

- Ensure `main` is green in CI (`.github/workflows/ci.yml`).
- Pick the next version using SemVer.
- Decide whether you are publishing to crates.io or doing a GitHub-only release.

## Step-by-step release (recommended)

### 1) Prepare a release PR

Make a branch/PR that does the release bookkeeping:

- **Bump crate version** in `Cargo.toml` to the target version (no leading `v`).
- **Finalize `CHANGELOG.md`:**
  - Move items from `[Unreleased]` into a new section `[X.Y.Z] - YYYY-MM-DD` (or
    `[X.Y.Z-alpha] - YYYY-MM-DD` for a pre-release).
  - Keep an empty `[Unreleased]` section for follow-up work.
- **Update docs as needed** (design/docs/policy notes, etc.).
- **Update benchmark docs (optional):**
  - Run benchmarks locally: `cargo bench`
  - Update `docs/benchmarks.md` from Criterion output:
    - `scripts/update_docs_benchmarks.sh target/criterion docs/benchmarks.md`

Suggested local validation for the PR:

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features --all-targets
RUSTDOCFLAGS='-Dwarnings' cargo doc --no-deps --all-features
```

### 2) Merge the release PR

Merge once CI passes. Releases should always be cut from a commit on `main`.

### 3) Tag the release

Create an annotated tag on `main` and push it:

```bash
git checkout main
git pull --ff-only
git tag -a vX.Y.Z -m "cachekit vX.Y.Z"
git push origin vX.Y.Z
```

### 4) Monitor the release workflow

On tag push, `.github/workflows/release.yml` runs:

- **Validate:** fmt, clippy, tests, docs, `cargo package`, `cargo publish --dry-run`
- **GitHub Release:** creates a GitHub Release (currently using auto-generated notes)
- **crates.io publish (optional):** runs `cargo publish` if `CARGO_REGISTRY_TOKEN` is set

Notes:

- The workflow verifies the tag commit is reachable from `origin/main`.
- Publishing uses `CARGO_REGISTRY_TOKEN` from GitHub repository variables (`vars`).
  If you prefer GitHub Secrets, update the workflow accordingly.

### 5) Post-release

- Verify:
  - GitHub Release exists and has correct notes/tag.
  - crates.io shows the new version (if published).
  - Docs site updates (GitHub Pages builds from `docs/` on pushes to `main` via
    `.github/workflows/jekyll-gh-pages.yml`).
- Start the next development cycle:
  - Bump `Cargo.toml` to the next dev/pre-release version (optional).
  - Add new entries under `[Unreleased]` in `CHANGELOG.md`.

## Troubleshooting

- **Release workflow didn’t run:** confirm the tag matches `v*.*.*` and was pushed
  to the upstream repository (not just locally).
- **Publish job skipped:** ensure `CARGO_REGISTRY_TOKEN` is configured as a GitHub
  repository variable (or adjust the workflow to use secrets).
- **Workflow fails fetching actions:** if GitHub Actions reports it can’t resolve an
  action version (for example `actions/checkout@v6`), update the workflow to a
  published major version (for example `actions/checkout@v4`).
