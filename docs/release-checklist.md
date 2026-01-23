# Release Checklist

- For the full workflow (release PR → tag → automation), see [Releasing CacheKit](releasing.md).

- Update `CHANGELOG.md` with final date/version and review entries.
- Run `cargo fmt` if needed.
- Run `cargo clippy --all-targets --all-features -- -D warnings`.
- Run `cargo test`.
- Run key benchmarks (`cargo bench --bench lru`, `--bench lru_k`, `--bench lfu`).
- Verify docs with `cargo doc --no-deps` and spot-check new module docs.
- Confirm `Cargo.toml` version matches the release tag.
- Tag and push:
  - Stable: `git tag -a v0.1.0 -m "cachekit v0.1.0" && git push origin v0.1.0`
  - Pre-release: `git tag -a v0.1.0-alpha -m "cachekit v0.1.0-alpha" && git push origin v0.1.0-alpha`
    - Note: the default `.github/workflows/release.yml` trigger matches `v*.*.*` only.
