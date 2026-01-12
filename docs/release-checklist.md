# Release Checklist

- Update `CHANGELOG.md` with final date/version and review entries.
- Run `cargo fmt` if needed.
- Run `cargo clippy --all-targets --all-features -- -D warnings`.
- Run `cargo test`.
- Run key benchmarks (`cargo bench --bench lru`, `--bench lru_k`, `--bench lfu`).
- Verify docs with `cargo doc --no-deps` and spot-check new module docs.
- Confirm `Cargo.toml` version matches the release tag.
- Tag and push (e.g., `git tag v0.1.0`).
