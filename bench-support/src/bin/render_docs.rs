//! Renders benchmark results from JSON to GitHub Pages Markdown +
//! charts page.
//!
//! ## Architecture
//! Thin binary entry point. The actual pipeline lives in three
//! sibling modules so each layer can be tested and reasoned about in
//! isolation:
//! - [`templates`]: bundled `charts_template.{html,js,css}` assets,
//!   sentinel substitution, schema-version + renderer-stamp logic.
//! - [`markdown`]: pure `BenchmarkArtifact → String` rendering for
//!   `index.md`.
//! - [`io`]: argv parsing, atomic file writes, the `run`/`run_with_paths`
//!   orchestrator, and bounded JSON reads.
//! - [`test_helpers`]: `#[cfg(test)]`-only fixtures shared between the
//!   three modules above.
//!
//! ## Core Operations
//! `main` exits non-zero on render failure so a CI job that reuses
//! `render_docs` in a script halts on the first bad artifact instead
//! of silently publishing stale or partial docs.
//!
//! ## Usage
//! ```text
//! cargo run --package bench-support --bin render_docs -- \
//!     <results.json> [output-dir]
//! ```
//!
//! `output-dir` defaults to `docs/benchmarks/latest` (relative to
//! cwd).

// Submodules live alongside this file in `render_docs/`. Cargo's
// `bin` target doesn't auto-discover sibling-folder submodules, so
// each declaration carries an explicit `#[path]` instead of moving
// the binary entry point to `render_docs/main.rs` (which would
// require either renaming the artifact or pinning a `[[bin]] path`
// override in `Cargo.toml`).

#[path = "render_docs/io.rs"]
mod io;

#[path = "render_docs/markdown.rs"]
mod markdown;

#[path = "render_docs/templates.rs"]
mod templates;

#[cfg(test)]
#[path = "render_docs/test_helpers.rs"]
mod test_helpers;

use std::process::ExitCode;

fn main() -> ExitCode {
    match io::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        },
    }
}
