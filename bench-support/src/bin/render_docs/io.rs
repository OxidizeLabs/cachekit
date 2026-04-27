//! ## Architecture
//! Glue between the user (argv + stdio) and the pure rendering layer
//! (`templates`, `markdown`). Owns:
//! - argv parsing ([`parse_args`] → [`ParsedArgs`]).
//! - the in-process pipeline ([`run_with_paths`]) that drives every
//!   renderer once per invocation.
//! - atomic file replacement so a crashed render leaves the published
//!   docs tree in a consistent state, never half-written.
//! - input bound checks ([`MAX_RESULTS_JSON_BYTES`]).
//!
//! ## Performance Trade-offs
//! Atomic writes use `rename` over a sibling tempfile (one syscall on
//! Unix, atomic on the same filesystem). The cost is one extra
//! `fs::write` per output; for ~5 KiB-class outputs this is below
//! noise.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use bench_support::json_results::BenchmarkArtifact;
use bench_support::registry::POLICIES;

use crate::markdown::generate_markdown;
use crate::templates::{CHARTS_CSS, check_schema_version, render_charts_html, render_charts_js};

// ============================================================================
// CLI front-end
// ============================================================================

/// Parsed CLI invocation, ready to drive [`run_with_paths`].
pub(crate) struct CliArgs {
    pub(crate) json_path: PathBuf,
    pub(crate) output_dir: PathBuf,
}

/// Outcome of [`parse_args`]: the three branches `run` distinguishes
/// (help, error, valid invocation). Carrying `prog` through every
/// branch keeps usage messages anchored to the actual `argv[0]` the
/// user typed, which differs between `cargo run` and a direct binary
/// invocation.
pub(crate) enum ParsedArgs {
    Run(CliArgs),
    HelpRequested(String),
    Error { prog: String, message: String },
}

/// Parse the renderer's argv. Testable seam — [`run`] just routes the
/// outcome to stdio. Accepts at most one positional `<results.json>`
/// and an optional `[output-dir]`; anything else (extra positionals,
/// unrecognized leading flags) is rejected with a clear usage error
/// rather than silently dropped on the floor.
pub(crate) fn parse_args(args: &[String]) -> ParsedArgs {
    let prog = args
        .first()
        .map(String::as_str)
        .unwrap_or("render_docs")
        .to_string();

    if args.iter().any(|a| a == "-h" || a == "--help") {
        return ParsedArgs::HelpRequested(prog);
    }

    let positionals: Vec<&str> = args.iter().skip(1).map(String::as_str).collect();

    if let Some(flag) = positionals
        .iter()
        .find(|a| a.starts_with("--") || a.starts_with('-'))
    {
        return ParsedArgs::Error {
            prog,
            message: format!(
                "unrecognized flag {flag:?}; render_docs only takes positional \
                 arguments. See --help for usage."
            ),
        };
    }

    if positionals.is_empty() {
        return ParsedArgs::Error {
            prog,
            message: "missing <results.json> argument".to_string(),
        };
    }

    if positionals.len() > 2 {
        return ParsedArgs::Error {
            prog,
            message: format!(
                "too many positional arguments ({}, expected at most 2: \
                 <results.json> [output-dir])",
                positionals.len(),
            ),
        };
    }

    let json_path = PathBuf::from(positionals[0]);
    let output_dir = positionals
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/benchmarks/latest"));

    ParsedArgs::Run(CliArgs {
        json_path,
        output_dir,
    })
}

fn print_usage(prog: &str, out: &mut dyn std::io::Write) -> std::io::Result<()> {
    writeln!(out, "Usage: {prog} <results.json> [output-dir]")?;
    writeln!(out)?;
    writeln!(out, "Example:")?;
    writeln!(
        out,
        "  {prog} target/benchmarks/latest/results.json docs/benchmarks/latest"
    )?;
    writeln!(out)?;
    writeln!(
        out,
        "output-dir defaults to 'docs/benchmarks/latest' (relative to cwd)."
    )?;
    Ok(())
}

/// Top-level entry point used by `fn main`. Stdio chatter lives here;
/// [`run_with_paths`] is the silent core.
pub(crate) fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let parsed = match parse_args(&args) {
        ParsedArgs::HelpRequested(prog) => {
            print_usage(&prog, &mut std::io::stdout())?;
            return Ok(());
        },
        ParsedArgs::Error { prog, message } => {
            let mut stderr = std::io::stderr();
            print_usage(&prog, &mut stderr)?;
            return Err(message.into());
        },
        ParsedArgs::Run(parsed) => parsed,
    };

    println!(
        "Reading benchmark results from: {}",
        parsed.json_path.display()
    );
    let outputs = run_with_paths(&parsed.json_path, &parsed.output_dir)?;

    println!("Generated documentation:");
    for path in outputs.iter() {
        println!("   - {}", path.display());
    }
    Ok(())
}

// ============================================================================
// Pipeline
// ============================================================================

/// Paths to the files [`run_with_paths`] produces, in a fixed schema.
///
/// Returned by name (not position) so a future caller adding a sixth
/// output can't silently shift `outputs[2]` to mean a different file.
/// Tests and the binary both iterate via [`RenderOutputs::iter`] and
/// look up specific files by field name.
pub(crate) struct RenderOutputs {
    pub(crate) index: PathBuf,
    pub(crate) results: PathBuf,
    pub(crate) charts_html: PathBuf,
    pub(crate) charts_css: PathBuf,
    pub(crate) charts_js: PathBuf,
}

impl RenderOutputs {
    /// Iterate all output paths in the order the renderer wrote them.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &Path> {
        [
            self.index.as_path(),
            self.results.as_path(),
            self.charts_html.as_path(),
            self.charts_css.as_path(),
            self.charts_js.as_path(),
        ]
        .into_iter()
    }
}

/// Core renderer: produce the five output files from a parsed JSON
/// path.
///
/// Split out from [`run`] (which owns argv parsing and stdout chatter)
/// so integration tests can drive the full pipeline against a temp
/// directory and inspect the resulting files.
pub(crate) fn run_with_paths(
    json_path: &Path,
    output_dir: &Path,
) -> Result<RenderOutputs, Box<dyn Error>> {
    let artifact = read_artifact(json_path)?;
    check_schema_version(&artifact.schema_version)?;

    fs::create_dir_all(output_dir)
        .map_err(|e| format!("creating {}: {e}", output_dir.display()))?;

    let markdown = generate_markdown(&artifact);

    let index = output_dir.join("index.md");
    write_atomic(&index, markdown.as_bytes())?;

    let results = output_dir.join("results.json");
    copy_json(json_path, &results)?;

    let charts_html = output_dir.join("charts.html");
    let charts_html_body = render_charts_html()?;
    write_atomic(&charts_html, charts_html_body.as_bytes())?;

    let charts_css = output_dir.join("charts.css");
    write_atomic(&charts_css, CHARTS_CSS.as_bytes())?;

    let charts_js = output_dir.join("charts.js");
    let charts_js_body = render_charts_js(POLICIES)?;
    write_atomic(&charts_js, charts_js_body.as_bytes())?;

    Ok(RenderOutputs {
        index,
        results,
        charts_html,
        charts_css,
        charts_js,
    })
}

// ============================================================================
// Atomic writes
// ============================================================================

/// Replace `path` with the result of `produce(tmp)` atomically: stage
/// the new contents in a sibling tempfile, then `rename` over the
/// destination.
///
/// On Unix, `rename` is a single atomic syscall — a crash leaves
/// either the old file or the new, never a half-written one. The
/// tempfile lives in the same directory so the rename stays on one
/// filesystem (cross-FS renames are not atomic and would fail with
/// `EXDEV` anyway). On error, the tempfile is best-effort removed so
/// a half-finished render leaves no `.tmp-*` debris in the published
/// docs tree.
fn atomic_replace<F>(path: &Path, produce: F) -> Result<(), Box<dyn Error>>
where
    F: FnOnce(&Path) -> std::io::Result<()>,
{
    let parent = path
        .parent()
        .ok_or_else(|| format!("atomic write to {} has no parent directory", path.display()))?;
    let basename = path
        .file_name()
        .ok_or_else(|| format!("atomic write to {} has no filename", path.display()))?;
    let tmp_path = parent.join(format!(".tmp-{}", basename.to_string_lossy()));

    if let Err(e) = produce(&tmp_path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(format!("staging {}: {e}", tmp_path.display()).into());
    }
    if let Err(e) = fs::rename(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(format!("renaming {} -> {}: {e}", tmp_path.display(), path.display()).into());
    }
    Ok(())
}

/// Atomically write `bytes` to `path`. See [`atomic_replace`] for
/// semantics.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    atomic_replace(path, |tmp| fs::write(tmp, bytes))
}

/// Skips the copy when source and destination resolve to the same
/// file; otherwise replaces `dest` atomically (see
/// [`atomic_replace`]).
///
/// Handles the first-run case where `dest` does not exist yet by
/// canonicalising the destination's parent directory (which
/// `create_dir_all` has just made) and re-joining the filename,
/// instead of canonicalising `dest` directly.
fn copy_json(src: &Path, dest: &Path) -> Result<(), Box<dyn Error>> {
    if same_file(src, dest) {
        return Ok(());
    }
    atomic_replace(dest, |tmp| fs::copy(src, tmp).map(|_| ()))
}

fn same_file(src: &Path, dest: &Path) -> bool {
    let Ok(src_abs) = src.canonicalize() else {
        return false;
    };
    if let Ok(dest_abs) = dest.canonicalize() {
        return src_abs == dest_abs;
    }
    let (Some(parent), Some(name)) = (dest.parent(), dest.file_name()) else {
        return false;
    };
    match parent.canonicalize() {
        Ok(parent_abs) => src_abs == parent_abs.join(name),
        Err(_) => false,
    }
}

// ============================================================================
// Bounded JSON read
// ============================================================================

/// Generous cap for the input JSON artifact. Real benchmark runs
/// produce well under 10 MiB; this exists to fail fast (instead of
/// OOM) when pointed at a wrong path — a multi-GB log, `/dev/zero`,
/// etc.
pub(crate) const MAX_RESULTS_JSON_BYTES: u64 = 256 * 1024 * 1024;

fn read_artifact(path: &Path) -> Result<BenchmarkArtifact, Box<dyn Error>> {
    read_artifact_with_limit(path, MAX_RESULTS_JSON_BYTES)
}

/// Internal seam for [`read_artifact`] that lets tests exercise the
/// over-limit branch with a small synthetic file. Reading via
/// `read_to_string` + `from_str` is materially faster than
/// `from_reader` for small JSON (per the serde docs) and gives us a
/// length we can check up-front.
pub(crate) fn read_artifact_with_limit(
    path: &Path,
    limit_bytes: u64,
) -> Result<BenchmarkArtifact, Box<dyn Error>> {
    let metadata = fs::metadata(path).map_err(|e| format!("stat {}: {e}", path.display()))?;
    if metadata.len() > limit_bytes {
        return Err(format!(
            "artifact {} is {} bytes, exceeds {} byte limit (point at the right results.json)",
            path.display(),
            metadata.len(),
            limit_bytes,
        )
        .into());
    }
    let body = fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let artifact: BenchmarkArtifact =
        serde_json::from_str(&body).map_err(|e| format!("parsing {}: {e}", path.display()))?;
    Ok(artifact)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bench_support::json_results::SCHEMA_VERSION;

    use crate::templates::{
        CHART_JS_SRI_PLACEHOLDER, CHART_JS_VERSION_PLACEHOLDER, POLICY_COLORS_PLACEHOLDER,
        RENDERER_STAMP_PLACEHOLDER, SCHEMA_MAJOR_PLACEHOLDER, renderer_stamp,
    };
    use crate::test_helpers::{hit_metrics, metadata, row, unique_temp_dir};

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn same_file_detects_existing_paths() {
        let dir = unique_temp_dir("same-file");
        let p = dir.join("a.json");
        std::fs::write(&p, b"{}").unwrap();

        assert!(same_file(&p, &p));

        let q = dir.join("b.json");
        std::fs::write(&q, b"{}").unwrap();
        assert!(!same_file(&p, &q));

        let alt = dir.join(".").join("a.json");
        assert!(same_file(&p, &alt));

        let missing = dir.join("never-created.json");
        assert!(!same_file(&p, &missing));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn run_with_paths_writes_all_five_outputs() {
        let tmp = unique_temp_dir("e2e");

        let json_path = tmp.join("results.json");
        let mut artifact = BenchmarkArtifact::new(metadata());
        artifact.add_result(row("LRU", "Zipf", "hit_rate", hit_metrics(0.5)));
        std::fs::write(&json_path, serde_json::to_vec(&artifact).unwrap()).expect("write fixture");

        let out_dir = tmp.join("out");
        let outputs = run_with_paths(&json_path, &out_dir).expect("render");

        assert_eq!(outputs.index, out_dir.join("index.md"));
        assert_eq!(outputs.results, out_dir.join("results.json"));
        assert_eq!(outputs.charts_html, out_dir.join("charts.html"));
        assert_eq!(outputs.charts_css, out_dir.join("charts.css"));
        assert_eq!(outputs.charts_js, out_dir.join("charts.js"));
        for path in outputs.iter() {
            assert!(path.exists(), "output file missing: {}", path.display());
            let bytes = std::fs::read(path).expect("read output");
            assert!(!bytes.is_empty(), "output file empty: {}", path.display());
        }

        let html = std::fs::read_to_string(&outputs.charts_html).unwrap();
        assert!(!html.contains(CHART_JS_VERSION_PLACEHOLDER));
        assert!(!html.contains(CHART_JS_SRI_PLACEHOLDER));
        assert!(!html.contains(RENDERER_STAMP_PLACEHOLDER));
        let stamp = renderer_stamp();
        assert!(
            html.contains(&format!(r#"<meta name="generator" content="{stamp}">"#)),
            "rendered charts.html missing <meta name=\"generator\"> with stamp {stamp:?}",
        );
        let js = std::fs::read_to_string(&outputs.charts_js).unwrap();
        assert!(!js.contains(POLICY_COLORS_PLACEHOLDER));
        assert!(!js.contains(SCHEMA_MAJOR_PLACEHOLDER));

        let md = std::fs::read_to_string(&outputs.index).unwrap();
        assert!(
            md.contains(&format!("Generated by `{stamp}`")),
            "rendered index.md missing renderer stamp footer; got tail: {:?}",
            md.lines().rev().take(3).collect::<Vec<_>>(),
        );

        let leftover: Vec<_> = std::fs::read_dir(&out_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(".tmp-"))
            .collect();
        assert!(
            leftover.is_empty(),
            "found stray atomic-write tempfiles: {:?}",
            leftover.iter().map(|e| e.path()).collect::<Vec<_>>(),
        );

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn run_with_paths_is_byte_idempotent() {
        // The renderer uses BTreeMap/BTreeSet today (deterministic
        // iteration), but a future refactor that switches to HashMap
        // would silently produce byte-different output across runs
        // with no test failure. Run the pipeline twice and byte-
        // compare the four deterministic outputs.
        let tmp = unique_temp_dir("idempotence");
        let json_path = tmp.join("results.json");
        let mut artifact = BenchmarkArtifact::new(metadata());
        artifact.add_result(row("LRU", "Zipf", "hit_rate", hit_metrics(0.5)));
        artifact.add_result(row("LFU", "Uniform", "hit_rate", hit_metrics(0.4)));
        std::fs::write(&json_path, serde_json::to_vec(&artifact).unwrap()).unwrap();

        let out_a = tmp.join("a");
        let out_b = tmp.join("b");
        let outputs_a = run_with_paths(&json_path, &out_a).expect("render a");
        let outputs_b = run_with_paths(&json_path, &out_b).expect("render b");

        for (label, a, b) in [
            ("index.md", &outputs_a.index, &outputs_b.index),
            (
                "charts.html",
                &outputs_a.charts_html,
                &outputs_b.charts_html,
            ),
            ("charts.css", &outputs_a.charts_css, &outputs_b.charts_css),
            ("charts.js", &outputs_a.charts_js, &outputs_b.charts_js),
        ] {
            let bytes_a = std::fs::read(a).unwrap();
            let bytes_b = std::fs::read(b).unwrap();
            assert_eq!(
                bytes_a,
                bytes_b,
                "{label} differs across runs ({} vs {} bytes); a non-deterministic \
                 collection (HashMap/HashSet?) leaked into the renderer",
                bytes_a.len(),
                bytes_b.len(),
            );
        }
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn parse_args_accepts_one_or_two_positionals() {
        match parse_args(&args(&["render_docs", "in.json"])) {
            ParsedArgs::Run(p) => {
                assert_eq!(p.json_path, PathBuf::from("in.json"));
                assert_eq!(p.output_dir, PathBuf::from("docs/benchmarks/latest"));
            },
            other => panic!("expected Run, got {:?}", std::mem::discriminant(&other)),
        }
        match parse_args(&args(&["render_docs", "in.json", "out"])) {
            ParsedArgs::Run(p) => {
                assert_eq!(p.json_path, PathBuf::from("in.json"));
                assert_eq!(p.output_dir, PathBuf::from("out"));
            },
            other => panic!("expected Run, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn parse_args_help_short_circuits_in_either_position() {
        for argv in [
            args(&["render_docs", "--help"]),
            args(&["render_docs", "-h"]),
            args(&["render_docs", "foo", "--help"]),
        ] {
            assert!(
                matches!(parse_args(&argv), ParsedArgs::HelpRequested(_)),
                "expected HelpRequested for {argv:?}",
            );
        }
    }

    #[test]
    fn parse_args_rejects_extra_positionals() {
        let p = parse_args(&args(&["render_docs", "in.json", "out", "extra"]));
        match p {
            ParsedArgs::Error { message, .. } => assert!(
                message.contains("too many positional arguments"),
                "got: {message}",
            ),
            _ => panic!("expected Error for extra positional"),
        }
    }

    #[test]
    fn parse_args_rejects_unknown_leading_flags() {
        let p = parse_args(&args(&["render_docs", "--output", "docs", "in.json"]));
        match p {
            ParsedArgs::Error { message, .. } => {
                assert!(message.contains("unrecognized flag"), "got: {message}",)
            },
            _ => panic!("expected Error for unknown flag"),
        }
    }

    #[test]
    fn parse_args_requires_at_least_one_positional() {
        let p = parse_args(&args(&["render_docs"]));
        match p {
            ParsedArgs::Error { message, .. } => {
                assert!(message.contains("missing <results.json>"), "got: {message}",)
            },
            _ => panic!("expected Error for missing positional"),
        }
    }

    #[test]
    fn parse_args_carries_prog_name_through_branches() {
        match parse_args(&args(&["my-render", "--help"])) {
            ParsedArgs::HelpRequested(prog) => assert_eq!(prog, "my-render"),
            _ => panic!(),
        }
        match parse_args(&args(&["my-render"])) {
            ParsedArgs::Error { prog, .. } => assert_eq!(prog, "my-render"),
            _ => panic!(),
        }
    }

    #[test]
    fn write_atomic_replaces_existing_file_and_cleans_tempfile() {
        let tmp = unique_temp_dir("write-atomic");
        let target = tmp.join("a.txt");
        std::fs::write(&target, b"old").unwrap();

        write_atomic(&target, b"new").expect("atomic replace");
        assert_eq!(std::fs::read(&target).unwrap(), b"new");
        assert!(
            !tmp.join(".tmp-a.txt").exists(),
            "tempfile must be renamed away on success",
        );
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn write_atomic_creates_new_file() {
        let tmp = unique_temp_dir("write-atomic-new");
        let target = tmp.join("fresh.txt");
        assert!(!target.exists());
        write_atomic(&target, b"hello").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"hello");
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn read_artifact_with_limit_rejects_oversize_input() {
        let tmp = unique_temp_dir("bounded-read");
        let path = tmp.join("big.json");
        std::fs::write(&path, vec![0u8; 4 * 1024]).unwrap();
        let err = read_artifact_with_limit(&path, 1024)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("exceeds") && err.contains("byte limit"),
            "error should explain the cap: {err}",
        );
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn read_artifact_with_limit_accepts_valid_input_under_cap() {
        let tmp = unique_temp_dir("bounded-read-ok");
        let path = tmp.join("ok.json");
        let artifact = BenchmarkArtifact::new(metadata());
        std::fs::write(&path, serde_json::to_vec(&artifact).unwrap()).unwrap();
        let parsed =
            read_artifact_with_limit(&path, MAX_RESULTS_JSON_BYTES).expect("parse under cap");
        assert_eq!(parsed.schema_version, SCHEMA_VERSION);
        std::fs::remove_dir_all(&tmp).unwrap();
    }
}
