//! Renders benchmark results from JSON to GitHub Pages Markdown.
//!
//! Pipeline: parse `BenchmarkArtifact` → group rows by `case_id` → emit one
//! Markdown section per case (pivoting policy × workload tables where
//! applicable) → write `index.md`, copy `results.json`, embed `charts.html`.
//!
//! Usage:
//!   cargo run --package bench-support --bin render_docs -- \
//!       <results.json> [output-dir]
//!
//! `output-dir` defaults to `docs/benchmarks/latest` (relative to cwd).

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use bench_support::json_results::{BenchmarkArtifact, ResultRow, SCHEMA_VERSION, case_id};
use bench_support::registry::{POLICIES, PolicyMeta};

/// Markdown table for the static policy selection guide.
const POLICY_GUIDE_MD: &str = include_str!("policy_guide.md");

/// HTML shell for the charts page. Loads Chart.js (CDN, SRI-pinned), the
/// sibling `charts.js` script, and the sibling `charts.css` stylesheet.
/// Inline scripts and inline styles were intentionally split out so the page
/// can run under a strict CSP (no `unsafe-inline`, no `unsafe-eval`).
const CHARTS_HTML: &str = include_str!("charts_template.html");

/// Behavior for the charts page; sibling script of [`CHARTS_HTML`].
const CHARTS_JS: &str = include_str!("charts_template.js");

/// Presentation for the charts page; sibling stylesheet of [`CHARTS_HTML`].
/// Carries `.no-js #loading` and `.hidden` rules that take the place of the
/// previous inline `<style>` and `style="display: none"` attributes.
const CHARTS_CSS: &str = include_str!("charts_template.css");

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        },
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let prog = args.first().map(String::as_str).unwrap_or("render_docs");

    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage(prog, &mut std::io::stdout())?;
        return Ok(());
    }

    if args.len() < 2 {
        let mut stderr = std::io::stderr();
        print_usage(prog, &mut stderr)?;
        return Err("missing <results.json> argument".into());
    }

    let json_path = PathBuf::from(&args[1]);
    let output_dir = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/benchmarks/latest"));

    println!("Reading benchmark results from: {}", json_path.display());

    let artifact = read_artifact(&json_path)?;
    check_schema_version(&artifact.schema_version)?;

    fs::create_dir_all(&output_dir)
        .map_err(|e| format!("creating {}: {e}", output_dir.display()))?;

    let markdown = generate_markdown(&artifact);

    let index_path = output_dir.join("index.md");
    fs::write(&index_path, markdown)
        .map_err(|e| format!("writing {}: {e}", index_path.display()))?;

    let json_dest = output_dir.join("results.json");
    copy_json(&json_path, &json_dest)?;

    let charts_html_path = output_dir.join("charts.html");
    fs::write(&charts_html_path, CHARTS_HTML)
        .map_err(|e| format!("writing {}: {e}", charts_html_path.display()))?;

    let charts_css_path = output_dir.join("charts.css");
    fs::write(&charts_css_path, CHARTS_CSS)
        .map_err(|e| format!("writing {}: {e}", charts_css_path.display()))?;

    let charts_js_path = output_dir.join("charts.js");
    let charts_js = inject_policy_colors(CHARTS_JS, POLICIES)?;
    fs::write(&charts_js_path, charts_js)
        .map_err(|e| format!("writing {}: {e}", charts_js_path.display()))?;

    println!("Generated documentation:");
    println!("   - {}", index_path.display());
    println!("   - {}", json_dest.display());
    println!("   - {}", charts_html_path.display());
    println!("   - {}", charts_css_path.display());
    println!("   - {}", charts_js_path.display());
    Ok(())
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

fn read_artifact(path: &Path) -> Result<BenchmarkArtifact, Box<dyn Error>> {
    let file = fs::File::open(path).map_err(|e| format!("opening {}: {e}", path.display()))?;
    let reader = BufReader::new(file);
    let artifact: BenchmarkArtifact =
        serde_json::from_reader(reader).map_err(|e| format!("parsing {}: {e}", path.display()))?;
    Ok(artifact)
}

/// Refuses results whose major schema version differs from this binary's.
fn check_schema_version(found: &str) -> Result<(), Box<dyn Error>> {
    let major = |v: &str| v.split('.').next().and_then(|s| s.parse::<u32>().ok());
    match (major(found), major(SCHEMA_VERSION)) {
        (Some(f), Some(e)) if f == e => Ok(()),
        (Some(f), Some(e)) => Err(format!(
            "schema version mismatch: artifact is {found}, renderer expects {SCHEMA_VERSION} (major {e}, got {f})"
        )
        .into()),
        _ => Err(format!(
            "unrecognized schema version {found:?} (renderer expects {SCHEMA_VERSION})"
        )
        .into()),
    }
}

/// Skips the copy when source and destination resolve to the same file.
///
/// Handles the first-run case where `dest` does not exist yet by canonicalising
/// the destination's parent directory (which `create_dir_all` has just made)
/// and re-joining the filename, instead of canonicalising `dest` directly.
fn copy_json(src: &Path, dest: &Path) -> Result<(), Box<dyn Error>> {
    if same_file(src, dest) {
        return Ok(());
    }
    fs::copy(src, dest)
        .map_err(|e| format!("copying {} -> {}: {e}", src.display(), dest.display()))?;
    Ok(())
}

/// Sentinel substituted in the bundled charts script at render time.
///
/// The exact byte sequence must match the placeholder in
/// `charts_template.js`. Keeping it as a JS comment + empty object literal
/// means the template is still syntactically valid (and would render with
/// FNV fallback colors only) if substitution ever fails — but we'd rather
/// fail loudly, hence the explicit `Err` below.
const POLICY_COLORS_PLACEHOLDER: &str = "/* @POLICY_COLORS@ */ {}";

/// Replace [`POLICY_COLORS_PLACEHOLDER`] in the charts script with a JS
/// object literal sourced from `policies`. Errors when the placeholder is
/// missing (template malformed) or appears more than once (would lead to
/// non-deterministic output).
fn inject_policy_colors(template: &str, policies: &[PolicyMeta]) -> Result<String, Box<dyn Error>> {
    let count = template.matches(POLICY_COLORS_PLACEHOLDER).count();
    match count {
        0 => Err(format!(
            "charts template is missing the policy-colors sentinel `{POLICY_COLORS_PLACEHOLDER}`; \
             refusing to render with stale colors"
        )
        .into()),
        1 => {
            let literal = render_policy_colors_literal(policies);
            Ok(template.replacen(POLICY_COLORS_PLACEHOLDER, &literal, 1))
        },
        n => Err(
            format!("charts template has {n} policy-colors sentinels; expected exactly 1").into(),
        ),
    }
}

/// Render a JS object literal `{ "Display": "#hex", ... }` from `policies`.
/// Display names are emitted as JSON strings so any special characters are
/// safely escaped.
fn render_policy_colors_literal(policies: &[PolicyMeta]) -> String {
    let mut out = String::from("{\n");
    for (i, meta) in policies.iter().enumerate() {
        let key = serde_json::to_string(meta.display_name).expect("display_name is valid UTF-8");
        let value = serde_json::to_string(meta.color).expect("color is valid UTF-8");
        let comma = if i + 1 == policies.len() { "" } else { "," };
        let _ = writeln!(out, "            {key}: {value}{comma}");
    }
    out.push_str("        }");
    out
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

fn generate_markdown(artifact: &BenchmarkArtifact) -> String {
    let mut md = String::with_capacity(4 * 1024);

    writeln!(md, "# Benchmark Results\n").unwrap();
    writeln!(
        md,
        "**Quick Links**: [Interactive Charts](charts.html) | [Raw JSON](results.json)\n"
    )
    .unwrap();
    writeln!(md, "---\n").unwrap();

    write_environment(&mut md, artifact);
    write_configuration(&mut md, artifact);

    let by_case = artifact.results_by_case();

    if let Some(rows) = by_case.get(case_id::HIT_RATE) {
        write_pivot_section(
            &mut md,
            "Hit Rate Comparison",
            rows,
            |r| r.metrics.hit_stats.as_ref().map(|s| s.hit_rate),
            |v| format!("{:.2}%", v * 100.0),
        );
    }

    if let Some(rows) = by_case.get(case_id::COMPREHENSIVE) {
        write_pivot_section(
            &mut md,
            "Throughput (Million ops/sec)",
            rows,
            |r| {
                r.metrics
                    .throughput
                    .as_ref()
                    .map(|t| t.ops_per_sec / 1_000_000.0)
            },
            |v| format!("{v:.2}"),
        );
        write_pivot_section(
            &mut md,
            "Latency P99 (nanoseconds)",
            rows,
            |r| r.metrics.latency.as_ref().map(|l| l.p99_ns),
            |v| v.to_string(),
        );
    }

    if let Some(rows) = by_case.get(case_id::SCAN_RESISTANCE) {
        write_scan_resistance_section(&mut md, rows);
    }

    if let Some(rows) = by_case.get(case_id::ADAPTATION) {
        write_adaptation_section(&mut md, rows);
    }

    writeln!(md, "## Policy Selection Guide\n").unwrap();
    md.push_str(POLICY_GUIDE_MD);
    if !POLICY_GUIDE_MD.ends_with('\n') {
        md.push('\n');
    }
    md.push('\n');

    writeln!(md, "---\n").unwrap();
    writeln!(
        md,
        "*Generated from `results.json` (schema v{})*",
        artifact.schema_version
    )
    .unwrap();

    md
}

fn write_environment(md: &mut String, artifact: &BenchmarkArtifact) {
    let m = &artifact.metadata;
    writeln!(md, "## Environment\n").unwrap();
    writeln!(md, "- **Date**: {}", m.timestamp).unwrap();
    if let Some(commit) = &m.git_commit {
        writeln!(md, "- **Commit**: `{commit}`").unwrap();
    }
    if let Some(branch) = &m.git_branch {
        writeln!(md, "- **Branch**: `{branch}`").unwrap();
    }
    writeln!(md, "- **Dirty**: {}", m.git_dirty).unwrap();
    writeln!(md, "- **Rustc**: {}", m.rustc_version).unwrap();
    writeln!(md, "- **Host**: {}", m.host_triple).unwrap();
    if let Some(cpu) = &m.cpu_model {
        writeln!(md, "- **CPU**: {cpu}").unwrap();
    }
    md.push('\n');
}

fn write_configuration(md: &mut String, artifact: &BenchmarkArtifact) {
    let c = &artifact.metadata.config;
    writeln!(md, "## Configuration\n").unwrap();
    writeln!(md, "- **Capacity**: {}", c.capacity).unwrap();
    writeln!(md, "- **Universe**: {}", c.universe).unwrap();
    writeln!(md, "- **Operations**: {}", c.operations).unwrap();
    writeln!(md, "- **Seed**: {}", c.seed).unwrap();
    md.push('\n');
}

/// Pivots `rows` into a policy × workload matrix and emits a Markdown table.
///
/// `extract` returns `Some(value)` for rows that contribute to this metric.
/// Rows for which it returns `None` are skipped entirely (they do not even
/// contribute their workload column). Duplicate `(policy, workload)` pairs
/// emit a stderr warning and keep the first occurrence.
fn write_pivot_section<V, E, F>(
    md: &mut String,
    title: &str,
    rows: &[&ResultRow],
    extract: E,
    fmt_cell: F,
) where
    E: Fn(&ResultRow) -> Option<V>,
    F: Fn(&V) -> String,
{
    writeln!(md, "## {title}\n").unwrap();

    // BTreeMap gives deterministic, sorted policy iteration.
    // BTreeSet gives deterministic, sorted column ordering.
    let mut by_policy: BTreeMap<&str, BTreeMap<&str, V>> = BTreeMap::new();
    let mut workloads: BTreeSet<&str> = BTreeSet::new();

    for row in rows {
        let Some(value) = extract(row) else { continue };
        workloads.insert(row.workload_name.as_str());
        let cell = by_policy
            .entry(row.policy_name.as_str())
            .or_default()
            .entry(row.workload_name.as_str());
        match cell {
            std::collections::btree_map::Entry::Vacant(v) => {
                v.insert(value);
            },
            std::collections::btree_map::Entry::Occupied(_) => {
                eprintln!(
                    "warning: duplicate ({}, {}) in section {title:?}; keeping first",
                    row.policy_name, row.workload_name
                );
            },
        }
    }

    if by_policy.is_empty() {
        writeln!(md, "_No data._\n").unwrap();
        return;
    }

    write!(md, "| Policy |").unwrap();
    for w in &workloads {
        write!(md, " {w} |").unwrap();
    }
    md.push('\n');

    write!(md, "|--------|").unwrap();
    for _ in &workloads {
        write!(md, "-------:|").unwrap();
    }
    md.push('\n');

    for (policy, cells) in &by_policy {
        write!(md, "| **{policy}** |").unwrap();
        for w in &workloads {
            match cells.get(w) {
                Some(v) => write!(md, " {} |", fmt_cell(v)).unwrap(),
                None => md.push_str(" - |"),
            }
        }
        md.push('\n');
    }
    md.push('\n');
}

fn write_scan_resistance_section(md: &mut String, rows: &[&ResultRow]) {
    writeln!(md, "## Scan Resistance\n").unwrap();

    let mut sorted: Vec<&ResultRow> = rows
        .iter()
        .copied()
        .filter(|r| r.metrics.scan_resistance.is_some())
        .collect();
    if sorted.is_empty() {
        writeln!(md, "_No data._\n").unwrap();
        return;
    }
    sorted.sort_unstable_by(|a, b| a.policy_name.cmp(&b.policy_name));

    md.push_str("| Policy | Baseline | During Scan | Recovery | Score |\n");
    md.push_str("|--------|---------:|------------:|---------:|------:|\n");

    for row in sorted {
        let s = row
            .metrics
            .scan_resistance
            .as_ref()
            .expect("filtered to Some above");
        let score = match s.resistance_score {
            Some(v) => format!("{v:.3}"),
            None => "n/a".to_string(),
        };
        writeln!(
            md,
            "| **{}** | {:.2}% | {:.2}% | {:.2}% | {} |",
            row.policy_name,
            s.baseline_hit_rate * 100.0,
            s.scan_hit_rate * 100.0,
            s.recovery_hit_rate * 100.0,
            score,
        )
        .unwrap();
    }
    writeln!(
        md,
        "\n*Score = Recovery/Baseline (1.0 = perfect recovery, n/a = baseline too low to compare)*\n"
    )
    .unwrap();
}

fn write_adaptation_section(md: &mut String, rows: &[&ResultRow]) {
    writeln!(md, "## Adaptation Speed\n").unwrap();

    let mut sorted: Vec<&ResultRow> = rows
        .iter()
        .copied()
        .filter(|r| r.metrics.adaptation.is_some())
        .collect();
    if sorted.is_empty() {
        writeln!(md, "_No data._\n").unwrap();
        return;
    }
    sorted.sort_unstable_by(|a, b| a.policy_name.cmp(&b.policy_name));

    let any_curve = sorted.iter().any(|r| {
        !r.metrics
            .adaptation
            .as_ref()
            .unwrap()
            .hit_rate_curve
            .is_empty()
    });

    if any_curve {
        md.push_str("| Policy | Stable Hit Rate | Ops to 50% | Ops to 80% | Curve |\n");
        md.push_str("|--------|----------------:|-----------:|-----------:|:------|\n");
    } else {
        md.push_str("| Policy | Stable Hit Rate | Ops to 50% | Ops to 80% |\n");
        md.push_str("|--------|----------------:|-----------:|-----------:|\n");
    }

    for row in &sorted {
        let a = row
            .metrics
            .adaptation
            .as_ref()
            .expect("filtered to Some above");
        if any_curve {
            writeln!(
                md,
                "| **{}** | {:.2}% | {} | {} | `{}` |",
                row.policy_name,
                a.stable_hit_rate * 100.0,
                a.ops_to_50_percent,
                a.ops_to_80_percent,
                sparkline(&a.hit_rate_curve),
            )
            .unwrap();
        } else {
            writeln!(
                md,
                "| **{}** | {:.2}% | {} | {} |",
                row.policy_name,
                a.stable_hit_rate * 100.0,
                a.ops_to_50_percent,
                a.ops_to_80_percent,
            )
            .unwrap();
        }
    }

    if any_curve {
        let sample = sorted.iter().find_map(|r| {
            let a = r.metrics.adaptation.as_ref().unwrap();
            (a.window_size > 0).then_some((a.window_size, a.hit_rate_curve.len()))
        });
        let curve_note = match sample {
            Some((window, len)) if len > 0 => format!(
                " Curve = per-window hit rate after the workload shift, low → high (`▁` ≈ 0%, `█` ≈ 100%); each cell is {window} ops, total {} ops measured.",
                window * len,
            ),
            _ => " Curve = per-window hit rate after the workload shift, low → high (`▁` ≈ 0%, `█` ≈ 100%).".into(),
        };
        writeln!(
            md,
            "\n*Lower ops-to-X% is better (faster adaptation).{curve_note}*\n",
        )
        .unwrap();
    } else {
        writeln!(md, "\n*Lower ops-to-X% is better (faster adaptation)*\n").unwrap();
    }
}

/// Render a hit-rate curve as a Unicode block sparkline. Each cell maps
/// `[0.0, 1.0]` to one of eight block heights so a long curve still fits in
/// a Markdown table cell.
fn sparkline(values: &[f64]) -> String {
    const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if values.is_empty() {
        return String::new();
    }
    values
        .iter()
        .map(|&v| {
            let clamped = v.clamp(0.0, 1.0);
            // 0.0 → block 0, 1.0 → block 7; midpoints round nearest.
            let idx = (clamped * (BLOCKS.len() as f64 - 1.0)).round() as usize;
            BLOCKS[idx.min(BLOCKS.len() - 1)]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bench_support::json_results::{
        AdaptationStats, BenchmarkConfig, HitStats, LatencyStats, Metrics, RunMetadata,
        ScanResistanceStats, ThroughputStats,
    };

    fn metadata() -> RunMetadata {
        RunMetadata {
            timestamp: "2026-01-01T00:00:00Z".into(),
            git_commit: None,
            git_branch: None,
            git_dirty: false,
            rustc_version: "rustc test".into(),
            host_triple: "x86_64-unknown-linux-gnu".into(),
            cpu_model: None,
            config: BenchmarkConfig {
                capacity: 1024,
                universe: 10_000,
                operations: 100_000,
                seed: 42,
            },
        }
    }

    fn empty_metrics() -> Metrics {
        Metrics {
            hit_stats: None,
            throughput: None,
            latency: None,
            eviction: None,
            scan_resistance: None,
            adaptation: None,
        }
    }

    fn row(policy: &str, workload: &str, case: &str, metrics: Metrics) -> ResultRow {
        ResultRow {
            policy_id: policy.to_lowercase(),
            policy_name: policy.into(),
            workload_id: workload.to_lowercase(),
            workload_name: workload.into(),
            case_id: case.into(),
            metrics,
        }
    }

    fn hit_metrics(hit_rate: f64) -> Metrics {
        Metrics {
            hit_stats: Some(HitStats {
                hits: 0,
                misses: 0,
                inserts: 0,
                updates: 0,
                hit_rate,
                miss_rate: 1.0 - hit_rate,
            }),
            ..empty_metrics()
        }
    }

    #[test]
    fn pivot_section_renders_sorted_rows_and_columns() {
        let rows = [
            row("LRU", "Zipf", "hit_rate", hit_metrics(0.42)),
            row("LFU", "Uniform", "hit_rate", hit_metrics(0.15)),
            row("LFU", "Zipf", "hit_rate", hit_metrics(0.51)),
        ];
        let refs: Vec<&ResultRow> = rows.iter().collect();
        let mut md = String::new();
        write_pivot_section(
            &mut md,
            "Hit Rate Comparison",
            &refs,
            |r| r.metrics.hit_stats.as_ref().map(|s| s.hit_rate),
            |v| format!("{:.2}%", v * 100.0),
        );

        // Header alphabetised: Uniform before Zipf.
        assert!(md.contains("| Policy | Uniform | Zipf |"));
        // Policies alphabetised: LFU before LRU.
        let lfu = md.find("| **LFU** |").expect("LFU row present");
        let lru = md.find("| **LRU** |").expect("LRU row present");
        assert!(lfu < lru, "LFU should sort before LRU");
        // LRU is missing the Uniform workload, so column shows '-'.
        assert!(md.contains("| **LRU** | - | 42.00% |"));
        assert!(md.contains("| **LFU** | 15.00% | 51.00% |"));
    }

    #[test]
    fn pivot_section_empty_emits_no_data_marker() {
        let mut md = String::new();
        write_pivot_section(
            &mut md,
            "Hit Rate Comparison",
            &[],
            |r: &ResultRow| r.metrics.hit_stats.as_ref().map(|s| s.hit_rate),
            |v| format!("{v}"),
        );
        assert!(md.contains("_No data._"));
        assert!(!md.contains("| Policy |"));
    }

    #[test]
    fn scan_resistance_section_skips_rows_without_metric() {
        let rows = [
            row("LRU", "Scan", "scan_resistance", empty_metrics()),
            row(
                "S3-FIFO",
                "Scan",
                "scan_resistance",
                Metrics {
                    scan_resistance: Some(ScanResistanceStats {
                        baseline_hit_rate: 0.80,
                        scan_hit_rate: 0.10,
                        recovery_hit_rate: 0.78,
                        resistance_score: Some(0.975),
                    }),
                    ..empty_metrics()
                },
            ),
        ];
        let refs: Vec<&ResultRow> = rows.iter().collect();
        let mut md = String::new();
        write_scan_resistance_section(&mut md, &refs);
        assert!(md.contains("**S3-FIFO**"));
        assert!(!md.contains("**LRU**"));
        assert!(md.contains("0.975"));
    }

    #[test]
    fn scan_resistance_section_renders_n_a_for_missing_score() {
        let rows = [row(
            "LIFO",
            "Scan",
            "scan_resistance",
            Metrics {
                scan_resistance: Some(ScanResistanceStats {
                    baseline_hit_rate: 0.001,
                    scan_hit_rate: 0.0,
                    recovery_hit_rate: 0.001,
                    resistance_score: None,
                }),
                ..empty_metrics()
            },
        )];
        let refs: Vec<&ResultRow> = rows.iter().collect();
        let mut md = String::new();
        write_scan_resistance_section(&mut md, &refs);
        assert!(md.contains("| **LIFO** |"));
        assert!(
            md.contains(" n/a |"),
            "expected n/a placeholder, got:\n{md}"
        );
    }

    #[test]
    fn adaptation_section_renders_counts() {
        let rows = [row(
            "LRU",
            "Shift",
            "adaptation",
            Metrics {
                adaptation: Some(AdaptationStats {
                    stable_hit_rate: 0.62,
                    ops_to_50_percent: 1234,
                    ops_to_80_percent: 9999,
                    hit_rate_curve: Vec::new(),
                    window_size: 0,
                }),
                ..empty_metrics()
            },
        )];
        let refs: Vec<&ResultRow> = rows.iter().collect();
        let mut md = String::new();
        write_adaptation_section(&mut md, &refs);
        assert!(md.contains("| **LRU** | 62.00% | 1234 | 9999 |"));
    }

    #[test]
    fn adaptation_section_renders_sparkline_when_curve_present() {
        let rows = [
            row(
                "LRU",
                "Shift",
                "adaptation",
                Metrics {
                    adaptation: Some(AdaptationStats {
                        stable_hit_rate: 0.85,
                        ops_to_50_percent: 500,
                        ops_to_80_percent: 1500,
                        hit_rate_curve: vec![0.0, 0.25, 0.5, 0.75, 1.0],
                        window_size: 256,
                    }),
                    ..empty_metrics()
                },
            ),
            row(
                "FIFO",
                "Shift",
                "adaptation",
                Metrics {
                    adaptation: Some(AdaptationStats {
                        stable_hit_rate: 0.4,
                        ops_to_50_percent: 800,
                        ops_to_80_percent: 4000,
                        hit_rate_curve: Vec::new(),
                        window_size: 0,
                    }),
                    ..empty_metrics()
                },
            ),
        ];
        let refs: Vec<&ResultRow> = rows.iter().collect();
        let mut md = String::new();
        write_adaptation_section(&mut md, &refs);

        assert!(
            md.contains("| Curve |"),
            "expected Curve column header, got:\n{md}",
        );
        // LRU has a 5-point curve from 0.0 to 1.0 stepping by 0.25:
        // each value × 7 then rounded → 0, 2, 4 (round-half-away), 5, 7.
        assert!(
            md.contains("`▁▃▅▆█`"),
            "expected sparkline for LRU, got:\n{md}",
        );
        // FIFO has no curve; cell should be an empty backtick pair.
        assert!(
            md.contains("4000 | `` |"),
            "expected empty sparkline cell for FIFO, got:\n{md}",
        );
        // Footnote reports window size when at least one row supplied it.
        assert!(
            md.contains("each cell is 256 ops"),
            "expected window-size note, got:\n{md}",
        );
    }

    #[test]
    fn sparkline_maps_extremes_and_midpoints() {
        assert_eq!(sparkline(&[]), "");
        assert_eq!(sparkline(&[0.0, 1.0]), "▁█");
        // Out-of-range inputs are clamped, not panicked on.
        assert_eq!(sparkline(&[-0.5, 1.5]), "▁█");
        // Midpoint rounds to the middle bucket.
        assert_eq!(sparkline(&[0.5]).chars().count(), 1);
    }

    #[test]
    fn schema_version_matches_when_major_equal() {
        assert!(check_schema_version(SCHEMA_VERSION).is_ok());
        assert!(check_schema_version("1.99.0").is_ok());
        assert!(check_schema_version("2.0.0").is_err());
        assert!(check_schema_version("not a version").is_err());
    }

    #[test]
    fn pivot_section_keeps_first_on_duplicate_pair() {
        // Two rows with the same (policy, workload); first hit_rate is 0.10,
        // second is 0.99. The renderer must keep the first.
        let rows = [
            row("LRU", "Zipf", "hit_rate", hit_metrics(0.10)),
            row("LRU", "Zipf", "hit_rate", hit_metrics(0.99)),
        ];
        let refs: Vec<&ResultRow> = rows.iter().collect();
        let mut md = String::new();
        write_pivot_section(
            &mut md,
            "Hit Rate Comparison",
            &refs,
            |r| r.metrics.hit_stats.as_ref().map(|s| s.hit_rate),
            |v| format!("{:.2}%", v * 100.0),
        );
        assert!(
            md.contains("| **LRU** | 10.00% |"),
            "expected first-wins (10.00%), got:\n{md}"
        );
        assert!(!md.contains("99.00%"));
    }

    #[test]
    fn same_file_detects_existing_paths() {
        let dir = std::env::temp_dir().join("cachekit-render-docs-same-file");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.json");
        std::fs::write(&p, b"{}").unwrap();

        // Self-comparison: same physical file.
        assert!(same_file(&p, &p));

        // Different file in same dir, neither pointing at the other.
        let q = dir.join("b.json");
        std::fs::write(&q, b"{}").unwrap();
        assert!(!same_file(&p, &q));

        // Same file reached via a non-canonical path (uses parent canonicalisation).
        let alt = dir.join(".").join("a.json");
        assert!(same_file(&p, &alt));

        // Nonexistent dest with a different filename in the same dir.
        let missing = dir.join("never-created.json");
        assert!(!same_file(&p, &missing));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn render_policy_colors_literal_emits_valid_js_object() {
        let metas = [
            PolicyMeta {
                id: "lru",
                display_name: "LRU",
                color: "#3498db",
            },
            PolicyMeta {
                id: "two_q",
                display_name: "2Q",
                color: "#e67e22",
            },
        ];
        let lit = render_policy_colors_literal(&metas);
        // Quoted keys (JSON-safe even for "2Q") and the trailing item has no comma.
        assert!(lit.contains("\"LRU\": \"#3498db\","), "got:\n{lit}");
        assert!(lit.contains("\"2Q\": \"#e67e22\""), "got:\n{lit}");
        assert!(!lit.contains("#e67e22\","), "trailing comma in last entry");
        assert!(lit.starts_with('{') && lit.trim_end().ends_with('}'));
    }

    #[test]
    fn inject_policy_colors_replaces_sentinel_once() {
        let template = "head\nconst POLICY_COLORS = /* @POLICY_COLORS@ */ {};\ntail";
        let metas = [PolicyMeta {
            id: "lru",
            display_name: "LRU",
            color: "#3498db",
        }];
        let out = inject_policy_colors(template, &metas).expect("substitution");
        assert!(out.contains("\"LRU\": \"#3498db\""));
        // Sentinel must be gone after substitution.
        assert!(!out.contains("@POLICY_COLORS@"));
        // Surrounding lines preserved.
        assert!(out.starts_with("head\n"));
        assert!(out.ends_with("\ntail"));
    }

    #[test]
    fn inject_policy_colors_errors_when_sentinel_missing() {
        let err = inject_policy_colors("no sentinel here", &[]).unwrap_err();
        assert!(
            err.to_string()
                .contains("missing the policy-colors sentinel"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn inject_policy_colors_errors_when_sentinel_duplicated() {
        let template = "/* @POLICY_COLORS@ */ {} and again /* @POLICY_COLORS@ */ {}";
        let err = inject_policy_colors(template, &[]).unwrap_err();
        assert!(
            err.to_string().contains("expected exactly 1"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn bundled_charts_script_contains_the_sentinel() {
        // Cheap guard: the include_str! script must actually carry the
        // sentinel. If someone hand-edits charts_template.js and removes it,
        // every render_docs run would fail at the substitution step; this test
        // catches the regression at `cargo test` time instead.
        assert!(
            CHARTS_JS.contains(POLICY_COLORS_PLACEHOLDER),
            "charts_template.js no longer contains `{POLICY_COLORS_PLACEHOLDER}`",
        );
        // The HTML, on the other hand, must NOT carry the sentinel — that
        // would silently leave a `{}` literal in markup if the JS rename ever
        // regressed, and break colors with no error.
        assert!(
            !CHARTS_HTML.contains(POLICY_COLORS_PLACEHOLDER),
            "charts_template.html unexpectedly contains the sentinel; \
             substitution should target charts_template.js only",
        );
    }

    #[test]
    fn bundled_charts_html_is_csp_safe() {
        // The page must not ship inline scripts, inline styles, or
        // string-eval patterns; it should pull in the sibling charts.js
        // and charts.css instead. These checks pin the CSP posture so a
        // future hand-edit can't silently reintroduce inline content or
        // opt out of the protections.
        assert!(
            CHARTS_HTML.contains("src=\"charts.js\""),
            "charts_template.html no longer references the external charts.js",
        );
        assert!(
            CHARTS_HTML.contains("href=\"charts.css\""),
            "charts_template.html no longer references the external charts.css",
        );
        assert!(
            CHARTS_HTML.contains("Content-Security-Policy"),
            "charts_template.html is missing its CSP <meta>",
        );
        // The CSP must include all the strict directives we rely on.
        // `style-src 'self'` (without 'unsafe-inline') is the whole point of
        // extracting charts.css; pin it here so a careless edit can't quietly
        // re-add `'unsafe-inline'` to make a stray inline style "work".
        for required in [
            "default-src 'none'",
            "frame-ancestors 'none'",
            "style-src 'self'",
        ] {
            assert!(
                CHARTS_HTML.contains(required),
                "charts_template.html CSP no longer asserts `{required}`",
            );
        }
        for forbidden in ["unsafe-eval", "unsafe-inline", "unsafe-hashes"] {
            assert!(
                !CHARTS_HTML.contains(forbidden),
                "charts_template.html must not opt in to `{forbidden}`",
            );
        }
        // script-src must allow Chart.js by both hash (CSP3, strict) and
        // host (compat fallback for browsers that don't honor hash-source
        // for external scripts). Either drift would silently break loads.
        for required in [
            "'sha384-9nhczxUqK87bcKHh20fSQcTGD4qq5GhayNYSYWqwBkINBhOfQLg/P5HG5lF1urn4'",
            "https://cdn.jsdelivr.net",
        ] {
            assert!(
                CHARTS_HTML.contains(required),
                "charts_template.html script-src no longer lists `{required}`",
            );
        }
        // No inline <style> or <script> bodies, and no `style=""` attribute
        // anywhere — those would all require `'unsafe-inline'` to load.
        assert!(
            !CHARTS_HTML.contains("<style>"),
            "charts_template.html contains an inline <style> block",
        );
        assert!(
            !CHARTS_HTML.contains("style=\""),
            "charts_template.html contains an inline `style=\"...\"` attribute",
        );

        // Substring guards for string-eval patterns. Each canonical form is
        // checked across all three quoting styles JS supports for the
        // string argument (`"..."`, `'...'`, `` `...` ``). Substring matches
        // can false-positive in comments — keep these out of the templates
        // (mention them in render_docs.rs prose instead).
        let string_eval_calls = ["setTimeout(", "setInterval(", "execScript("];
        let quote_chars = ['"', '\'', '`'];
        let mut forbidden_patterns: Vec<String> = vec!["eval(".into(), "new Function(".into()];
        for call in string_eval_calls {
            for q in quote_chars {
                forbidden_patterns.push(format!("{call}{q}"));
            }
        }
        for pat in &forbidden_patterns {
            assert!(
                !CHARTS_HTML.contains(pat.as_str()),
                "charts_template.html contains string-eval pattern {pat:?}",
            );
            assert!(
                !CHARTS_JS.contains(pat.as_str()),
                "charts_template.js contains string-eval pattern {pat:?}",
            );
        }
    }

    #[test]
    fn generate_markdown_smoke() {
        let mut artifact = BenchmarkArtifact::new(metadata());
        artifact.add_result(row("LRU", "Zipf", "hit_rate", hit_metrics(0.5)));
        artifact.add_result(row(
            "LRU",
            "Zipf",
            "comprehensive",
            Metrics {
                throughput: Some(ThroughputStats {
                    duration_ms: 100.0,
                    ops_per_sec: 5_000_000.0,
                    gets_per_sec: 4_000_000.0,
                    inserts_per_sec: 1_000_000.0,
                }),
                latency: Some(LatencyStats {
                    sample_count: 1000,
                    min_ns: 10,
                    p50_ns: 50,
                    p95_ns: 200,
                    p99_ns: 500,
                    max_ns: 9999,
                    mean_ns: 75,
                }),
                ..empty_metrics()
            },
        ));
        let md = generate_markdown(&artifact);
        assert!(md.contains("# Benchmark Results"));
        assert!(md.contains("## Hit Rate Comparison"));
        assert!(md.contains("## Throughput (Million ops/sec)"));
        assert!(md.contains("## Latency P99 (nanoseconds)"));
        assert!(md.contains("5.00")); // 5M ops/sec
        assert!(md.contains("500")); // p99
        assert!(md.contains("schema v"));
    }
}
