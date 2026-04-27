//! ## Architecture
//! Owns the on-disk asset templates (`charts_template.{html,js,css}`),
//! their sentinels, the substitution engine that turns each template
//! into a rendered string, and the build-time-derived metadata
//! (schema version + renderer stamp) that gets woven in.
//!
//! ## Key Components
//! - [`CHARTS_HTML`] / [`CHARTS_JS`] / [`CHARTS_CSS`]: bundled assets,
//!   loaded via `include_str!` so a stale checkout can't ship a
//!   surprise.
//! - [`Substitution`] + [`apply_substitutions`]: count-checked
//!   sentinel replacement so a hand-edit that drops or duplicates a
//!   token is a render-time error, not a silent breakage.
//! - [`render_charts_html`] / [`render_charts_js`]: the only two
//!   entry points the IO orchestrator calls.
//! - [`renderer_stamp`]: provenance string woven into the rendered
//!   HTML's `<meta name="generator">` and the markdown footer.
//!
//! ## Performance Trade-offs
//! Substitution does `String::replace` per sentinel (O(N) over the
//! template). N is ~5 KiB and substitutions run once per render, so
//! a streaming replacement is not worth the extra moving parts.
//!
//! ## Thread Safety
//! All public functions are pure; the constants are `&'static str`.

use std::error::Error;
use std::fmt::Write as _;

use bench_support::json_results::SCHEMA_VERSION;
use bench_support::registry::PolicyMeta;

// ============================================================================
// Bundled assets
// ============================================================================

/// HTML shell for the charts page. Carries sentinels for the Chart.js
/// version + SRI hash (substituted from [`CHART_JS_VERSION`] /
/// [`CHART_JS_SRI`]) so a Chart.js bump only touches Rust constants.
/// Runs under a strict CSP (no `unsafe-eval`, no `unsafe-inline`); the
/// only inline content is the tiny `no-js` remover, whose body
/// ([`NO_JS_REMOVER_BODY`]) is substituted into the template and whose
/// CSP hash-source is computed at render time by
/// [`compute_no_js_remover_hash`] — so the two cannot drift.
pub(crate) const CHARTS_HTML: &str = include_str!("../charts_template.html");

/// Behavior for the charts page; sibling script of [`CHARTS_HTML`].
/// Carries sentinels for the policy color map and the expected schema
/// major (substituted from [`bench_support::registry::POLICIES`] and
/// [`SCHEMA_VERSION`]) so the JS can never drift from the Rust schema.
pub(crate) const CHARTS_JS: &str = include_str!("../charts_template.js");

/// Presentation for the charts page; sibling stylesheet of
/// [`CHARTS_HTML`]. Hosts the `.no-js #loading` and `.hidden` rules
/// that let the page run under `style-src 'self'` (no inline `<style>`
/// or `style="…"` attributes).
pub(crate) const CHARTS_CSS: &str = include_str!("../charts_template.css");

// ============================================================================
// Pinned external dependencies
// ============================================================================

/// Pinned Chart.js release. Bump this to upgrade — no other source
/// needs editing; [`render_charts_html`] substitutes it into the URL,
/// the `integrity=` attribute, and the CSP `script-src` hash-source.
pub(crate) const CHART_JS_VERSION: &str = "4.4.1";

/// Subresource Integrity hash for the Chart.js release pinned by
/// [`CHART_JS_VERSION`]. Recompute with
/// `curl -s https://cdn.jsdelivr.net/npm/chart.js@VERSION/dist/chart.umd.min.js \
///    | openssl dgst -sha384 -binary | openssl base64 \
///    | (printf 'sha384-'; cat)`.
pub(crate) const CHART_JS_SRI: &str =
    "sha384-9nhczxUqK87bcKHh20fSQcTGD4qq5GhayNYSYWqwBkINBhOfQLg/P5HG5lF1urn4";

/// Body of the parser-blocking inline `<script>` substituted into
/// [`CHARTS_HTML`]'s `<head>` at the [`NO_JS_REMOVER_BODY_PLACEHOLDER`]
/// sentinel. The CSP `script-src` hash-source authorizing it is
/// computed (not pinned) by [`compute_no_js_remover_hash`] from these
/// exact bytes — so a drive-by edit to either side is a non-issue:
/// only this constant is the source of truth.
pub(crate) const NO_JS_REMOVER_BODY: &str = "document.documentElement.classList.remove('no-js')";

// ============================================================================
// Sentinels
// ============================================================================

/// Sentinel substituted in `charts_template.js` for the policy color
/// map. Wrapped as a JS comment + empty object literal so the
/// unsubstituted template is still syntactically valid (it would parse
/// to `const POLICY_COLORS = {};` and render fallback colors).
pub(crate) const POLICY_COLORS_PLACEHOLDER: &str = "/* @POLICY_COLORS@ */ {}";

/// Sentinel substituted in `charts_template.js` for the expected JSON
/// schema major. Wrapped so the unsubstituted template still parses
/// (it would yield `EXPECTED_SCHEMA_MAJOR = '0'` and reject every real
/// artifact, failing loud at runtime). The Rust-side
/// [`SCHEMA_VERSION`] is the single source of truth.
pub(crate) const SCHEMA_MAJOR_PLACEHOLDER: &str = "/* @SCHEMA_MAJOR@ */ '0'";

/// Sentinel substituted in `charts_template.html` for the Chart.js
/// release tag. Appears once, in the `<script src="…/chart.js@…/…">`
/// URL.
pub(crate) const CHART_JS_VERSION_PLACEHOLDER: &str = "@CHART_JS_VERSION@";

/// Sentinel substituted in `charts_template.html` for the Chart.js SRI
/// hash. Appears twice — once in `integrity="…"` on the `<script>` tag
/// and once in the CSP `script-src` hash-source. The substitution
/// helper asserts both occurrences, so a future template edit can't
/// drop one silently.
pub(crate) const CHART_JS_SRI_PLACEHOLDER: &str = "@CHART_JS_SRI@";

/// Sentinel substituted in `charts_template.html` for the body of the
/// parser-blocking inline `<script>`. Replaced with
/// [`NO_JS_REMOVER_BODY`] at render time so the constant is the one
/// source of truth.
pub(crate) const NO_JS_REMOVER_BODY_PLACEHOLDER: &str = "@NO_JS_REMOVER_BODY@";

/// Sentinel substituted in `charts_template.html` for the CSP
/// `script-src` hash-source authorizing the inline `no-js` remover.
/// Replaced with `sha256-<base64(SHA-256(NO_JS_REMOVER_BODY))>`
/// computed at render time, so the body and the entry that authorizes
/// it cannot drift apart — there's nothing to keep in sync, just a
/// function.
pub(crate) const NO_JS_REMOVER_HASH_PLACEHOLDER: &str = "@NO_JS_REMOVER_HASH@";

/// Sentinel substituted in `charts_template.html` (HTML `<meta
/// name="generator">`) and woven into the `index.md` footer for the
/// renderer's name + version stamp. Substituted with
/// [`renderer_stamp`].
pub(crate) const RENDERER_STAMP_PLACEHOLDER: &str = "@RENDERER_STAMP@";

// ============================================================================
// Renderer provenance
// ============================================================================

/// Renderer crate name (from `Cargo.toml`). Surfaced in the `index.md`
/// footer and the `<meta name="generator">` tag so a reader hitting a
/// stale or surprising rendered file can trace it back to the exact
/// binary that produced it.
pub(crate) const RENDERER_NAME: &str = env!("CARGO_PKG_NAME");

/// Renderer crate version (from `Cargo.toml`). See [`RENDERER_NAME`].
pub(crate) const RENDERER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Optional git SHA, embedded only when `RENDER_DOCS_GIT_SHA` is set
/// in the build environment (e.g. via CI). When unset, the stamp omits
/// the git suffix entirely so dev builds don't lie about provenance.
pub(crate) const RENDERER_GIT_SHA: Option<&str> = option_env!("RENDER_DOCS_GIT_SHA");

/// Compose the renderer provenance stamp.
///
/// Returns `"<name> v<version>"`, optionally suffixed with
/// `" (git <short-sha>)"` when [`RENDERER_GIT_SHA`] is present at
/// build time. The stamp is restricted to a safe character set
/// (`[A-Za-z0-9._() -]`) so it can be interpolated into either a
/// Markdown footer or an HTML attribute value with no escaping; the
/// `renderer_stamp_is_html_attribute_safe` test enforces this
/// invariant.
pub(crate) fn renderer_stamp() -> String {
    match RENDERER_GIT_SHA {
        Some(sha) if !sha.is_empty() => {
            // Truncate to the conventional 7-char short SHA so the
            // stamp stays compact in both the markdown footer and the
            // `<meta>` attribute.
            let short: String = sha.chars().take(7).collect();
            format!("{RENDERER_NAME} v{RENDERER_VERSION} (git {short})")
        },
        _ => format!("{RENDERER_NAME} v{RENDERER_VERSION}"),
    }
}

/// Returns `true` if `c` is safe to embed verbatim in an HTML
/// attribute value and a Markdown paragraph without escaping. The
/// renderer stamp is built from `Cargo.toml` and an optional git SHA,
/// both of which should naturally produce only this subset; we assert
/// it at render time so a malformed `RENDER_DOCS_GIT_SHA` env var
/// can't smuggle `<script>` into the rendered HTML.
pub(crate) fn is_renderer_stamp_safe_char(c: char) -> bool {
    matches!(c, 'a'..='z' | 'A'..='Z' | '0'..='9' | '.' | '_' | '(' | ')' | ' ' | '-')
}

// ============================================================================
// Substitution engine
// ============================================================================

/// One sentinel-substitution rule.
pub(crate) struct Substitution<'a> {
    pub(crate) sentinel: &'a str,
    pub(crate) value: String,
    /// Exact number of times `sentinel` must appear in the template;
    /// any other count is a substitution-time error so a hand-edit
    /// can't silently drop or duplicate the token.
    pub(crate) expected_count: usize,
}

/// Apply all `subs` to `template`, in order, returning the rendered
/// string. Each substitution is fully resolved before the next runs,
/// so a value containing another sentinel won't be interpreted
/// (substitutions don't chain). Errors when any sentinel's actual
/// occurrence count differs from its `expected_count`.
pub(crate) fn apply_substitutions(
    template: &str,
    subs: &[Substitution],
) -> Result<String, Box<dyn Error>> {
    let mut out = template.to_string();
    for sub in subs {
        let count = out.matches(sub.sentinel).count();
        if count != sub.expected_count {
            return Err(format!(
                "template substitution failed: sentinel `{}` appears {} time(s), expected {}",
                sub.sentinel, count, sub.expected_count,
            )
            .into());
        }
        out = out.replace(sub.sentinel, &sub.value);
    }
    Ok(out)
}

// ============================================================================
// Schema version
// ============================================================================

/// Parse the leading numeric component of a `<major>.<minor>.<patch>`
/// version. Returns `None` for missing or non-numeric majors. Shared
/// between [`check_schema_version`] and [`schema_major`] so the two
/// always agree on what "major" means.
pub(crate) fn parse_major(version: &str) -> Option<u32> {
    version.split('.').next()?.parse::<u32>().ok()
}

/// Major component of [`SCHEMA_VERSION`] (e.g. `1` for `"1.2.0"`),
/// substituted into `charts_template.js` so the JS schema check is
/// derived from Rust at render time. Shares parsing with
/// [`check_schema_version`] via [`parse_major`]; the
/// `schema_version_constant_is_well_formed` test pins the invariant
/// so this never panics in production.
pub(crate) fn schema_major() -> u32 {
    parse_major(SCHEMA_VERSION).expect(
        "SCHEMA_VERSION must start with a numeric major; \
         pinned by schema_version_constant_is_well_formed",
    )
}

/// Refuses results whose major schema version differs from this
/// binary's.
///
/// Distinguishes three error cases so users can act on the message:
/// 1. the renderer's own [`SCHEMA_VERSION`] constant is malformed
///    (internal bug — shouldn't happen, pinned by a startup test);
/// 2. the artifact's `schema_version` field doesn't parse as
///    `<u32>.…`;
/// 3. both parse but the majors differ.
pub(crate) fn check_schema_version(found: &str) -> Result<(), Box<dyn Error>> {
    let renderer_major = parse_major(SCHEMA_VERSION).ok_or_else(|| {
        format!(
            "internal error: render_docs's SCHEMA_VERSION constant `{SCHEMA_VERSION}` does not \
             start with a numeric major (this is a bug in bench-support, not in your input)"
        )
    })?;
    let found_major = parse_major(found).ok_or_else(|| {
        format!(
            "unrecognized schema version `{found}` in artifact (expected `<major>.<minor>.<patch>` \
             form, matching renderer's {SCHEMA_VERSION})"
        )
    })?;
    if found_major == renderer_major {
        Ok(())
    } else {
        Err(format!(
            "schema version mismatch: artifact is {found} (major {found_major}), \
             renderer expects {SCHEMA_VERSION} (major {renderer_major})"
        )
        .into())
    }
}

// ============================================================================
// Rendering
// ============================================================================

/// Render `charts_template.js` with all sentinels substituted.
pub(crate) fn render_charts_js(policies: &[PolicyMeta]) -> Result<String, Box<dyn Error>> {
    let subs = [
        Substitution {
            sentinel: POLICY_COLORS_PLACEHOLDER,
            value: render_policy_colors_literal(policies),
            expected_count: 1,
        },
        Substitution {
            sentinel: SCHEMA_MAJOR_PLACEHOLDER,
            value: format!("'{}'", schema_major()),
            expected_count: 1,
        },
    ];
    apply_substitutions(CHARTS_JS, &subs)
}

/// Render `charts_template.html` with all sentinels substituted.
pub(crate) fn render_charts_html() -> Result<String, Box<dyn Error>> {
    let stamp = renderer_stamp();
    if let Some(bad) = stamp.chars().find(|c| !is_renderer_stamp_safe_char(*c)) {
        return Err(format!(
            "renderer stamp {stamp:?} contains unsafe character {bad:?}; \
             refuse to interpolate into HTML attribute"
        )
        .into());
    }
    let subs = [
        Substitution {
            sentinel: CHART_JS_VERSION_PLACEHOLDER,
            value: CHART_JS_VERSION.to_string(),
            expected_count: 1,
        },
        Substitution {
            sentinel: CHART_JS_SRI_PLACEHOLDER,
            value: CHART_JS_SRI.to_string(),
            expected_count: 2,
        },
        Substitution {
            sentinel: NO_JS_REMOVER_BODY_PLACEHOLDER,
            value: NO_JS_REMOVER_BODY.to_string(),
            expected_count: 1,
        },
        Substitution {
            sentinel: NO_JS_REMOVER_HASH_PLACEHOLDER,
            value: compute_no_js_remover_hash(NO_JS_REMOVER_BODY),
            expected_count: 1,
        },
        Substitution {
            sentinel: RENDERER_STAMP_PLACEHOLDER,
            value: stamp,
            expected_count: 1,
        },
    ];
    apply_substitutions(CHARTS_HTML, &subs)
}

/// CSP `script-src` hash-source for an inline script body, as
/// `sha256-<base64(SHA-256(body))>`. The standard base64 alphabet
/// (with `+`/`/` and padding) is what browsers expect for hash-source
/// values; some SRI tools emit URL-safe base64, which CSP rejects.
pub(crate) fn compute_no_js_remover_hash(body: &str) -> String {
    use base64::Engine;
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(body.as_bytes());
    format!(
        "sha256-{}",
        base64::engine::general_purpose::STANDARD.encode(digest)
    )
}

/// Render a JS object literal `{ "Display": "#hex", ... }` from
/// `policies`. Display names are emitted as JSON strings so any
/// special characters are safely escaped.
pub(crate) fn render_policy_colors_literal(policies: &[PolicyMeta]) -> String {
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bench_support::registry::POLICIES;
    use std::collections::BTreeSet;

    use crate::markdown::POLICY_GUIDE_MD;

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
        assert!(lit.contains("\"LRU\": \"#3498db\","), "got:\n{lit}");
        assert!(lit.contains("\"2Q\": \"#e67e22\""), "got:\n{lit}");
        assert!(!lit.contains("#e67e22\","), "trailing comma in last entry");
        assert!(lit.starts_with('{') && lit.trim_end().ends_with('}'));
    }

    #[test]
    fn apply_substitutions_replaces_each_sentinel_correct_count() {
        let template = "head [A] middle [B] [B] tail";
        let subs = [
            Substitution {
                sentinel: "[A]",
                value: "alpha".into(),
                expected_count: 1,
            },
            Substitution {
                sentinel: "[B]",
                value: "beta".into(),
                expected_count: 2,
            },
        ];
        let out = apply_substitutions(template, &subs).expect("substitution");
        assert_eq!(out, "head alpha middle beta beta tail");
    }

    #[test]
    fn apply_substitutions_errors_when_sentinel_missing() {
        let err = apply_substitutions(
            "no sentinel here",
            &[Substitution {
                sentinel: "[A]",
                value: "x".into(),
                expected_count: 1,
            }],
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("appears 0 time(s), expected 1"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn apply_substitutions_errors_when_sentinel_count_wrong() {
        let err = apply_substitutions(
            "[A][A]",
            &[Substitution {
                sentinel: "[A]",
                value: "x".into(),
                expected_count: 1,
            }],
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("appears 2 time(s), expected 1"),
            "unexpected error: {err}",
        );
        let err = apply_substitutions(
            "[A]",
            &[Substitution {
                sentinel: "[A]",
                value: "x".into(),
                expected_count: 2,
            }],
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("appears 1 time(s), expected 2"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn apply_substitutions_does_not_chain() {
        // After [A]→[B] the template has two `[B]`s, but the second
        // substitution expects exactly 1 — so this errors. That's the
        // safe failure mode: chaining is opt-out.
        let template = "[A] [B]";
        let subs = [
            Substitution {
                sentinel: "[A]",
                value: "[B]".into(),
                expected_count: 1,
            },
            Substitution {
                sentinel: "[B]",
                value: "ZZ".into(),
                expected_count: 1,
            },
        ];
        assert!(apply_substitutions(template, &subs).is_err());
    }

    #[test]
    fn schema_major_returns_leading_component_of_schema_version() {
        let expected: u32 = SCHEMA_VERSION.split('.').next().unwrap().parse().unwrap();
        assert_eq!(schema_major(), expected);
        // Today's schema is "1.x.y"; pin it so a future major bump
        // forces a deliberate review of the JS-side coupling.
        assert_eq!(schema_major(), 1);
    }

    #[test]
    fn schema_version_constant_is_well_formed() {
        assert!(
            parse_major(SCHEMA_VERSION).is_some(),
            "SCHEMA_VERSION = {SCHEMA_VERSION:?} must start with a numeric major \
             (e.g. \"1.0.0\"); update SCHEMA_VERSION accordingly",
        );
        assert!(
            !SCHEMA_VERSION.is_empty(),
            "SCHEMA_VERSION must not be empty",
        );
    }

    #[test]
    fn renderer_stamp_starts_with_crate_name_and_version() {
        let stamp = renderer_stamp();
        assert!(
            stamp.starts_with(&format!("{RENDERER_NAME} v{RENDERER_VERSION}")),
            "stamp {stamp:?} must start with `{RENDERER_NAME} v{RENDERER_VERSION}`",
        );
        assert!(!RENDERER_NAME.is_empty(), "CARGO_PKG_NAME is empty");
        assert!(!RENDERER_VERSION.is_empty(), "CARGO_PKG_VERSION is empty");
    }

    #[test]
    fn renderer_stamp_is_html_attribute_safe() {
        let stamp = renderer_stamp();
        for c in stamp.chars() {
            assert!(
                is_renderer_stamp_safe_char(c),
                "renderer stamp contains unsafe character {c:?}; would need HTML \
                 escaping before interpolation: {stamp:?}",
            );
        }
    }

    #[test]
    fn render_charts_html_rejects_unsafe_renderer_stamp() {
        for unsafe_ch in ['<', '>', '"', '\'', '&', '\n', '\t', '\\', '/'] {
            assert!(
                !is_renderer_stamp_safe_char(unsafe_ch),
                "{unsafe_ch:?} must be rejected by the renderer-stamp safety predicate",
            );
        }
    }

    #[test]
    fn check_schema_version_blames_renderer_when_constant_is_broken() {
        let renderer_major = parse_major(SCHEMA_VERSION).expect("constant pinned by sibling test");
        assert!(check_schema_version(&format!("{renderer_major}.0.0")).is_ok());
        let wrong = format!("{}.0.0", renderer_major + 1);
        let err = check_schema_version(&wrong).unwrap_err().to_string();
        assert!(
            err.contains(&wrong) && err.contains(SCHEMA_VERSION),
            "mismatch error should name both versions: {err}",
        );
        let err = check_schema_version("not-a-version")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("unrecognized schema version") && !err.contains("internal error"),
            "malformed input should be attributed to the user, got: {err}",
        );
    }

    #[test]
    fn schema_version_matches_when_major_equal() {
        assert!(check_schema_version(SCHEMA_VERSION).is_ok());
        assert!(check_schema_version("1.99.0").is_ok());
        assert!(check_schema_version("2.0.0").is_err());
        assert!(check_schema_version("not a version").is_err());
    }

    #[test]
    fn raw_charts_js_carries_required_sentinels() {
        for sentinel in [POLICY_COLORS_PLACEHOLDER, SCHEMA_MAJOR_PLACEHOLDER] {
            assert!(
                CHARTS_JS.contains(sentinel),
                "charts_template.js no longer contains `{sentinel}`",
            );
        }
        for sentinel in [POLICY_COLORS_PLACEHOLDER, SCHEMA_MAJOR_PLACEHOLDER] {
            assert!(
                !CHARTS_HTML.contains(sentinel),
                "charts_template.html unexpectedly contains JS-only sentinel `{sentinel}`",
            );
        }
    }

    #[test]
    fn raw_charts_html_carries_required_sentinels() {
        assert_eq!(
            CHARTS_HTML.matches(CHART_JS_VERSION_PLACEHOLDER).count(),
            1,
            "charts_template.html must reference `{CHART_JS_VERSION_PLACEHOLDER}` exactly once",
        );
        assert_eq!(
            CHARTS_HTML.matches(CHART_JS_SRI_PLACEHOLDER).count(),
            2,
            "charts_template.html must reference `{CHART_JS_SRI_PLACEHOLDER}` exactly \
             twice (CSP hash-source + <script integrity>)",
        );
        assert_eq!(
            CHARTS_HTML.matches(NO_JS_REMOVER_BODY_PLACEHOLDER).count(),
            1,
            "charts_template.html must reference `{NO_JS_REMOVER_BODY_PLACEHOLDER}` exactly once",
        );
        assert_eq!(
            CHARTS_HTML.matches(NO_JS_REMOVER_HASH_PLACEHOLDER).count(),
            1,
            "charts_template.html must reference `{NO_JS_REMOVER_HASH_PLACEHOLDER}` exactly once",
        );
        assert_eq!(
            CHARTS_HTML.matches(RENDERER_STAMP_PLACEHOLDER).count(),
            1,
            "charts_template.html must reference `{RENDERER_STAMP_PLACEHOLDER}` exactly \
             once (the <meta name=\"generator\"> attribute)",
        );
        for sentinel in [
            CHART_JS_VERSION_PLACEHOLDER,
            CHART_JS_SRI_PLACEHOLDER,
            NO_JS_REMOVER_BODY_PLACEHOLDER,
            NO_JS_REMOVER_HASH_PLACEHOLDER,
            RENDERER_STAMP_PLACEHOLDER,
        ] {
            assert!(
                !CHARTS_JS.contains(sentinel),
                "charts_template.js unexpectedly contains HTML-only sentinel `{sentinel}`",
            );
            assert!(
                !CHARTS_CSS.contains(sentinel),
                "charts_template.css unexpectedly contains HTML-only sentinel `{sentinel}`",
            );
        }
    }

    #[test]
    fn render_charts_js_substitutes_all_sentinels() {
        let metas = [PolicyMeta {
            id: "lru",
            display_name: "LRU",
            color: "#3498db",
        }];
        let out = render_charts_js(&metas).expect("substitution");
        assert!(
            !out.contains(POLICY_COLORS_PLACEHOLDER),
            "POLICY_COLORS sentinel survived: {out}",
        );
        assert!(
            !out.contains(SCHEMA_MAJOR_PLACEHOLDER),
            "SCHEMA_MAJOR sentinel survived: {out}",
        );
        assert!(out.contains("\"LRU\": \"#3498db\""));
        assert!(out.contains(&format!("EXPECTED_SCHEMA_MAJOR = '{}'", schema_major())));
    }

    #[test]
    fn render_charts_html_substitutes_all_sentinels() {
        let out = render_charts_html().expect("substitution");
        assert!(!out.contains(CHART_JS_VERSION_PLACEHOLDER));
        assert!(!out.contains(CHART_JS_SRI_PLACEHOLDER));
        assert!(!out.contains(NO_JS_REMOVER_BODY_PLACEHOLDER));
        assert!(!out.contains(NO_JS_REMOVER_HASH_PLACEHOLDER));
        assert!(!out.contains(RENDERER_STAMP_PLACEHOLDER));
        let stamp = renderer_stamp();
        assert!(
            out.contains(&format!(r#"<meta name="generator" content="{stamp}">"#)),
            "rendered HTML missing <meta name=\"generator\"> with stamp {stamp:?}",
        );
        assert!(out.contains(&format!(
            "chart.js@{CHART_JS_VERSION}/dist/chart.umd.min.js"
        )));
        assert!(out.contains(&format!("integrity=\"{CHART_JS_SRI}\"")));
        assert!(out.contains(&format!("'{CHART_JS_SRI}'")));
        assert!(out.contains(&format!("<script>{NO_JS_REMOVER_BODY}</script>")));
        let expected_hash = compute_no_js_remover_hash(NO_JS_REMOVER_BODY);
        assert!(
            out.contains(&format!("'{expected_hash}'")),
            "rendered HTML CSP missing computed hash `{expected_hash}`",
        );
    }

    fn rendered_html() -> String {
        render_charts_html().expect("html substitution")
    }

    fn rendered_js() -> String {
        render_charts_js(POLICIES).expect("js substitution")
    }

    #[test]
    fn rendered_html_links_external_assets() {
        let html = rendered_html();
        assert!(
            html.contains("src=\"charts.js\""),
            "rendered HTML no longer references the external charts.js",
        );
        assert!(
            html.contains("href=\"charts.css\""),
            "rendered HTML no longer references the external charts.css",
        );
        assert!(
            html.contains("Content-Security-Policy"),
            "rendered HTML is missing its CSP <meta>",
        );
    }

    #[test]
    fn rendered_html_csp_carries_required_directives() {
        let html = rendered_html();
        for required in [
            "default-src 'none';",
            "frame-ancestors 'none'",
            "style-src 'self';",
            "img-src 'self' data:;",
            "connect-src 'self';",
            "base-uri 'none';",
            "form-action 'none';",
        ] {
            assert!(
                html.contains(required),
                "rendered HTML CSP no longer asserts `{required}`",
            );
        }
    }

    #[test]
    fn rendered_html_csp_does_not_opt_in_to_unsafe_sources() {
        let html = rendered_html();
        for forbidden in ["unsafe-eval", "unsafe-inline", "unsafe-hashes"] {
            assert!(
                !html.contains(forbidden),
                "rendered HTML opts in to `{forbidden}`; this defeats the strict CSP",
            );
        }
    }

    #[test]
    fn rendered_html_csp_pins_chart_js_by_hash_and_host() {
        let html = rendered_html();
        assert!(
            html.contains(&format!("'{CHART_JS_SRI}'")),
            "script-src no longer lists Chart.js SRI hash `{CHART_JS_SRI}`",
        );
        assert!(
            html.contains("https://cdn.jsdelivr.net"),
            "script-src no longer lists the jsdelivr host (compat fallback for older browsers)",
        );
    }

    #[test]
    fn rendered_html_chart_js_sri_appears_in_csp_and_integrity() {
        let html = rendered_html();
        assert!(
            html.contains(&format!("'{CHART_JS_SRI}'")),
            "Chart.js SRI hash missing from CSP `script-src`",
        );
        assert!(
            html.contains(&format!("integrity=\"{CHART_JS_SRI}\"")),
            "Chart.js SRI hash missing from <script integrity>",
        );
        let occurrences = html.matches(CHART_JS_SRI).count();
        assert_eq!(
            occurrences, 2,
            "Chart.js SRI hash appears {occurrences} times in rendered HTML; expected exactly 2"
        );
    }

    #[test]
    fn rendered_html_pins_no_js_class_and_noscript() {
        let html = rendered_html();
        assert!(
            html.contains("<html lang=\"en\" class=\"no-js\">"),
            "rendered HTML no longer marks <html> with `class=\"no-js\"`; the inline \
             remover, `.no-js #loading`, and <noscript> all depend on this token",
        );
        assert!(
            html.contains("<noscript>"),
            "rendered HTML no longer carries a <noscript> fallback",
        );
        assert!(
            html.contains("JavaScript is required"),
            "<noscript> fallback wording changed; pin the user-visible string \
             so wording shifts are deliberate",
        );
    }

    #[test]
    fn rendered_html_carries_inline_no_js_remover_and_matching_hash() {
        let html = rendered_html();
        let inline_script = format!("<script>{NO_JS_REMOVER_BODY}</script>");
        assert!(
            html.contains(&inline_script),
            "rendered HTML no longer carries the inline no-js remover verbatim \
             ({inline_script:?}); render_charts_html may be stale",
        );
        let expected_hash = compute_no_js_remover_hash(NO_JS_REMOVER_BODY);
        assert!(
            html.contains(&format!("'{expected_hash}'")),
            "rendered HTML CSP no longer lists `'{expected_hash}'`; browsers \
             will block the inline script. Computed body length = {}.",
            NO_JS_REMOVER_BODY.len(),
        );
    }

    #[test]
    fn compute_no_js_remover_hash_is_deterministic_and_well_shaped() {
        let body = "abc";
        let h1 = compute_no_js_remover_hash(body);
        let h2 = compute_no_js_remover_hash(body);
        assert_eq!(h1, h2, "hash must be a pure function of the body");
        assert_canonical_hash_source("compute_no_js_remover_hash(\"abc\")", &h1, "sha256", 32);
        // SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        // base64 standard:   ungWv48Bz+pBQUDeXa4iI7ADYaOWF3qctBD/YfIAFa0=
        assert_eq!(
            h1, "sha256-ungWv48Bz+pBQUDeXa4iI7ADYaOWF3qctBD/YfIAFa0=",
            "SHA-256(\"abc\") base64 changed — hashing or encoding broken",
        );
    }

    /// Validate that an integrity / hash-source string `s` has shape
    /// `<algo>-<base64>` and the base64 body decodes to exactly
    /// `expected_digest_bytes`.
    fn assert_canonical_hash_source(
        label: &str,
        s: &str,
        expected_algo: &str,
        expected_digest_bytes: usize,
    ) {
        use base64::Engine;
        let prefix = format!("{expected_algo}-");
        assert!(
            s.starts_with(&prefix),
            "{label} must start with `{prefix}`, got `{s}`",
        );
        let body = &s[prefix.len()..];
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(body)
            .unwrap_or_else(|e| panic!("{label} base64 body `{body}` does not decode: {e}"));
        assert_eq!(
            decoded.len(),
            expected_digest_bytes,
            "{label} decoded to {} bytes, expected {} ({}-byte digest)",
            decoded.len(),
            expected_digest_bytes,
            expected_algo,
        );
    }

    #[test]
    fn chart_js_sri_has_canonical_sha384_shape() {
        assert_canonical_hash_source("CHART_JS_SRI", CHART_JS_SRI, "sha384", 48);
    }

    #[test]
    fn computed_no_js_remover_hash_has_canonical_sha256_shape() {
        let computed = compute_no_js_remover_hash(NO_JS_REMOVER_BODY);
        assert_canonical_hash_source(
            "compute_no_js_remover_hash(NO_JS_REMOVER_BODY)",
            &computed,
            "sha256",
            32,
        );
    }

    #[test]
    fn rendered_html_has_no_inline_styles() {
        let html = rendered_html();
        assert!(
            !html.contains("<style>"),
            "rendered HTML contains an inline <style> block",
        );
        assert!(
            !html.contains("style=\""),
            "rendered HTML contains an inline `style=\"…\"` attribute",
        );
    }

    #[test]
    fn rendered_html_and_js_have_no_string_eval_patterns() {
        let html = rendered_html();
        let js = rendered_js();
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
                !html.contains(pat.as_str()),
                "rendered HTML contains string-eval pattern {pat:?}",
            );
            assert!(
                !js.contains(pat.as_str()),
                "rendered JS contains string-eval pattern {pat:?}",
            );
        }
    }

    #[test]
    fn rendered_css_uses_no_remote_or_imported_assets() {
        assert!(
            !CHARTS_CSS.contains("@import"),
            "charts_template.css uses @import; under default-src 'none' the import \
             will be blocked, and adding it to the allowlist erodes the strict posture",
        );
        assert!(
            !CHARTS_CSS.contains("url(http"),
            "charts_template.css references a remote URL via url(); strict CSP \
             would block the fetch, and loosening it would regress the same-origin guarantee",
        );
    }

    #[test]
    fn js_get_element_by_id_calls_have_matching_html_ids() {
        let js = rendered_js();
        let html = rendered_html();
        let ids = scan_get_element_by_id_calls(&js);
        assert!(
            !ids.is_empty(),
            "no getElementById calls scanned from charts.js; the scanner is wrong",
        );
        for id in &ids {
            let needle = format!("id=\"{id}\"");
            assert!(
                html.contains(&needle),
                "charts.js calls getElementById({id:?}) but rendered HTML has no `{needle}`",
            );
        }
    }

    /// Hand-rolled scanner (no regex dep). Extracts the literal string
    /// passed to each `getElementById('…')` or `getElementById("…")`
    /// call.
    fn scan_get_element_by_id_calls(js: &str) -> BTreeSet<String> {
        let needle = "getElementById(";
        let mut out = BTreeSet::new();
        let bytes = js.as_bytes();
        let mut i = 0;
        while i + needle.len() <= js.len() {
            let Some(rel) = js[i..].find(needle) else {
                break;
            };
            let start = i + rel + needle.len();
            if start >= js.len() {
                break;
            }
            let quote = bytes[start];
            if quote != b'\'' && quote != b'"' {
                i = start;
                continue;
            }
            let body_start = start + 1;
            let Some(end_rel) = js[body_start..].find(quote as char) else {
                break;
            };
            out.insert(js[body_start..body_start + end_rel].to_string());
            i = body_start + end_rel + 1;
        }
        out
    }

    #[test]
    fn templates_contain_no_liquid_or_front_matter() {
        // Jekyll on GitHub Pages would process any HTML/JS/CSS/MD file
        // that contains Liquid tokens or front matter, mutating its
        // bytes and (for the HTML asset) breaking the SHA-256
        // hash-source for the inline no-js remover. Pin against
        // accidental introduction in any of the bundled assets,
        // including the appended policy_guide.md fragment.
        for (label, body) in [
            ("charts_template.html", CHARTS_HTML),
            ("charts_template.js", CHARTS_JS),
            ("charts_template.css", CHARTS_CSS),
            ("policy_guide.md", POLICY_GUIDE_MD),
        ] {
            for token in ["{{", "{%"] {
                assert!(
                    !body.contains(token),
                    "{label} contains Liquid token {token:?}; Jekyll on GitHub Pages \
                     would rewrite the file and break either the inline-script SHA-256 \
                     or the rendered Markdown",
                );
            }
            assert!(
                !body.starts_with("---"),
                "{label} starts with YAML front matter; Jekyll would run it through a \
                 layout and break the rendered output",
            );
        }
    }
}
