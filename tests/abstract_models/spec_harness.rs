//! Spec-first harness metadata for contributors.
//!
//! Documentation-first module — not runtime enforcement. See the canonical policy index at
//! [`docs/testing/specs/matrix.md`](../../docs/testing/specs/matrix.md).
//!
//! ## Spec-change checklist
//!
//! When a policy spec changes, update in order:
//!
//! 1. Operational spec in `docs/testing/specs/<policy>.md`
//! 2. `reference/<policy>.rs` (if a reference model exists)
//! 3. Cross-model test expectations (if behavior changed)
//! 4. `exact/<policy>.rs` if the exact model was wrong
//! 5. TLA+ module and alignment notes (if applicable)
//! 6. Row in `docs/testing/specs/matrix.md`
//!
//! ## Harness modes
//!
//! | Mode | When to use |
//! |------|-------------|
//! | [`HarnessMode::DualRun`] | Exact, mirror, composed — `PolicyModel` vs impl |
//! | [`HarnessMode::CrossModel`] | Independent `reference/` vs `exact/` (FIFO, LRU, Fast-LRU, LIFO, LFU, MRU, Heap-LFU, MFU) |
//! | [`HarnessMode::InvariantOnly`] | Bounded — structural invariants only |
//!
//! A policy may use multiple modes (e.g. FIFO: DualRun + CrossModel).
//!
//! ## Cross-model availability
//!
//! `cross_model_available` is true for all exact-tier policies with a `reference/` model. Use
//! [`driver::assert_models_agree`] or [`driver::assert_models_agree_with_recency`] when
//! [`driver::ModelRecencyRank`] is implemented (LRU family).
//!
//! ## Related
//!
//! - [template.md](../../docs/testing/specs/template.md) — new policy spec skeleton
//! - [tla-guide.md](../../docs/testing/specs/tla-guide.md) — optional TLA+ specs

/// Root path for operational specs (grep-friendly).
pub const SPEC_ROOT: &str = "docs/testing/specs";

/// Model tier in the harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecTier {
    /// Deterministic victim and residency (LFU, LIFO, LRU, …).
    Exact,
    /// DS-shaped oracle transcribed from implementation (Clock, 2Q, SLRU, NRU).
    Mirror,
    /// Adaptive victim — invariant-only tests today (ARC, CAR, Clock-PRO, S3-FIFO).
    Bounded,
    /// Decorator over inner policy (TTL over LRU).
    Composed,
}

/// How the harness validates a policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarnessMode {
    /// `exact/` `PolicyModel` dual-run vs implementation.
    DualRun,
    /// `reference/` vs `exact/` agreement (FIFO, LRU, Fast-LRU, LIFO, LFU, MRU, Heap-LFU, MFU).
    CrossModel,
    /// `len <= capacity` + invariant checks (bounded tier).
    InvariantOnly,
}

/// Whether an independent `reference/` model exists for cross-model tests.
pub fn cross_model_available(policy: &str) -> bool {
    policy == "fifo"
        || policy == "lru"
        || policy == "fast-lru"
        || policy == "lifo"
        || policy == "lfu"
        || policy == "mru"
        || policy == "heap-lfu"
        || policy == "mfu"
        || policy == "lru-k"
}

/// Per-policy harness metadata (documentation-oriented).
#[derive(Debug, Clone, Copy)]
pub struct PolicyHarnessMeta {
    pub name: &'static str,
    pub tier: SpecTier,
    pub modes: &'static [HarnessMode],
    pub spec_doc: &'static str,
    pub notes: &'static str,
}
