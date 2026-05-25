# Documentation Style Guide

This style guide covers two related but distinct concerns:

- **Rustdoc style** for module- and item-level documentation inside
  `src/`. The audience is a Rust developer reading the API.
- **Design-doc style** for the prose docs in `docs/design/`. The
  audience is a contributor (or future maintainer) trying to
  understand why a piece of cachekit looks the way it does.

Both styles share the same goals: make behaviour, invariants, and
trade-offs clear without verbosity, and keep examples compile-ready
and focused.

## Rustdoc style

### Goals

- Keep module docs consistent across the codebase.
- Make behavior, invariants, and trade-offs clear without verbosity.
- Ensure examples compile and demonstrate a single, focused use case.

### Module doc layout

Use `//!` and follow this order:

- Architecture
- Key Components
- Core Operations
- Performance Trade-offs
- When to Use
- Example Usage
- Type Constraints
- Thread Safety
- Implementation Notes

### Item docstrings

Use `///` with a one-sentence summary. Mention invariants or complexity
only when they matter. Avoid Args/Returns sections unless behavior is
non-obvious.

### Template

```rust
//! ## Architecture
//! ...
//!
//! ## Key Components
//! ...
//!
//! ## Core Operations
//! ...
//!
//! ## Performance Trade-offs
//! ...
//!
//! ## When to Use
//! ...
//!
//! ## Example Usage
//! ```rust
//! // ...
//! ```
//!
//! ## Type Constraints
//! ...
//!
//! ## Thread Safety
//! ...
//!
//! ## Implementation Notes
//! ...

/// Brief summary of behavior.
```

## Design-doc style

Files in `docs/design/` follow a shared shape so a reader who has
finished one knows what to expect from the next. The shape is not a
strict template — sections are added or omitted as the topic
warrants — but the meta-conventions below are uniform.

### Status preamble

Every design doc opens with a blockquote that names what the doc
covers and links its immediate siblings. The convention is
`> Status: <one-sentence framing>. Companion to <links>.`

```markdown
> Status: design rationale for the concurrent surface that ships today
> behind the `concurrency` feature flag. Companion to the cross-cutting
> principles in [`docs/design/design.md`](design.md) §3 and the trait
> rationale in [`docs/design/trait-hierarchy.md`](trait-hierarchy.md).
```

The preamble does three things in one paragraph:

- Names the doc's scope.
- States the implementation status (shipped, partially shipped, deferred).
- Anchors the doc in the wider design corpus by linking siblings.

### Section structure

- Numbered top-level sections (`§1`, `§2`, …) are encouraged when other
  docs may want to cross-reference specific sections. `design.md`,
  `ttl.md`, and `trait-hierarchy.md` all do this.
- Closer sections, in order, when relevant:
    - **Trade-offs** — explicit tables or side-by-side prose comparing
      alternatives.
    - **Failure modes** — what breaks under stress, panic, contention.
    - **Future directions** / **Roadmap** — what is deferred, in rough
      priority order.
    - **Adding a new X** — checklist for the most common contribution
      pattern (new policy, new capability trait, new metric, etc.).
    - **When not to use X** — explicit boundaries for users.
    - **See also** — links to sibling design docs, source files, and
      external references.

### Tables for trade-offs

When two or more options have different trade-offs, use a table rather
than a bulleted list. Tables make it easy to scan one column for one
property and force the writer to give every option the same set of
properties.

```markdown
| Property | Option A | Option B |
|----------|----------|----------|
| Cost     | …        | …        |
| Memory   | …        | …        |
```

### Diagrams

Use fenced code blocks tagged `text` for ASCII diagrams. Avoid Mermaid
or other rich diagram formats — plain text renders in every tool
(rustdoc, GitHub, terminal `less`) without configuration. See the
hierarchy diagram in `trait-hierarchy.md` for the conventional shape.

```text
            ┌──────────────────┐
            │  Cache<K, V>     │
            └────────┬─────────┘
                     │ extends
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   Capability1   Capability2   Capability3
```

### Source citations

Every concrete claim that names a type, trait, or method should link
the source file. Use relative paths
(`[`src/policy/lru.rs`](../../src/policy/lru.rs)`) so the docs work
both on GitHub and in local clones. When citing a specific feature
gate, name the feature inline (`gated by `#[cfg(feature = "ttl")]`).

### Cross-references

- Refer to sibling design docs by filename, not display title:
  `[concurrency](concurrency.md)` rather than `[Concurrency design](...)`.
  This survives renames better and matches the rest of the corpus.
- When citing a specific section, append the section number or anchor:
  `[design.md §3](design.md)`, `[concurrency.md §"Failure modes"](concurrency.md#failure-modes)`.

### Tone

- Direct, declarative prose. "The wrapper takes a write lock", not
  "The wrapper will take a write lock".
- Trade-offs are stated explicitly, not buried in passive voice.
- Marketing language is out of place. "Excellent", "powerful",
  "blazing fast" — replace with the property that motivated the
  adjective.
- It is acceptable, and often correct, to say "this is a known sharp
  edge" or "this is the wrong trait for that surface" when it is.

### `See Also` closer

Every design doc ends with a `## See Also` section. The conventional
order is:

1. **Sibling design docs** with a one-sentence framing of each link.
2. **Source files** that contain the canonical implementation.
3. **External references** (Rust API Guidelines, research papers,
   Wikipedia entries) when relevant.

The framing matters: a bare list of links is less useful than a list
where each entry says why the reader might follow it.

### Adding a new design doc

Checklist:

1. **Pick a clear single topic.** If you are documenting two concerns,
   split into two docs and link them.
2. **Write the status preamble first.** Naming the scope up front
   keeps the rest of the doc honest.
3. **Number top-level sections** if they're likely to be
   cross-referenced from elsewhere.
4. **Add a `See Also` block** to siblings that should know about the
   new doc, and add a corresponding bullet to
   [`docs/index.md`](../index.md).
5. **Link from `design.md`'s See Also block** so the new doc is
   reachable from the index design overview.
6. **Mirror trade-offs as tables** when there are alternatives.
7. **Close with `When not to use X`** (or the equivalent) — explicit
   boundaries are part of the contract.
