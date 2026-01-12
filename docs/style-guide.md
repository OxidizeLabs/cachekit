# Documentation Style Guide

## Goals
- Keep module docs consistent across the codebase.
- Make behavior, invariants, and trade-offs clear without verbosity.
- Ensure examples compile and demonstrate a single, focused use case.

## Module Doc Layout
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

## Item Docstrings
Use `///` with a one-sentence summary. Mention invariants or complexity only when
they matter. Avoid Args/Returns sections unless behavior is non-obvious.

## Template
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
///
/// Brief summary of behavior.
```
