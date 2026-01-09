# Contributing to cachekit

Thank you for your interest in contributing to cachkit! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to ferrite.db@gmail.com.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cachekit.git
   cd cachekit
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/OxidizeLabs/cachekit.git
   ```
4. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- **Rust toolchain**: Install via [rustup](https://rustup.rs/) (stable channel)
- **Git**: For version control
- **Make** (optional): For using Makefile shortcuts

### Building

```bash
# Debug build
cargo build

# Release build
cargo build --release

# Build with all features
cargo build --all-features
```

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run tests with output
cargo test -- --nocapture

# Run integration tests only
cargo test --test integration_tests
```

### Running Benchmarks

```bash
cargo bench
```

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When filing a bug report, include:

- **Clear title** describing the issue
- **Steps to reproduce** the behavior
- **Expected behavior** vs. actual behavior
- **Environment details** (OS, Rust version, cachekit version)
- **Relevant logs or error messages**
- **Minimal reproduction case** if possible

### Suggesting Features

Feature requests are welcome! Please:

- **Check existing issues** to avoid duplicates
- **Describe the use case** clearly
- **Explain the expected behavior**
- **Consider implementation complexity**

### Contributing Code

1. **Find or create an issue** for the work you want to do
2. **Comment on the issue** to let others know you're working on it
3. **Fork and branch** from `main`
4. **Write your code** following our [coding standards](#coding-standards)
5. **Add tests** for new functionality
6. **Update documentation** as needed
7. **Submit a pull request**

### Good First Issues

Look for issues labeled `good first issue` - these are curated for newcomers and typically have clear scope and guidance.

## Pull Request Process

1. **Ensure your code compiles** without warnings:
   ```bash
   cargo build --all-features
   ```

2. **Run the full test suite**:
   ```bash
   cargo test
   ```

3. **Format your code**:
   ```bash
   cargo fmt
   ```

4. **Run clippy** and address any warnings:
   ```bash
   cargo clippy -- -D warnings
   ```

5. **Update documentation** if you changed public APIs

6. **Write a clear PR description** that:
   - References the related issue(s)
   - Describes what changes were made and why
   - Notes any breaking changes
   - Includes testing instructions if relevant

7. **Request review** from maintainers

8. **Address review feedback** promptly

### PR Title Convention

Use conventional commit format for PR titles:
- `feat: add new B+ tree implementation`
- `fix: resolve deadlock in lock manager`
- `docs: update architecture documentation`
- `test: add integration tests for recovery`
- `refactor: simplify buffer pool logic`
- `perf: optimize page serialization`
- `chore: update dependencies`

## Coding Standards

### Rust Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` for consistent formatting
- Use `cargo clippy` for linting
- Prefer explicit types over `impl Trait` in public APIs

### Naming Conventions

- `snake_case` for functions, variables, modules
- `PascalCase` for types, traits, enums
- `SCREAMING_SNAKE_CASE` for constants
- Use descriptive names: `page_id` not `pid`, `frame_id` not `fid`

### Error Handling

- Define specific error types using `thiserror`
- Implement `From` conversions between error types
- Use `Result<T, DBError>` as the primary return type
- Log errors appropriately: `error!()` for serious issues, `warn!()` for recoverable

### Concurrency

- Use `parking_lot::RwLock` for performance-critical locks
- Prefer `Arc<RwLock<T>>` for shared mutable state
- Document lock ordering to prevent deadlocks
- Use `AtomicU64` for counters and IDs

### Documentation

- Document all public APIs with `///` comments
- Include examples in documentation where helpful
- Explain complex algorithms and invariants
- Keep documentation up to date with code changes

## Testing Guidelines

### Unit Tests

- Write tests for each public function
- Test both success and error paths
- Use descriptive test names: `test_buffer_pool_evicts_unpinned_pages`
- Mock external dependencies when appropriate

### Integration Tests

- Test complete workflows end-to-end
- Test transaction commit/abort scenarios
- Test recovery after simulated crashes
- Test concurrent access patterns

### Property-Based Testing

Consider using `proptest` for testing complex invariants.

### Test Organization

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_happy_path() {
        // Test normal operation
    }

    #[test]
    fn test_feature_error_condition() {
        // Test error handling
    }
}
```

## Documentation

- **Code comments**: Explain "why", not "what"
- **API docs**: Document all public items
- **Architecture docs**: Update `docs/` when making architectural changes
- **README**: Keep getting started instructions current
- **CHANGELOG**: Add entries for notable changes

## Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion

### Recognition

All contributors will be recognized in release notes. Significant contributions may be acknowledged in the README.

## License

By contributing to cachekit, you agree that your contributions will be licensed under the same terms as the project (MIT or Apache-2.0, at your option).

---

Thank you for contributing to cachekit! Your efforts help make database technology more accessible and reliable.
