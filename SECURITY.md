# Security Policy

## Supported Versions

| Version | Supported |
|--------|-----------|
| 0.1.x  | ✔️        |

---

## Reporting a Vulnerability

We take security and correctness issues in **cachekit** seriously.  
If you discover a potential vulnerability, soundness issue, or misuse of `unsafe` code, please report it responsibly.

### How to Report

**Please do NOT report security issues via public GitHub issues.**

Instead, report privately via email:

**cachekit.security@gmail.com**  
(or the maintainer contact listed in the repository)

### What to Include

Please include as much of the following as possible:

- Type of issue  
  (e.g. soundness bug, data race, undefined behavior, logic error, DoS via pathological inputs)
- Affected source file(s) and module(s)
- Git tag / branch / commit hash
- Minimal reproduction steps or test case
- Explanation of the impact:
    - incorrect eviction
    - memory safety violation
    - violation of documented invariants
    - performance degradation exploitable by input patterns
- Proof-of-concept code, if available

---

## Response Timeline

- **Acknowledgement**: within 48 hours
- **Initial assessment**: within 7 days
- **Fix or mitigation plan**: as soon as practical, with priority given to:
    1. soundness or memory safety issues
    2. data races or `unsafe` misuse
    3. correctness bugs with security implications

---

## Disclosure Policy

- We will coordinate privately to understand and resolve the issue
- We will keep reporters informed of progress
- We will credit reporters in release notes or advisories (unless anonymity is requested)
- We ask for reasonable time to address the issue before public disclosure

---

## Safe Harbor

Security research conducted under this policy is considered:

- Authorized and welcome
- Performed in good faith
- Beneficial to users of the crate

We will not pursue legal action against researchers who:

- Follow responsible disclosure
- Avoid unnecessary data corruption or denial of service
- Do not exploit issues beyond what is needed to demonstrate impact

---

## Security Model and Scope

### What cachekit *is*

- An **in-process Rust library**
- Provides cache policies (FIFO/LRU/etc.) and optional tiering
- Does **not** perform IO, networking, authentication, or encryption
- Does **not** manage persistence or durability

### What cachekit *does not protect against*

- Attacks on host applications
- Malicious callers with arbitrary code execution
- Logic bugs introduced by incorrect integration
- Misuse outside documented invariants

---

## Security-Relevant Design Considerations

### Memory Safety

- cachekit is written in Rust and relies on Rust’s memory safety guarantees
- `unsafe` code is used **sparingly and intentionally**, primarily for:
    - performance-critical metrics with external synchronization
- All `unsafe` blocks are documented with explicit invariants

### Concurrency

- Thread safety is achieved through **external synchronization**
- Some internal components assume:
    - exclusive access via `Mutex` / `RwLock`
    - no concurrent mutation without higher-level locking
- Violating these assumptions is considered undefined behavior

### Denial of Service Considerations

- Certain cache policies have O(n) paths (e.g. scans for eviction or ranking)
- Adversarial or pathological access patterns may degrade performance
- These are documented and treated as **correctness-preserving but performance-sensitive** behaviors

---

## Guidance for Users

### Safe Integration

- Respect documented locking and concurrency requirements
- Do not share cache instances across threads without synchronization
- Avoid holding locks across unbounded or user-controlled work

### Development Best Practices

- Run `cargo audit` regularly
- Enable `clippy` and `-D warnings`
- Review `unsafe` blocks carefully when modifying internals
- Add regression tests for any discovered correctness or soundness issue

---

## Security Updates

Security-relevant fixes will be communicated via:

- GitHub Security Advisories
- Release notes
- Changelog entries

---

Thank you for helping keep **cachekit** correct, sound, and dependable.
