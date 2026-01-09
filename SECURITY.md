# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue in Ferrite, please report it responsibly.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **ferrite.db@gmail.com**

### What to Include

Please include as much of the following information as possible:

- Type of vulnerability (e.g., buffer overflow, SQL injection, privilege escalation)
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: Within 48 hours, we will acknowledge receipt of your report
- **Status Update**: Within 7 days, we will provide an initial assessment
- **Resolution**: We aim to resolve critical vulnerabilities within 30 days

### Disclosure Policy

- We will work with you to understand and resolve the issue quickly
- We will keep you informed of our progress
- We will credit you in the security advisory (unless you prefer to remain anonymous)
- We ask that you give us reasonable time to address the issue before public disclosure

### Safe Harbor

We consider security research conducted in accordance with this policy to be:

- Authorized and lawful
- Helpful to the security of our users
- Conducted in good faith

We will not pursue legal action against researchers who:

- Follow this responsible disclosure policy
- Make a good faith effort to avoid privacy violations and data destruction
- Do not exploit vulnerabilities beyond what is necessary for testing

## Security Best Practices for Users

### Production Deployment

1. **Network Security**: Run Ferrite behind a firewall; do not expose directly to the internet without proper authentication
2. **Authentication**: Enable authentication when available; use strong credentials
3. **Encryption**: Use TLS for network connections when available
4. **Updates**: Keep Ferrite updated to the latest version
5. **Backups**: Regularly backup your data and test recovery procedures
6. **Monitoring**: Monitor logs for suspicious activity

### Development

1. **Dependencies**: Run `cargo audit` regularly to check for vulnerable dependencies
2. **Secrets**: Never commit credentials or secrets to version control
3. **Testing**: Test security-sensitive code paths thoroughly

## Known Security Considerations

As a database system, Ferrite handles sensitive data. Current security considerations:

- **Memory Safety**: Ferrite is written in Rust, providing memory safety guarantees
- **No Unsafe in Critical Paths**: We avoid `unsafe` code in security-critical areas
- **Input Validation**: SQL parsing includes input validation to prevent injection

## Security Updates

Security updates will be announced through:

- GitHub Security Advisories
- Release notes
- The project's security mailing list (when established)

---

Thank you for helping keep Ferrite and its users safe!
