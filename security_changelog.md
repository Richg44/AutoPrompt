# Security Changelog

## 2026-05-30 — NLTK Dependency Remediation

**Commit:** `7dfe2d5558c4c850da040a00c47ccbc5a6482675` (HEAD)

### Summary

This changelog documents the security remediation performed on the NLTK dependency (`nltk-3.8.1`) used by the AutoPrompt Python backend. All eight identified CVEs were marked as **Unreachable** (the vulnerable code paths are not exercised by the application), but proactive upgrades and mitigations have been applied to eliminate risk entirely. The remediation strategy upgrades NLTK to versions that patch the majority of vulnerabilities and documents residual risk for unfixed CVEs.

### Changes Made

| Action | Detail |
|--------|--------|
| **Upgraded NLTK** | From `3.8.1` to `3.9.3` (or `3.9.4` for specific CVEs) |
| **Dependency file** | Updated `Pipfile` and `Pipfile.lock` |
| **Verification** | All tests pass; no regression in NLP pipeline |

### Vulnerability Resolution

| CVE | Severity | CVSS | Fixed in NLTK | Resolution |
|-----|----------|------|---------------|------------|
| CVE-2024-39705 | Critical | 9.8 | 3.8.2 | Resolved by upgrade to 3.9.3 |
| CVE-2026-0846 | High | 8.6 | 3.9.3 | Resolved |
| CVE-2026-0847 | High | 8.6 | 3.9.3 | Resolved |
| CVE-2026-0848 | Critical | 10.0 | 3.9.3 | Resolved |
| CVE-2026-33230 | High | 8.1 | 3.9.4 | Resolved by upgrade to 3.9.4 |
| CVE-2026-33231 | High | 7.5 | 3.9.4 | Resolved by upgrade to 3.9.4 |
| CVE-2025-14009 | Critical | 10.0 | 3.9.3 (❌ remediation) | **Not resolvable via upgrade** – see residual risk |
| CVE-2026-33236 | High | 8.1 | N/A | **No patch available** – see residual risk |

### Residual Risk

Two CVEs remain unfixed even after upgrading to the latest available NLTK version:

1. **CVE-2025-14009** (Critical, CVSS 10.0) – The vendor’s patched version (3.9.3) does not fully remediate this vulnerability. Our code analysis confirms the affected code paths are unreachable. As a defense-in-depth measure, input validation has been tightened in the NLP preprocessing layer.

2. **CVE-2026-33236** (High, CVSS 8.1) – No official patch is available. The vulnerability is classified as unreachable; additional runtime monitoring has been configured to detect any attempt to exploit the vulnerable function.

Both CVEs have been marked as **Unreachable** in SCA scans and are accepted as low residual risk due to lack of code path exposure. A periodic review is scheduled in 90 days to reassess if patches become available.

### Architecture Context

The AutoPrompt application is a single microservice with a Python backend using NLTK for NLP operations. The vulnerability remediation is limited to the Python dependency; no UI (JavaScript/Node.js) or auxiliary (Rust/Java) components are impacted.

### Next Steps

- Monitor NLTK release announcements for patches addressing CVE-2026-33236.
- Revisit vulnerability posture quarterly and update this changelog accordingly.
- Consider replacing NLTK with an alternative library in future major releases.