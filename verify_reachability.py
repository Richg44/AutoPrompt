#!/usr/bin/env python3
"""
verify_reachability.py

Run Mend CLI reachability analysis after patching to confirm all known
vulnerabilities are either fixed (by version upgrade) or remain unreachable
(documented as acceptable risk).

Usage:
    python verify_reachability.py [--json]

Requirements:
    - Mend CLI (whitesource) binary in PATH or configured via MEND_CLI_PATH env var.
    - transformers >= 4.48.0 (patched) installed in the current environment.
    - git + patch utilities available for any future monkey‑patch workflows.
"""

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and known vulnerability data
# ---------------------------------------------------------------------------
# Known vulnerabilities from transformers 4.38.2 (extracted from spec).
# Each entry: (CVE_ID, severity, fixed_version_str, remediation_possible)
KNOWN_VULNERABILITIES: List[Tuple[str, str, Optional[str], bool]] = [
    ("CVE-2024-11394", "High", "4.48.0", True),
    ("CVE-2024-11393", "High", "4.48.0", True),
    ("CVE-2026-4372",  "High", "5.3.0",  True),
    ("CVE-2025-14930", "High", None,     False),
    ("CVE-2025-14929", "High", None,     False),
    ("CVE-2025-14928", "High", None,     False),
    ("CVE-2025-14927", "High", None,     False),
]

# Minimum patched version for transformers (covers all fixable CVEs)
MIN_TRANSFORMERS_VERSION = (4, 48, 0)

# Mend CLI binary lookup (env var override)
MEND_CLI_EXECUTABLE = os.environ.get(
    "MEND_CLI_PATH",
    "mend"  # falls back to PATH
)

# ---------------------------------------------------------------------------
# Utility: parse version strings
# ---------------------------------------------------------------------------
def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse a dot-separated version string into a tuple of integers."""
    try:
        return tuple(int(part) for part in version_str.split("."))
    except (ValueError, AttributeError):
        raise ValueError(f"Cannot parse version string: {version_str}")

# ---------------------------------------------------------------------------
# Check installed transformers version
# ---------------------------------------------------------------------------
def get_installed_transformers_version() -> Optional[str]:
    """Return the installed transformers version or None if not found."""
    try:
        import importlib.metadata as importlib_metadata
        return importlib_metadata.version("transformers")
    except importlib_metadata.PackageNotFoundError:
        logger.warning("transformers package not found.")
        return None

def check_transformers_version() -> bool:
    """Verify that transformers version >= MIN_TRANSFORMERS_VERSION."""
    version_str = get_installed_transformers_version()
    if version_str is None:
        logger.error("transformers is not installed in the current environment.")
        return False

    installed = parse_version(version_str)
    required = MIN_TRANSFORMERS_VERSION
    if installed < required:
        logger.error(
            f"transformers version is {version_str}, "
            f"requires at least {'.'.join(map(str, required))}. "
            "Upgrade before running reachability verification."
        )
        return False
    logger.info(f"transformers version {version_str} meets minimum requirement.")
    return True

# ---------------------------------------------------------------------------
# Run Mend CLI
# ---------------------------------------------------------------------------
def run_mend_cli(project_dir: Optional[Path] = None) -> Optional[str]:
    """
    Execute Mend CLI reachability scan and return raw output (JSON expected).
    Returns None if the command fails or output cannot be captured.
    """
    if project_dir is None:
        project_dir = Path.cwd()

    cmd = [
        MEND_CLI_EXECUTABLE,
        "scan",
        "--no-color",
        "-f", "json",  # request JSON output
        "--reachability", "true",
        str(project_dir),
    ]

    logger.info(f"Running Mend CLI: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes should be enough
            check=False,  # Do not raise on non-zero exit
        )
    except FileNotFoundError:
        logger.error(f"Mend CLI not found: {MEND_CLI_EXECUTABLE}. "
                      "Install Mend CLI or set MEND_CLI_PATH environment variable.")
        return None
    except subprocess.TimeoutExpired:
        logger.error("Mend CLI scan timed out after 600 seconds.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error running Mend CLI: {e}")
        return None

    if result.returncode != 0:
        logger.error(
            f"Mend CLI exited with code {result.returncode}:\n"
            f"stdout: {result.stdout[:2000]}\n"
            f"stderr: {result.stderr[:2000]}"
        )
        return None

    return result.stdout

# ---------------------------------------------------------------------------
# Parse Mend CLI output (assumed JSON)
# ---------------------------------------------------------------------------
def parse_mend_output(raw_output: str) -> Optional[List[Dict[str, Any]]]:
    """
    Try to parse Mend CLI JSON output containing vulnerability list.
    Returns list of vulnerability dicts, or None on failure.
    """
    # Mend CLI typically returns a JSON object with a "vulnerabilities" key
    # or an array directly. Try both.
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Mend CLI JSON output: {e}")
        return None

    # If parsed is a dict, try "vulnerabilities" or "alerts" key
    if isinstance(parsed, dict):
        for key in ("vulnerabilities", "alerts", "results"):
            if key in parsed:
                return parsed[key]
        # Otherwise wrap the whole dict as a single item (unlikely)
        logger.warning("Unexpected Mend CLI JSON structure – treating as single record.")
        return [parsed]

    # If parsed is a list, assume it's the vulnerability list
    if isinstance(parsed, list):
        return parsed

    logger.error("Mend CLI output is neither a list nor an object.")
    return None

# ---------------------------------------------------------------------------
# Verification logic
# ---------------------------------------------------------------------------
def verify_vulnerabilities(
    mend_vulns: List[Dict[str, Any]],
    known: List[Tuple[str, str, Optional[str], bool]],
) -> Tuple[bool, List[str], List[str]]:
    """
    Compare Mend results against expected known vulnerabilities.

    Returns:
        (all_ok, errors, warnings)
    - all_ok: True if every known vulnerability is either fixed or unreachable.
    - errors: list of violation messages.
    - warnings: list of informational/advisory messages.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Map known CVEs by ID for quick lookup
    known_map: Dict[str, Tuple[str, Optional[str], bool]] = {
        cve: (severity, fixed_ver, fixable)
        for cve, severity, fixed_ver, fixable in known
    }

    # Map Mend results by CVE (assuming "CVE-XXXX" in vulnerabilityId field)
    mend_map: Dict[str, Dict[str, Any]] = {}
    for vuln in mend_vulns:
        vuln_id = vuln.get("vulnerabilityId") or vuln.get("id") or vuln.get("name")
        if vuln_id:
            # Extract CVE ID if present (some vendors embed in string)
            cve_match = re.search(r"CVE-\d{4}-\d+", str(vuln_id))
            if cve_match:
                cve = cve_match.group(0)
                mend_map[cve] = vuln

    # Process each known vulnerability
    for cve, severity, fixed_ver, fixable in known:
        if cve not in mend_map:
            # Mend might not report fixed CVEs at all – that's okay if fixable
            if fixable and fixed_ver:
                warnings.append(
                    f"{cve}: Not found in Mend results (assumed fixed by upgrade)."
                )
            else:
                # Non-fixable CVEs *must* appear – if missing, we can't confirm unreachable
                errors.append(
                    f"{cve}: Not found in Mend results, but has no fix version. "
                    "Unable to confirm unreachable status."
                )
            continue

        mend_entry = mend_map[cve]
        reachability = (mend_entry.get("reachability") or "").lower()
        fixed_version_mend = mend_entry.get("fixedVersion") or mend_entry.get("fixVersion")
        severity_mend = mend_entry.get("severity", "")

        # Determine if Mend considers it reachable
        is_reachable = "reachable" in reachability

        # Check fix status
        installed_ver = get_installed_transformers_version()
        if fixable and fixed_ver:
            # Expect fixed (version >= fixed_ver) OR unreachable
            if installed_ver and parse_version(installed_ver) >= parse_version(fixed_ver):
                warnings.append(
                    f"{cve}: Fixed by upgrade to transformers {installed_ver} "
                    f"(fix requires >= {fixed_ver})."
                )
            elif not is_reachable and reachability:
                warnings.append(
                    f"{cve}: Unreachable (reachability='{reachability}'). Documented as acceptable."
                )
            else:
                errors.append(
                    f"{cve}: Remediation is possible (fix version {fixed_ver}) but "
                    f"current transformers version is {installed_ver or 'unknown'} "
                    f"and vulnerability is reachable (reachability='{reachability}'). "
                    "Either upgrade or apply mitigation to make unreachable."
                )
        else:
            # No fix available – must be unreachable
            if not is_reachable:
                warnings.append(
                    f"{cve}: Unreachable (as expected – no fix available)."
                )
            else:
                errors.append(
                    f"{cve}: No fix available, but Mend reports it as reachable "
                    f"(reachability='{reachability}'). Security risk – must mitigate."
                )

    return len(errors) == 0, errors, warnings

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify vulnerability reachability after patching using Mend CLI."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for CI consumption).",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Project directory to scan (default: current working directory).",
    )
    args = parser.parse_args()

    # 1. Check transformers version
    if not check_transformers_version():
        logger.error("Transformers version check failed.")
        return 1

    # 2. Run Mend CLI scan
    raw_output = run_mend_cli(project_dir=args.project_dir)
    if raw_output is None:
        logger.error("Mend CLI scan failed. Aborting.")
        return 1

    # 3. Parse Mend output
    mend_vulns = parse_mend_output(raw_output)
    if mend_vulns is None:
        logger.error("Failed to parse Mend output. Aborting.")
        return 1

    logger.info(f"Parsed {len(mend_vulns)} vulnerability records from Mend CLI.")

    # 4. Verify against known list
    all_ok, errors, warnings = verify_vulnerabilities(
        mend_vulns, KNOWN_VULNERABILITIES
    )

    # 5. Build result object
    result = {
        "status": "pass" if all_ok else "fail",
        "errors": errors,
        "warnings": warnings,
        "installed_transformers": get_installed_transformers_version(),
        "vulnerabilities_checked": len(KNOWN_VULNERABILITIES),
        "mend_records_found": len(mend_vulns),
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if all_ok:
            logger.info("✅ All known vulnerabilities are fixed or unreachable.")
        else:
            logger.error("❌ Some vulnerabilities require attention:")
            for err in errors:
                logger.error(f"  - {err}")
        for w in warnings:
            logger.info(f"ℹ️  {w}")

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())