#!/usr/bin/env python3
"""
apply_patches.py - Apply all patches from /patches/ to the installed transformers library.

This script is designed for use in Dockerfile or CI build steps to apply
custom vulnerability patches to the installed transformers package.
It locates the transformers site-packages directory and applies each
.patch file using the Unix `patch` command (with -p1 for typical diffs).

Exit codes:
    0 - All patches applied successfully
    1 - No patches found or script execution error
    2 - One or more patch applications failed
"""

import logging
import subprocess
import sys
from pathlib import Path

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
PATCHES_DIR = Path("/patches")
PATCH_LEVEL = 1  # Typical for patches generated from git diff with prefix a/b
LOGGING_FORMAT = "[%(asctime)s] %(levelname)s - %(message)s"

# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger("apply_patches")


def _get_transformers_location() -> Path | None:
    """Return the directory containing the transformers package source files."""
    try:
        import transformers
    except ImportError:
        logger.error("transformers is not installed in the current environment.")
        return None

    # __file__ points to e.g. .../site-packages/transformers/__init__.py
    transformers_init = Path(transformers.__file__).resolve()
    transformers_dir = transformers_init.parent
    if not transformers_dir.is_dir():
        logger.error("Could not locate transformers directory at %s", transformers_dir)
        return None
    logger.info("Found transformers installation at: %s", transformers_dir)
    return transformers_dir


def _find_patches() -> list[Path]:
    """Return sorted list of .patch files in PATCHES_DIR."""
    if not PATCHES_DIR.is_dir():
        logger.warning("Patches directory %s does not exist. No patches to apply.", PATCHES_DIR)
        return []
    patches = sorted(PATCHES_DIR.glob("*.patch"))
    logger.info("Found %d patch file(s) in %s", len(patches), PATCHES_DIR)
    return patches


def _apply_patch(patch_file: Path, target_dir: Path) -> bool:
    """Apply a single .patch file to the target directory using the `patch` command."""
    logger.info("Applying patch: %s", patch_file.name)
    try:
        result = subprocess.run(
            ["patch", f"-p{PATCH_LEVEL}", "-i", str(patch_file)],
            cwd=str(target_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        logger.error("Patch %s timed out after 120 seconds.", patch_file.name)
        return False
    except FileNotFoundError:
        logger.error("'patch' command not found. Ensure the 'patch' utility is installed.")
        return False
    except Exception as exc:
        logger.error("Unexpected error while applying patch %s: %s", patch_file.name, exc)
        return False

    if result.returncode != 0:
        logger.error(
            "Patch %s failed with exit code %d.\nSTDERR:\n%s\nSTDOUT:\n%s",
            patch_file.name,
            result.returncode,
            result.stderr,
            result.stdout,
        )
        return False

    logger.info("Patch %s applied successfully.", patch_file.name)
    return True


def _log_patch_summary(success_count: int, fail_count: int, total: int) -> None:
    """Log the final status of patch application."""
    results = f"Applied {success_count}/{total} patches"
    if fail_count > 0:
        logger.error("%s with %d failure(s).", results, fail_count)
    else:
        logger.info("%s — all patches applied cleanly.", results)


def main() -> int:
    """Main entry point for patch application."""
    logger.info("Starting patch application process...")

    # 1. Locate transformers installation
    transformers_dir = _get_transformers_location()
    if not transformers_dir:
        return 1

    # 2. Find all patches
    patches = _find_patches()
    if not patches:
        logger.info("No patches to apply. Exiting successfully.")
        return 0

    # 3. Apply each patch
    total = len(patches)
    success_count = 0
    fail_count = 0

    for patch_file in patches:
        if _apply_patch(patch_file, transformers_dir):
            success_count += 1
        else:
            fail_count += 1

    # 4. Log summary and return appropriate exit code
    _log_patch_summary(success_count, fail_count, total)
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())