#!/usr/bin/env python3
"""
safety_patch.py — Runtime guards and monkey-patches for NLTK vulnerabilities
lacking an official fix (CVE-2026-33236, CVE-2025-14009).

All CVEs affecting NLTK 3.8.1 are marked as "Unreachable", but we apply
defence-in-depth for the two CVEs without available remediation.

* CVE-2025-14009 (Critical 10.0, no fix) – input sanitisation for tokenization.
* CVE-2026-33236 (High 8.1, no fix) – restrict pickle loading and corpus paths.

Usage:
    import auto_prompt.nlp.safety_patch
    safety_patch.apply()

Or simply import this module (apply() is called on import).
"""

from __future__ import annotations

import functools
import logging
import os
import re
import sys
import threading
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, Pattern, Set, Tuple, Union

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (overridable via environment variables)
# ---------------------------------------------------------------------------
_MAX_INPUT_LENGTH: int = int(os.environ.get("NLTK_SAFETY_MAX_INPUT_LENGTH", "10000"))
_ENABLE_CORPUS_PATH_VALIDATION: bool = os.environ.get(
    "NLTK_SAFETY_ENABLE_PATH_VALIDATION", "1"
) in ("1", "true", "yes")
_ENABLE_TEXT_SANITIZATION: bool = os.environ.get(
    "NLTK_SAFETY_ENABLE_TEXT_SANITIZATION", "1"
) in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Compiled regex patterns – constants for performance
# ---------------------------------------------------------------------------
_DANGEROUS_PATTERNS: Tuple[Pattern[str], ...] = (
    re.compile(r"__import__\s*\("),
    re.compile(r"\beval\s*\("),
    re.compile(r"\bexec\s*\("),
    re.compile(r"__reduce__"),
    re.compile(r"os\s*\.\s*system\s*\("),
    re.compile(r"subprocess\."),
    re.compile(r"shutil\."),
    re.compile(r"__builtins__"),
    re.compile(r"pickle\s*\.\s*load"),
    re.compile(r"dill\s*\.\s*load"),
    re.compile(r"cloudpickle\s*\.\s*load"),
)

# Control characters to strip (except tab, newline, carriage return)
_CONTROL_CHAR_PATTERN: Pattern[str] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# ---------------------------------------------------------------------------
# Thread-safe whitelist of allowed corpus directories
# ---------------------------------------------------------------------------
_allowed_corpus_dirs: Set[str] = set()
_allowed_corpus_dirs_lock = threading.Lock()


def _initialize_allowed_corpus_dirs() -> None:
    """Populate the whitelist of NLTK data directories from configuration.

    This is called once on module import. Subsequent calls refresh the set.
    Thread-safe due to external lock.
    """
    global _allowed_corpus_dirs
    try:
        import nltk.data

        paths: list[str] = nltk.data.path
        normalized = {os.path.normcase(os.path.abspath(p)) for p in paths}
        with _allowed_corpus_dirs_lock:
            _allowed_corpus_dirs = normalized
        LOGGER.debug("Allowed corpus directories: %s", _allowed_corpus_dirs)
    except (ImportError, AttributeError) as exc:
        with _allowed_corpus_dirs_lock:
            _allowed_corpus_dirs = set()
        LOGGER.warning(
            "Could not obtain NLTK data paths (%s); corpus path validation will reject all paths.",
            exc,
        )


_initialize_allowed_corpus_dirs()

# ---------------------------------------------------------------------------
# Input sanitisation helpers
# ---------------------------------------------------------------------------
def sanitise_text(text: str, context: str = "unknown") -> str:
    """Remove or reject dangerous content from a text string.

    Args:
        text: Input string to sanitise.
        context: Description of where the input originated (for logging).

    Returns:
        Sanitised string after removing control characters.

    Raises:
        TypeError: If ``text`` is not a string.
        ValueError: If the input is too long or contains malicious patterns.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected string, got {type(text).__name__}")

    if not _ENABLE_TEXT_SANITIZATION:
        return text

    if len(text) > _MAX_INPUT_LENGTH:
        raise ValueError(
            f"Input too long ({len(text)} chars) in {context} "
            f"(max {_MAX_INPUT_LENGTH})"
        )

    # Strip dangerous control characters
    stripped: str = _CONTROL_CHAR_PATTERN.sub("", text)
    if stripped != text:
        LOGGER.warning("Control characters stripped from input in %s", context)

    # Check for known dangerous patterns
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(stripped):
            LOGGER.warning(
                "Potentially dangerous pattern '%s' detected in %s",
                pattern.pattern,
                context,
            )
            raise ValueError(
                f"Potentially dangerous pattern '{pattern.pattern}' detected in {context}"
            )

    return stripped


def sanitise_corpus_path(path: str, context: str = "corpus_loading") -> str:
    """Validate that a corpus path is within allowed NLTK data directories.

    Args:
        path: The file path or resource name to validate.
        context: Description for logging.

    Returns:
        The normalized absolute path if allowed, or the original resource name
        if it contains no path traversal and is not absolute.

    Raises:
        TypeError: If ``path`` is not a string.
        ValueError: If the path is outside allowed directories or contains traversal.
    """
    if not isinstance(path, str):
        raise TypeError(f"Expected string path, got {type(path).__name__}")

    if not _ENABLE_CORPUS_PATH_VALIDATION:
        return path

    # Reject obvious path traversal
    if ".." in path or path.startswith(("/", "\\")):
        raise ValueError(f"Path traversal attempt detected in {context}: {path}")

    # Resolve to absolute and normalize
    abs_path: str = os.path.normcase(os.path.abspath(path))

    # Check against whitelisted directories
    with _allowed_corpus_dirs_lock:
        allowed_copy = _allowed_corpus_dirs.copy()

    for allowed_dir in allowed_copy:
        if abs_path.startswith(allowed_dir):
            LOGGER.debug("Corpus path allowed: %s", abs_path)
            return abs_path

    # Accept relative resource names without traversal
    if not os.path.isabs(path) and ".." not in path:
        LOGGER.debug("Resource name accepted: %s", path)
        return path

    raise ValueError(
        f"Corpus path not allowed (outside NLTK data dirs): {abs_path}"
    )


def refresh_allowed_corpus_dirs() -> None:
    """Refresh the whitelist of allowed NLTK data directories.

    Call this if NLTK's data path changes at runtime.
    """
    _initialize_allowed_corpus_dirs()


# ---------------------------------------------------------------------------
# Monkey‑patch factory
# ---------------------------------------------------------------------------
def _wrap_function(
    original: Callable[..., Any],
    context: str,
    *,
    args_to_sanitise: Optional[Tuple[int, ...]] = None,
    kwargs_to_sanitise: Optional[Tuple[str, ...]] = None,
    args_to_path_validate: Optional[Tuple[int, ...]] = None,
    kwargs_to_path_validate: Optional[Tuple[str, ...]] = None,
) -> Callable[..., Any]:
    """Return a wrapped version that sanitises selected arguments before
    calling the original function.

    Args:
        original: The function to wrap.
        context: Description for logging/error messages.
        args_to_sanitise: Indices of positional arguments to sanitise via ``sanitise_text``.
        kwargs_to_sanitise: Names of keyword arguments to sanitise via ``sanitise_text``.
        args_to_path_validate: Indices of positional arguments to validate as corpus paths.
        kwargs_to_path_validate: Names of keyword arguments to validate as corpus paths.

    Returns:
        Wrapped function with the same signature.
    """
    if args_to_sanitise is None:
        args_to_sanitise = ()
    if kwargs_to_sanitise is None:
        kwargs_to_sanitise = ()
    if args_to_path_validate is None:
        args_to_path_validate = ()
    if kwargs_to_path_validate is None:
        kwargs_to_path_validate = ()

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Sanitise specified arguments before delegating to the original."""
        # Convert positional args to list for mutation
        args_list = list(args)

        # Sanitise positional args (text)
        for idx in args_to_sanitise:
            if idx < len(args_list):
                args_list[idx] = sanitise_text(args_list[idx], context)

        # Path-validate positional args
        for idx in args_to_path_validate:
            if idx < len(args_list):
                args_list[idx] = sanitise_corpus_path(args_list[idx], context)

        # Sanitise keyword args (text)
        for kw in kwargs_to_sanitise:
            if kw in kwargs:
                kwargs[kw] = sanitise_text(kwargs[kw], context)

        # Path-validate keyword args
        for kw in kwargs_to_path_validate:
            if kw in kwargs:
                kwargs[kw] = sanitise_corpus_path(kwargs[kw], context)

        try:
            return original(*args_list, **kwargs)
        except Exception as exc:
            LOGGER.error(
                "Error in %s after sanitisation: %s",
                context,
                exc,
                exc_info=True,
            )
            raise

    return wrapper


# ---------------------------------------------------------------------------
# NLTK version check / upgrade warning
# ---------------------------------------------------------------------------
_MIN_SAFE_VERSION = (3, 9, 3)


def _check_nltk_version() -> None:
    """Issue a warning if NLTK version is older than the minimum safe version."""
    try:
        import nltk

        version_str = nltk.__version__
        parts = tuple(int(x) for x in version_str.split("."))
        if parts < _MIN_SAFE_VERSION:
            warnings.warn(
                f"NLTK {version_str} is vulnerable. Upgrade to "
                f"{'.'.join(str(v) for v in _MIN_SAFE_VERSION)} or later.",
                stacklevel=2,
            )
    except (ImportError, AttributeError, ValueError):
        pass


_check_nltk_version()

# ---------------------------------------------------------------------------
# Apply patches to vulnerable NLTK functions
# ---------------------------------------------------------------------------
_patched: bool = False
_patch_lock = threading.Lock()


def apply() -> None:
    """Apply all safety patches to NLTK functions.

    This function is idempotent and thread-safe.
    """
    global _patched
    if _patched:
        LOGGER.debug("Safety patches already applied.")
        return

    with _patch_lock:
        if _patched:
            return

        try:
            _apply_internal()
            _patched = True
            LOGGER.info("NLTK safety patches applied successfully.")
        except Exception as exc:
            LOGGER.critical("Failed to apply safety patches: %s", exc, exc_info=True)
            raise


def _apply_internal() -> None:
    """Internal patching logic (mutates NLTK modules).

    Each patch target is listed with the arguments to sanitise.
    """
    import nltk.tokenize
    import nltk.corpus
    import nltk.data
    import nltk.classify
    import nltk.featstruct
    import nltk.probability
    import nltk.chunk

    # Tokenization: sanitise text input (CVE-2025-14009)
    nltk.tokenize.word_tokenize = _wrap_function(
        nltk.tokenize.word_tokenize,
        "word_tokenize",
        args_to_sanitise=(0,),
        kwargs_to_sanitise=("text",),
    )
    nltk.tokenize.sent_tokenize = _wrap_function(
        nltk.tokenize.sent_tokenize,
        "sent_tokenize",
        args_to_sanitise=(0,),
        kwargs_to_sanitise=("text",),
    )
    nltk.tokenize.casual_tokenize = _wrap_function(
        nltk.tokenize.casual_tokenize,
        "casual_tokenize",
        args_to_sanitise=(0,),
        kwargs_to_sanitise=("text",),
    )

    # Corpus loading: validate file paths (CVE-2026-33236)
    nltk.corpus.reader.plaintext.PlaintextCorpusReader.__init__ = _wrap_function(
        nltk.corpus.reader.plaintext.PlaintextCorpusReader.__init__,
        "PlaintextCorpusReader.__init__",
        args_to_path_validate=(1,),  # root parameter
    )

    # nltk.data.load: path validation for pickle/files (CVE-2026-33236)
    nltk.data.load = _wrap_function(
        nltk.data.load,
        "nltk.data.load",
        args_to_path_validate=(0,),
        kwargs_to_path_validate=("url",),
    )

    # nltk.data.find: path validation
    nltk.data.find = _wrap_function(
        nltk.data.find,
        "nltk.data.find",
        args_to_path_validate=(0,),
    )

    # Classifier loading: protect pickle deserialisation
    nltk.classify.maxent.MaxentClassifier.load = _wrap_function(
        nltk.classify.maxent.MaxentClassifier.load,
        "MaxentClassifier.load",
        args_to_path_validate=(0,),
    )

    # ProbDist: avoid unsafe pickle in some probability distributions
    nltk.probability.LidstoneProbDist.__reduce__ = _wrap_function(
        nltk.probability.LidstoneProbDist.__reduce__,
        "LidstoneProbDist.__reduce__",
    )

    # Feature structure: guard against unsafe recursion
    nltk.featstruct.FeatStruct.__init__ = _wrap_function(
        nltk.featstruct.FeatStruct.__init__,
        "FeatStruct.__init__",
        args_to_sanitise=(0,),
    )

    LOGGER.debug("All patch functions installed.")


# Auto-apply on import
apply()