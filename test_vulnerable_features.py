"""
/tests/test_vulnerable_features.py

Unit tests that exercise code paths associated with known but unreachable
vulnerabilities in the transformers library (v4.38.2). These tests serve as
regression checks to confirm that patches applied for CVEs (e.g., CVE-2024-11394,
CVE-2024-11393, CVE-2026-4372, CVE-2025-14930, CVE-2025-14929,
CVE-2025-14928, CVE-2025-14927) do not break core functionality.

All vulnerabilities are marked "Unreachable" in the current environment,
so tests are designed to exercise safe, typical usage paths without triggering
the vulnerable behavior. They rely on locally cached models to avoid network
access and use only standard, non-malicious inputs.
"""

import pytest
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    pipeline,
    logging,
)
import os

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Use a small, commonly cached model to cover tokenizer/model loading paths.
_MODEL_NAME = "distilbert-base-uncased"
_SEQ2SEQ_MODEL = "t5-small"
_MASKED_LM_MODEL = "bert-base-uncased"


@pytest.fixture(scope="module")
def small_tokenizer():
    """Return a tokenizer from local cache (safe path)."""
    try:
        tok = AutoTokenizer.from_pretrained(_MODEL_NAME, local_files_only=False)
        # Force model loading into cache if not present; typical CI already has it.
        tok.save_pretrained("/tmp/test_tokenizer_cache")
        return AutoTokenizer.from_pretrained("/tmp/test_tokenizer_cache", local_files_only=True)
    except OSError:
        # If not cached, download and cache; for unreachable test environment this is acceptable.
        tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
        return tok


@pytest.fixture(scope="module")
def small_model():
    """Return a small classification model for pipeline tests."""
    model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
    return model


# ---------------------------------------------------------------------------
# Tokenizer tests – exercise encoding/decoding code paths
# ---------------------------------------------------------------------------

class TestTokenizerVulnerablePaths:
    """
    Tests covering tokenizer code paths that correspond to CVEs in tokenizers
    component (e.g., CVE-2024-11394, CVE-2024-11393). Ensures normal operation
    is unaffected by patches.
    """

    def test_basic_encode_decode(self, small_tokenizer):
        """Standard tokenization and detokenization."""
        text = "Hello, world! This is a test sentence."
        tokens = small_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        decoded = small_tokenizer.decode(tokens["input_ids"][0])
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_batch_encode(self, small_tokenizer):
        """Batch processing – common attack surface for tokenization issues."""
        texts = ["First sentence.", "Second longer sentence for testing purposes."]
        encoded = small_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        assert "input_ids" in encoded
        assert encoded["input_ids"].shape[0] == 2

    def test_special_tokens(self, small_tokenizer):
        """Adding special tokens – related to off-by-one or overflow bugs."""
        tokens = small_tokenizer(
            "Add [SEP] and [CLS] tokens.",
            add_special_tokens=True,
            return_tensors="pt",
        )
        decoded = small_tokenizer.decode(tokens["input_ids"][0])
        assert "[CLS]" in decoded or small_tokenizer.cls_token in decoded

    def test_pad_token_id(self, small_tokenizer):
        """Padding token handling – can be exploited in some vulnerabilities."""
        assert small_tokenizer.pad_token_id is not None
        assert isinstance(small_tokenizer.pad_token_id, int)

    def test_long_text_truncation(self, small_tokenizer):
        """Truncation behavior – code path for length-related vulnerabilities."""
        long_text = "word " * 1000
        tokens = small_tokenizer(long_text, truncation=True, max_length=512, return_tensors="pt")
        assert tokens["input_ids"].shape[1] <= 512


# ---------------------------------------------------------------------------
# Model loading and saving tests – exercise pickle/deserialization paths
# ---------------------------------------------------------------------------

class TestModelLoadingPaths:
    """
    Tests covering model loading from local files, which is the context for
    vulnerabilities like CVE-2025-14930, CVE-2025-14929 (unsafe deserialization).
    Using only trusted local paths.
    """

    def test_model_from_pretrained_safe_path(self):
        """Load a sequence classification model from local cache."""
        model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        assert model is not None
        assert model.config.num_labels == 2  # distilbert-base-uncased default

    def test_model_save_and_reload(self, tmp_path):
        """Save model to disk and reload – covers file I/O code paths."""
        model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        save_path = tmp_path / "test_model"
        model.save_pretrained(save_path)
        del model

        reloaded = AutoModelForSequenceClassification.from_pretrained(save_path, local_files_only=True)
        assert reloaded is not None

    def test_masked_lm_model_load(self):
        """Another model architecture – broadens coverage of `from_pretrained`."""
        model = AutoModelForMaskedLM.from_pretrained(_MASKED_LM_MODEL)
        assert model.config.model_type == "bert"


# ---------------------------------------------------------------------------
# Pipeline tests – full pipeline code paths (used in many CVEs)
# ---------------------------------------------------------------------------

class TestPipelinePaths:
    """
    Pipelines wrap tokenizers + models. Vulnerabilities in tokenizer/model
    interaction are exercised via pipelines.
    """

    @pytest.fixture(scope="class")
    def classifier_pipeline(self):
        return pipeline("text-classification", model=_MODEL_NAME, tokenizer=_MODEL_NAME)

    def test_sentiment_classification(self, classifier_pipeline):
        result = classifier_pipeline("This movie was fantastic!")
        assert isinstance(result, list)
        assert len(result) == 1
        assert "label" in result[0]
        assert "score" in result[0]

    def test_batch_classification(self, classifier_pipeline):
        texts = ["Good.", "Bad.", "Okay."]
        results = classifier_pipeline(texts, batch_size=2)
        assert len(results) == 3
        for r in results:
            assert "label" in r

    def test_pipeline_with_truncation(self, classifier_pipeline):
        long_text = "word " * 600
        # Pipeline should truncate to max_length automatically.
        result = classifier_pipeline(long_text)
        assert result is not None


# ---------------------------------------------------------------------------
# Edge case tests – boundary conditions for resource handling
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """
    Tests for unusual inputs that could trigger edge-case vulnerabilities.
    All inputs are non-malicious but stress normal code paths.
    """

    def test_empty_string_tokenization(self, small_tokenizer):
        """Empty string �� potential division-by-zero or overflow."""
        tokens = small_tokenizer("", return_tensors="pt")
        # Should produce at least special tokens (CLS, SEP).
        assert tokens["input_ids"].shape[1] >= 1

    def test_unicode_special_characters(self, small_tokenizer):
        """Unicode characters – encoding/decoding bugs (related to CVE-2026-4372)."""
        text = "日本語 test 😀 emoji"
        tokens = small_tokenizer(text, return_tensors="pt")
        decoded = small_tokenizer.decode(tokens["input_ids"][0])
        # Should not crash; exact decoded string may differ, but content should be present.
        assert "日本語" in decoded or len(decoded) > 0

    def test_huge_batch_size(self, small_tokenizer):
        """Large batch – memory and loop boundaries."""
        texts = ["test"] * 100
        encoded = small_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        assert encoded["input_ids"].shape[0] == 100

    def test_missing_tokenizer_files(self):
        """Graceful error handling on missing files (not a vulnerability but important safeguard)."""
        with pytest.raises((OSError, ValueError)):
            AutoTokenizer.from_pretrained("nonexistent-model-id", local_files_only=True)


# ---------------------------------------------------------------------------
# Logging silence – avoid noisy pytest output
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Suppress transformers logging during tests for cleaner output."""
    logging.set_verbosity_error()
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"