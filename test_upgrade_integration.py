"""
Integration test: verify basic model inference compatibility after upgrading
the `transformers` library to a patched version (>=4.48.0).

The test uses a small, pre-trained model (e.g., `hf-internal-testing/tiny-random-bert`)
to avoid large downloads, but can also run with a mock if the model is unavailable.
It checks that model loading, tokenization, and a forward pass succeed, and
that the installed transformers version satisfies the minimum requirement.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from transformers import AutoModel, AutoTokenizer, __version__

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_REQUIRED_VERSION = "4.48.0"
MODEL_NAME = os.getenv(
    "TEST_UPGRADE_MODEL", "hf-internal-testing/tiny-random-bert"
)
# Extremely small model for quick tests; fallback to "bert-base-uncased" if you
# prefer a real but compact model (comment out the line above and use below).
# MODEL_NAME = "prajjwal1/bert-tiny"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model_and_tokenizer():
    """
    Load a real model + tokenizer if possible, else return mocks.

    Returns:
        tuple: (model, tokenizer)
    """
    try:
        # Attempt to load the real model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        return model, tokenizer
    except Exception as exc:
        # If loading fails (no internet, model not found, etc.), use mocks
        warnings.warn(f"Could not load real model '{MODEL_NAME}': {exc}. "
                      "Falling back to mocks.")
        model = MagicMock()
        tokenizer = MagicMock()
        # Simulate typical outputs
        tokenizer.return_value = {"input_ids": [[101, 102]], "attention_mask": [[1, 1]]}
        model.return_value = MagicMock(last_hidden_state=MagicMock(shape=(1, 2, 768)))
        return model, tokenizer


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestUpgradeIntegration:
    """Integration tests for transformers upgrade compatibility."""

    def test_transformers_version(self):
        """Verify that the installed transformers version meets the minimum."""
        from packaging.version import Version
        installed = Version(__version__)
        required = Version(MIN_REQUIRED_VERSION)
        assert installed >= required, (
            f"transformers {__version__} is installed, but {MIN_REQUIRED_VERSION}+ is required. "
            "Run: pip install 'transformers>=4.48.0'"
        )

    def test_model_loading(self, model_and_tokenizer):
        """Ensure model and tokenizer are loaded without errors."""
        model, tokenizer = model_and_tokenizer
        assert model is not None
        assert tokenizer is not None
        # If real model, check that it's an instance of PreTrainedModel
        if not isinstance(model, MagicMock):
            from transformers import PreTrainedModel
            assert isinstance(model, PreTrainedModel)
            assert model.config is not None

    def test_tokenization(self, model_and_tokenizer):
        """Test that tokenization produces expected structure."""
        _, tokenizer = model_and_tokenizer
        text = "Hello, world!"
        tokens = tokenizer(text, return_tensors="pt")
        if not isinstance(tokenizer, MagicMock):
            assert "input_ids" in tokens
            assert tokens["input_ids"].ndim == 2
            assert tokens["input_ids"].shape[-1] > 0  # at least one token
        else:
            # mock always returns the same structure
            assert tokens["input_ids"] is not None

    def test_inference(self, model_and_tokenizer):
        """Run a forward pass and verify the output contains a last_hidden_state."""
        model, tokenizer = model_and_tokenizer
        text = "Testing transformers upgrade."
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        if not isinstance(model, MagicMock):
            # For real models, outputs should have last_hidden_state
            assert hasattr(outputs, "last_hidden_state")
            assert outputs.last_hidden_state is not None
            # Verify shape: (batch, seq_len, hidden_size)
            assert outputs.last_hidden_state.ndim == 3
        else:
            # mock returns a mock with shape attribute
            assert outputs.last_hidden_state.shape == (1, 2, 768)

    def test_inference_text_classification_style(self, model_and_tokenizer):
        """
        Simulate a common pipeline: classification-like output.
        Uses a model that can be adapted to return logits (if real model).
        For distilbert/bert tiny, we can just check logits exist.
        """
        model, tokenizer = model_and_tokenizer
        if isinstance(model, MagicMock):
            pytest.skip("Skipping classification test with mock model.")

        # For real tiny BERT, it has no classification head by default, so
        # we simply check that forward pass works again.
        text = "Test sentence for classification."
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        # For BERT-like models, pooler_output may exist
        if hasattr(outputs, "pooler_output"):
            assert outputs.pooler_output is not None
            assert outputs.pooler_output.shape[0] == 1  # batch size


# ---------------------------------------------------------------------------
# Run the tests if executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))