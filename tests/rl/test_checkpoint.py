"""Tests for src/rl/checkpoint.py.

Acceptance criteria (issue 04):
  - Save produces a dict containing state-dict, config snapshot, and layout version.
  - Load returns the bundle when the layout version matches the encoder constant.
  - Load raises a descriptive error (naming both versions) on mismatch.
  - Loading a legacy bare state-dict file fails with a "legacy checkpoint, retrain" message
    rather than a KeyError.
  - No torch-architecture coupling beyond the state_dict itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.rl.checkpoint import load, save
from src.rl.set_encoder import OBS_LAYOUT_VERSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_state_dict() -> dict:
    """Return a minimal state_dict-shaped dict (no actual nn.Module needed)."""
    return {
        "net.0.weight": torch.randn(64, 16),
        "net.0.bias": torch.zeros(64),
        "log_std": torch.zeros(3),
    }


def _dummy_config() -> dict:
    """Return a minimal training config snapshot."""
    return {
        "lr": 3e-4,
        "n_steps": 128,
        "K_active": 5,
        "episode_length": 180,
    }


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Round-trip tests for the checkpoint module."""

    def test_save_creates_file(self, tmp_path: Path):
        """save() creates the checkpoint file."""
        path = tmp_path / "ckpt.pt"
        save(_dummy_state_dict(), _dummy_config(), path)
        assert path.exists(), "Checkpoint file not created"

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        """save() creates intermediate parent directories."""
        path = tmp_path / "nested" / "deep" / "ckpt.pt"
        save(_dummy_state_dict(), _dummy_config(), path)
        assert path.exists(), "Checkpoint not created in nested directory"

    def test_load_returns_bundle_with_correct_keys(self, tmp_path: Path):
        """load() returns a dict with state_dict, config, and layout_version keys."""
        path = tmp_path / "ckpt.pt"
        save(_dummy_state_dict(), _dummy_config(), path)
        bundle = load(path)
        assert "state_dict" in bundle, "Missing 'state_dict' key"
        assert "config" in bundle, "Missing 'config' key"
        assert "layout_version" in bundle, "Missing 'layout_version' key"

    def test_load_state_dict_matches(self, tmp_path: Path):
        """The loaded state_dict is identical to the saved one."""
        sd = _dummy_state_dict()
        path = tmp_path / "ckpt.pt"
        save(sd, _dummy_config(), path)
        bundle = load(path)
        loaded_sd = bundle["state_dict"]
        for key in sd:
            assert key in loaded_sd, f"Key {key!r} missing from loaded state_dict"
            assert torch.allclose(sd[key], loaded_sd[key]), (
                f"Tensor mismatch for key {key!r}"
            )

    def test_load_config_matches(self, tmp_path: Path):
        """The loaded config snapshot matches the saved config."""
        cfg = _dummy_config()
        path = tmp_path / "ckpt.pt"
        save(_dummy_state_dict(), cfg, path)
        bundle = load(path)
        assert bundle["config"] == cfg, "Config snapshot mismatch"

    def test_load_layout_version_matches_encoder(self, tmp_path: Path):
        """The loaded layout_version matches the current encoder constant."""
        path = tmp_path / "ckpt.pt"
        save(_dummy_state_dict(), _dummy_config(), path)
        bundle = load(path)
        assert bundle["layout_version"] == OBS_LAYOUT_VERSION, (
            f"Layout version mismatch: {bundle['layout_version']} vs {OBS_LAYOUT_VERSION}"
        )

    def test_save_accepts_string_path(self, tmp_path: Path):
        """save() accepts both str and Path for the path argument."""
        path_str = str(tmp_path / "ckpt_str.pt")
        save(_dummy_state_dict(), _dummy_config(), path_str)
        assert Path(path_str).exists()

    def test_load_accepts_string_path(self, tmp_path: Path):
        """load() accepts both str and Path for the path argument."""
        path = tmp_path / "ckpt.pt"
        save(_dummy_state_dict(), _dummy_config(), path)
        bundle = load(str(path))  # pass as string
        assert "layout_version" in bundle


# ---------------------------------------------------------------------------
# Version mismatch
# ---------------------------------------------------------------------------


class TestVersionMismatch:
    """load() raises descriptive errors on version mismatches."""

    def test_raises_on_layout_version_mismatch(self, tmp_path: Path):
        """load() raises ValueError naming both versions on mismatch."""
        path = tmp_path / "stale_ckpt.pt"
        # Manually build a bundle with a wrong layout version.
        stale_version = OBS_LAYOUT_VERSION + 99  # definitely wrong
        bundle = {
            "state_dict": _dummy_state_dict(),
            "config": _dummy_config(),
            "layout_version": stale_version,
        }
        torch.save(bundle, path)

        with pytest.raises(ValueError) as exc_info:
            load(path)

        msg = str(exc_info.value)
        # Error message must name the stored version.
        assert str(stale_version) in msg, (
            f"Error message does not name stored version {stale_version}: {msg}"
        )
        # Error message must name the current version.
        assert str(OBS_LAYOUT_VERSION) in msg, (
            f"Error message does not name current version {OBS_LAYOUT_VERSION}: {msg}"
        )

    def test_mismatch_error_is_descriptive(self, tmp_path: Path):
        """The mismatch error message is human-readable (not just a version number)."""
        path = tmp_path / "stale.pt"
        bundle = {
            "state_dict": _dummy_state_dict(),
            "config": _dummy_config(),
            "layout_version": OBS_LAYOUT_VERSION - 1,  # one version behind
        }
        torch.save(bundle, path)

        with pytest.raises(ValueError) as exc_info:
            load(path)

        msg = str(exc_info.value).lower()
        # Should contain meaningful words, not just numbers.
        assert "layout" in msg or "version" in msg or "mismatch" in msg, (
            f"Error message not descriptive: {exc_info.value}"
        )


# ---------------------------------------------------------------------------
# Legacy checkpoint detection
# ---------------------------------------------------------------------------


class TestLegacyCheckpoint:
    """load() rejects legacy bare state-dict files with a clear message."""

    def test_legacy_state_dict_raises_value_error(self, tmp_path: Path):
        """A bare state_dict (no bundle wrapper) raises ValueError, not KeyError."""
        path = tmp_path / "legacy.pt"
        # Save a bare state dict — the old format.
        torch.save(_dummy_state_dict(), path)

        with pytest.raises(ValueError) as exc_info:
            load(path)

        msg = str(exc_info.value).lower()
        # Message should mention "legacy" and "retrain".
        assert "legacy" in msg, f"Expected 'legacy' in message: {exc_info.value}"
        assert "retrain" in msg, f"Expected 'retrain' in message: {exc_info.value}"

    def test_legacy_does_not_raise_key_error(self, tmp_path: Path):
        """A legacy checkpoint must NOT propagate a raw KeyError."""
        path = tmp_path / "legacy.pt"
        torch.save(_dummy_state_dict(), path)

        # Should be ValueError, not KeyError.
        with pytest.raises(ValueError):
            load(path)

    def test_legacy_ordered_dict_also_raises(self, tmp_path: Path):
        """An OrderedDict state_dict (torch default) is also detected as legacy."""
        import collections
        path = tmp_path / "legacy_ordered.pt"
        ordered_sd = collections.OrderedDict(_dummy_state_dict())
        torch.save(ordered_sd, path)

        with pytest.raises(ValueError) as exc_info:
            load(path)
        assert "legacy" in str(exc_info.value).lower()
