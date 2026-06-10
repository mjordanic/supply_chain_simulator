"""Checkpoint I/O: self-describing checkpoints with layout-version validation.

A checkpoint is a dict bundling:

- ``state_dict``: the actor ``state_dict()``.
- ``config``: a training config snapshot (plain dict, JSON-serialisable).
- ``layout_version``: the observation-layout version integer sourced from
  ``src.rl.set_encoder.OBS_LAYOUT_VERSION``.

Loading validates the stored layout version against the encoder's current
constant and raises a descriptive error on mismatch — a stale checkpoint
fails loudly instead of with a torch shape error or silently.

Loading a legacy bare state-dict file (saved before this module existed)
raises a ``ValueError`` with a clear "legacy checkpoint, retrain" message.

This module does **not** construct networks — it is architecture-agnostic
beyond the ``state_dict`` key.  The train loop writes new checkpoints
(issue 06), and eval / notebooks read them (issues 07 and 09).

Public API
----------
``save(state_dict, config, path)``
``load(path) → dict``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.rl.set_encoder import OBS_LAYOUT_VERSION

__all__ = ["save", "load"]

# Key names used in the checkpoint dict.
_KEY_STATE_DICT = "state_dict"
_KEY_CONFIG = "config"
_KEY_LAYOUT_VERSION = "layout_version"


def save(
    state_dict: dict[str, Any],
    config: dict[str, Any],
    path: str | Path,
) -> None:
    """Save a self-describing checkpoint.

    Parameters
    ----------
    state_dict:
        The actor's ``state_dict()`` (or any ``nn.Module``'s state dict).
    config:
        Training config snapshot as a plain dict.  Should be JSON-serialisable,
        but no validation is enforced here.
    path:
        Destination file path.  Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        _KEY_STATE_DICT: state_dict,
        _KEY_CONFIG: config,
        _KEY_LAYOUT_VERSION: OBS_LAYOUT_VERSION,
    }
    torch.save(bundle, path)


def load(path: str | Path) -> dict[str, Any]:
    """Load and validate a checkpoint.

    Parameters
    ----------
    path:
        Path to a checkpoint file written by :func:`save`.

    Returns
    -------
    dict
        ``{"state_dict": ..., "config": ..., "layout_version": int}``

    Raises
    ------
    ValueError
        If the file is a legacy bare state-dict (does not contain the
        expected keys), or if the stored layout version does not match
        ``OBS_LAYOUT_VERSION``.
    """
    path = Path(path)
    raw = torch.load(path, map_location="cpu", weights_only=False)

    # Detect legacy bare state-dict: a plain dict whose values are tensors
    # (or an OrderedDict), not a structured bundle.
    if not isinstance(raw, dict) or _KEY_LAYOUT_VERSION not in raw:
        raise ValueError(
            f"Legacy checkpoint detected at '{path}'. "
            "This file was saved before layout-version validation was introduced. "
            "Please retrain to produce a compatible checkpoint."
        )

    stored_version: int = raw[_KEY_LAYOUT_VERSION]
    if stored_version != OBS_LAYOUT_VERSION:
        raise ValueError(
            f"Checkpoint layout version mismatch: "
            f"checkpoint has version {stored_version}, "
            f"but the current encoder uses version {OBS_LAYOUT_VERSION}. "
            "Please retrain or use a checkpoint produced with the current encoder layout."
        )

    return raw
