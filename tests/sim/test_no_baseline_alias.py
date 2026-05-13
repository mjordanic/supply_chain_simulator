"""Smoke test: ``from src.sim.policy import BaselinePolicy`` must raise ImportError.

This is the auditable proof that the hard-break property holds after the
rename ``BaselinePolicy`` → ``HeuristicPolicy`` (issue 02).
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest


def test_baseline_policy_not_importable():
    """``from src.sim.policy import BaselinePolicy`` raises ``ImportError``.

    The rename in issue 02 is a hard break — no compatibility alias is
    permitted by the PRD.  This test pins that invariant.
    """
    # Force a fresh import so cached module state doesn't hide a deletion.
    mod_name = "src.sim.policy"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    with pytest.raises(ImportError):
        from src.sim.policy import BaselinePolicy  # noqa: F401


def test_heuristic_policy_is_importable():
    """``HeuristicPolicy`` is the successor and must be importable."""
    mod_name = "src.sim.policy"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    from src.sim.policy import HeuristicPolicy  # noqa: F401

    assert HeuristicPolicy is not None
