"""Smoke test: ``from src.sim.policy import BaselinePolicy`` must raise ImportError.

This is the auditable proof that the hard-break property holds after the
rename ``BaselinePolicy`` → ``HeuristicPolicy`` (issue 02).
"""

from __future__ import annotations

import src.sim.policy


def test_baseline_policy_not_importable():
    """``BaselinePolicy`` must not be exposed by ``src.sim.policy``.

    The rename in issue 02 is a hard break — no compatibility alias is
    permitted by the PRD.  This test pins that invariant.

    Checked via ``hasattr`` rather than deleting from ``sys.modules``
    and re-importing: a re-import gives the module's classes new
    identity, which breaks ``isinstance`` checks in any later test that
    captured the original class objects at collection time.
    """
    assert not hasattr(src.sim.policy, "BaselinePolicy")


def test_heuristic_policy_is_importable():
    """``HeuristicPolicy`` is the successor and must be importable."""
    assert src.sim.policy.HeuristicPolicy is not None
