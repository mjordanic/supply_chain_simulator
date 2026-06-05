"""Smoke test: dead policy aliases must not be exposed by ``src.sim.policy``.

Pins the hard-break property for names that were removed or renamed.
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


def test_heuristic_policy_not_importable():
    """``HeuristicPolicy`` was removed in issue 07 — it must not be importable."""
    assert not hasattr(src.sim.policy, "HeuristicPolicy")


def test_noop_policy_not_importable():
    """``NoopPolicy`` was removed in issue 07 — it must not be importable."""
    assert not hasattr(src.sim.policy, "NoopPolicy")
