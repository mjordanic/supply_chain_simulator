"""Unit tests for the pure-function ``FreshnessCurve.multiplier``.

The freshness curve is

    m(τ) = 1 + α · exp(−τ / β)

These tests pin the math properties enumerated in PRD § Testing
Decisions T1: peak at τ=0, baseline limit as τ → ∞, neutrality when
α = 0, monotone decay in τ for α > 0.
"""

from __future__ import annotations

import math

import pytest

from src.sim import freshness_curve


def test_multiplier_at_tau_zero_is_one_plus_alpha():
    """Immediately after activation, hype is at its peak."""
    assert freshness_curve.multiplier(alpha=0.4, decay=30, ticks_since_activation=0) == pytest.approx(1.4)
    assert freshness_curve.multiplier(alpha=0.1, decay=15, ticks_since_activation=0) == pytest.approx(1.1)


def test_multiplier_as_tau_large_approaches_one():
    """As ticks since activation grow large, the multiplier decays to 1.0."""
    assert freshness_curve.multiplier(alpha=0.4, decay=30, ticks_since_activation=10_000) == pytest.approx(1.0)
    # Within 5% of baseline by τ ≈ 3β.
    assert abs(freshness_curve.multiplier(alpha=0.4, decay=30, ticks_since_activation=90) - 1.0) < 0.05


def test_multiplier_with_alpha_zero_is_always_one():
    """With α=0, the curve is identically 1.0 regardless of τ or β."""
    for tau in (0, 1, 30, 1_000, 1_000_000):
        for decay in (1, 15, 30, 100):
            assert freshness_curve.multiplier(alpha=0.0, decay=decay, ticks_since_activation=tau) == 1.0


def test_multiplier_monotone_decreasing_in_tau_for_positive_alpha():
    """For α > 0, m(τ) is strictly decreasing in τ."""
    alpha = 0.4
    decay = 30
    prev = freshness_curve.multiplier(alpha, decay, 0)
    for tau in range(1, 200):
        current = freshness_curve.multiplier(alpha, decay, tau)
        assert current < prev
        prev = current


def test_multiplier_matches_closed_form():
    """Spot-check the formula against the closed-form expression."""
    alpha, decay, tau = 0.4, 30, 15
    expected = 1.0 + alpha * math.exp(-tau / decay)
    assert freshness_curve.multiplier(alpha, decay, tau) == pytest.approx(expected)
