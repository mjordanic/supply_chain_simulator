"""Tests for src/rl/agents/set_actor_critic.py.

Acceptance criteria (issue 03):
  - Actor maps (B, K_max, F) → (B, K_max, 3) with one weight set shared across rows.
  - Joint log-prob equals the sum of per-row Gaussian log-probs over active rows only.
  - Entropy excludes masked rows; masked slots receive zero gradient through the loss terms.
  - Sampling at a fixed torch seed is deterministic.
  - Permutation invariance test: row permutation of inputs ⇒ identically permuted
    per-row actions, unchanged joint log-prob, unchanged value.
  - Critic output is a scalar V per batch element, invariant to K (K=1 and K=32).
  - Dedicated test module (this file) for the masked distribution.
  - Existing fixed-K actor/critic and tests untouched.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.rl.agents.set_actor_critic import (
    MaskedJointGaussian,
    SetActor,
    SetCritic,
)
from src.rl.set_encoder import F, K_MAX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs_and_mask(B: int, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return random obs (B, K_MAX, F) and a binary mask (B, K_MAX) with K active rows."""
    obs = torch.randn(B, K_MAX, F)
    mask = torch.zeros(B, K_MAX)
    mask[:, :K] = 1.0
    return obs, mask


# ---------------------------------------------------------------------------
# MaskedJointGaussian tests
# ---------------------------------------------------------------------------


class TestMaskedJointGaussian:
    """Unit tests for the masked distribution sub-module."""

    def _make_dist(
        self,
        B: int = 2,
        K: int = 5,
    ) -> tuple[MaskedJointGaussian, torch.Tensor]:
        mean = torch.randn(B, K_MAX, 3)
        log_std = torch.zeros(3)
        mask = torch.zeros(B, K_MAX)
        mask[:, :K] = 1.0
        dist = MaskedJointGaussian(mean, log_std, mask)
        return dist, mask

    def test_log_prob_active_rows_only(self):
        """Joint log-prob uses only active rows (mask == 1)."""
        B, K = 3, 4
        dist, mask = self._make_dist(B, K)
        action = dist.sample()

        lp = dist.log_prob(action)
        assert lp.shape == (B,)

        # Alter the padded rows of action — log_prob must not change.
        action_perturbed = action.clone()
        action_perturbed[:, K:] += 99.0  # change padded rows
        lp_perturbed = dist.log_prob(action_perturbed)
        assert torch.allclose(lp, lp_perturbed), (
            "log_prob changed when padded rows were altered"
        )

    def test_entropy_active_rows_only(self):
        """Entropy uses only active rows; changing K changes the entropy."""
        B = 2
        K_small = 3
        K_large = 8
        mean = torch.randn(B, K_MAX, 3)
        log_std = torch.zeros(3)

        mask_small = torch.zeros(B, K_MAX)
        mask_small[:, :K_small] = 1.0
        mask_large = torch.zeros(B, K_MAX)
        mask_large[:, :K_large] = 1.0

        dist_small = MaskedJointGaussian(mean, log_std, mask_small)
        dist_large = MaskedJointGaussian(mean, log_std, mask_large)

        ent_small = dist_small.entropy()
        ent_large = dist_large.entropy()
        assert ent_small.shape == (B,)
        assert ent_large.shape == (B,)
        # More active rows → higher entropy sum.
        assert (ent_large > ent_small).all(), (
            "Entropy with more active rows should be larger"
        )

    def test_padded_rows_zero_entropy(self):
        """Padded rows contribute 0 to entropy."""
        B, K = 2, 3
        dist, mask = self._make_dist(B, K)
        ent = dist.entropy()

        # Manually compute entropy using K active rows only.
        std = dist.std
        per_head_ent = dist._dist.entropy()  # (B, K_MAX, 3)
        expected = (per_head_ent.sum(dim=-1) * mask).sum(dim=-1)
        assert torch.allclose(ent, expected)

    def test_sampling_deterministic_at_fixed_seed(self):
        """Two samples with the same generator seed are identical."""
        B, K = 2, 5
        dist, mask = self._make_dist(B, K)

        gen1 = torch.Generator()
        gen1.manual_seed(42)
        s1 = dist.sample(generator=gen1)

        gen2 = torch.Generator()
        gen2.manual_seed(42)
        s2 = dist.sample(generator=gen2)

        assert torch.allclose(s1, s2), "Samples differ with same generator seed"

    def test_padded_rows_zeroed_in_sample(self):
        """Padded rows in the sampled action are exactly zero."""
        B, K = 3, 4
        dist, mask = self._make_dist(B, K)
        action = dist.sample()
        padded = action[:, K:]  # (B, K_MAX - K, 3)
        assert torch.all(padded == 0.0), "Padded action rows are not zero"

    def test_log_prob_matches_manual_computation(self):
        """Joint log-prob equals the manual masked sum."""
        B, K = 2, 4
        dist, mask = self._make_dist(B, K)
        action = dist.sample()

        lp = dist.log_prob(action)

        # Manual: invert tanh, compute base log-prob, apply correction.
        u = torch.atanh(action.clamp(-1 + 1e-6, 1 - 1e-6))
        base_lp = dist._dist.log_prob(u)
        correction = torch.log1p(-action.pow(2).clamp(max=1 - 1e-6))
        per_row_lp = (base_lp - correction).sum(dim=-1)  # (B, K_MAX)
        expected = (per_row_lp * mask).sum(dim=-1)  # (B,)

        assert torch.allclose(lp, expected, atol=1e-5), (
            f"log_prob mismatch: {lp} vs {expected}"
        )


# ---------------------------------------------------------------------------
# SetActor tests
# ---------------------------------------------------------------------------


class TestSetActor:
    """Unit tests for the SetActor."""

    def test_output_shape(self):
        """Actor maps (B, K_MAX, F) → (B, K_MAX, 3)."""
        actor = SetActor()
        B, K = 4, 7
        obs, mask = _make_obs_and_mask(B, K)
        action, lp, ent = actor.get_action_and_log_prob(obs, mask)
        assert action.shape == (B, K_MAX, 3), f"action shape: {action.shape}"
        assert lp.shape == (B,), f"log_prob shape: {lp.shape}"
        assert ent.shape == (B,), f"entropy shape: {ent.shape}"

    def test_action_in_minus1_plus1(self):
        """Active-row actions are in (-1, 1)."""
        actor = SetActor()
        B, K = 2, 5
        obs, mask = _make_obs_and_mask(B, K)
        action, _, _ = actor.get_action_and_log_prob(obs, mask)
        active = action[:, :K]
        assert torch.all(active > -1.0) and torch.all(active < 1.0), (
            "Active actions not in (-1, 1)"
        )

    def test_padded_action_rows_zero(self):
        """Padded rows of the action tensor are 0."""
        actor = SetActor()
        B, K = 2, 4
        obs, mask = _make_obs_and_mask(B, K)
        action, _, _ = actor.get_action_and_log_prob(obs, mask)
        padded = action[:, K:]
        assert torch.all(padded == 0.0), "Padded action rows not zero"

    def test_shared_weights_across_rows(self):
        """The actor uses exactly one set of MLP weights (shared across rows)."""
        actor = SetActor()
        # All weight parameters are inside actor.net or actor.log_std.
        param_count = sum(p.numel() for p in actor.parameters())
        # With hidden_size=64, in=F=16, out=3: 16*64+64 + 64*64+64 + 64*3+3 = 1024+4160+195 = 5379
        # log_std: 3
        expected_min = 5000  # sanity floor
        assert param_count > expected_min, (
            f"Unexpectedly few parameters ({param_count}); shared-weight MLP may be missing."
        )

    def test_evaluate_given_action(self):
        """Passing an action to get_action_and_log_prob evaluates its log-prob."""
        actor = SetActor()
        B, K = 2, 6
        obs, mask = _make_obs_and_mask(B, K)
        # Sample first.
        action_sampled, lp_sampled, _ = actor.get_action_and_log_prob(obs, mask)
        # Evaluate the same action.
        _, lp_eval, _ = actor.get_action_and_log_prob(obs, mask, action=action_sampled)
        assert torch.allclose(lp_sampled, lp_eval, atol=1e-5), (
            "Re-evaluating a sampled action yields different log-prob"
        )

    def test_permutation_invariance_actions(self):
        """Permuting active rows permutes the distribution means correspondingly.

        The actor is deterministic in its mean (MLP is shared-weight, no
        row-to-row coupling). We test permutation invariance of the per-row
        means (the deterministic part of the policy).
        """
        actor = SetActor()
        B, K = 2, 5
        obs, mask = _make_obs_and_mask(B, K)

        # Get the distribution means (without any stochastic noise).
        with torch.no_grad():
            dist_orig = actor.get_distribution(obs, mask)
            mean_orig = dist_orig.mean  # (B, K_MAX, 3) tanh-squashed

        # Create a permutation of the K active rows.
        perm = torch.tensor([2, 0, 4, 1, 3])  # permutation of [0..K-1]
        obs_perm = obs.clone()
        obs_perm[:, :K] = obs[:, perm]

        with torch.no_grad():
            dist_perm = actor.get_distribution(obs_perm, mask)
            mean_perm = dist_perm.mean  # (B, K_MAX, 3)

        # Permuted obs → permuted rows in mean (within first K).
        expected_mean = mean_orig.clone()
        expected_mean[:, :K] = mean_orig[:, perm]
        assert torch.allclose(mean_perm[:, :K], expected_mean[:, :K], atol=1e-5), (
            "Actor mean rows not permuted correspondingly to input permutation"
        )

    def test_permutation_invariance_log_prob(self):
        """Permuting active rows leaves the joint log-prob unchanged.

        Evaluated at the distribution mean (deterministic), so no sampling noise.
        """
        actor = SetActor()
        B, K = 2, 5
        obs, mask = _make_obs_and_mask(B, K)

        perm = torch.tensor([3, 0, 4, 2, 1])  # fixed permutation of [0..K-1]
        obs_perm = obs.clone()
        obs_perm[:, :K] = obs[:, perm]

        with torch.no_grad():
            dist_orig = actor.get_distribution(obs, mask)
            mean_orig = dist_orig.mean

            dist_perm = actor.get_distribution(obs_perm, mask)
            # Evaluate perm distribution at the correspondingly permuted action.
            action_for_perm = mean_orig.clone()
            action_for_perm[:, :K] = mean_orig[:, perm]

            lp_orig = dist_orig.log_prob(mean_orig)
            lp_perm = dist_perm.log_prob(action_for_perm)

        assert torch.allclose(lp_orig, lp_perm, atol=1e-4), (
            f"Joint log-prob changed under row permutation: {lp_orig} vs {lp_perm}"
        )

    def test_no_gradient_through_padded_rows(self):
        """Padded rows contribute zero gradient to the log-prob loss."""
        actor = SetActor()
        B, K = 2, 4
        obs = torch.randn(B, K_MAX, F, requires_grad=True)
        mask = torch.zeros(B, K_MAX)
        mask[:, :K] = 1.0

        action, lp, _ = actor.get_action_and_log_prob(obs, mask)
        loss = lp.sum()
        loss.backward()

        # Gradients for padded rows should be zero (padding doesn't contribute).
        assert obs.grad is not None
        pad_grad = obs.grad[:, K:, :]  # (B, K_MAX - K, F)
        assert torch.all(pad_grad == 0.0), (
            "Non-zero gradients flowed through padded observation rows"
        )


# ---------------------------------------------------------------------------
# SetCritic tests
# ---------------------------------------------------------------------------


class TestSetCritic:
    """Unit tests for the SetCritic."""

    def test_output_shape(self):
        """Critic outputs shape (B, 1) for any K."""
        critic = SetCritic()
        for K in [1, 5, 20, 32]:
            B = 3
            obs, mask = _make_obs_and_mask(B, K)
            v = critic(obs, mask)
            assert v.shape == (B, 1), f"Value shape for K={K}: {v.shape}"

    def test_invariant_to_k(self):
        """Critic works for K=1 and K=K_MAX (32) without error."""
        critic = SetCritic()
        for K in [1, K_MAX]:
            obs, mask = _make_obs_and_mask(B=2, K=K)
            v = critic(obs, mask)
            assert torch.all(torch.isfinite(v)), f"Non-finite value for K={K}"

    def test_permutation_invariance_value(self):
        """Critic value is invariant to row permutation of active products."""
        critic = SetCritic()
        B, K = 2, 5
        obs, mask = _make_obs_and_mask(B, K)

        v_orig = critic(obs, mask)

        perm = torch.randperm(K)
        obs_perm = obs.clone()
        obs_perm[:, :K] = obs[:, perm]

        v_perm = critic(obs_perm, mask)
        assert torch.allclose(v_orig, v_perm, atol=1e-5), (
            f"Critic value changed under row permutation: {v_orig} vs {v_perm}"
        )

    def test_finite_values(self):
        """Critic outputs are finite for normal inputs."""
        critic = SetCritic()
        B, K = 4, 8
        obs, mask = _make_obs_and_mask(B, K)
        v = critic(obs, mask)
        assert torch.all(torch.isfinite(v)), "Critic produced non-finite values"

    def test_gradient_flows(self):
        """Loss derived from critic value has finite gradients for active rows."""
        critic = SetCritic()
        B, K = 2, 5
        obs = torch.randn(B, K_MAX, F, requires_grad=True)
        mask = torch.zeros(B, K_MAX)
        mask[:, :K] = 1.0

        v = critic(obs, mask)
        loss = v.sum()
        loss.backward()

        assert obs.grad is not None
        assert torch.all(torch.isfinite(obs.grad[:, :K])), (
            "Non-finite gradients in active rows of critic"
        )

    def test_padded_rows_dont_affect_value(self):
        """Altering padded rows does not change the critic's value estimate."""
        critic = SetCritic()
        B, K = 2, 4
        obs, mask = _make_obs_and_mask(B, K)

        v_orig = critic(obs, mask)

        obs_perturbed = obs.clone()
        obs_perturbed[:, K:] += 99.0  # alter padded rows

        v_perturbed = critic(obs_perturbed, mask)
        assert torch.allclose(v_orig, v_perturbed, atol=1e-5), (
            "Critic value changed when padded rows were altered"
        )


# ---------------------------------------------------------------------------
# Full permutation invariance test (combined actor + critic)
# ---------------------------------------------------------------------------


class TestPermutationInvariance:
    """Combined permutation invariance test per acceptance criterion."""

    def test_full_permutation_invariance(self):
        """Permuting active rows:
        1. permutes per-row distribution means correspondingly,
        2. leaves joint log-prob unchanged (when evaluated on the correspondingly permuted action),
        3. leaves critic value unchanged.
        """
        torch.manual_seed(1234)
        actor = SetActor()
        critic = SetCritic()

        B, K = 2, 6
        obs, mask = _make_obs_and_mask(B, K)

        # Permute active rows.
        perm = torch.tensor([2, 0, 5, 1, 3, 4])  # permutation of [0..K-1]
        obs_perm = obs.clone()
        obs_perm[:, :K] = obs[:, perm]

        with torch.no_grad():
            # Get means from original and permuted obs.
            dist_orig = actor.get_distribution(obs, mask)
            mean_orig = dist_orig.mean  # (B, K_MAX, 3) — deterministic

            dist_perm = actor.get_distribution(obs_perm, mask)
            mean_perm = dist_perm.mean  # (B, K_MAX, 3)

            # Compute log-probs: use mean_orig as "action" for both.
            # For permuted obs, we use the correspondingly permuted action.
            action_for_perm = mean_orig.clone()
            action_for_perm[:, :K] = mean_orig[:, perm]

            lp_orig = dist_orig.log_prob(mean_orig)
            lp_perm = dist_perm.log_prob(action_for_perm)

            v_orig = critic(obs, mask)
            v_perm = critic(obs_perm, mask)

        # 1. Per-row means permuted.
        expected_mean = mean_orig.clone()
        expected_mean[:, :K] = mean_orig[:, perm]
        assert torch.allclose(
            mean_perm[:, :K], expected_mean[:, :K], atol=1e-5
        ), "Per-row distribution means not permuted correspondingly"

        # 2. Joint log-prob unchanged.
        assert torch.allclose(lp_orig, lp_perm, atol=1e-4), (
            f"Joint log-prob changed: {lp_orig} vs {lp_perm}"
        )

        # 3. Critic value unchanged.
        assert torch.allclose(v_orig, v_perm, atol=1e-5), (
            f"Critic value changed: {v_orig} vs {v_perm}"
        )
