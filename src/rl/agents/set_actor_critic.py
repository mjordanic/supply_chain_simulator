"""Masked set actor-critic for variable-K RL (ADR 0021, Decision 1).

Implements:

- ``SetActor``  — shared-weight MLP applied row-wise, ``(B, K_max, F) → (B, K_max, 3)``.
  Tanh-squashed means; state-independent log-std per head.
- ``MaskedJointGaussian`` — masked joint distribution; joint log-prob is the
  masked sum of per-product diagonal Gaussian log-probs; entropy excludes
  masked rows; sampling is deterministic at a fixed seed.
- ``SetCritic`` — DeepSets-style critic; per-product embedding MLP, masked
  mean-pool, concatenated with the global block, then a scalar V head.
  Permutation-invariant and K-agnostic by construction.

Both actor and critic are permutation-invariant: the structural guarantee that
replaces the deleted slot-shuffle (ADR 0004 Decision 2).

This module co-exists with ``src/rl/agents/ppo.py`` (the legacy flat Actor /
Critic).  The legacy networks are retired in issue 06.

Public API
----------
``SetActor``, ``SetCritic``, ``MaskedJointGaussian``
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from src.rl.set_encoder import F, K_MAX

# ---------------------------------------------------------------------------
# Constants (mirror the set_encoder layout)
# ---------------------------------------------------------------------------

_GLOBAL_DIM: int = 4
"""Number of global features broadcast per row (ROW_CASH..ROW_COS in set_encoder)."""

_PER_PRODUCT_DIM: int = F - _GLOBAL_DIM - 1  # 16 - 4 - 1(mask) = 11, but mask col is index 15
# We keep all F features as actor input; the mask channel acts as a gate, not
# as a feature to be split. The split is only needed by the critic.
_ACTOR_IN_DIM: int = F
"""Feature dimension per row for the actor (full F, including mask channel)."""

# DeepSets critic splits the observation row into per-product and global parts.
# Global: ROW_CASH(9), ROW_TOTAL_INV(10), ROW_SIN(11), ROW_COS(12) = indices 9-12
# Mask: ROW_MASK(15)
# We treat all F features as per-product embedding input for simplicity, and
# extract the 4 global features (indices 9-12) separately for the global block.
_GLOBAL_FEATURE_START: int = 9
_GLOBAL_FEATURE_END: int = 13  # exclusive — indices 9,10,11,12
_CRITIC_EMBED_DIM: int = F  # embed full row
_CRITIC_GLOBAL_DIM: int = _GLOBAL_FEATURE_END - _GLOBAL_FEATURE_START  # 4

# Number of action heads per row
_ACT_HEADS: int = 3


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _layer_init(layer: nn.Linear, std: float = 0.01, bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal init + constant bias — standard CleanRL practice."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# ---------------------------------------------------------------------------
# MaskedJointGaussian
# ---------------------------------------------------------------------------


class MaskedJointGaussian:
    """Masked joint Gaussian distribution over a ``(B, K_max, 3)`` action space.

    Each active row contributes independently to the joint log-prob, entropy,
    and gradient.  Padded rows (mask == 0) are zeroed out.

    Parameters
    ----------
    mean:
        Shape ``(B, K_max, 3)``. Pre-squashed means; NOT passed through tanh
        yet — ``sample`` and ``log_prob`` apply tanh squash internally.
    log_std:
        Shape ``(3,)`` or broadcastable. State-independent log-std per head.
    mask:
        Shape ``(B, K_max)`` or ``(K_max,)``. 1.0 for active rows, 0.0 for padding.

    Notes
    -----
    Log-prob uses the change-of-variables correction for the tanh squash:

        log p(a) = log p(u) − Σ log(1 − tanh²(u))

    where ``u`` is the pre-squash sample and ``a = tanh(u)``.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        self.mean = mean  # (B, K_max, 3)
        self.std = log_std.exp()  # (3,)
        # Ensure mask is (B, K_max)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(mean.shape[0], -1)
        self.mask = mask  # (B, K_max)
        self._dist = Normal(mean, self.std.expand_as(mean))

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def sample(self, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Draw a sample; squash via tanh.  Padded rows remain all-zero.

        Parameters
        ----------
        generator:
            Optional ``torch.Generator`` for deterministic sampling at a fixed seed.

        Returns
        -------
        torch.Tensor
            Shape ``(B, K_max, 3)``, in ``(-1, 1)`` via tanh.  Padded rows are 0.
        """
        # Draw standard normal noise, scale by std, shift by mean.
        noise = torch.randn(self.mean.shape, generator=generator, dtype=self.mean.dtype)
        raw = self.mean + self.std * noise  # (B, K_max, 3)
        action = torch.tanh(raw)
        # Zero out padded rows: mask (B, K_max) → (B, K_max, 1)
        action = action * self.mask.unsqueeze(-1)
        return action

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Compute masked joint log-prob.

        Parameters
        ----------
        action:
            Shape ``(B, K_max, 3)``, assumed in ``(-1, 1)`` (tanh-squashed).

        Returns
        -------
        torch.Tensor
            Shape ``(B,)`` — sum of per-row, per-head log-probs over active rows.
        """
        # Invert tanh to recover pre-squash sample.
        u = torch.atanh(action.clamp(-1 + 1e-6, 1 - 1e-6))

        # Per-head log-prob from the base Gaussian, shape (B, K_max, 3).
        base_lp = self._dist.log_prob(u)

        # Tanh change-of-variables correction: -log(1 - tanh²(u))
        # = -log(1 - action²), computed stably.
        correction = torch.log1p(-action.pow(2).clamp(max=1 - 1e-6))
        per_head_lp = base_lp - correction  # (B, K_max, 3)

        # Sum heads → per-row log-prob, shape (B, K_max).
        per_row_lp = per_head_lp.sum(dim=-1)

        # Mask out padded rows, then sum over K_max → (B,).
        return (per_row_lp * self.mask).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Masked sum of per-row diagonal-Gaussian entropies.

        Returns
        -------
        torch.Tensor
            Shape ``(B,)``.  Padded rows contribute 0.
        """
        # Per-head entropy, shape (B, K_max, 3).
        per_head_ent = self._dist.entropy()
        # Sum heads → per-row entropy, shape (B, K_max).
        per_row_ent = per_head_ent.sum(dim=-1)
        # Mask and sum → (B,).
        return (per_row_ent * self.mask).sum(dim=-1)


# ---------------------------------------------------------------------------
# SetActor
# ---------------------------------------------------------------------------


class SetActor(nn.Module):
    """Shared-weight MLP actor applied row-wise over the ``(B, K_max, F)`` obs.

    Maps ``(B, K_max, F) → (B, K_max, 3)`` with one weight set shared across
    all rows and batch elements.  Tanh-squashed means; state-independent
    log-std per head.

    Parameters
    ----------
    hidden_size:
        Width of each hidden layer.
    in_features:
        Input feature dimension per row (default ``F`` from set_encoder).
    """

    def __init__(
        self,
        hidden_size: int = 64,
        in_features: int = _ACTOR_IN_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(in_features, hidden_size), std=(2.0 ** 0.5)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, hidden_size), std=(2.0 ** 0.5)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, _ACT_HEADS), std=0.01),
        )
        # State-independent log-std per action head.
        self.log_std = nn.Parameter(torch.zeros(_ACT_HEADS))

    def get_distribution(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> MaskedJointGaussian:
        """Return the masked joint distribution conditioned on ``obs``.

        Parameters
        ----------
        obs:
            Shape ``(B, K_max, F)``.
        mask:
            Shape ``(B, K_max)`` or ``(K_max,)``.  1.0 active, 0.0 padding.
        """
        B, K, _F = obs.shape
        # Flatten to (B*K, F), apply shared MLP, reshape back.
        flat = obs.reshape(B * K, _F)
        raw_mean = self.net(flat).reshape(B, K, _ACT_HEADS)  # (B, K_max, 3)
        mean = torch.tanh(raw_mean)
        return MaskedJointGaussian(mean, self.log_std, mask)

    def get_action_and_log_prob(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or evaluate) and return action, log-prob, entropy.

        Parameters
        ----------
        obs:
            Shape ``(B, K_max, F)``.
        mask:
            Shape ``(B, K_max)`` — 1.0 active, 0.0 padding.
        action:
            If given, evaluate log-prob for this action.  Shape ``(B, K_max, 3)``.

        Returns
        -------
        action:
            Shape ``(B, K_max, 3)``, in ``(-1, 1)`` via tanh.  Padded rows are 0.
        log_prob:
            Shape ``(B,)`` — masked sum.
        entropy:
            Shape ``(B,)`` — masked sum.
        """
        dist = self.get_distribution(obs, mask)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        ent = dist.entropy()
        return action, log_prob, ent


# ---------------------------------------------------------------------------
# SetCritic
# ---------------------------------------------------------------------------


class SetCritic(nn.Module):
    """DeepSets-style critic: masked mean-pool of per-product embeddings.

    Architecture:
    1. Per-product embedding MLP: ``(F,) → (embed_dim,)`` applied row-wise.
    2. Masked mean-pool over active rows → ``(embed_dim,)`` per batch elem.
    3. Concatenate with the global block ``(4 global features)`` extracted
       from row 0 (the value is broadcast so row 0 == any active row).
    4. Head MLP → scalar V per batch element.

    Permutation-invariant by construction: step 2 is a masked mean-pool
    that is invariant to the order of active rows.

    Parameters
    ----------
    embed_dim:
        Output dimension of the per-product embedding MLP.
    head_hidden_size:
        Hidden size of the final value head.
    in_features:
        Input feature dimension per row (default ``F``).
    """

    def __init__(
        self,
        embed_dim: int = 64,
        head_hidden_size: int = 64,
        in_features: int = _CRITIC_EMBED_DIM,
    ) -> None:
        super().__init__()
        # Per-product embedding MLP (shared weights, applied row-wise).
        self.embed_net = nn.Sequential(
            _layer_init(nn.Linear(in_features, embed_dim), std=(2.0 ** 0.5)),
            nn.Tanh(),
            _layer_init(nn.Linear(embed_dim, embed_dim), std=(2.0 ** 0.5)),
            nn.Tanh(),
        )
        # Head that maps [pooled_embed ‖ global_block] → scalar V.
        head_in = embed_dim + _CRITIC_GLOBAL_DIM
        self.head = nn.Sequential(
            _layer_init(nn.Linear(head_in, head_hidden_size), std=(2.0 ** 0.5)),
            nn.Tanh(),
            _layer_init(nn.Linear(head_hidden_size, 1), std=1.0),
        )

    def forward(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return scalar value estimates.

        Parameters
        ----------
        obs:
            Shape ``(B, K_max, F)``.
        mask:
            Shape ``(B, K_max)`` — 1.0 active, 0.0 padding.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 1)``.
        """
        B, K, _F = obs.shape

        # 1. Per-product embeddings.
        flat = obs.reshape(B * K, _F)
        embeds = self.embed_net(flat).reshape(B, K, -1)  # (B, K_max, embed_dim)

        # 2. Masked mean-pool.
        mask_3d = mask.unsqueeze(-1)  # (B, K_max, 1)
        active_count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1)
        pooled = (embeds * mask_3d).sum(dim=1) / active_count  # (B, embed_dim)

        # 3. Global block: masked mean of global features across active rows.
        # Global features (indices 9-12) are broadcast identically to all active
        # rows, so any active row has the correct value. The masked mean is
        # permutation-invariant and handles K=1 correctly.
        global_feats = obs[:, :, _GLOBAL_FEATURE_START:_GLOBAL_FEATURE_END]  # (B, K_max, 4)
        global_block = (global_feats * mask.unsqueeze(-1)).sum(dim=1) / active_count  # (B, 4)

        # 4. Head.
        combined = torch.cat([pooled, global_block], dim=-1)  # (B, embed_dim + 4)
        return self.head(combined)  # (B, 1)


__all__ = ["SetActor", "SetCritic", "MaskedJointGaussian"]
