"""CleanRL-style single-file PPO for the supply-chain RL env.

Adapted from CleanRL's ``ppo_continuous_action.py`` with the following
changes:

- Actor and Critic are separate small MLPs (64-64 hidden, tanh).  Actor
  outputs a ``Normal`` distribution over actions; samples are squashed to
  ``[-1, 1]`` via tanh at collection time.
- All hyperparameters are read from ``RLConfig`` — no magic numbers buried
  inline.
- The training entry-point ``train_ppo(envs, config, writer, eval_fn)``
  owns the rollout loop and update loop.  The ``train.py`` driver (issue 08)
  constructs ``envs``, ``writer``, and ``eval_fn`` and calls into this
  function.
- TensorBoard scalars logged every update:
    train/episodic_return, train/episodic_length,
    losses/value_loss, losses/policy_loss, losses/entropy,
    losses/approx_kl, losses/clipfrac,
    charts/learning_rate, charts/SPS.

Architecture
------------
The Actor / Critic boundary is the *only* place a future heavier model
(transformer, attention-over-SKUs) needs to change.  Everything around it
(rollout collection, GAE, update loop, TensorBoard logging) stays the same.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from src.rl.configs.default import RLConfig


# ---------------------------------------------------------------------------
# Network definitions
# ---------------------------------------------------------------------------


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal init + constant bias — standard CleanRL practice."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """MLP actor that parameterises a diagonal-Gaussian action distribution.

    The output mean is passed through tanh to keep the mean in ``(-1, 1)``.
    The log-std is a learned parameter vector (independent of the obs).

    Parameters
    ----------
    obs_dim:
        Dimensionality of the flat observation vector.
    act_dim:
        Dimensionality of the continuous action vector.
    hidden_size:
        Width of each hidden layer (default 64).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, act_dim), std=0.01),
        )
        # Log-std as a stand-alone learnable parameter (scalar-broadcast per dim).
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        """Return the action distribution conditioned on ``obs``."""
        mean = torch.tanh(self.net(obs))
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action_and_log_prob(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action (or evaluate a given one) and return log-prob + entropy.

        Parameters
        ----------
        obs:
            Observation batch, shape ``(B, obs_dim)``.
        action:
            If given, evaluate log-prob for this action.  Otherwise, sample.

        Returns
        -------
        action:
            Action batch, shape ``(B, act_dim)``, in ``(-1, 1)`` via tanh.
        log_prob:
            Per-sample sum of log-probs, shape ``(B,)``.
        entropy:
            Per-sample sum of entropies, shape ``(B,)``.
        """
        dist = self.get_distribution(obs)
        if action is None:
            raw = dist.rsample()
            action = torch.tanh(raw)
        else:
            # Invert tanh to recover the pre-squash sample for log-prob.
            # Clamp to avoid log(0) from atanh at ±1.
            raw = torch.atanh(action.clamp(-1 + 1e-6, 1 - 1e-6))
        log_prob = dist.log_prob(raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy


class Critic(nn.Module):
    """MLP critic that estimates the state-value function V(s).

    Parameters
    ----------
    obs_dim:
        Dimensionality of the flat observation vector.
    hidden_size:
        Width of each hidden layer (default 64).
    """

    def __init__(self, obs_dim: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return value estimates, shape ``(B, 1)``."""
        return self.net(obs)


# ---------------------------------------------------------------------------
# Rollout buffer helpers
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """Fixed-size buffer for PPO rollout storage.

    Holds ``n_steps × n_envs`` transitions collected from a sync-vec-env.
    Advantages are computed in-place via ``compute_gae`` before the update
    loop reads from the buffer.

    Parameters
    ----------
    n_steps:
        Number of env steps per env per rollout (from ``config.n_steps``).
    n_envs:
        Number of parallel envs.
    obs_dim:
        Flat observation size.
    act_dim:
        Flat action size.
    device:
        PyTorch device for all stored tensors.
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
    ) -> None:
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        self.obs = torch.zeros(n_steps, n_envs, obs_dim, device=device)
        self.actions = torch.zeros(n_steps, n_envs, act_dim, device=device)
        self.log_probs = torch.zeros(n_steps, n_envs, device=device)
        self.rewards = torch.zeros(n_steps, n_envs, device=device)
        self.dones = torch.zeros(n_steps, n_envs, device=device)
        self.values = torch.zeros(n_steps, n_envs, device=device)

        # Filled by compute_gae.
        self.advantages = torch.zeros(n_steps, n_envs, device=device)
        self.returns = torch.zeros(n_steps, n_envs, device=device)

        self._ptr: int = 0

    def store(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Store one step's data at the current pointer, then advance."""
        t = self._ptr
        self.obs[t] = obs
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.rewards[t] = reward
        self.dones[t] = done
        self.values[t] = value
        self._ptr += 1

    def compute_gae(
        self,
        next_obs: torch.Tensor,
        next_done: torch.Tensor,
        critic: Critic,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE-λ advantages and Monte-Carlo returns in-place.

        Populates ``self.advantages`` and ``self.returns``.
        """
        with torch.no_grad():
            next_value = critic(next_obs).squeeze(-1)  # (n_envs,)
            last_gae = torch.zeros(self.n_envs, device=self.device)
            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    next_non_terminal = 1.0 - next_done.float()
                    v_next = next_value
                else:
                    next_non_terminal = 1.0 - self.dones[t + 1]
                    v_next = self.values[t + 1]
                delta = self.rewards[t] + gamma * v_next * next_non_terminal - self.values[t]
                last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
                self.advantages[t] = last_gae
            self.returns = self.advantages + self.values

    def reset(self) -> None:
        """Reset the write pointer after a full rollout has been consumed."""
        self._ptr = 0

    def get_flat(self) -> tuple[torch.Tensor, ...]:
        """Return all stored data flattened to ``(n_steps * n_envs, ...)``.

        Returns
        -------
        obs, actions, log_probs_old, advantages, returns
        """
        b = self.n_steps * self.n_envs
        return (
            self.obs.view(b, self.obs_dim),
            self.actions.view(b, self.act_dim),
            self.log_probs.view(b),
            self.advantages.view(b),
            self.returns.view(b),
        )


# ---------------------------------------------------------------------------
# Training entry-point
# ---------------------------------------------------------------------------


def train_ppo(
    envs,
    config: RLConfig,
    writer: SummaryWriter,
    eval_fn: Callable[[Actor], dict[str, float]] | None = None,
    *,
    device: torch.device | None = None,
    seed: int = 0,
) -> tuple[Actor, Critic]:
    """Run a full PPO training loop.

    The training driver (``train.py``, issue 08) constructs ``envs``,
    ``writer``, and ``eval_fn`` and then calls this function.  This
    function owns the rollout loop, GAE computation, and minibatch update
    loop.

    Parameters
    ----------
    envs:
        A ``gymnasium.vector.VectorEnv`` (sync or async).  Must expose
        ``envs.single_observation_space`` and ``envs.single_action_space``.
    config:
        ``RLConfig`` instance carrying all hyperparameters.
    writer:
        A ``torch.utils.tensorboard.SummaryWriter`` open for writing.
        Scalars are flushed every update step.
    eval_fn:
        Optional callable ``(actor: Actor) → dict[str, float]``.  Called
        every ``config.eval_cadence_env_steps`` global env steps; the
        returned dict is logged under ``eval/*`` keys.  When ``None``,
        no eval runs.
    device:
        Torch device.  When ``None``, uses CUDA if available, else CPU.
    seed:
        Seed for torch / numpy RNG used inside the training loop.

    Returns
    -------
    actor, critic
        Trained network objects (on ``device``).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Derive shapes from the vec-env.
    obs_dim: int = int(np.prod(envs.single_observation_space.shape))
    act_dim: int = int(np.prod(envs.single_action_space.shape))
    n_envs: int = envs.num_envs

    # Networks.
    actor = Actor(obs_dim, act_dim).to(device)
    critic = Critic(obs_dim).to(device)
    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=config.lr,
        eps=1e-5,
    )

    # Rollout buffer.
    buffer = RolloutBuffer(config.n_steps, n_envs, obs_dim, act_dim, device)

    # Derived training constants.
    batch_size: int = config.n_steps * n_envs
    minibatch_size: int = batch_size // config.n_minibatches
    n_updates: int = config.total_env_steps // batch_size

    # Episode-return tracking for TensorBoard.
    ep_return_buf: list[float] = []
    ep_length_buf: list[int] = []
    ep_return_running: list[float] = []  # per-env running accum
    ep_length_running: list[int] = []

    # Initialise vectorised env.
    obs_np, _ = envs.reset(seed=seed)
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
    done = torch.zeros(n_envs, device=device)
    ep_ret = [0.0] * n_envs
    ep_len = [0] * n_envs

    next_eval_step: int = config.eval_cadence_env_steps
    global_step: int = 0
    start_time: float = time.time()

    for update in range(1, n_updates + 1):
        # ----------------------------------------------------------------
        # Anneal learning rate (linear decay to 0).
        # ----------------------------------------------------------------
        frac = 1.0 - (update - 1.0) / n_updates
        lr_now = frac * config.lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # ----------------------------------------------------------------
        # Rollout collection.
        # ----------------------------------------------------------------
        buffer.reset()
        for _step in range(config.n_steps):
            global_step += n_envs

            with torch.no_grad():
                action, log_prob, _ = actor.get_action_and_log_prob(obs)
                value = critic(obs).squeeze(-1)

            action_np = action.cpu().numpy()
            obs_next_np, reward_np, terminated_np, truncated_np, infos = envs.step(action_np)
            done_next_np = np.logical_or(terminated_np, truncated_np)

            reward = torch.tensor(reward_np, dtype=torch.float32, device=device)
            done_next = torch.tensor(done_next_np, dtype=torch.float32, device=device)

            buffer.store(obs, action, log_prob, reward, done, value)

            obs = torch.tensor(obs_next_np, dtype=torch.float32, device=device)
            done = done_next

            # Track episode returns.
            for i in range(n_envs):
                ep_ret[i] += float(reward_np[i])
                ep_len[i] += 1
                if done_next_np[i]:
                    ep_return_buf.append(ep_ret[i])
                    ep_length_buf.append(ep_len[i])
                    ep_ret[i] = 0.0
                    ep_len[i] = 0

        # ----------------------------------------------------------------
        # GAE computation.
        # ----------------------------------------------------------------
        buffer.compute_gae(obs, done, critic, config.gamma, config.gae_lambda)

        # ----------------------------------------------------------------
        # PPO update.
        # ----------------------------------------------------------------
        b_obs, b_actions, b_log_probs_old, b_advantages, b_returns = buffer.get_flat()

        # Normalise advantages over the full batch.
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Indices for minibatch permutation.
        indices = np.arange(batch_size)

        # Accumulate update statistics for TensorBoard.
        value_losses: list[float] = []
        policy_losses: list[float] = []
        entropy_losses: list[float] = []
        approx_kls: list[float] = []
        clip_fracs: list[float] = []
        kl_early_stop: bool = False

        for _epoch in range(config.n_epochs):
            if kl_early_stop:
                break
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                mb_obs = b_obs[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_log_probs_old = b_log_probs_old[mb_idx]
                mb_advantages = b_advantages[mb_idx]
                mb_returns = b_returns[mb_idx]

                _, new_log_prob, entropy = actor.get_action_and_log_prob(mb_obs, mb_actions)
                new_value = critic(mb_obs).squeeze(-1)

                log_ratio = new_log_prob - mb_log_probs_old
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    approx_kls.append(approx_kl)
                    clip_frac = ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                    clip_fracs.append(clip_frac)

                # KL early stop.
                if config.target_kl is not None and approx_kl > config.target_kl:
                    kl_early_stop = True
                    break

                # Policy loss (clipped surrogate).
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped).
                v_loss_unclipped = (new_value - mb_returns) ** 2
                v_clipped = b_returns[mb_idx] + torch.clamp(
                    new_value - b_returns[mb_idx],
                    -config.clip_coef,
                    config.clip_coef,
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Entropy bonus.
                entropy_loss = entropy.mean()

                loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    config.max_grad_norm,
                )
                optimizer.step()

                value_losses.append(v_loss.item())
                policy_losses.append(pg_loss.item())
                entropy_losses.append(entropy_loss.item())

        # ----------------------------------------------------------------
        # TensorBoard logging.
        # ----------------------------------------------------------------
        sps = int(global_step / (time.time() - start_time))

        if ep_return_buf:
            writer.add_scalar("train/episodic_return", np.mean(ep_return_buf), global_step)
            writer.add_scalar("train/episodic_length", np.mean(ep_length_buf), global_step)
            ep_return_buf.clear()
            ep_length_buf.clear()

        writer.add_scalar("losses/value_loss", np.mean(value_losses) if value_losses else 0.0, global_step)
        writer.add_scalar("losses/policy_loss", np.mean(policy_losses) if policy_losses else 0.0, global_step)
        writer.add_scalar("losses/entropy", np.mean(entropy_losses) if entropy_losses else 0.0, global_step)
        writer.add_scalar("losses/approx_kl", np.mean(approx_kls) if approx_kls else 0.0, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clip_fracs) if clip_fracs else 0.0, global_step)
        writer.add_scalar("charts/learning_rate", lr_now, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.flush()

        # ----------------------------------------------------------------
        # Optional eval pass.
        # ----------------------------------------------------------------
        if eval_fn is not None and global_step >= next_eval_step:
            eval_metrics = eval_fn(actor)
            for k, v in eval_metrics.items():
                writer.add_scalar(k, v, global_step)
            writer.flush()
            next_eval_step += config.eval_cadence_env_steps

    return actor, critic


__all__ = ["Actor", "Critic", "RolloutBuffer", "train_ppo"]
