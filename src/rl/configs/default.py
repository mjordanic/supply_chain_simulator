"""Default RL configuration dataclass.

``RLConfig`` captures every knob the agent, env, sampler, and eval will
read. It is a frozen dataclass so configs can be safely shared across
vector envs without aliasing bugs.

All defaults match the PRD specification exactly.  Instantiating with no
arguments ``RLConfig()`` yields a fully-valid config ready for a training
run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.sim.distributions import Distribution


# ---------------------------------------------------------------------------
# Helper: lightweight Uniform placeholder so the default factory for
# capacity_dist / balance_dist does not import any heavy dependencies at
# import time.  The actual runtime sampling goes through the Distribution
# ABC in src.sim.distributions.
# ---------------------------------------------------------------------------


def _default_capacity_dist() -> Distribution:
    """Return ``Uniform(150, 400)`` — default per-episode capacity distribution."""
    from src.sim.distributions import Uniform

    return Uniform(150, 400)


def _default_balance_dist() -> Distribution:
    """Return ``Uniform(15000, 40000)`` — default per-episode balance distribution."""
    from src.sim.distributions import Uniform

    return Uniform(15000, 40000)


@dataclass(frozen=True)
class RLConfig:
    """Immutable configuration for a single RL training run.

    Fields are grouped by concern:

    - **Episode shape** — basic sizing parameters for one episode.
    - **Episode randomisation** — distributions sampled fresh each reset.
    - **Simulator knobs** — passed verbatim to StoreTemplate / Scenario.
    - **PPO hyperparameters** — CleanRL-compatible knobs.
    - **Training driver** — vec-env count and step budget.
    - **Eval** — cadence, seed count, and offset.
    - **Output** — TensorBoard log dir, checkpoint dir, experiment name.
    - **World** — archetype label and optional cache path.
    """

    # ------------------------------------------------------------------
    # Episode shape
    # ------------------------------------------------------------------
    episode_length: int = 180
    """Number of ticks per episode (half-year at daily resolution)."""

    K_active: int = 5
    """Number of active SKUs per episode (sampled from the catalog)."""

    K_catalog: int = 100
    """Total size of the product universe from which K_active are sampled."""

    # ------------------------------------------------------------------
    # Episode randomisation
    # ------------------------------------------------------------------
    capacity_dist: Distribution = field(default_factory=_default_capacity_dist)
    """Per-episode store capacity sampled from this distribution."""

    balance_dist: Distribution = field(default_factory=_default_balance_dist)
    """Per-episode opening cash balance sampled from this distribution."""

    # ------------------------------------------------------------------
    # Simulator knobs
    # ------------------------------------------------------------------
    delivery_lag: int = 3
    """Fixed order delivery lead time in ticks."""

    holding_rate: float = 0.01
    """Per-tick holding cost rate (fraction of unit cost per unit)."""

    order_fee: float = 50.0
    """Fixed fee charged per non-zero order (discourages micro-orders)."""

    # ------------------------------------------------------------------
    # PPO hyperparameters
    # ------------------------------------------------------------------
    lr: float = 3e-4
    """Adam learning rate."""

    n_steps: int = 128
    """Number of env steps collected per update per env (rollout length)."""

    n_epochs: int = 10
    """Number of PPO optimisation epochs per update."""

    n_minibatches: int = 4
    """Number of minibatches to split the rollout buffer into per epoch."""

    clip_coef: float = 0.2
    """PPO clipping coefficient epsilon."""

    ent_coef: float = 0.0
    """Entropy bonus coefficient (0 = disabled for the first cut)."""

    vf_coef: float = 0.5
    """Value function loss coefficient."""

    gae_lambda: float = 0.95
    """GAE lambda for advantage estimation."""

    gamma: float = 0.99
    """Discount factor."""

    max_grad_norm: float = 0.5
    """Global gradient clipping norm."""

    target_kl: Optional[float] = None
    """KL divergence threshold for early-stopping PPO updates (None = disabled)."""

    # ------------------------------------------------------------------
    # Training driver
    # ------------------------------------------------------------------
    n_envs: int = 8
    """Number of parallel vectorised envs in the sync vec-env."""

    total_env_steps: int = 1_000_000
    """Total environment steps budget for a training run."""

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------
    eval_cadence_env_steps: int = 50_000
    """Run CRN evaluation every this many env steps."""

    n_eval_seeds: int = 32
    """Number of held-out CRN seeds used per evaluation pass."""

    eval_seed_offset: int = 10_000_000
    """First eval seed — kept well above the training seed space."""

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    tb_log_dir: str = "runs"
    """Root directory for TensorBoard SummaryWriter output."""

    checkpoint_dir: str = "runs"
    """Root directory under which checkpoint subdirs are created."""

    experiment_name: str = "rl_ppo"
    """Label used for the TensorBoard sub-directory and checkpoint prefix."""

    # ------------------------------------------------------------------
    # Order-up-to decoder parameters
    # ------------------------------------------------------------------
    target_centre_lead_times: int = 15
    """Order-up-to target (in lead-times) when ``order_raw = 0``.

    Matches ``OrderUpToPolicy.S / rate`` at the default policy kwargs
    (``delivery_lag + safety_lead_ticks + cover_horizon_ticks = 3 + 2 + 10``).
    """

    target_half_span_lead_times: int = 15
    """Half-width of the action range around the centre (in lead-times).

    The raw action ``order_raw ∈ [-1, 1]`` maps to a target of
    ``target_centre ± target_half_span`` lead-times before clipping.
    """

    target_max_lead_times: int = 30
    """Upper clip for the order-up-to target (in lead-times).

    Prevents pathologically large target requests on extreme positive actions.
    """

    # ------------------------------------------------------------------
    # World
    # ------------------------------------------------------------------
    world_archetype: str = "rl_train"
    """Archetype label passed to the world builder / cache loader."""

    world_cache_path: Optional[str] = None
    """Optional explicit path to a cached ``world.json``; overrides auto-lookup."""
