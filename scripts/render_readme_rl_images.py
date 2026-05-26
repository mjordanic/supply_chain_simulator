"""Render the RL-section KPI bar chart for the README.

Outputs:
    docs/images/rl_vs_baseline_kpis.png

Mirrors notebooks/06-compare_rl_vs_baseline.ipynb section 9 (business KPI
side-by-side bars). Loads the latest checkpoint under
runs/fashion_run/checkpoints/, runs the CRN-paired evaluator on the held-out
32-seed set against fashion_retail_250, and plots aggregate KPI means.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.rl.agents.ppo import Actor
from src.rl.configs.default import RLConfig
from src.rl.encoders import action_dim, observation_dim
from src.rl.eval import build_eval_seeds, evaluate
from src.sim.policy import OrderUpToPolicy
from src.sim.world_loader import load_world

OUT_DIR = REPO / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT = "fashion_run"
WORLD_ARCHETYPE = "fashion_retail_250"
WORLD_CACHE_PATH = REPO / "data" / "worlds" / WORLD_ARCHETYPE / "world.json"

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 130


def _pick_checkpoint() -> Path:
    ckpt_dir = REPO / "runs" / EXPERIMENT / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("actor_step*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {ckpt_dir}")
    return ckpts[-1]


def _build_rl_policy_fn(actor: Actor, device: torch.device):
    def fn(obs_np):
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t = torch.tanh(actor.get_distribution(obs_t).mean)
        return action_t.squeeze(0).cpu().numpy().astype(np.float32)

    return fn


def main() -> None:
    config = RLConfig()
    checkpoint = _pick_checkpoint()
    print(f"using checkpoint: {checkpoint}")

    catalog, base_template, world_market_params, disruption_params = load_world(
        archetype=WORLD_ARCHETYPE,
        cache_path=str(WORLD_CACHE_PATH) if WORLD_CACHE_PATH.exists() else None,
        delivery_lag=config.delivery_lag,
        holding_rate=config.holding_rate,
        order_fee=config.order_fee,
        K_active=config.K_active,
        synthetic_fallback=True,
        synthetic_catalog_size=config.K_catalog,
    )

    obs_dim = observation_dim(config.K_active)
    act_dim = action_dim(config.K_active)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(obs_dim, act_dim).to(device)
    actor.load_state_dict(torch.load(checkpoint, map_location=device))
    actor.eval()

    rl_policy_fn = _build_rl_policy_fn(actor, device)

    def baseline_factory():
        return OrderUpToPolicy()

    eval_specs = build_eval_seeds(
        catalog,
        base_template,
        config,
        market_params=world_market_params,
        disruption_params=disruption_params,
        lifecycle_params=None,
    )
    print(f"built {len(eval_specs)} eval specs")

    agg = evaluate(
        rl_policy_fn=rl_policy_fn,
        baseline_policy_factory=baseline_factory,
        eval_specs=eval_specs,
        config=config,
    )
    print(f"paired_uplift = {agg['eval/paired_uplift']:.2f}")
    print(f"win_rate      = {agg['eval/win_rate']:.2%}")

    kpis = [
        ("service_level", "Service level", True),
        ("stockout_rate", "Stockout rate", False),
        ("inventory_turnover", "Inventory turnover", True),
        ("mean_price_pct_of_msrp", "Mean price % MSRP", None),
        ("revenue", "Revenue", True),
    ]

    fig, axes = plt.subplots(1, len(kpis), figsize=(3.2 * len(kpis), 4.5))
    for ax, (key, label, higher_better) in zip(axes, kpis):
        rl_val = agg[f"eval/rl_{key}"]
        bl_val = agg[f"eval/baseline_{key}"]
        bars = ax.bar(["RL", "Baseline"], [rl_val, bl_val], color=["tab:blue", "tab:orange"])
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.3, axis="y")
        for b, v in zip(bars, [rl_val, bl_val]):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{v:.3g}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        if higher_better is True:
            ax.set_xlabel("higher is better")
        elif higher_better is False:
            ax.set_xlabel("lower is better")

    fig.suptitle(
        f"RL vs OrderUpToPolicy — mean KPIs across {len(eval_specs)} CRN-paired seeds",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = OUT_DIR / "rl_vs_baseline_kpis.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
