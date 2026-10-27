"""Track T1 workflow on the ``hidden_regimes`` world family.

    T1="uv run python -m src.perfect_supplier.t1 --out <dir>"
    $T1 g1 --scores <dev eval dir>   # gate G1 on a harness dev eval (Bayes / oracle planner vs textbook)
    $T1 collect                      # behaviour-policy shop banks: train, tune (HMM order), dev (G2)
    $T1 fit --seeds 0 1 2            # joint JEPA and direct, one model per training seed
    $T1 fit-hmm --states 2 3 4       # Baum-Welch on train sales; order chosen on the tune bank
    $T1 g2                           # dev forecast metrics, JEPA vs direct, the gate G2
    $T1 arms --seeds 0 1 2           # planner arm configs; the sampler of each learned forecaster
                                     # (native paths or a copula) picked on tune worlds

Every step is a line in ``experiments/log/T1-hidden-regimes.jsonl``. The
closed-loop steps (tune, eval, report, freeze) run through the harness CLI
with ``--world-family hidden_regimes``.

G2 scores forecasts on dev shops run by behaviour policies, against true
demand, at every ``G2_EVERY``-th decision day. Each forecaster draws
``G2_PATHS`` paths, and three pinball losses are taken over
``G2_LEVELS``, each divided by the product's context-mean scale:

- ``daily``: every (day, product) of the next 35 days;
- ``lead_time``: each product's demand summed over the next ``lag + 7`` days;
- ``shop_lead_time``: the shop total over the next ``lag + 7`` days.

The gate rule is fixed in :data:`G2_RULE` before the metrics are computed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from joblib import Parallel, delayed

from src.perfect_supplier.bench import log, stats
from src.perfect_supplier.bench.protocol import Protocol
from src.perfect_supplier.behavior import sample_behavior
from src.perfect_supplier.sim import simulate
from src.perfect_supplier.world import HISTORY_DAYS, demand_path, sample_world

TRACK = "T1-hidden-regimes"
FAMILY = "hidden_regimes"
BANKS = {"train": 2000, "tune": 200, "dev": 300}
G2_EVERY = 7
G2_PATHS = 256
G2_LEVELS = (0.1, 0.5, 0.9, 0.95, 0.99)
G1_PTS = 1.0
G1_RULE = ("G1 passes when the known-model Bayes planner's mean dev margin capture beats both the protocol "
           "reference (the textbook arm with the best tune score) and the best textbook arm on dev by >= 1.00 pt")
G2_RULE = ("G2 passes when JEPA minus direct (each averaged over its training seeds) on the dev lead_time "
           "pinball has a world-paired 95% BCa interval entirely below 0, and the point estimates on daily and "
           "shop_lead_time are <= 0")


def _workers() -> int:
    return int(os.environ.get("PS_WORKERS", 3))


def _shop(seed: int, split: str, regimes: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world = sample_world(seed, split, family=FAMILY, regimes=regimes)
    frame = simulate(world, sample_behavior(seed)).frame
    P = world.n_products
    demand = demand_path(world)
    sales = np.concatenate([demand[:HISTORY_DAYS], frame["sales"].to_numpy().reshape(-1, P)])
    cens = np.concatenate([np.zeros((HISTORY_DAYS, P), bool), frame["censored"].to_numpy(bool).reshape(-1, P)])
    return sales, cens, demand


def g1(scores_dir: Path, protocol: Protocol, log_dir=log.LOG_DIR) -> str:
    """Score gate G1 on a harness dev eval; one ``kind: gate`` log line."""
    import pandas as pd

    from src.perfect_supplier.bench.arms import TEXTBOOK_KINDS
    from src.perfect_supplier.bench.evaluate import score_matrix
    from src.perfect_supplier.bench.report import best_textbook

    configs = json.loads((scores_dir / "arms.json").read_text())
    m = score_matrix(pd.read_parquet(scores_dir / "scores.parquet"))
    textbook = [c["name"] for c in configs if c["kind"] in TEXTBOOK_KINDS]
    refs = {"by tune score": best_textbook(configs), "best on dev": m[textbook].mean().idxmax()}
    lines = [f"# T1 gate G1 ({len(m)} dev worlds)", "", f"Rule (fixed before the run): {G1_RULE}.", "",
             "| arm − reference | reference | diff, pts [95% BCa] | win rate |", "| --- | --- | --- | --- |"]
    result = {}
    for arm in ("planner_bayes", "planner_oracle"):
        for how, ref in refs.items():
            d = (m[arm] - m[ref]).to_numpy()
            lo, hi = stats.bca_ci(d, level=0.95, n_resamples=protocol.stats["resamples"])
            result[f"{arm} - {how}"] = {"ref": ref, "diff_pts": 100 * float(d.mean()), "ci95": [100 * lo, 100 * hi]}
            lines.append(f"| {arm} − {how} | `{ref}` | {100 * d.mean():+.3f} [{100 * lo:+.3f}, {100 * hi:+.3f}] "
                         f"| {(d > 0).mean():.0%} |")
    passed = all(result[f"planner_bayes - {h}"]["diff_pts"] >= G1_PTS for h in refs)
    lines += ["", "Mean margin capture: " + ", ".join(f"`{a}` {v:.4f}" for a, v in
                                                       m.mean().sort_values(ascending=False).items()), "",
              f"**G1 {'passes' if passed else 'fails'}.**", ""]
    text = "\n".join(lines)
    (scores_dir / "g1.md").write_text(text)
    log.append(TRACK, {"kind": "gate", "arm": "G1", "world_family": FAMILY, "variant": "base", "split": "dev",
                       "config": {"rule": G1_RULE, "scores": str(scores_dir),
                                  "regimes": protocol.world_params(FAMILY)},
                       "result": {"gate": "G1", "passed": passed, **result},
                       "artifacts": [{"path": str(scores_dir / "g1.md"),
                                      "sha256": log.file_sha256(scores_dir / "g1.md")}],
                       "status": "kept"}, protocol, log_dir)
    return text


def collect(out: Path, protocol: Protocol, sizes: dict[str, int] = BANKS, log_dir=log.LOG_DIR) -> None:
    from src.perfect_supplier.joint import ShopBank

    regimes = protocol.world_params(FAMILY)
    for split, n in sizes.items():
        seeds = protocol.seeds(split, n)
        entry = {"kind": "fit", "arm": "behaviour_log", "world_family": FAMILY, "variant": "base", "split": split,
                 "seeds": {"first": seeds[0], "count": n}, "note": "behaviour-policy shop bank (sample_behavior)"}
        with log.logged_run(TRACK, entry, protocol, log_dir):
            rows = Parallel(n_jobs=_workers())(delayed(_shop)(s, split, regimes) for s in seeds)
            bank = ShopBank(*(np.stack([r[i] for r in rows]).astype(t) for i, t in
                              ((0, np.float32), (1, bool), (2, np.float32))), np.asarray(seeds))
            path = out / f"bank-{split}.npz"
            bank.save(path)
            entry["result"] = {"censored_share": float(bank.censored[:, HISTORY_DAYS:].mean())}
            entry["artifacts"] = [{"path": str(path), "sha256": log.file_sha256(path)}]
        print(split, path, entry["result"], flush=True)


def _fit_one(out: Path, arm: str, seed: int, steps: int, protocol: Protocol, log_dir) -> None:
    import torch

    from src.perfect_supplier import joint

    torch.set_num_threads(1)
    bank = joint.ShopBank.load(out / "bank-train.npz")
    rng = np.random.default_rng(0)
    val = rng.random(len(bank.sales)) < 0.1  # the train block's own 10% for early stopping
    part = lambda m: joint.ShopBank(bank.sales[m], bank.censored[m], bank.demand[m], bank.world_seed[m])  # noqa: E731
    config = joint.JointConfig(jepa=arm == "jepa", steps=steps, seed=seed)
    name = f"{arm}-s{seed}"
    entry = {"kind": "fit", "arm": name, "world_family": FAMILY, "variant": "base", "split": "train",
             "seeds": {"first": int(bank.world_seed[0]), "count": len(bank.world_seed)},
             "config": {"model": "joint", **{k: v for k, v in vars(config).items()}}}
    with log.logged_run(TRACK, entry, protocol, log_dir):
        model = joint.fit(part(~val), part(val), config, log_path=out / "models" / f"{name}_fit.json")
        path = out / "models" / f"{name}.pt"
        joint.save(model, path)
        entry["result"] = {"val_nll": json.loads((out / "models" / f"{name}_fit.json").read_text())["history"][-1]}
        entry["artifacts"] = [{"path": str(path), "sha256": log.file_sha256(path)}]


def fit(out: Path, seeds: list[int], steps: int, protocol: Protocol, log_dir=log.LOG_DIR) -> None:
    jobs = [(arm, s) for s in seeds for arm in ("jepa", "direct")]
    Parallel(n_jobs=min(_workers(), len(jobs)))(delayed(_fit_one)(out, a, s, steps, protocol, log_dir)
                                               for a, s in jobs)


def fit_hmm(out: Path, states: list[int], protocol: Protocol, log_dir=log.LOG_DIR) -> None:
    """Fit one HMM per order on train sales; keep the order with the best tune-bank log-likelihood."""
    from src.perfect_supplier.joint import ShopBank
    from src.perfect_supplier.regimes import _forward, fit_hmm_on_sales, save_hmm, shop_log_sales

    train, tune = ShopBank.load(out / "bank-train.npz"), ShopBank.load(out / "bank-tune.npz")
    entry = {"kind": "fit", "arm": "hmm", "world_family": FAMILY, "variant": "base", "split": "train",
             "seeds": {"first": int(train.world_seed[0]), "count": len(train.world_seed)},
             "config": {"states": states, "selection": "tune-bank log-likelihood"}}
    with log.logged_run(TRACK, entry, protocol, log_dir):
        scores = {}
        for k in states:
            hmm = fit_hmm_on_sales(list(zip(train.sales, train.censored)), k)
            ll = sum(np.log(_forward(hmm.emission(shop_log_sales(s, c)[0]), hmm.trans, hmm.start)[1]).sum()
                     for s, c in zip(tune.sales, tune.censored))
            scores[k] = float(ll)
            save_hmm(hmm, out / "models" / f"hmm-k{k}.json")
            print(f"hmm k={k}: tune log-likelihood {ll:.1f}", flush=True)
        best = max(scores, key=scores.get)
        path = out / "models" / "hmm.json"
        path.write_bytes((out / "models" / f"hmm-k{best}.json").read_bytes())
        entry["result"] = {"tune_loglik": scores, "chosen_states": best}
        entry["artifacts"] = [{"path": str(path), "sha256": log.file_sha256(path)}]


def _pinball(draws: np.ndarray, truth: np.ndarray) -> float:
    """Mean pinball over :data:`G2_LEVELS` of the empirical quantiles of ``draws`` (first axis)."""
    q = np.quantile(draws, G2_LEVELS, axis=0)
    diff = truth[None] - q
    lv = np.asarray(G2_LEVELS).reshape((-1,) + (1,) * truth.ndim)
    return float(np.maximum(lv * diff, (lv - 1) * diff).mean())


def _world_metrics(forecasters: dict, bank, i: int, protocol: Protocol) -> dict[str, dict[str, float]]:
    seed = int(bank.world_seed[i])
    world = sample_world(seed, "dev", family=FAMILY, regimes=protocol.world_params(FAMILY))
    span = world.delivery_lag + 7
    made = {name: make(world) for name, make in forecasters.items()}
    out = {name: {"daily": [], "lead_time": [], "shop_lead_time": []} for name in made}
    for day in range(0, world.n_days - 35, G2_EVERY):
        end = HISTORY_DAYS + day
        obs = SimpleNamespace(day=day, weekday=end % 7, sales=bank.sales[i, : end + 1].astype(np.int64),
                              censored=bank.censored[i, : end + 1], n_products=world.n_products)
        truth = bank.demand[i, end + 1 : end + 36]  # (H, P)
        scale = obs.sales[-35:].mean(axis=0) + 1.0
        for name, fc in made.items():
            paths = fc.paths(obs, G2_PATHS, np.random.default_rng([seed, day]))  # (M, H, P)
            out[name]["daily"].append(_pinball(paths / scale, truth / scale))
            out[name]["lead_time"].append(_pinball(paths[:, :span].sum(1) / (scale * span),
                                                   truth[:span].sum(0) / (scale * span)))
            out[name]["shop_lead_time"].append(_pinball(paths[:, :span].sum((1, 2)) / (scale.sum() * span),
                                                        np.asarray(truth[:span].sum() / (scale.sum() * span))))
    return {name: {k: float(np.mean(v)) for k, v in m.items()} for name, m in out.items()}


def _maker(kind: str, path: str | None = None):
    def make(world):
        if kind == "bayes":
            from src.perfect_supplier.regimes import BayesRegimeForecaster

            return BayesRegimeForecaster(world)
        if kind == "hmm":
            from src.perfect_supplier.regimes import HmmForecaster, load_hmm

            return HmmForecaster(load_hmm(path))
        if kind == "moving_average":
            from src.perfect_supplier.forecast import MovingAverageForecaster
            from src.perfect_supplier.regimes import CopulaSampler

            return CopulaSampler(MovingAverageForecaster(), 0.0)
        import torch

        from src.perfect_supplier import joint

        torch.set_num_threads(1)
        return joint.JointForecaster(joint.load(path), name=kind)

    return make


def g2(out: Path, seeds: list[int], protocol: Protocol, n_worlds: int, log_dir=log.LOG_DIR) -> str:
    from src.perfect_supplier.joint import ShopBank

    bank = ShopBank.load(out / "bank-dev.npz")
    n = min(n_worlds, len(bank.world_seed))
    forecasters = {f"{a}-s{s}": _maker(a, str(out / "models" / f"{a}-s{s}.pt")) for a in ("jepa", "direct")
                   for s in seeds}
    forecasters |= {"hmm": _maker("hmm", str(out / "models" / "hmm.json")), "bayes": _maker("bayes"),
                    "moving_average": _maker("moving_average")}
    entry = {"kind": "diagnostic", "arm": "G2", "world_family": FAMILY, "variant": "base", "split": "dev",
             "seeds": {"first": int(bank.world_seed[0]), "count": n}, "config": {
                 "rule": G2_RULE, "every": G2_EVERY, "paths": G2_PATHS, "levels": G2_LEVELS,
                 "models": {k: log.file_sha256(out / "models" / f"{k}.pt") for k in forecasters if "-s" in k}}}
    with log.logged_run(TRACK, entry, protocol, log_dir):
        rows = Parallel(n_jobs=_workers())(delayed(_world_metrics)(forecasters, bank, i, protocol) for i in range(n))
        metrics = ("daily", "lead_time", "shop_lead_time")
        per = {name: {m: np.array([r[name][m] for r in rows]) for m in metrics} for name in forecasters}
        for a in ("jepa", "direct"):
            per[a] = {m: np.mean([per[f"{a}-s{s}"][m] for s in seeds], axis=0) for m in metrics}
        lines = [f"# T1 gate G2: dev forecast metrics ({n} dev shops, behaviour policies, vs true demand)", "",
                 f"Rule (fixed before the run): {G2_RULE}.", "",
                 "Mean pinball over levels " + ", ".join(map(str, G2_LEVELS)) + ", scaled; lower is better.", "",
                 "| forecaster | " + " | ".join(metrics) + " |", "| --- |" + " --- |" * len(metrics)]
        for name in ["jepa", "direct", *forecasters]:
            lines.append(f"| {name} | " + " | ".join(f"{per[name][m].mean():.5f}" for m in metrics) + " |")
        lines += ["", "| JEPA − direct | mean | 95% BCa | JEPA better in |", "| --- | --- | --- | --- |"]
        result = {}
        for m in metrics:
            d = per["jepa"][m] - per["direct"][m]
            lo, hi = stats.bca_ci(d, level=0.95, n_resamples=protocol.stats["resamples"])
            result[m] = {"diff": float(d.mean()), "ci95": [float(lo), float(hi)], "win": float((d < 0).mean())}
            lines.append(f"| {m} | {d.mean():+.5f} | [{lo:+.5f}, {hi:+.5f}] | {(d < 0).mean():.0%} of shops |")
        passed = (result["lead_time"]["ci95"][1] < 0 and result["daily"]["diff"] <= 0
                  and result["shop_lead_time"]["diff"] <= 0)
        lines += ["", f"**G2 {'passes' if passed else 'fails'}.**", ""]
        text = "\n".join(lines)
        (out / "g2.md").write_text(text)
        entry["result"] = {"gate": "G2", "passed": passed, "jepa_minus_direct": result,
                           "means": {k: {m: float(v[m].mean()) for m in metrics} for k, v in per.items()}}
        entry["artifacts"] = [{"path": str(out / "g2.md"), "sha256": log.file_sha256(out / "g2.md")}]
    return text


SAMPLERS = ({}, {"copula_rho": 0.0}, {"copula_rho": 0.5})


def _planner(name: str, fc: str, model: dict | None = None, **params) -> dict:
    config = {"name": name, "kind": "planner", "variant": "base", "world_family": FAMILY,
              "params": {"forecaster": fc, **params}}
    return config | ({"model": model} if model else {})


def pick_sampler(config: dict, protocol: Protocol, n_worlds: int = 32, log_dir=log.LOG_DIR) -> dict:
    """The best of :data:`SAMPLERS` for a planner arm, by mean score on ``n_worlds`` tune worlds."""
    from src.perfect_supplier.bench.evaluate import run_episodes

    seeds = protocol.seeds("tune", n_worlds)
    grid = [{**config, "name": f"{config['name']}-g{i}", "params": {**config["params"], **g}}
            for i, g in enumerate(SAMPLERS)]
    entry = {"kind": "tune", "arm": config["name"], "world_family": FAMILY, "variant": "base", "split": "tune",
             "seeds": {"first": seeds[0], "count": n_worlds}, "config": {"base": config, "grid": SAMPLERS},
             "budget": {"trials": len(SAMPLERS), "worlds_per_trial": n_worlds,
                        "episodes": len(SAMPLERS) * n_worlds}}
    with log.logged_run(TRACK, entry, protocol, log_dir):
        frame = run_episodes(grid, seeds, "tune", "base", _workers(), FAMILY, protocol.world_params(FAMILY))
        scores = frame.groupby("arm")["score"].mean()
        best = int(np.argmax([scores[g["name"]] for g in grid]))
        entry["result"] = {"metric": "margin_capture", "scores": {str(SAMPLERS[i]): float(scores[g["name"]])
                                                                  for i, g in enumerate(grid)},
                           "chosen": SAMPLERS[best]}
    return SAMPLERS[best]


def arms(out: Path, seeds: list[int], protocol: Protocol, n_worlds: int = 32, log_dir=log.LOG_DIR) -> list[dict]:
    """Write ``arms/*.json``: JEPA and direct per training seed, the HMM, Bayes, oracle, and classical planners.

    Each learned forecaster's sampler is picked on tune worlds with its
    seed-0 model (JEPA, direct) or its one model (HMM), and kept for every seed.
    """
    from src.perfect_supplier.bench.arms import model_spec

    (out / "arms").mkdir(parents=True, exist_ok=True)
    configs = []
    for fc in ("jepa", "direct"):
        spec = lambda s: model_spec(out / "models" / f"{fc}-s{s}.pt", "joint")  # noqa: E731
        chosen = pick_sampler(_planner(f"{fc}-s{seeds[0]}", fc, spec(seeds[0])), protocol, n_worlds, log_dir)
        configs += [_planner(f"{fc}-s{s}", fc, spec(s), **chosen) | {"seed_group": fc} for s in seeds]
    hmm = _planner("planner_hmm", "hmm", model_spec(out / "models" / "hmm.json", "hmm"))
    hmm["params"].update(pick_sampler(hmm, protocol, n_worlds, log_dir))
    configs += [hmm, _planner("planner_bayes", "bayes"), _planner("planner_oracle", "oracle"),
                _planner("planner_moving_average", "moving_average"),
                {"name": "seasonal_ets", "kind": "seasonal_ets", "variant": "base", "world_family": FAMILY,
                 "params": {}}]
    for c in configs:
        (out / "arms" / f"{c['name']}.json").write_text(json.dumps(c, indent=1))
    return configs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.perfect_supplier.t1")
    parser.add_argument("--out", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    g1p = sub.add_parser("g1")
    g1p.add_argument("--scores", type=Path, required=True)
    sub.add_parser("collect")
    f = sub.add_parser("fit")
    f.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    f.add_argument("--steps", type=int, default=12_000)
    h = sub.add_parser("fit-hmm")
    h.add_argument("--states", type=int, nargs="+", default=[2, 3, 4])
    g = sub.add_parser("g2")
    g.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    g.add_argument("--n-worlds", type=int, default=BANKS["dev"])
    a = sub.add_parser("arms")
    a.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a.add_argument("--n-worlds", type=int, default=32)
    args = parser.parse_args(argv)
    protocol = Protocol.load()
    (args.out / "models").mkdir(parents=True, exist_ok=True)
    if args.command == "g1":
        print(g1(args.scores, protocol))
    elif args.command == "collect":
        collect(args.out, protocol)
    elif args.command == "fit":
        fit(args.out, args.seeds, args.steps, protocol)
    elif args.command == "fit-hmm":
        fit_hmm(args.out, args.states, protocol)
    elif args.command == "g2":
        print(g2(args.out, args.seeds, protocol, args.n_worlds))
    elif args.command == "arms":
        for c in arms(args.out, args.seeds, protocol, args.n_worlds):
            print(c["name"], c["params"])


if __name__ == "__main__":
    main()
