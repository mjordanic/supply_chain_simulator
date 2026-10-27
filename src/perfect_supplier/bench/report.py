"""Markdown report for one paired evaluation directory (``scores.parquet`` + ``arms.json``).

Sections: arm means with the cost breakdown; the primary family (each learned
arm against the reference, Holm across the family, protocol labels); the
secondary family (pairs, Holm within it); training seeds (two-level bootstrap)
for arms that share a ``seed_group``; strata (BH-adjusted, exploratory). Margin
capture differences are in points; cost reduction sits next to each.

The families and strata come from ``comparisons`` in the protocol. The reference
is the textbook arm with the best tune score, so it is chosen on tuning data,
never on the worlds it is compared on. A reference, primary, or secondary
family passed by the caller makes the report exploratory, and says so. Every
report is a log line.

The primary family spans every (track, world family, variant), so a single
report's Holm is provisional and prints no labels. :func:`write_pooled_primary`
pools the primary comparisons of the protocol's declared primary cells from
their logged test evals, runs Holm once over the whole family, and writes the
final labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.perfect_supplier.bench import log, stats
from src.perfect_supplier.bench.arms import TEXTBOOK_KINDS
from src.perfect_supplier.bench.evaluate import score_matrix
from src.perfect_supplier.bench.protocol import Protocol


def best_textbook(configs: list[dict]) -> str:
    tuned = [c for c in configs if c["kind"] in TEXTBOOK_KINDS and c.get("tune_score") is not None]
    if not tuned:
        raise ValueError("no tuned textbook arm with a tune score; pass the reference explicitly")
    return max(tuned, key=lambda c: c["tune_score"])["name"]


def role(config: dict) -> str:
    """The name the protocol's comparison families use for an arm."""
    return config.get("params", {}).get("forecaster", "") if config["kind"] == "planner" else config["kind"]


def arms_by_role(configs: list[dict]) -> dict[str, list[str]]:
    """Role -> arms in the report (a seed group counts once)."""
    out: dict[str, list[str]] = {}
    for c in configs:
        arms = out.setdefault(role(c), [])
        if (c.get("seed_group") or c["name"]) not in arms:
            arms.append(c.get("seed_group") or c["name"])
    return out


def breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    """Margin capture and where the rest of the margin went, per arm (mean over worlds)."""
    f = frame.copy()
    margin = f["total_demand"] * (f["price"] - f["unit_cost"])
    f["lost_sales_share"] = f["lost_units"] * (f["price"] - f["unit_cost"]) / margin
    f["fee_share"] = f["fees"] / margin
    f["holding_share"] = f["holding_cost"] / margin
    cols = ["score", "lost_sales_share", "fee_share", "holding_share", "order_days", "fill_rate_demand", "mean_on_hand"]
    return f.groupby("arm")[cols].mean().sort_values("score", ascending=False)


def markdown_table(df: pd.DataFrame, index: bool = True) -> str:
    if index:
        df = df.reset_index()
    rows = [[str(v) for v in df.columns]] + [[str(v) for v in r] for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * len(rows[0])]
                     + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def _pts(x: float) -> str:
    return f"{100 * x:+.3f}"


def _ci(ci) -> str:
    return f"[{_pts(ci[0])}, {_pts(ci[1])}]"


def _vectors(m: pd.DataFrame, groups: dict[str, list[str]], name: str) -> np.ndarray:
    return m[groups[name]].mean(axis=1).to_numpy() if name in groups else m[name].to_numpy()


def _family_table(family: list[stats.Comparison], labels: bool = True) -> list[str]:
    lines = ["| comparison | diff, pts [95% BCa] | Holm-adjusted interval | cost reduction [95%] | win rate "
             "| Wilcoxon p | Holm p | TOST p | CVaR10, pts |" + (" label |" if labels else ""),
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |" + (" --- |" if labels else "")]
    for c in family:
        lines.append(
            f"| {c.arm} − {c.ref} | {_pts(c.diff)} {_ci(c.ci)} | {_ci(c.ci_adj)} | "
            f"{100 * c.cost_reduction:+.1f}% [{100 * c.cost_reduction_ci[0]:+.1f}, {100 * c.cost_reduction_ci[1]:+.1f}] | "
            f"{c.win_rate:.0%} | {c.p_wilcoxon:.2g} | {c.p_adj:.2g} | {c.p_tost:.2g} | {_pts(c.cvar10)} |"
            + (f" {c.label} |" if labels else ""))
    return lines


def _seed_groups(configs: list[dict]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for c in configs:
        if c.get("seed_group"):
            groups.setdefault(c["seed_group"], []).append(c["name"])
    return groups


def write_report(
    scores_dir: str | Path,
    protocol: Protocol,
    out: str | Path,
    track: str,
    *,
    reference: str | None = None,
    primary: list[str] | None = None,
    secondary: list[tuple[str, str]] | None = None,
    strata: list[str] | None = None,
    title: str = "Paired evaluation",
    n_resamples: int | None = None,
    log_dir: str | Path = log.LOG_DIR,
) -> str:
    scores_dir = Path(scores_dir)
    frame = pd.read_parquet(scores_dir / "scores.parquet")
    configs = json.loads((scores_dir / "arms.json").read_text())
    st = protocol.stats
    n_resamples = n_resamples or st["resamples"]
    level, alpha, sesoi = st["level"], st["alpha"], protocol.sesoi
    comp = protocol.raw["comparisons"]
    exploratory = reference is not None or primary is not None or secondary is not None
    by_tune_score = reference is None
    reference = reference or best_textbook(configs)
    by_role = arms_by_role(configs)
    if primary is None:
        primary = [a for r in comp["primary"]["arms"] for a in by_role.get(r, [])]
    if secondary is None:
        secondary = [(a, b) for ra, rb in comp["secondary"]["pairs"]
                     for a in by_role.get(ra, []) for b in by_role.get(rb, [])]
    if strata is None:
        strata = comp["strata"]["keys"]
    tag = " (exploratory: families chosen by the caller)" if exploratory else ""
    groups = _seed_groups(configs)
    m = score_matrix(frame)
    split = frame["split"].iloc[0] if "split" in frame else "?"
    worlds = frame.drop_duplicates("world_seed").set_index("world_seed").reindex(m.index)

    def run_family(pairs):
        return _decide([(m, groups, a, b) for a, b in pairs], protocol, n_resamples)

    lines = [
        f"# {title}", "",
        f"- split `{split}`{' (sealed test)' if protocol.is_sealed(split) else ' (not held-out)'}, "
        f"{len(m)} worlds, seeds {m.index.min()}–{m.index.max()}, variant `{frame['variant'].iloc[0]}`",
        f"- protocol sha256 `{protocol.sha256[:16]}…`; SESOI {protocol.raw['sesoi_pts']} pts; "
        f"BCa {n_resamples} resamples; Holm at α = {alpha}",
        f"- reference: `{reference}`" + (" (best tune score among tuned textbook arms)"
                                          if by_tune_score else ""),
        "- families: " + ("**exploratory**, chosen by the caller; not the pre-registered ones" if exploratory
                          else "pre-registered (`comparisons` in the protocol)"),
        "- margin capture is the primary metric; cost reduction is `1 − (1 − s_arm)/(1 − s_ref)`, a ratio of means",
        "", "## Arms", "",
        markdown_table(breakdown(frame).round(4)), "",
        f"## Primary family{tag} (provisional: Holm within this report, not over the whole pre-registered "
        "family; final labels come from the pooled primary report)", "",
        *_family_table(run_family([(a, reference) for a in primary if a != reference]), labels=False), "",
    ]
    if secondary:
        lines += [f"## Secondary family{tag}", "", *_family_table(run_family(secondary)), ""]
    if groups:
        lines += ["## Training seeds (two-level bootstrap: worlds and seeds resampled)", "",
                  "| group − reference | mean, pts [95%] | per-seed estimates, pts |", "| --- | --- | --- |"]
        for g, arms in groups.items():
            d = m[arms].to_numpy() - m[[reference]].to_numpy()
            mean, lo, hi = stats.two_level_ci(d, level=level, n_resamples=n_resamples)
            lines.append(f"| {g} − {reference} | {_pts(mean)} [{_pts(lo)}, {_pts(hi)}] | "
                         + ", ".join(f"{a}: {_pts(d[:, i].mean())}" for i, a in enumerate(arms)) + " |")
        lines.append("")
    if strata:
        rows = []
        for a in [p for p in primary if p != reference]:
            d = pd.Series(_vectors(m, groups, a) - _vectors(m, groups, reference), index=m.index)
            for key in strata:
                for value, part in d.groupby(worlds[key]):
                    rows.append({"comparison": f"{a} − {reference}", "stratum": f"{key} = {value:.6g}",
                                 "n": len(part), "diff, pts": round(100 * part.mean(), 3),
                                 "p (t)": stats.ttest_p(part.to_numpy())})
        if rows:
            t = pd.DataFrame(rows)
            t["BH q"] = stats.bh(t["p (t)"].to_numpy())
            t["p (t)"] = t["p (t)"].map(lambda p: f"{p:.2g}")
            t["BH q"] = t["BH q"].map(lambda p: f"{p:.2g}")
            lines += [f"## Strata (exploratory, BH at q = {st['bh_q']})", "", markdown_table(t, index=False), ""]
    text = "\n".join(lines)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text)
    log.append(track, {"kind": "report", "scores": scores_dir, "split": split, "reference": reference,
                       "primary": primary, "secondary": secondary, "strata": strata,
                       "families": "exploratory" if exploratory else "pre-registered",
                       "artifacts": [{"path": str(out), "sha256": log.file_sha256(out)}], "status": "kept"},
               protocol, log_dir)
    return text


def _decide(pairs: list[tuple[pd.DataFrame, dict, str, str]], protocol: Protocol,
            n_resamples: int) -> list[stats.Comparison]:
    """Compare each (scores, seed groups, arm, ref) and apply Holm and the labels across all of them."""
    st, sesoi = protocol.stats, protocol.sesoi
    fam = [stats.compare(_vectors(m, g, a), _vectors(m, g, b), (a, b), sesoi, level=st["level"],
                         n_resamples=n_resamples) for m, g, a, b in pairs]
    diffs = [_vectors(m, g, a) - _vectors(m, g, b) for m, g, a, b in pairs]
    return stats.decide(fam, diffs, sesoi, alpha=st["alpha"], n_resamples=n_resamples) if fam else []


def write_pooled_primary(
    protocol: Protocol,
    out: str | Path,
    track: str,
    *,
    title: str = "Primary family (pooled)",
    n_resamples: int | None = None,
    log_dir: str | Path = log.LOG_DIR,
) -> str:
    """Holm once over every primary comparison of the protocol's declared primary cells.

    A cell is a (track, world family, variant) listed under
    ``comparisons.primary.cells``; its scores are every kept test eval logged
    for it, the latest run of an arm winning. Test evals of undeclared cells
    are ignored. Raises :class:`ValueError` while a declared cell has no test
    eval or lacks a primary arm of the protocol.
    """
    n_resamples = n_resamples or protocol.stats["resamples"]
    cells: dict[tuple, list[dict]] = {(c["track"], c["world_family"], c["variant"]): []
                                      for c in protocol.raw["comparisons"]["primary"]["cells"]}
    for row in log.read(log_dir=log_dir):
        cell = (row.get("track"), row.get("world_family"), row.get("variant"))
        if row.get("kind") == "test-eval" and row.get("status") == "kept" and cell in cells:
            cells[cell].append(row)
    missing = [f"{'/'.join(cell)}: no test eval" for cell, rows in cells.items() if not rows]
    pairs, names = [], []
    for cell, rows in cells.items():
        if not rows:
            continue
        frames = [pd.read_parquet(Path(a["path"])) for row in rows for a in row["artifacts"]
                  if Path(a["path"]).name == "scores.parquet"]
        configs = {c["name"]: c for row in rows for a in row["artifacts"] if Path(a["path"]).name == "arms.json"
                   for c in json.loads(Path(a["path"]).read_text())}
        frame = pd.concat(frames).drop_duplicates(["arm", "world_seed"], keep="last")
        m, groups = score_matrix(frame), _seed_groups(list(configs.values()))
        reference, by_role = best_textbook(list(configs.values())), arms_by_role(list(configs.values()))
        for r in protocol.raw["comparisons"]["primary"]["arms"]:
            if r not in by_role:
                missing.append(f"{'/'.join(cell)}: {r}")
            for a in by_role.get(r, []):
                pairs.append((m, groups, a, reference))
                names.append("/".join(cell))
    if missing:
        raise ValueError(f"pooled primary family incomplete; missing: {', '.join(missing)}")
    family = _decide(pairs, protocol, n_resamples)
    for c, cell in zip(family, names):
        c.arm = f"{cell}: {c.arm}"
    lines = [f"# {title}", "",
             f"- protocol sha256 `{protocol.sha256[:16]}…`; SESOI {protocol.raw['sesoi_pts']} pts; "
             f"BCa {n_resamples} resamples; Holm at α = {protocol.stats['alpha']} over all {len(family)} comparisons",
             f"- cells (from the protocol): {', '.join('/'.join(c) for c in cells)}; sealed test evals only; "
             "final labels", "",
             *_family_table(family), ""]
    text = "\n".join(lines)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text)
    log.append(track, {"kind": "report", "cells": ["/".join(c) for c in cells], "split": "test", "families": "pre-registered",
                       "primary": [c.arm for c in family], "result": {c.arm: {"label": c.label, "p_holm": c.p_adj} for c in family},
                       "artifacts": [{"path": str(out), "sha256": log.file_sha256(out)}], "status": "kept"},
               protocol, log_dir)
    return text
