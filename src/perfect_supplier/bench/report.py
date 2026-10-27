"""Markdown report for one paired evaluation directory (``scores.parquet`` + ``arms.json``).

Sections: arm means with the cost breakdown; the primary family (each named arm
against the reference, Holm across the family, protocol labels); the secondary
family (named pairs, Holm within it); training seeds (two-level bootstrap) for
arms that share a ``seed_group``; strata (BH-adjusted, exploratory). Margin
capture differences are in points; cost reduction sits next to each.

The reference defaults to the textbook arm with the best tune score, so it is
chosen on tuning data, never on the worlds it is compared on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.perfect_supplier.bench import stats
from src.perfect_supplier.bench.arms import TEXTBOOK_KINDS
from src.perfect_supplier.bench.evaluate import score_matrix
from src.perfect_supplier.bench.protocol import Protocol


def best_textbook(configs: list[dict]) -> str:
    tuned = [c for c in configs if c["kind"] in TEXTBOOK_KINDS and c.get("tune_score") is not None]
    if not tuned:
        raise ValueError("no tuned textbook arm with a tune score; pass the reference explicitly")
    return max(tuned, key=lambda c: c["tune_score"])["name"]


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


def _family_table(family: list[stats.Comparison]) -> list[str]:
    lines = ["| comparison | diff, pts [95% BCa] | Holm-adjusted interval | cost reduction [95%] | win rate "
             "| Wilcoxon p | Holm p | TOST p | CVaR10, pts | label |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in family:
        lines.append(
            f"| {c.arm} − {c.ref} | {_pts(c.diff)} {_ci(c.ci)} | {_ci(c.ci_adj)} | "
            f"{100 * c.cost_reduction:+.1f}% [{100 * c.cost_reduction_ci[0]:+.1f}, {100 * c.cost_reduction_ci[1]:+.1f}] | "
            f"{c.win_rate:.0%} | {c.p_wilcoxon:.2g} | {c.p_adj:.2g} | {c.p_tost:.2g} | {_pts(c.cvar10)} | {c.label} |")
    return lines


def write_report(
    scores_dir: str | Path,
    protocol: Protocol,
    out: str | Path,
    *,
    reference: str | None = None,
    primary: list[str] | None = None,
    secondary: list[tuple[str, str]] | None = None,
    strata: list[str] | None = None,
    title: str = "Paired evaluation",
    n_resamples: int | None = None,
) -> str:
    scores_dir = Path(scores_dir)
    frame = pd.read_parquet(scores_dir / "scores.parquet")
    configs = json.loads((scores_dir / "arms.json").read_text())
    st = protocol.stats
    n_resamples = n_resamples or st["resamples"]
    level, alpha, sesoi = st["level"], st["alpha"], protocol.sesoi
    reference = reference or best_textbook(configs)
    groups: dict[str, list[str]] = {}
    for c in configs:
        if c.get("seed_group"):
            groups.setdefault(c["seed_group"], []).append(c["name"])
    m = score_matrix(frame)
    split = frame["split"].iloc[0] if "split" in frame else "?"
    worlds = frame.drop_duplicates("world_seed").set_index("world_seed").reindex(m.index)
    primary = primary or list(dict.fromkeys(
        c.get("seed_group") or c["name"] for c in configs if c["kind"] not in TEXTBOOK_KINDS + ("never_order",)))

    def run_family(pairs):
        fam = [stats.compare(_vectors(m, groups, a), _vectors(m, groups, b), (a, b), sesoi, level=level,
                             n_resamples=n_resamples) for a, b in pairs]
        diffs = [_vectors(m, groups, a) - _vectors(m, groups, b) for a, b in pairs]
        return stats.decide(fam, diffs, sesoi, alpha=alpha, n_resamples=n_resamples) if fam else []

    lines = [
        f"# {title}", "",
        f"- split `{split}`{' (sealed test)' if protocol.is_sealed(split) else ' (not held-out)'}, "
        f"{len(m)} worlds, seeds {m.index.min()}–{m.index.max()}, variant `{frame['variant'].iloc[0]}`",
        f"- protocol sha256 `{protocol.sha256[:16]}…`; SESOI {protocol.raw['sesoi_pts']} pts; "
        f"BCa {n_resamples} resamples; Holm at α = {alpha}",
        f"- reference: `{reference}`" + (" (best tune score among tuned textbook arms)"
                                          if reference == best_textbook(configs) else ""),
        "- margin capture is the primary metric; cost reduction is `1 − (1 − s_arm)/(1 − s_ref)`, a ratio of means",
        "", "## Arms", "",
        markdown_table(breakdown(frame).round(4)), "",
        "## Primary family", "",
        *_family_table(run_family([(a, reference) for a in primary if a != reference])), "",
    ]
    if secondary:
        lines += ["## Secondary family", "", *_family_table(run_family(secondary)), ""]
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
    return text
