"""Where simulation output lives and how to read it back.

Layout under one run root (default ``runs/perfect-supplier``)::

    timeseries/split=<split>/method=<method>/part-<n>.parquet   one row per day x product
    summary/<split>-<method>-<n>.parquet                         one row per (world, method)

``load_timeseries`` reads any slice by split, method, and world seed without
loading the rest; ``load_summary`` concatenates every summary part.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path("runs/perfect-supplier")


def _part_name(root: Path, split: str, method: str, part: int) -> tuple[Path, Path]:
    ts = root / "timeseries" / f"split={split}" / f"method={method}" / f"part-{part:05d}.parquet"
    sm = root / "summary" / f"{split}-{method}-{part:05d}.parquet"
    return ts, sm


def write_part(
    root: str | Path,
    split: str,
    method: str,
    part: int,
    frames: Sequence[pd.DataFrame],
    summaries: Sequence[dict],
) -> None:
    """Write one batch of episodes. ``split`` and ``method`` become partition keys."""
    root = Path(root)
    ts, sm = _part_name(root, split, method, part)
    ts.parent.mkdir(parents=True, exist_ok=True)
    sm.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(frames, ignore_index=True).drop(columns=["split", "method"])
    frame.to_parquet(ts, index=False)
    pd.DataFrame(list(summaries)).assign(split=split, method=method).to_parquet(sm, index=False)


def clear(root: str | Path, split: str, method: str) -> None:
    """Delete one (split, method) slice so a rerun does not mix with old parts."""
    root = Path(root)
    for path in (root / "timeseries" / f"split={split}" / f"method={method}").glob("part-*.parquet"):
        path.unlink()
    for path in (root / "summary").glob(f"{split}-{method}-*.parquet"):
        path.unlink()


def load_timeseries(
    root: str | Path = DEFAULT_ROOT,
    *,
    split: str | Iterable[str] | None = None,
    method: str | Iterable[str] | None = None,
    world_seed: int | Iterable[int] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read day x product rows, filtered on split, method, and world seed.

    Each (split, method) partition is read on its own and the results are
    concatenated, so decision columns that only one method writes survive
    (missing columns are NaN for the other methods).
    """

    def wanted(value):
        if value is None:
            return None
        return {str(v) for v in ([value] if isinstance(value, (str, int)) else value)}

    splits, methods = wanted(split), wanted(method)
    seeds = None if world_seed is None else (
        [world_seed] if isinstance(world_seed, int) else [int(s) for s in world_seed]
    )
    filters = [("world_seed", "in", seeds)] if seeds is not None else None
    frames = []
    for split_dir in sorted((Path(root) / "timeseries").glob("split=*")):
        split_name = split_dir.name.removeprefix("split=")
        if splits is not None and split_name not in splits:
            continue
        for method_dir in sorted(split_dir.glob("method=*")):
            method_name = method_dir.name.removeprefix("method=")
            if methods is not None and method_name not in methods:
                continue
            cols = None if columns is None else [c for c in columns if c not in ("split", "method")]
            part = pd.read_parquet(method_dir, filters=filters, columns=cols)
            if len(part):
                frames.append(part.assign(split=split_name, method=method_name))
    if not frames:
        return pd.DataFrame(columns=columns)
    frame = pd.concat(frames, ignore_index=True)
    return frame if columns is None else frame[[c for c in columns if c in frame.columns]]


def load_summary(root: str | Path = DEFAULT_ROOT) -> pd.DataFrame:
    parts = sorted((Path(root) / "summary").glob("*.parquet"))
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
