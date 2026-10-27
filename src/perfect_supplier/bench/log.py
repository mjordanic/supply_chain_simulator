"""Append-only JSONL experiment log, one file per track: ``experiments/log/<track>.jsonl``.

The first line of a new file records the protocol hash. Every later line is
one variant run, written by the harness itself, failed and abandoned runs
included (:func:`logged_run`). Files from several tracks concatenate; dedupe
on ``run_id``.
"""

from __future__ import annotations

import json
import secrets
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.perfect_supplier.bench.protocol import EXPERIMENTS, Protocol, sha256_bytes

LOG_DIR = EXPERIMENTS / "log"
REPO = EXPERIMENTS.parent


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") + "-" + secrets.token_hex(2)


def _git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, check=False).stdout


def git_state() -> dict:
    """HEAD and a hash of everything that differs from it: tracked changes plus untracked, non-ignored files."""
    diff = _git("diff", "HEAD")
    for name in _git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0"):
        path = REPO / name.decode()
        if name and path.is_file():
            diff += b"\0untracked\0" + name + b"\0" + path.read_bytes()
    return {
        "git_sha": _git("rev-parse", "HEAD").decode().strip() or None,
        "git_dirty_diff_sha256": sha256_bytes(diff) if diff else None,
    }


def file_sha256(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def log_path(track: str, log_dir: str | Path = LOG_DIR) -> Path:
    return Path(log_dir) / f"{track}.jsonl"


def append(track: str, entry: dict, protocol: Protocol, log_dir: str | Path = LOG_DIR) -> dict:
    """Write ``entry`` with the run id, track, git state, and protocol hash filled in.

    Git state already in ``entry`` (captured when a run started) is kept.
    """
    path = log_path(track, log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        header = {"kind": "protocol", "track": track, "protocol_sha256": protocol.sha256,
                  "protocol_path": str(protocol.path.relative_to(REPO)) if protocol.path.is_relative_to(REPO)
                  else str(protocol.path)}
        path.write_text(json.dumps(header) + "\n")
    full = {"run_id": new_run_id(), "track": track, **git_state(), "protocol_sha256": protocol.sha256, **entry}
    with path.open("a") as f:
        f.write(json.dumps(full, default=_json_default) + "\n")
    return full


def read(track: str | None = None, log_dir: str | Path = LOG_DIR) -> list[dict]:
    """Every run line (headers skipped), across all tracks unless ``track`` is given."""
    paths = [log_path(track, log_dir)] if track else sorted(Path(log_dir).glob("*.jsonl"))
    out, seen = [], set()
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            row = json.loads(line)
            if row.get("kind") == "protocol" or row.get("run_id") in seen:
                continue
            seen.add(row.get("run_id"))
            out.append(row)
    return out


@contextmanager
def logged_run(track: str, entry: dict, protocol: Protocol, log_dir: str | Path = LOG_DIR):
    """Log ``entry`` when the block ends: ``kept`` with ``entry['result']`` filled in, or ``failed``.

    The block fills ``entry`` in place (result, artifacts, note).
    """
    t0 = time.time()
    entry.update(git_state())  # the code that runs, not the tree when the run ends
    try:
        yield entry
    except BaseException as exc:
        append(track, {**entry, "status": "failed", "wall_s": round(time.time() - t0, 1),
                       "note": f"{type(exc).__name__}: {exc}"}, protocol, log_dir)
        raise
    entry.setdefault("status", "kept")
    append(track, {**entry, "wall_s": round(time.time() - t0, 1)}, protocol, log_dir)


def _json_default(o):
    import numpy as np

    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")
