"""The pre-registered protocol (``experiments/protocol.yaml``) and the test gate.

:class:`Protocol` holds the approved constants, the seed-block registry, and
the burnt-seed list. :func:`Protocol.seeds` is the only way the harness gets
world seeds, so a run cannot draw from a burnt range or past the end of a
block. :func:`check_test_gate` refuses a sealed-split run unless every arm's
config hash is listed in ``experiments/frozen.json``, and refuses a second
test run of the same (arm, variant) without a rerun reason.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

EXPERIMENTS = Path(__file__).resolve().parents[3] / "experiments"
PROTOCOL_PATH = EXPERIMENTS / "protocol.yaml"
FROZEN_PATH = EXPERIMENTS / "frozen.json"

LEARNED_KINDS = ("planner", "seasonal_ets")


class GateError(RuntimeError):
    """A run the protocol does not allow."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def config_sha256(config: dict) -> str:
    return sha256_bytes(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())


def arm_sha256(config: dict) -> str:
    """What the arm is, whatever it is called: the config hash without its name."""
    return config_sha256({k: v for k, v in config.items() if k != "name"})


@dataclass(frozen=True)
class Protocol:
    raw: dict
    sha256: str
    path: Path

    @classmethod
    def load(cls, path: str | Path = PROTOCOL_PATH) -> Protocol:
        path = Path(path)
        text = path.read_bytes()
        raw = yaml.safe_load(text)
        budgets = raw["budgets"]
        learned = budgets["learned"]["trials"] * budgets["learned"]["worlds_per_trial"]
        textbook = budgets["textbook"]["trials"] * budgets["textbook"]["worlds_per_trial"]
        if textbook < budgets["textbook_multiplier"] * learned:
            raise ValueError(f"{path}: textbook budget {textbook} < {budgets['textbook_multiplier']} x learned {learned}")
        return cls(raw=raw, sha256=sha256_bytes(text), path=path)

    @property
    def sesoi(self) -> float:
        """SESOI as a score difference (the yaml holds points)."""
        return self.raw["sesoi_pts"] / 100

    @property
    def stats(self) -> dict:
        return self.raw["statistics"]

    def workers(self) -> int:
        return int(os.environ.get("PS_WORKERS", self.raw["compute"]["max_workers"]))

    def budget(self, kind: str) -> dict:
        """``{trials, worlds_per_trial}`` for an arm kind."""
        b = self.raw["budgets"]
        return dict(b["learned"] if kind in LEARNED_KINDS else b["textbook"])

    def world_params(self, family: str) -> dict:
        """The declared parameters of a world family (``regimes`` for ``sample_world``)."""
        return dict(self.raw["world_families"][family].get("params") or {})

    def search_space(self, kind: str) -> dict:
        return self.raw["search_spaces"][kind]

    def is_sealed(self, split: str) -> bool:
        return bool(self.raw["seed_blocks"].get(split, {}).get("sealed", False))

    def burnt(self, seed: int) -> str | None:
        for b in self.raw["burnt"]:
            if b["first"] <= seed <= b["last"]:
                return b["why"]
        return None

    def seeds(self, split: str, n: int, offset: int = 0) -> list[int]:
        """``n`` world seeds of ``split`` starting ``offset`` into its block."""
        block = self.raw["seed_blocks"][split]
        if offset < 0 or offset + n > block["size"]:
            raise GateError(f"{split} block holds {block['size']} seeds; asked for {offset}..{offset + n - 1}")
        seeds = [block["first"] + offset + i for i in range(n)]
        hit = [(s, self.burnt(s)) for s in seeds if self.burnt(s)]
        if hit:
            raise GateError(f"burnt seed {hit[0][0]}: {hit[0][1]}")
        return seeds


def load_frozen(path: str | Path = FROZEN_PATH) -> dict:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {"configs": {}}


def freeze(config: dict, note: str, protocol: Protocol, path: str | Path = FROZEN_PATH) -> str:
    """List ``config`` in ``frozen.json``; return its hash."""
    frozen = load_frozen(path)
    sha = config_sha256(config)
    frozen["configs"][sha] = {
        "arm": config["name"],
        "kind": config["kind"],
        "protocol_sha256": protocol.sha256,
        "note": note,
        "config": config,
    }
    Path(path).write_text(json.dumps(frozen, indent=1, sort_keys=True) + "\n")
    return sha


def check_test_gate(
    protocol: Protocol,
    split: str,
    configs: list[dict],
    variant: str,
    previous_test_runs: list[dict],
    rerun_reason: str | None = None,
    frozen_path: str | Path = FROZEN_PATH,
) -> None:
    """Raise :class:`GateError` unless a run on ``split`` is allowed.

    Non-sealed splits always pass. On a sealed split every config must be
    frozen under the current protocol, and an (arm, variant) that already has
    a test-eval log line, under its name or under another name with the same
    :func:`arm_sha256`, needs ``rerun_reason``.
    """
    if not protocol.is_sealed(split):
        return
    frozen = load_frozen(frozen_path)["configs"]
    for config in configs:
        sha = config_sha256(config)
        if sha not in frozen:
            raise GateError(f"refusing {split} run: config of arm {config['name']!r} ({sha[:12]}) is not frozen")
        if frozen[sha]["protocol_sha256"] != protocol.sha256:
            raise GateError(f"refusing {split} run: arm {config['name']!r} was frozen under another protocol")
        seen = [r for r in previous_test_runs
                if (config["name"] in r.get("arms", []) or arm_sha256(config) in r.get("arm_sha256", {}).values())
                and r.get("variant") == variant and r.get("status") == "kept"]
        if seen and not rerun_reason:
            raise GateError(
                f"refusing {split} run: arm {config['name']!r} already has {len(seen)} test run(s) on {variant!r}; "
                "pass a rerun reason"
            )
