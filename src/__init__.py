"""Top-level package for the retail market simulator.

The codebase is split into four sub-packages:

- ``src.sim``    — the core discrete-event simulator: scenarios, stores,
  policies, markets, events, item lifecycle, freshness curves,
  distributions, and the data exporter that persists run results.
- ``src.llm``    — the LLM-driven world builder pipeline. Generates
  catalog + market + store templates for an archetype string using
  structured OpenAI completions, validated through Pydantic schemas.
- ``src.tuning`` — Optuna-based hyperparameter tuner for ``Policy``
  subclasses; consumes ``src.sim`` rollout primitives.
- ``src.rl``     — PPO training stack that wraps the simulator as a
  Gymnasium environment and evaluates against the textbook baseline
  with Common Random Numbers.

This ``__init__.py`` intentionally stays empty so the sub-packages can
be imported on demand (``from src.sim.runner import Runner``) without
pulling in optional dependencies (matplotlib, openai, torch) at
import time.
"""
