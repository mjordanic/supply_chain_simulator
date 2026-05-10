"""Top-level package for the retail market simulator.

The codebase is split into two main sub-packages:

- ``src.sim``  — the core discrete-event simulator: scenarios, stores,
  policies, markets, events, item lifecycle, freshness curves,
  distributions, and the data exporter that persists run results.
- ``src.llm``  — the LLM-driven world builder pipeline. Generates
  catalog + market + store templates for an archetype string using
  structured OpenAI completions, validated through Pydantic schemas.

A third sub-package, ``src.forecasting``, holds a small ARIMA-based
sales predictor that is not currently wired into the live simulator
but is kept available for downstream/offline analysis.

This ``__init__.py`` intentionally stays empty so the simulator and
LLM modules can be imported on demand (``from src.sim.runner import
Runner``) without pulling in optional dependencies (matplotlib,
statsmodels, openai) at import time.
"""
