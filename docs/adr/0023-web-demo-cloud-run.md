# Streamlit CRN lab on Cloud Run; slim image; sibling of src.sim

Status: Accepted.

The web demo is a **sibling consumer of `src.sim`** (same ADR 0010 rule as `src.tuning/` and `src.rl/`): it loads a setup directory, overlays a few knobs in memory, and calls `Runner(..., policy_overrides={"shop-1": pol})`. It does not grow a second tick loop and does not import `src.rl`, `src.tuning`, or `src.llm`.

**UI.** Streamlit + Plotly. Compare batch-runs the selected textbook families, then the browser auto-plays the focal Run Log on the DAG (Plotly animation frames + Play / Skip to end). Not a websocket, not a live `Simulation.tick()` stream.

**Host.** Cloud Run scale-to-zero, unauthenticated, 512Mi–1Gi / 1 CPU / concurrency 1. **Not Fargate** — an always-on task plus a load balancer is tens of dollars/month idle. The demo image is **not** `uv sync` of the root project: torch/transformers would make a multi-GB image. `Dockerfile.web` installs only Streamlit + the `src.sim` runtime deps and copies `src/sim`, `src/web`, and the two small setup dirs.
