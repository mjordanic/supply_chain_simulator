# Per-stage transition probabilities and a `dead` stage with terminal-by-default cycle

Status: Superseded by ADR 0017 (product life-cycle removed for the PoC; see `TODO.md`)

Lifecycle stages are `[introduction, growth, maturity, decline, dead]`. Transitions are governed by a per-stage dictionary `stage_change_probs: dict[str, float | Distribution]` keyed by *current* stage (not a single scalar): a fashion category needs fast `decline → dead`, a staple category needs slow `maturity → decline`, and one scalar can't express both. `dead` carries `stage_multipliers["dead"] = 0.05` (5%-of-baseline trickle demand) and a default `dead → introduction = 0.003` per tick (~one-year mean comeback time) — products *can* re-launch, rarely.

We chose this over (a) **strict-terminal** (no `dead → introduction` at all) which is too rigid for occasional re-launches and useful comeback dynamics in long simulations; (b) **slowing the existing scalar** — still cycles unrealistically and conflates "fast decay" with "fast hype"; (c) **per-product fixed durations** — loses the stochastic-transition flavour of the world model and is harder for an LLM to author meaningfully. Per-`Ware` overrides on `init_stage` and `stage_change_probs` are LLM-authored per category. Setting `dead → introduction = 0` on a `Ware` recovers strict-terminal behaviour for that product.
