# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Behavioral rules and the skill/agent registry live in `AGENTS.md` — keep them there, not here.

@AGENTS.md

## This project

A small, hackable **multi-echelon** supply-chain simulator. A scenario is a validated directed
acyclic graph of typed nodes — factories produce, intermediate nodes (warehouses / shops) hold
inventory and route orders across multiple upstream suppliers, and demand sinks generate the only
new cash in the system — all sharing one stochastic world (regional supply/demand, seasonal cycles,
disruption events). It ships a family of textbook inventory policies as
baselines, plus three add-ons: an LLM world generator, an Optuna hyperparameter tuner, and a PPO
reinforcement-learning stack. It is a demo project — the goal is to be readable and easy to extend,
not production-grade.

Deep reference: `README.md`, `CONTEXT.md` (domain & architecture glossary), and `docs/adr/`.
