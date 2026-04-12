---
name: facet-prompt-optimizer
description: Use when bootstrapping a facet ground-truth dataset, running initial facet evals, or later optimizing a facet prompt in this repo.
---

# Instructions

Follow the workflow in [README.md](../../../README.md). Treat the README as the source of truth for commands, defaults, and sequencing.

# Agent guardrails

- Do not duplicate workflow instructions here. Update the README first when the workflow changes.
- Use the latest `bt` CLI and call `bt eval` directly for evals.
- Keep customer-owned prompts and generated datasets in local ignored paths such as `.local/facet-optimizer`.
- Do not treat production facet outputs as ground truth; use generated or reviewed `expected` values.
- Do not proceed from dataset creation to prompt optimization until the user has reviewed the dataset balance and initial evals.
