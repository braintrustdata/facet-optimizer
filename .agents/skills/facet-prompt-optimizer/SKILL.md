---
name: facet-prompt-optimizer
description: Use when turning production Braintrust facet executions into a versioned prompt YAML plus a ground-truth dataset, then running `bt eval` against `brain-facet-1` on `braintrustproxy.com/v1` to improve the facet prompt without storing the prompt text in dataset rows.
---

# Purpose

Use this skill when the goal is to improve a customer-owned facet prompt by:

1. sampling positive and negative production executions
2. extracting the stable facet prompt into a versioned YAML file
3. building dataset rows with only `facet_name` and `preprocessed_text`
4. uploading those rows into a Braintrust dataset
5. iterating on the prompt with `bt eval`

# Workflow

1. Sample roots with [scripts/sample_facet_roots.py](../../../scripts/sample_facet_roots.py).
2. Build the versioned prompt YAML and local ground-truth seed dataset with [scripts/build_ground_truth_dataset.py](../../../scripts/build_ground_truth_dataset.py).
3. Review the generated dataset locally and replace seeded `expected` values with real ground truth where needed.
4. Upload the reviewed dataset with [scripts/upload_ground_truth_dataset.py](../../../scripts/upload_ground_truth_dataset.py).
5. Run [eval_prompt.py](../../../eval_prompt.py) with `bt eval` to score prompt changes against the dataset.

# Guardrails

- Dataset rows should contain only the factored substrate: `facet_name` and `preprocessed_text`.
- Keep the prompt itself in a versioned YAML file under `prompts/`.
- Treat production facet values as seed labels, not automatic gold labels.
- If prompt factoring cannot identify a single varying message, rerun with `--input-message-index` and inspect the generated summary.
- Keep the sampling SQL window explicit; do not quietly mix time ranges.

