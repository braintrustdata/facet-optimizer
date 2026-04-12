# facet-optimizer

`facet-optimizer` turns production Braintrust facet executions into a prompt-optimization loop:

1. sample positive and negative facet executions from a project
2. factor the stable facet prompt into a versioned YAML file
3. build a local ground-truth-ready dataset with only `facet_name` and `preprocessed_text`
4. upload the reviewed dataset to Braintrust
5. run `bt eval` against `brain-facet-1` on `https://braintrustproxy.com/v1`
6. edit the prompt YAML and rerun until the scores improve

The prompt stays in a versioned file like `prompts/facet_<name>_v1.yaml`. The dataset rows do not store the prompt text.

## Install

1. Install the Braintrust CLI.
   Official docs: https://www.braintrust.dev/docs/reference/cli
2. Authenticate the CLI.

```bash
bt auth login
bt status
```

3. Install Python dependencies with `uv`.

```bash
uv sync
cp .env.example .env
```

Set `BRAINTRUST_API_KEY` in `.env`. The same key is used for:
- Braintrust SDK calls
- `braintrustproxy.com/v1` model calls

## Repository Layout

- `scripts/sample_facet_roots.py`: sample positive and negative automation traces
- `scripts/build_ground_truth_dataset.py`: fetch source LLM spans, factor the prompt, and build the seed dataset
- `scripts/upload_ground_truth_dataset.py`: upload the reviewed dataset into Braintrust
- `eval_prompt.py`: run `bt eval` against `brain-facet-1`
- `.agents/skills/facet-prompt-optimizer/SKILL.md`: reusable Codex skill for this workflow

## Step 1: Sample Production Facet Executions

This samples automation traces where the facet is present (`positive`) and absent or `no_match` (`negative`).

```bash
uv run python scripts/sample_facet_roots.py \
  --prefer-profile \
  --profile ai-replit \
  --project-id ee38835c-d0d1-423b-abfb-cfd2c96e52a6 \
  --facet-name emergent-issues-20260409 \
  --positive-limit 300 \
  --negative-limit 100 \
  --output-dir /tmp/facet-optimizer/emergent-issues-20260409
```

This writes:
- `sample_manifest.json`
- `sampled_roots.json`
- `positive_roots.json`
- `negative_roots.json`
- `queries/*.sql`

## Step 2: Factor the Prompt and Build a Seed Dataset

This pulls the source `brain-facet-1` LLM spans, infers the stable prompt structure, writes a versioned prompt YAML, and creates dataset rows with only the factored input substrate:

```json
{
  "input": {
    "facet_name": "emergent-issues-20260409",
    "preprocessed_text": "..."
  }
}
```

Run:

```bash
uv run python scripts/build_ground_truth_dataset.py \
  --manifest /tmp/facet-optimizer/emergent-issues-20260409/sample_manifest.json \
  --output-root . \
  --prompt-version v1 \
  --llm-model brain-facet-1
```

This writes:
- `prompts/facet_<name>_v1.yaml`
- `datasets/facet_<name>_v1_seed.json`
- `datasets/facet_<name>_v1_seed.jsonl`
- `artifacts/<name>_v1/source_llm_spans.json`
- `artifacts/<name>_v1/factoring_summary.json`

If the script cannot infer a single varying message, rerun with `--input-message-index`.

## Step 3: Review the Ground Truth Locally

The generated dataset is a seed. Its `expected` values are initialized from the production facet value. Review and edit them before upload.

Useful fields:
- `expected`: the label you want the prompt to produce
- `metadata.seed_expected`: the original production value
- `metadata.source_output_text`: the raw source model output, stripped of `<think>` tags
- `metadata.marked_positive`: whether the sampled automation span came from the positive bucket

## Step 4: Upload the Reviewed Dataset

```bash
uv run python scripts/upload_ground_truth_dataset.py \
  --input-file datasets/facet_emergent-issues-20260409_v1_seed.jsonl \
  --project "Facet Optimizer" \
  --dataset "emergent-issues-ground-truth-v1"
```

The upload script upserts by row `id`.

## Step 5: Run the Prompt Eval

Point the eval at the dataset and prompt YAML, then run it through the Braintrust proxy using `brain-facet-1`.

```bash
FACET_OPTIMIZER_PROJECT="Facet Optimizer" \
FACET_OPTIMIZER_DATASET="emergent-issues-ground-truth-v1" \
FACET_OPTIMIZER_PROMPT_PATH="prompts/facet_emergent-issues-20260409_v1.yaml" \
FACET_OPTIMIZER_MODEL="brain-facet-1" \
bt eval --env-file .env --language python --runner .venv/bin/python eval_prompt.py
```

Default scores:
- `output_nonempty`
- `none_decision_match`
- `normalized_exact_match`

## Step 6: Iterate on the Prompt

1. Edit the YAML under `prompts/`
2. Rerun `bt eval`
3. Compare experiments
4. Promote the next prompt version (`v2`, `v3`, ...)

The input dataset stays stable. Only the prompt file changes.

## Prompt File Shape

Generated prompt files look like this:

```yaml
schema_version: 1
facet_name: emergent-issues-20260409
prompt_version: v1
placeholder: preprocessed_text
messages:
  - role: system
    content: "..."
  - role: user
    content: "Here is the data to analyze:\n\n{{preprocessed_text}}"
suffix_messages:
  - - role: user
      content: "..."
```

Edit the prompt text in this YAML. Do not edit dataset rows to change the prompt.

