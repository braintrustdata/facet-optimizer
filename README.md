# Facet optimizer

Small local workflow for turning production Braintrust facet executions into a ground-truth dataset, then running baseline evals against `brain-facet-1` and `brain-facet-2`.

This repo does not implement the general dataset pipeline abstraction yet. It is the concrete version of that workflow for facets.

## Quick start

Give a coding agent this prompt from the repo root:

```text
Follow the facet optimizer README. Create a ground-truth dataset from production facet traces for source project "<My source project>", target project "Facet optimizer", and dataset "Facet groundtruth". After dataset creation, report the row counts, positive/negative balance, and train/validation split. Then bootstrap the facet definitions, run the initial brain-facet-1 eval, inspect failures with bt sql, make one conservative prompt optimization without overfitting, run a train smoke eval, run a validation eval, and summarize the prompt changes and metric deltas.
```

Minimal manual flow:

1. Set `BRAINTRUST_API_KEY` in `.env`.
2. Create the dataset with `scripts/create_facet_dataset.py`.
3. Bootstrap `.local/facet-optimizer/facet_definitions.yaml`.
4. Run the initial `brain-facet-1` eval with `bt eval`.
5. Inspect failures with `bt sql`.
6. Edit the facet prompt in `.local/facet-optimizer/facet_definitions.yaml`.
7. Run a small train eval, then a validation eval.
8. Keep the prompt only if validation improves without an unacceptable recall/precision tradeoff.

## Set up

Create a local `.env` from `.env.example` and set at least:

```bash
BRAINTRUST_API_KEY=...
FACET_OPTIMIZER_TARGET_PROJECT="Facet optimizer"
```

The dataset script uses `.env` by default. Source and target projects can differ. If the source or target org needs to be explicit, set `FACET_OPTIMIZER_SOURCE_ORG` and `FACET_OPTIMIZER_TARGET_ORG`.

Use the latest version of the `bt` CLI before running evals. The commands below use `bt eval`.

If `bt eval` cannot find a Python runner in your shell, add `--runner .venv/bin/python` after `bt eval`.

## Bootstrap a dataset

Create a ground-truth dataset from production facet LLM spans:

```bash
uv run python scripts/create_facet_dataset.py \
  --source-project "<My source project>" \
  --target-project "Facet optimizer" \
  --dataset "Facet groundtruth"
```

By default this searches source traces for:

```sql
span_attributes.type = 'llm'
AND (
  metadata.model = 'brain-facet-latest'
  OR metadata.model = 'brain-facet-1'
)
```

It samples up to `100` positive and `100` negative weak examples per facet, labels them with `gpt-5.4`, assigns a deterministic `train` / `validation` split, uploads rows to the target Braintrust dataset, and writes local artifacts under:

```text
.local/facet-optimizer/<run-id>/
```

Useful options:

```bash
# Add one specific trace by root span id.
uv run python scripts/create_facet_dataset.py \
  --source-project "<My source project>" \
  --target-project "Facet optimizer" \
  --dataset "Facet groundtruth" \
  --root-span-id "<root-span-id>"

# Adjust per-facet weak sampling limits.
uv run python scripts/create_facet_dataset.py \
  --source-project "<My source project>" \
  --target-project "Facet optimizer" \
  --dataset "Facet groundtruth" \
  --positive-limit 200 \
  --negative-limit 200

# Pull a narrower slice with source SQL filters.
uv run python scripts/create_facet_dataset.py \
  --source-project "<My source project>" \
  --target-project "Facet optimizer" \
  --dataset "Facet groundtruth" \
  --created-after-sql "NOW() - INTERVAL 30 DAY" \
  --created-before-sql "NOW() - INTERVAL 1 HOUR" \
  --extra-where-sql "metadata.some_attribute = 'some-value'"

# Change the validation holdout fraction.
uv run python scripts/create_facet_dataset.py \
  --source-project "<My source project>" \
  --target-project "Facet optimizer" \
  --dataset "Facet groundtruth" \
  --validation-fraction 0.2
```

After dataset creation, promote that run's captured facet definitions into the current local eval prompt file:

```bash
uv run python scripts/bootstrap_facet_definitions.py \
  --run-dir .local/facet-optimizer/<run-id>
```

This creates:

```text
.local/facet-optimizer/facet_definitions.yaml
```

That YAML contains the captured production wrapper messages plus the facet prompt in `suffix_messages`.

## Run initial evals

Run the baseline eval for `brain-facet-1`:

```bash
env FACET_OPTIMIZER_MODEL=brain-facet-1 \
  FACET_OPTIMIZER_DATASET="Facet groundtruth" \
  FACET_OPTIMIZER_PROMPT=.local/facet-optimizer/facet_definitions.yaml \
  bt eval --env-file .env eval_facet.py
```

You can also preview `brain-facet-2`. NOTE that it is not stably deployed yet, so you may see performance (speed) issues with it:

```bash
env FACET_OPTIMIZER_MODEL=brain-facet-2 \
  FACET_OPTIMIZER_DATASET="Facet groundtruth" \
  FACET_OPTIMIZER_PROMPT=.local/facet-optimizer/facet_definitions.yaml \
  bt eval --env-file .env eval_facet.py
```

The eval uses the same scorer configuration as `eval_facets_clean.py`, except the binary classification scorers are generalized across facets instead of being hardcoded to Issues only:

```python
binary_classification_scores
sentiment_label_correct
Factuality.partial(model="gpt-5.4")
```

Eval concurrency defaults to `16`. To run a smaller smoke test:

```bash
env FACET_OPTIMIZER_MODEL=brain-facet-1 \
  FACET_OPTIMIZER_MAX_ROWS=25 \
  FACET_OPTIMIZER_DATASET="Facet groundtruth" \
  FACET_OPTIMIZER_PROMPT=.local/facet-optimizer/facet_definitions.yaml \
  bt eval --env-file .env eval_facet.py
```

To scope to one facet:

```bash
env FACET_OPTIMIZER_MODEL=brain-facet-1 \
  FACET_OPTIMIZER_FACET_FILTER=issues \
  FACET_OPTIMIZER_DATASET="Facet groundtruth" \
  FACET_OPTIMIZER_PROMPT=.local/facet-optimizer/facet_definitions.yaml \
  bt eval --env-file .env eval_facet.py
```

To scope to one dataset split:

```bash
env FACET_OPTIMIZER_MODEL=brain-facet-1 \
  FACET_OPTIMIZER_SPLIT=validation \
  FACET_OPTIMIZER_DATASET="Facet groundtruth" \
  FACET_OPTIMIZER_PROMPT=.local/facet-optimizer/facet_definitions.yaml \
  bt eval --env-file .env eval_facet.py
```

## If scores look too high

Before changing the facet prompt, inspect the dataset and ground truth.

Start with the local run artifacts:

```text
.local/facet-optimizer/<run-id>/summary.json
.local/facet-optimizer/<run-id>/dataset_rows.jsonl
.local/facet-optimizer/<run-id>/parsed_calls.jsonl
```

For each suspicious row, check:

- `expected`: the generated ground-truth label
- `input.facet_name`
- `metadata.source_weak_bucket`
- `metadata.split`
- `metadata.production_output`
- `metadata.source_trace_permalink`

If the eval looks unrealistically good, common causes are:

- the ground-truth label is wrong or too close to the model output
- the dataset is mostly easy negatives
- the selected rows do not contain enough borderline positives
- a specific facet has too few examples

Fix the bad ground-truth values in the target Braintrust dataset, or recreate the dataset after changing the sampling settings. Then rerun both initial evals before editing the facet prompt.

## Prompt optimization

Use `brain-facet-1` for prompt iteration for now. `brain-facet-2` is useful for occasional comparison, but it is not ready for lots of optimization queries.

Before editing the prompt, decide the optimization target:

- minimize false positives: bias toward precision and avoid flagging normal cases
- minimize false negatives: bias toward recall and catch more true positives
- balanced: improve Factuality and the binary scores without strongly favoring either side

If you are using a coding agent and you do not know which target you want, say so. The agent should optimize the balanced target by default.

The editable prompt lives in:

```text
.local/facet-optimizer/facet_definitions.yaml
```

Change only the relevant facet's `suffix_messages` prompt unless you intentionally want to change the captured production wrapper.

Recommended loop:

1. Run a baseline eval on the validation split and save the experiment link.
2. Inspect failures from the current eval with `bt sql`. Prefer looking at validation failures for diagnosis, but make prompt edits against the training split.
3. Edit the facet prompt in `.local/facet-optimizer/facet_definitions.yaml`.
4. Run a smaller training eval:

   ```bash
   env FACET_OPTIMIZER_MODEL=brain-facet-1 \
     FACET_OPTIMIZER_SPLIT=train \
     FACET_OPTIMIZER_MAX_ROWS=50 \
     FACET_OPTIMIZER_DATASET="Facet groundtruth" \
     FACET_OPTIMIZER_PROMPT=.local/facet-optimizer/facet_definitions.yaml \
     bt eval --env-file .env eval_facet.py
   ```

5. When the training result looks better, run the full validation eval:

   ```bash
   env FACET_OPTIMIZER_MODEL=brain-facet-1 \
     FACET_OPTIMIZER_SPLIT=validation \
     FACET_OPTIMIZER_DATASET="Facet groundtruth" \
     FACET_OPTIMIZER_PROMPT=.local/facet-optimizer/facet_definitions.yaml \
     bt eval --env-file .env eval_facet.py
   ```

6. Keep the prompt change only if validation improves against the chosen target without large regressions in Factuality or the opposite error direction.

Useful `bt sql` queries:

```bash
# Get the latest experiments and ids.
bt experiments list --env-file .env -p "Facet optimizer" --json

# Per-facet binary metrics for an experiment.
bt sql --env-file .env -p "Facet optimizer" --json --non-interactive \
  "SELECT input.metadata.facet_name as facet,
          avg(scores.binary_decision_match) as binary_match,
          avg(scores.positive_recall) as positive_recall,
          avg(scores.negative_specificity) as negative_specificity,
          count(*) as n
   FROM experiment('<experiment-id>')
   WHERE span_attributes.name = 'binary_classification_scores'
   GROUP BY input.metadata.facet_name
   ORDER BY input.metadata.facet_name"

# Factuality by facet.
bt sql --env-file .env -p "Facet optimizer" --json --non-interactive \
  "SELECT input.metadata.facet_name as facet,
          avg(output.score) as factuality,
          count(*) as n
   FROM experiment('<experiment-id>')
   WHERE span_attributes.name = 'Factuality'
   GROUP BY input.metadata.facet_name
   ORDER BY input.metadata.facet_name"

# Example failed rows for one scorer.
bt sql --env-file .env -p "Facet optimizer" --json --non-interactive \
  "SELECT input.metadata.dataset_row_id as row_id,
          input.metadata.split as split,
          input.expected as expected,
          input.output as output
   FROM experiment('<experiment-id>')
   WHERE span_attributes.name = 'binary_classification_scores'
     AND input.metadata.facet_name = 'issues'
     AND scores.binary_decision_match = 0
   LIMIT 20"
```

Common first prompt improvement: add a clear data boundary. Many failures happen when the model treats the captured conversation as instructions and answers the user instead of producing the facet. A good facet prompt should explicitly say that the conversation is data, not a request to answer, and that the model should return only the requested facet value.

Avoid overfitting:

- do not tune wording around one or two specific examples
- do not repeatedly optimize against the validation split
- look for reusable failure patterns before editing
- keep prompt changes small and attributable
- rerun on fresh data before trusting a large gain

If there is not enough data, add more examples before optimizing. Prefer asking for specific root span ids for known edge cases, or pull more data with a wider time window or a targeted `--extra-where-sql` metadata filter.
