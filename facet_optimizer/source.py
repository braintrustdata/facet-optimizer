from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from braintrust.logger import login_to_state

from .models import FinalFacets, SourceSpan

UUID_RE = re.compile(r"^[0-9a-fA-F-]{32,36}$")


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def source_model_values(value: str) -> list[str]:
    models = [item.strip() for item in value.split(",") if item.strip()]
    return models or [value]


def source_model_clause(value: str) -> str:
    models = source_model_values(value)
    if len(models) == 1:
        return f"metadata.model = {sql_string(models[0])}"
    return "(" + " OR ".join(f"metadata.model = {sql_string(model)}" for model in models) + ")"


def run_bt_json(
    args: list[str],
    *,
    env_file: Path,
    org_name: str | None,
) -> Any:
    cmd = ["bt", *args, "--json", "--env-file", str(env_file), "--no-input"]
    if org_name:
        cmd.extend(["--org", org_name])
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "bt command failed"
        raise RuntimeError(message)
    stdout = completed.stdout.strip()
    if not stdout:
        return None
    return json.loads(stdout)


def run_bt_sql(
    query: str,
    *,
    env_file: Path,
    org_name: str | None,
) -> list[dict[str, Any]]:
    payload = run_bt_json(
        ["sql", "--non-interactive", query],
        env_file=env_file,
        org_name=org_name,
    )
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise RuntimeError("Unexpected bt sql JSON payload")


def resolve_project_id(project: str, *, env_file: Path, org_name: str | None) -> str:
    if UUID_RE.match(project):
        return project

    payload = run_bt_json(["projects", "list"], env_file=env_file, org_name=org_name)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected bt projects list JSON payload")
    matches = [row for row in payload if row.get("name") == project]
    if not matches:
        raise RuntimeError(f"Project {project!r} not found")
    if len(matches) > 1:
        raise RuntimeError(f"Project name {project!r} is ambiguous")
    project_id = matches[0].get("id")
    if not isinstance(project_id, str) or not project_id:
        raise RuntimeError(f"Project {project!r} did not include an id")
    return project_id


def resolve_login_org_name(
    *,
    api_key: str,
    app_url: str | None,
    org_name: str | None,
) -> str | None:
    if org_name:
        return org_name
    try:
        state = login_to_state(api_key=api_key, app_url=app_url, org_name=None)
    except Exception:
        return None
    resolved = state.org_name
    return resolved if isinstance(resolved, str) and resolved.strip() else None


def build_span_query(
    *,
    project_id: str,
    source_model: str,
    created_after_sql: str,
    created_before_sql: str,
    extra_where_sql: str | None,
    limit: int,
) -> str:
    extra_filter = f"\n  AND ({extra_where_sql})" if extra_where_sql else ""
    return f"""SELECT
  id,
  root_span_id,
  span_id,
  created,
  input,
  output,
  metadata,
  span_attributes
FROM project_logs({sql_string(project_id)})
WHERE span_attributes.type = 'llm'
  AND {source_model_clause(source_model)}
  AND created >= {created_after_sql}
  AND created < {created_before_sql}{extra_filter}
ORDER BY created DESC
LIMIT {limit}"""


def build_root_span_query(
    *,
    project_id: str,
    source_model: str,
    root_span_ids: list[str],
    extra_where_sql: str | None,
) -> str:
    in_clause = ", ".join(sql_string(root_id) for root_id in root_span_ids)
    extra_filter = f"\n  AND ({extra_where_sql})" if extra_where_sql else ""
    return f"""SELECT
  id,
  root_span_id,
  span_id,
  created,
  input,
  output,
  metadata,
  span_attributes
FROM project_logs({sql_string(project_id)})
WHERE span_attributes.type = 'llm'
  AND {source_model_clause(source_model)}
  AND root_span_id IN ({in_clause}){extra_filter}
ORDER BY created ASC"""


def build_final_facets_query(*, project_id: str, root_span_ids: list[str]) -> str:
    in_clause = ", ".join(sql_string(root_id) for root_id in root_span_ids)
    return f"""SELECT
  id,
  root_span_id,
  span_id,
  created,
  facets
FROM project_logs({sql_string(project_id)})
WHERE root_span_id IN ({in_clause})
  AND facets IS NOT NULL
ORDER BY created DESC"""


def jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def source_span_from_row(row: dict[str, Any]) -> SourceSpan | None:
    row_id = row.get("id")
    root_span_id = row.get("root_span_id")
    span_id = row.get("span_id") or row_id
    if not isinstance(row_id, str) or not isinstance(root_span_id, str):
        return None
    metadata_value = jsonish(row.get("metadata"))
    span_attributes_value = jsonish(row.get("span_attributes"))
    metadata = metadata_value if isinstance(metadata_value, dict) else {}
    span_attributes = (
        span_attributes_value if isinstance(span_attributes_value, dict) else {}
    )
    created = row.get("created")
    return SourceSpan(
        id=row_id,
        root_span_id=root_span_id,
        span_id=span_id if isinstance(span_id, str) else row_id,
        created=created if isinstance(created, str) else None,
        input=jsonish(row.get("input")),
        output=jsonish(row.get("output")),
        metadata=metadata,
        span_attributes=span_attributes,
    )


def final_facets_from_row(row: dict[str, Any]) -> FinalFacets | None:
    row_id = row.get("id")
    root_span_id = row.get("root_span_id")
    span_id = row.get("span_id") or row_id
    if not isinstance(row_id, str) or not isinstance(root_span_id, str):
        return None
    facets_value = jsonish(row.get("facets"))
    if not isinstance(facets_value, dict):
        return None
    created = row.get("created")
    return FinalFacets(
        row_id=row_id,
        root_span_id=root_span_id,
        span_id=span_id if isinstance(span_id, str) else row_id,
        created=created if isinstance(created, str) else None,
        facets=facets_value,
    )


def batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [
        items[index : index + batch_size]
        for index in range(0, len(items), batch_size)
    ]


def fetch_source_spans(
    *,
    env_file: Path,
    org_name: str | None,
    source_project: str | None = None,
    source_project_id: str | None = None,
    source_model: str,
    created_after_sql: str,
    created_before_sql: str,
    extra_where_sql: str | None,
    limit: int,
    root_span_ids: list[str],
) -> tuple[str, str, list[SourceSpan]]:
    if source_project_id:
        project_id = source_project_id
    elif source_project:
        project_id = resolve_project_id(
            source_project,
            env_file=env_file,
            org_name=org_name,
        )
    else:
        raise RuntimeError("Either source_project or source_project_id is required")
    if root_span_ids:
        query = build_root_span_query(
            project_id=project_id,
            source_model=source_model,
            root_span_ids=root_span_ids,
            extra_where_sql=extra_where_sql,
        )
    else:
        query = build_span_query(
            project_id=project_id,
            source_model=source_model,
            created_after_sql=created_after_sql,
            created_before_sql=created_before_sql,
            extra_where_sql=extra_where_sql,
            limit=limit,
        )
    rows = run_bt_sql(query, env_file=env_file, org_name=org_name)
    spans = [span for row in rows if (span := source_span_from_row(row)) is not None]
    return project_id, query, spans


def fetch_final_facets(
    *,
    env_file: Path,
    org_name: str | None,
    project_id: str,
    root_span_ids: list[str],
    batch_size: int = 100,
) -> tuple[list[str], dict[str, FinalFacets]]:
    queries: list[str] = []
    facets_by_root: dict[str, FinalFacets] = {}
    deduped_roots = list(dict.fromkeys(root_span_ids))
    for root_batch in batched(deduped_roots, batch_size):
        query = build_final_facets_query(project_id=project_id, root_span_ids=root_batch)
        queries.append(query)
        rows = run_bt_sql(query, env_file=env_file, org_name=org_name)
        for row in rows:
            final_facets = final_facets_from_row(row)
            if final_facets and final_facets.root_span_id not in facets_by_root:
                facets_by_root[final_facets.root_span_id] = final_facets
    return queries, facets_by_root
