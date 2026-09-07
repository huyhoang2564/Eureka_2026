from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import math

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def get_nested(doc: Dict[str, Any], path: str, default=None):
    """Lấy field lồng nhau bằng dotted path, ví dụ fields.project.key."""
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


import functools

@functools.lru_cache(maxsize=32768)
def _as_utc(value):
    if value is None:
        return pd.NaT
    # Optimize by directly converting assuming ISO 8601 if it's a string, falling back if needed.
    try:
        return pd.Timestamp(value).tz_convert('UTC') if pd.Timestamp(value).tz is not None else pd.Timestamp(value, tz='UTC')
    except:
        return pd.to_datetime(value, errors="coerce", utc=True)


def _text_len(value) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False))
    except Exception:
        return len(str(value))


def _normalize_field_name(x: Any) -> str:
    return str(x or "").strip().lower().replace("_", " ")


def _history_items(histories) -> Iterable[tuple[pd.Timestamp, dict]]:
    """Yield (created_time, item) cho từng changelog item."""
    if not isinstance(histories, list):
        return
    for h in histories:
        if not isinstance(h, dict):
            continue
        ts = _as_utc(h.get("created"))
        if pd.isna(ts):
            continue
        items = h.get("items") or []
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                yield ts, item


def initial_value_from_changelog(
    current_value: Any,
    histories: Any,
    aliases: Iterable[str],
):
    """
    Tái dựng giá trị ban đầu:
    - nếu field từng thay đổi: lấy fromString của lần thay đổi sớm nhất;
    - nếu chưa từng thay đổi: current_value được xem là giá trị từ đầu.
    """
    aliases = {_normalize_field_name(a) for a in aliases}
    candidates = []

    for ts, item in _history_items(histories):
        field = _normalize_field_name(item.get("field"))
        field_id = _normalize_field_name(item.get("fieldId"))
        if field in aliases or field_id in aliases:
            candidates.append((ts, item))

    if not candidates:
        return current_value

    candidates.sort(key=lambda x: x[0])
    first = candidates[0][1]
    value = first.get("fromString")
    if value is None:
        value = first.get("from")
    return value


def _comment_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        comments = value.get("comments")
        return comments if isinstance(comments, list) else []
    return []


def early_activity_features(
    created,
    histories,
    comments,
    landmark_days: int = 7,
) -> Dict[str, float]:
    created = _as_utc(created)
    if pd.isna(created):
        return {
            f"changes_{landmark_days}d": 0,
            f"status_changes_{landmark_days}d": 0,
            f"assignee_changes_{landmark_days}d": 0,
            f"priority_changes_{landmark_days}d": 0,
            f"comments_{landmark_days}d": 0,
            f"inactivity_days_at_{landmark_days}d": float(landmark_days),
        }

    cutoff = created + pd.Timedelta(days=landmark_days)

    n_changes = 0
    n_status = 0
    n_assignee = 0
    n_priority = 0
    activity_times = [created]

    for ts, item in _history_items(histories):
        if ts < created or ts > cutoff:
            continue
        n_changes += 1
        activity_times.append(ts)
        field = _normalize_field_name(item.get("field"))
        field_id = _normalize_field_name(item.get("fieldId"))

        if field == "status" or field_id == "status":
            n_status += 1
        if field == "assignee" or field_id == "assignee":
            n_assignee += 1
        if field == "priority" or field_id == "priority":
            n_priority += 1

    n_comments = 0
    for c in _comment_list(comments):
        if not isinstance(c, dict):
            continue
        ts = _as_utc(c.get("created"))
        if pd.isna(ts) or ts < created or ts > cutoff:
            continue
        n_comments += 1
        activity_times.append(ts)

    last_activity = max(activity_times)
    inactivity = (cutoff - last_activity).total_seconds() / 86400.0
    inactivity = max(0.0, min(float(landmark_days), inactivity))

    return {
        f"changes_{landmark_days}d": int(n_changes),
        f"status_changes_{landmark_days}d": int(n_status),
        f"assignee_changes_{landmark_days}d": int(n_assignee),
        f"priority_changes_{landmark_days}d": int(n_priority),
        f"comments_{landmark_days}d": int(n_comments),
        f"inactivity_days_at_{landmark_days}d": float(inactivity),
    }


def flatten_issue(
    doc: Dict[str, Any],
    repository: str,
    schema: Dict[str, str],
    landmark_days: int = 7,
) -> Dict[str, Any]:
    project = get_nested(doc, schema["project"])
    created = get_nested(doc, schema["created"])
    updated = get_nested(doc, schema["updated"])
    resolution = get_nested(doc, schema["resolution"])

    histories = get_nested(doc, schema["changelog_histories"], []) or []
    comments = get_nested(doc, schema["comments"], []) or []

    current_priority = get_nested(doc, schema["priority"])
    current_issue_type = get_nested(doc, schema["issue_type"])
    current_status = get_nested(doc, schema["status"])
    current_assignee = get_nested(doc, schema["assignee"])

    initial_priority = initial_value_from_changelog(
        current_priority, histories, aliases=["priority"]
    )
    initial_issue_type = initial_value_from_changelog(
        current_issue_type, histories, aliases=["issuetype", "issue type"]
    )
    initial_status = initial_value_from_changelog(
        current_status, histories, aliases=["status"]
    )
    initial_assignee = initial_value_from_changelog(
        current_assignee, histories, aliases=["assignee"]
    )

    summary = get_nested(doc, schema["summary"])
    description = get_nested(doc, schema["description"])
    labels = get_nested(doc, schema["labels"], [])
    components = get_nested(doc, schema["components"], [])

    row = {
        "repository": repository,
        "project": project,
        "issue_key": doc.get("key"),
        "created": created,
        "updated": updated,
        "resolution_date": resolution,
        "initial_priority": None if initial_priority is None else str(initial_priority),
        "initial_issue_type": None if initial_issue_type is None else str(initial_issue_type),
        "initial_status": None if initial_status is None else str(initial_status),
        "initial_assigned": int(initial_assignee is not None and str(initial_assignee).strip() not in {"", "None"}),
        # Các cột snapshot dưới đây được lưu để audit, nhưng mặc định KHÔNG dùng
        # trong strict Day-0 model vì có thể đã thay đổi sau khi issue được tạo.
        "snapshot_summary_len": _text_len(summary),
        "snapshot_description_len": _text_len(description),
        "snapshot_label_count": len(labels) if isinstance(labels, list) else 0,
        "snapshot_component_count": len(components) if isinstance(components, list) else 0,
    }

    row.update(
        early_activity_features(
            created=created,
            histories=histories,
            comments=comments,
            landmark_days=landmark_days,
        )
    )

    return row


def _projection(schema: Dict[str, str]) -> Dict[str, int]:
    fields = {
        "key",
        schema["project"],
        schema["created"],
        schema["updated"],
        schema["resolution"],
        schema["issue_type"],
        schema["priority"],
        schema["status"],
        schema["summary"],
        schema["description"],
        schema["labels"],
        schema["components"],
        schema["assignee"],
        schema["changelog_histories"],
        schema["comments"],
    }
    return {f: 1 for f in fields}


def extract_project_to_parquet(
    collection,
    repository: str,
    project_key: str,
    schema: Dict[str, str],
    out_file: str | Path,
    landmark_days: int = 7,
    batch_size: int = 5000,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Stream 1 project từ MongoDB -> flat Parquet.
    Không list() toàn bộ collection vào RAM.
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and not overwrite:
        return {
            "repository": repository,
            "project": project_key,
            "file": str(out_file),
            "status": "skipped_exists",
        }

    query = {schema["project"]: project_key}
    cursor = collection.find(query, _projection(schema), batch_size=batch_size)

    writer = None
    rows = []
    total = 0

    def flush():
        nonlocal writer, rows, total
        if not rows:
            return
        df = pd.DataFrame(rows)

        # Ép kiểu ổn định giữa các batch.
        str_cols = [
            "repository", "project", "issue_key",
            "created", "updated", "resolution_date",
            "initial_priority", "initial_issue_type", "initial_status",
        ]
        for c in str_cols:
            if c in df.columns:
                df[c] = df[c].astype("string")

        int_cols = [
            "initial_assigned",
            "snapshot_summary_len",
            "snapshot_description_len",
            "snapshot_label_count",
            "snapshot_component_count",
            f"changes_{landmark_days}d",
            f"status_changes_{landmark_days}d",
            f"assignee_changes_{landmark_days}d",
            f"priority_changes_{landmark_days}d",
            f"comments_{landmark_days}d",
        ]
        for c in int_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

        inactivity_col = f"inactivity_days_at_{landmark_days}d"
        if inactivity_col in df.columns:
            df[inactivity_col] = pd.to_numeric(df[inactivity_col], errors="coerce").astype("float64")

        table = pa.Table.from_pandas(df, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(out_file, table.schema, compression="snappy")

        writer.write_table(table)
        total += len(df)
        rows = []

    try:
        for doc in cursor:
            rows.append(
                flatten_issue(
                    doc,
                    repository=repository,
                    schema=schema,
                    landmark_days=landmark_days,
                )
            )
            if len(rows) >= batch_size:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()

    return {
        "repository": repository,
        "project": project_key,
        "rows": total,
        "file": str(out_file),
        "status": "written",
    }


def extract_selected_projects(
    db,
    selected_projects: pd.DataFrame,
    schema: Dict[str, str],
    out_dir: str | Path,
    landmark_days: int = 7,
    overwrite: bool = False,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for row in selected_projects.itertuples(index=False):
        repo = str(row.repository)
        project = str(row.project)
        safe_repo = repo.replace("/", "_").replace("\\", "_")
        safe_project = project.replace("/", "_").replace("\\", "_")
        out_file = out_dir / f"{safe_repo}__{safe_project}.parquet"

        print(f"Extracting {repo} / {project} ...")
        info = extract_project_to_parquet(
            collection=db[repo],
            repository=repo,
            project_key=project,
            schema=schema,
            out_file=out_file,
            landmark_days=landmark_days,
            overwrite=overwrite,
        )
        print("  ->", info["status"], info.get("rows", ""))
        manifest.append(info)

    return pd.DataFrame(manifest)
