from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from config import Settings

logger = logging.getLogger(__name__)


@dataclass
class TaskRecord:
    id: str
    created_at: str
    run_at: str
    kind: str
    channel: str
    payload: dict[str, Any]
    status: str
    last_error: str | None = None


TASK_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "task_schedule_ping",
            "description": "Schedule a future ping to the user (e.g., via Telegram) at a specific time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_at": {
                        "type": "string",
                        "description": "ISO 8601 datetime in the user's local timezone (the gateway will convert to UTC).",
                    },
                    "channel": {
                        "type": "string",
                        "enum": ["telegram"],
                        "description": "Notification channel; currently only 'telegram' is supported.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Message text to send when the task fires.",
                    },
                },
                "required": ["run_at", "channel", "text"],
            },
        },
    }
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_to_utc(value: str, default_tz: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(default_tz))
    return parsed.astimezone(timezone.utc)


def _task_path(settings: Settings) -> Path:
    return Path(settings.tasks_log_path)


def _read_tasks(settings: Settings) -> list[dict[str, Any]]:
    path = _task_path(settings)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except Exception as exc:
        logger.warning("task_engine read failed path=%s err=%r", str(path), exc)
    return rows


def _write_tasks(settings: Settings, rows: list[dict[str, Any]]) -> None:
    path = _task_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def add_task(settings: Settings, task: TaskRecord) -> None:
    rows = _read_tasks(settings)
    rows.append(asdict(task))
    _write_tasks(settings, rows)


def list_pending(settings: Settings, now: datetime | None = None) -> list[TaskRecord]:
    now_dt = now or _utc_now()
    rows = _read_tasks(settings)
    due: list[TaskRecord] = []
    for row in rows:
        if row.get("status") != "pending":
            continue
        run_at = row.get("run_at")
        if not isinstance(run_at, str):
            continue
        try:
            run_at_dt = _parse_iso_to_utc(run_at, "UTC")
        except Exception:
            continue
        if run_at_dt <= now_dt:
            due.append(TaskRecord(**row))
    return due


def _update_task(settings: Settings, task_id: str, *, status: str, error: str | None = None) -> bool:
    rows = _read_tasks(settings)
    updated = False
    for row in rows:
        if str(row.get("id") or "") == task_id:
            row["status"] = status
            row["last_error"] = error
            updated = True
            break
    if updated:
        _write_tasks(settings, rows)
    return updated


def mark_task_done(settings: Settings, task_id: str) -> bool:
    return _update_task(settings, task_id, status="done")


def mark_task_failed(settings: Settings, task_id: str, error: str) -> bool:
    safe_error = (error or "")[:240]
    return _update_task(settings, task_id, status="failed", error=safe_error)


def execute_task_tool(tool_name: str, arguments: dict[str, Any], *, settings: Settings, telegram_chat_id: str | None) -> dict[str, Any]:
    if tool_name != "task_schedule_ping":
        return {"ok": False, "error": f"unsupported tool: {tool_name}"}

    channel = str(arguments.get("channel") or "").strip().lower()
    if channel != "telegram":
        return {"ok": False, "error": "only channel=telegram is supported"}

    if not telegram_chat_id:
        return {"ok": False, "error": "telegram chat_id is unavailable"}

    text = str(arguments.get("text") or "").strip()
    if not text:
        return {"ok": False, "error": "text is required"}

    run_at_raw = str(arguments.get("run_at") or "").strip()
    if not run_at_raw:
        return {"ok": False, "error": "run_at is required"}

    try:
        run_at_utc = _parse_iso_to_utc(run_at_raw, settings.default_tz)
    except Exception:
        return {"ok": False, "error": "invalid run_at datetime"}

    now = _utc_now()
    task = TaskRecord(
        id=uuid4().hex,
        created_at=_to_iso(now),
        run_at=_to_iso(run_at_utc),
        kind="telegram_message",
        channel="telegram",
        payload={
            "chat_id": str(telegram_chat_id),
            "text": text,
            # future schema extension:
            # "condition": {"kind": "weather_condition_ping", ...}
        },
        status="pending",
        last_error=None,
    )
    add_task(settings, task)
    logger.info("task_engine task_schedule_ping id=%s run_at=%s kind=%s", task.id, task.run_at, task.kind)
    return {
        "ok": True,
        "task": {
            "id": task.id,
            "created_at": task.created_at,
            "run_at": task.run_at,
            "kind": task.kind,
            "channel": task.channel,
            "status": task.status,
        },
    }


def build_task_tool_message(
    tool_call_id: str,
    tool_name: str,
    arguments_json: str,
    *,
    settings: Settings,
    telegram_chat_id: str | None,
) -> dict[str, Any]:
    try:
        arguments = json.loads(arguments_json) if arguments_json else {}
        if not isinstance(arguments, dict):
            arguments = {}
    except Exception:
        arguments = {}

    payload = execute_task_tool(tool_name, arguments, settings=settings, telegram_chat_id=telegram_chat_id)
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": json.dumps(payload, ensure_ascii=False),
    }
