from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any

from config import Settings, get_settings


def _parse_iso(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None

async def get_latest_telemetry_snapshot(
    max_age_minutes: int = 60,
    settings: Settings | None = None,
) -> dict[str, Any] | None:
    cfg = settings or get_settings()
    path = Path(cfg.telemetry_log_path)
    if not path.exists():
        return None
    now = datetime.now().astimezone()
    threshold = now - timedelta(minutes=max(1, int(max_age_minutes)))

    latest: dict[str, Any] | None = None
    latest_at: datetime | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        rec_at = _parse_iso(str(obj.get("timestamp") or ""))
        if rec_at is None:
            continue
        if rec_at.tzinfo is None:
            rec_at = rec_at.astimezone()
        if rec_at < threshold:
            continue
        if latest_at is None or rec_at > latest_at:
            latest_at = rec_at
            latest = obj

    return latest

async def get_health_daily_summary(
    metrics: list[str],
    days: int = 7,
    settings: Settings | None = None,
) -> dict[str, Any]:
    from health_tools import run_health_query
    payload = await run_health_query(metrics=metrics, days=days, settings=settings)
    return payload
