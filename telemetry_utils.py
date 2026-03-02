from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from config import Settings, get_settings
from health_client import parse_health_record


def _parse_local_datetime(value: str, tz_name: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None

    local_tz = ZoneInfo(tz_name)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=local_tz)
    return dt.astimezone(local_tz)


async def get_latest_telemetry_snapshot(
    max_age_minutes: int = 60,
    settings: Settings | None = None,
) -> dict[str, Any] | None:
    cfg = settings or get_settings()
    path = Path(cfg.telemetry_log_path)
    if not path.exists():
        return None

    tz_name = cfg.default_tz
    local_tz = ZoneInfo(tz_name)
    now = datetime.now(local_tz)
    threshold = now - timedelta(minutes=max(1, int(max_age_minutes)))

    latest: dict[str, Any] | None = None
    latest_at: datetime | None = None

    try:
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
            rec_at = _parse_local_datetime(str(obj.get("received_at") or ""), tz_name)
            if rec_at is None or rec_at < threshold:
                continue
            payload = obj.get("payload")
            if not isinstance(payload, dict):
                continue
            if latest_at is None or rec_at > latest_at:
                latest_at = rec_at
                latest = {
                    "at": str(obj.get("received_at") or ""),
                    "payload": payload,
                }
    except Exception:
        return None

    return latest



async def get_health_daily_summary(
    metrics: list[str],
    days: int = 7,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Aggregate daily values by metric for the recent N days.
    Aggregation strategy: sum values for the same metric on the same day.
    """
    cfg = settings or get_settings()
    tz = ZoneInfo(cfg.default_tz)
    now = datetime.now(tz)
    day_count = max(1, int(days))
    start_day = (now - timedelta(days=day_count - 1)).date()

    wanted = [m.strip() for m in (metrics or []) if m and m.strip()]
    if not wanted:
        return {"metric": {}}

    start_dt = datetime.combine(start_day, datetime.min.time(), tz)
    health_path = Path(cfg.health_log_path)

    agg: dict[str, dict[str, float]] = {m: {} for m in wanted}
    if health_path.exists():
        for line in health_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            rec = parse_health_record(obj)
            if rec is None:
                continue

            metric = str(rec.metric or "")
            if metric not in agg:
                continue
            if rec.end < start_dt.astimezone(start_dt.tzinfo):
                continue
            if rec.start > now.astimezone(start_dt.tzinfo):
                continue

            value = rec.value
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    continue
            if not isinstance(value, (int, float)):
                continue

            local_day = rec.start.astimezone(tz).date().isoformat()
            agg[metric][local_day] = agg[metric].get(local_day, 0.0) + float(value)

    out: dict[str, list[dict[str, Any]]] = {}
    for metric in wanted:
        days_map = agg.get(metric, {})
        out[metric] = [
            {"date": d, "value": days_map[d]}
            for d in sorted(days_map.keys())
        ]

    return {"metric": out}
