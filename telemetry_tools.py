from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from statistics import mean
from typing import Any

from config import Settings, get_settings

TELEMETRY_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "telemetry_query",
            "description": (
                "Query recent device telemetry such as battery level, approximate "
                "location, and weather in the last N hours. "
                "Use this when you need to know my recent device conditions on my side "
                "to give more context-aware advice."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metrics": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["battery_pct", "location", "weather"],
                        },
                        "description": "Which telemetry metrics to query.",
                    },
                    "hours": {
                        "type": "integer",
                        "description": "How many recent hours to query (default 24, max 168).",
                        "default": 24,
                        "minimum": 1,
                        "maximum": 168,
                    },
                },
                "required": ["metrics"],
            },
        },
    }
]


def _parse_iso(ts: str) -> datetime | None:
    text = (ts or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


async def run_telemetry_query(
    metrics: list[str],
    hours: int = 24,
    settings: Settings | None = None,
) -> dict[str, Any]:
    cfg = settings or get_settings()
    path = Path(cfg.telemetry_log_path)
    if not path.exists():
        return {}

    allowed = {"battery_pct", "location", "weather"}
    wanted = [m for m in metrics if m in allowed]
    if not wanted:
        return {}

    hour_window = max(1, min(int(hours), 168))
    now = datetime.now().astimezone()
    threshold = now - timedelta(hours=hour_window)

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        dt = _parse_iso(str(rec.get("timestamp") or ""))
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.astimezone()
        if dt < threshold:
            continue
        rows.append(rec)

    rows.sort(key=lambda r: str(r.get("timestamp") or ""))
    out: dict[str, Any] = {}

    if "battery_pct" in wanted:
        vals: list[float] = []
        for r in rows:
            v = r.get("battery_pct")
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            out["battery_pct"] = {
                "unit": "fraction",
                "latest": round(vals[-1], 3),
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
                "avg": round(mean(vals), 3),
            }

    if "location" in wanted:
        latest_loc = None
        for r in rows:
            loc = r.get("location")
            if isinstance(loc, dict) and isinstance(loc.get("lat"), (int, float)) and isinstance(loc.get("lon"), (int, float)):
                latest_loc = {
                    "lat": float(loc.get("lat")),
                    "lon": float(loc.get("lon")),
                    "label": loc.get("label"),
                }
        if latest_loc is not None:
            out["location"] = {"latest": latest_loc}

    if "weather" in wanted:
        latest_weather = None
        temps: list[float] = []
        for r in rows:
            w = r.get("weather")
            if not isinstance(w, dict):
                continue
            t = w.get("temp_c")
            if isinstance(t, (int, float)):
                temps.append(float(t))
            cond = w.get("cond")
            if cond is not None or t is not None:
                latest_weather = {
                    "cond": cond,
                    "temp_c": t,
                }
        if latest_weather is not None:
            payload: dict[str, Any] = {"latest": latest_weather}
            if temps:
                payload["min_temp_c"] = round(min(temps), 3)
                payload["max_temp_c"] = round(max(temps), 3)
            out["weather"] = payload

    return out


async def execute_telemetry_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if tool_name != "telemetry_query":
        return {"ok": False, "error": f"unsupported tool: {tool_name}"}

    metrics_arg = arguments.get("metrics")
    if not isinstance(metrics_arg, list) or not metrics_arg:
        return {"ok": False, "error": "metrics must be a non-empty list"}

    hours = int(arguments.get("hours", 24) or 24)
    payload = await run_telemetry_query(metrics=[str(m) for m in metrics_arg], hours=hours)
    return {"ok": True, **payload}
