from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import logging
from statistics import mean
from zoneinfo import ZoneInfo
from statistics import mean
from typing import TYPE_CHECKING, Any
from config import get_settings
if TYPE_CHECKING:
    from config import Settings

from health_client import (
    SUPPORTED_HEALTH_METRICS,
    get_health_provider,
    parse_health_record,
)

logger = logging.getLogger(__name__)

HEALTH_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "health_log",
            "description": "Log one or more health records (sleep_hours, steps, heart_rate, period).",
            "parameters": {
                "type": "object",
                "properties": {
                    "records": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "metric": {"type": "string", "enum": sorted(SUPPORTED_HEALTH_METRICS)},
                                "value": {"oneOf": [{"type": "number"}, {"type": "string"}]},
                                "start": {"type": "string"},
                                "end": {"type": "string"},
                            },
                            "required": ["metric", "value", "start"],
                        },
                    }
                },
                "required": ["records"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "health_query",
            "description": "Query health metrics in a time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metrics": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["steps", "sleep_hours", "heart_rate", "period"],
                        },
                        "description": "Which health metrics to query.",
                    },
                    "days": {
                        "type": "integer",
                        "description": "How many recent days to query (default 7, max 60).",
                        "default": 7,
                        "minimum": 1,
                        "maximum": 60,
                    },
                },
                "required":["metrics"],
            },
        },
    },
]


UNIT_BY_METRIC = {
    "steps": "count",
    "sleep_hours": "hours",
    "heart_rate": "bpm",
    "period": "status",
}


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _trend(values: list[float]) -> str:
    if len(values) < 2:
        return "flat"
    first = values[0]
    last = values[-1]
    delta = last - first
    threshold = max(0.05, abs(first) * 0.03)
    if delta > threshold:
        return "up"
    if delta < -threshold:
        return "down"
    return "flat"


async def run_health_query(
    metrics: list[str],
    days: int = 7,
    settings: Settings | None = None,
) -> dict[str, Any]:
    cfg = settings or get_settings()
    tz = ZoneInfo(cfg.default_tz)

    wanted = [m for m in metrics if m in SUPPORTED_HEALTH_METRICS]
    if not wanted:
        return {"metrics": {}}

    day_window = max(1, min(int(days), 60))
    now = datetime.now(tz)
    start_dt = now - timedelta(days=day_window)

    provider = get_health_provider()
    rows = provider.list_metrics(
        start=start_dt.astimezone(start_dt.tzinfo),
        end=now.astimezone(start_dt.tzinfo),
        metrics=wanted,
        limit=5000,
    )

    daily_buckets: dict[str, dict[str, list[float]]] = {
        metric: defaultdict(list) for metric in wanted
    }

    for row in rows:
        metric = row.metric
        if metric not in daily_buckets:
            continue
        day = row.start.astimezone(tz).date().isoformat()
        v = _coerce_float(row.value)
        if v is None:
            continue
        daily_buckets[metric][day].append(v)

    out_metrics: dict[str, Any] = {}
    for metric in wanted:
        day_map = daily_buckets.get(metric, {})
        daily_points: list[dict[str, Any]] = []
        for day in sorted(day_map.keys()):
            values = day_map[day]
            agg_value = sum(values)
            daily_points.append({"date": day, "value": round(agg_value, 3)})

        values_only = [float(p["value"]) for p in daily_points]
        if values_only:
            summary = {
                "avg": round(mean(values_only), 3),
                "min": round(min(values_only), 3),
                "max": round(max(values_only), 3),
                "trend": _trend(values_only),
            }
        else: 
           summary = {
                "avg": None,
                "min": None,
                "max": None,
                "trend": "flat",
            }
        out_metrics[metric] = {
            "unit": UNIT_BY_METRIC.get(metric, "value"),
            "daily": daily_points,
            "summary": summary,
        }

    return {"metrics": out_metrics}

def _trend(values: list[float]) -> str:
    if len(values) < 2:
        return "flat"
    delta = values[-1] - values[0]
    if abs(delta) < 1e-6:
        return "flat"
    return "up" if delta > 0 else "down"


def _unit_for_metric(metric: str) -> str:
    if metric == "steps":
        return "count"
    if metric == "sleep_hours":
        return "hours"
    if metric == "heart_rate":
        return "bpm"
    return "state"


def _build_health_summary_by_days(metrics: list[str], days: int) -> dict[str, Any]:
    cfg = get_settings()
    tz = ZoneInfo(cfg.default_tz)
    now = datetime.now(tz)
    day_count = max(1, min(int(days or 7), 60))
    start_dt = (now - timedelta(days=day_count - 1)).replace(hour=0, minute=0, second=0, microsecond=0)

    provider = get_health_provider()
    rows = provider.list_metrics(
        start=start_dt.astimezone(timezone.utc),
        end=now.astimezone(timezone.utc),
        metrics=metrics,
        limit=5000,
    )

    grouped: dict[str, dict[str, list[float]]] = {m: defaultdict(list) for m in metrics}
    for row in rows:
        if row.metric not in grouped:
            continue
        if not isinstance(row.value, (int, float)):
            continue
        local_day = row.start.astimezone(tz).date().isoformat()
        grouped[row.metric][local_day].append(float(row.value))

    metrics_payload: dict[str, Any] = {}
    summary_chunks: list[str] = []
    for metric in metrics:
        day_map = grouped.get(metric, {})
        daily: list[dict[str, Any]] = []
        for day in sorted(day_map.keys()):
            values = day_map[day]
            daily.append({"date": day, "value": round(sum(values), 3)})

        agg_values = [float(d["value"]) for d in daily]
        if agg_values:
            summary = {
                "avg": round(sum(agg_values) / len(agg_values), 3),
                "min": round(min(agg_values), 3),
                "max": round(max(agg_values), 3),
                "trend": _trend(agg_values),
            }
        else:
            summary = {"avg": 0.0, "min": 0.0, "max": 0.0, "trend": "flat"}

        metrics_payload[metric] = {
            "unit": _unit_for_metric(metric),
            "daily": daily,
            "summary": summary,
        }
        if agg_values:
            summary_chunks.append(f"{metric} 近{day_count}天均值 {summary['avg']}")
        else:
            summary_chunks.append(f"{metric} 近{day_count}天暂无记录")

    return {
        "ok": True,
        "days": day_count,
        "metrics": metrics_payload,
        "summary_text": {"health": "；".join(summary_chunks)},
    }


def execute_health_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        provider = get_health_provider()

        if tool_name == "health_log":
            records = arguments.get("records")
            if not isinstance(records, list) or not records:
                return {"ok": False, "error": "records must be a non-empty list"}

            parsed = [parse_health_record(r) for r in records if isinstance(r, dict)]
            clean = [r for r in parsed if r is not None]
            if not clean:
                return {"ok": False, "error": "no valid records"}

            written = provider.append_records(clean)
            metric_names = sorted({r.metric for r in clean})
            logger.info("TOOL health_log written=%s metrics=%s", written, metric_names)
            print(f"TOOL health_log written={written} metrics={metric_names}")
            return {"ok": True, "written": written, "metrics": metric_names}

        if tool_name == "health_query":
            return {"ok": False, "error": "health_query is async-only; use execute_health_tool_async"}

        return {"ok": False, "error": f"unsupported tool: {tool_name}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

async def execute_health_tool_async(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "health_query":
        metrics_arg = arguments.get("metrics")
        if not isinstance(metrics_arg, list) or not metrics_arg:
            return {"ok": False, "error": "metrics must be a non-empty list"}
        metrics = [str(m) for m in metrics_arg]
        days = int(arguments.get("days", 7) or 7)
        payload = await run_health_query(metrics=metrics, days=days)
        return {"ok": True, **payload}
    return execute_health_tool(tool_name, arguments)


def build_health_tool_message(tool_call_id: str, tool_name: str, arguments_json: str) -> dict[str, Any]:
    try:
        arguments = json.loads(arguments_json) if arguments_json else {}
        if not isinstance(arguments, dict):
            arguments = {}
    except Exception:
        arguments = {}
    payload = execute_health_tool(tool_name, arguments)
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": json.dumps(payload, ensure_ascii=False),
    }
