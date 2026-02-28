from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import logging
from typing import Any

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
                    "start": {"type": "string", "description": "ISO 8601 datetime"},
                    "end": {"type": "string", "description": "ISO 8601 datetime"},
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string", "enum": sorted(SUPPORTED_HEALTH_METRICS)},
                    },
                    "aggregation": {"type": "string", "enum": ["raw", "daily"]},
                    "limit": {"type": "integer"},
                },
            },
        },
    },
]


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _aggregate_daily(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    passthrough: list[dict[str, Any]] = []

    for row in records:
        val = row.get("value")
        day = str(row.get("start", ""))[:10]
        if isinstance(val, (int, float)) and day:
            buckets[day].append(float(val))
        else:
            passthrough.append(row)

    out: list[dict[str, Any]] = []
    for day, values in sorted(buckets.items()):
        out.append(
            {
                "day": day,
                "count": len(values),
                "sum": round(sum(values), 3),
                "avg": round(sum(values) / len(values), 3),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
            }
        )
    out.extend(passthrough[:20])
    return out


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
            logger.info("health_log metrics=%s records=%s", metric_names, written)
            return {"ok": True, "written": written, "metrics": metric_names}

        if tool_name == "health_query":
            start = _parse_iso_datetime(arguments.get("start"))
            end = _parse_iso_datetime(arguments.get("end"))

            metrics_arg = arguments.get("metrics")
            metrics = None
            if isinstance(metrics_arg, list) and metrics_arg:
                metrics = [str(m) for m in metrics_arg if str(m) in SUPPORTED_HEALTH_METRICS]

            limit = int(arguments.get("limit", 200))
            limit = max(1, min(limit, 1000))

            rows = provider.list_metrics(start=start, end=end, metrics=metrics, limit=limit)

            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                grouped[row.metric].append(row.to_dict())

            aggregation = str(arguments.get("aggregation") or "raw")
            if aggregation == "daily":
                payload_metrics = {k: _aggregate_daily(v) for k, v in grouped.items()}
            else:
                payload_metrics = dict(grouped)

            total_points = sum(len(v) for v in payload_metrics.values())
            logger.info("health_query metrics=%s total_points=%s", list(payload_metrics.keys()), total_points)

            return {
                "ok": True,
                "metrics": payload_metrics,
                "count": total_points,
                "window": {
                    "start": start.isoformat() if start else None,
                    "end": end.isoformat() if end else None,
                },
            }

        return {"ok": False, "error": f"unsupported tool: {tool_name}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


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
