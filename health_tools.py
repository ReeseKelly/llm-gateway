from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
import logging
from typing import Any

from health_client import get_health_provider

logger = logging.getLogger(__name__)

HEALTH_TOOLS: list[dict[str, Any]] = [
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
                        "items": {"type": "string"},
                        "description": "e.g. ['steps','heart_rate']",
                    },
                    "aggregation": {
                        "type": "string",
                        "enum": ["raw", "hourly", "daily"],
                    },
                },
                "required": ["start", "end", "metrics"],
            },
        },
    }
]


def _parse_iso_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _aggregate_points(points: list[dict[str, Any]], aggregation: str) -> list[dict[str, Any]]:
    if aggregation == "raw":
        return points

    buckets: dict[str, list[float]] = defaultdict(list)
    for point in points:
        ts = point.get("timestamp")
        val = point.get("value")
        if not isinstance(ts, str) or not isinstance(val, (int, float)):
            continue
        normalized = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            continue
        key = dt.strftime("%Y-%m-%d") if aggregation == "daily" else dt.strftime("%Y-%m-%dT%H:00:00")
        buckets[key].append(float(val))

    out: list[dict[str, Any]] = []
    for key in sorted(buckets):
        values = buckets[key]
        out.append({"timestamp": key, "value": sum(values), "avg": sum(values) / len(values)})
    return out


def execute_health_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        if tool_name != "health_query":
            return {"ok": False, "error": f"unsupported tool: {tool_name}"}

        start = _parse_iso_datetime(str(arguments["start"]))
        end = _parse_iso_datetime(str(arguments["end"]))
        metrics = arguments.get("metrics", [])
        if not isinstance(metrics, list) or not metrics:
            return {"ok": False, "error": "metrics must be a non-empty list"}

        metric_names = [str(m) for m in metrics]
        aggregation = str(arguments.get("aggregation") or "raw")
        if aggregation not in {"raw", "hourly", "daily"}:
            aggregation = "raw"

        provider = get_health_provider()
        raw_data = provider.list_metrics(start=start, end=end, metrics=metric_names)

        aggregated: dict[str, list[dict[str, Any]]] = {}
        for metric, points in raw_data.items():
            aggregated[metric] = _aggregate_points(points, aggregation)

        total_points = sum(len(v) for v in aggregated.values())
        logger.info("health_query metrics=%s total_points=%s", list(aggregated.keys()), total_points)

        return {"ok": True, "metrics": aggregated}
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
