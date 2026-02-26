from __future__ import annotations

import json
from typing import Any

import httpx

from config import get_settings

WEATHER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "weather_query",
            "description": "Query current weather and short forecast for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "days": {"type": "integer", "minimum": 1, "maximum": 7},
                },
            },
        },
    }
]


def _normalize_weather_payload(data: dict[str, Any], fallback_location: str) -> dict[str, Any]:
    current_obj = data.get("current") if isinstance(data.get("current"), dict) else {}
    condition = None
    if isinstance(current_obj.get("weather"), list) and current_obj["weather"]:
        condition = (current_obj["weather"][0] or {}).get("description")

    forecast: list[dict[str, Any]] = []
    daily = data.get("daily") if isinstance(data.get("daily"), list) else []
    for day in daily:
        if not isinstance(day, dict):
            continue
        temp = day.get("temp") if isinstance(day.get("temp"), dict) else {}
        daily_condition = None
        if isinstance(day.get("weather"), list) and day["weather"]:
            daily_condition = (day["weather"][0] or {}).get("description")
        forecast.append(
            {
                "date": str(day.get("dt") or ""),
                "min": temp.get("min"),
                "max": temp.get("max"),
                "condition": daily_condition,
            }
        )

    return {
        "ok": True,
        "location": str(data.get("timezone") or fallback_location),
        "current": {
            "temp": current_obj.get("temp"),
            "condition": condition,
        },
        "forecast": forecast,
    }


async def execute_weather_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    try:
        if tool_name != "weather_query":
            return {"ok": False, "error": f"unsupported tool: {tool_name}"}
        if not settings.weather_api_url:
            return {"ok": False, "error": "WEATHER_API_URL is not configured"}

        days = max(1, min(int(arguments.get("days", 1)), 7))
        location = str(arguments.get("location") or settings.weather_default_location)

        params: dict[str, Any] = {"units": settings.weather_units, "days": days}
        if settings.weather_api_key:
            params["appid"] = settings.weather_api_key
            params["api_key"] = settings.weather_api_key

        lat = arguments.get("lat")
        lon = arguments.get("lon")
        if lat is not None and lon is not None:
            params["lat"] = lat
            params["lon"] = lon
        elif location:
            params["q"] = location
        else:
            return {"ok": False, "error": "location or lat/lon is required"}

        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
            resp = await client.get(settings.weather_api_url, params=params)
            resp.raise_for_status()
            data = resp.json()

        if not isinstance(data, dict):
            return {"ok": False, "error": "weather API returned non-object JSON"}
        return _normalize_weather_payload(data, location)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def build_weather_tool_message(tool_call_id: str, tool_name: str, arguments_json: str) -> dict[str, Any]:
    try:
        arguments = json.loads(arguments_json) if arguments_json else {}
        if not isinstance(arguments, dict):
            arguments = {}
    except Exception:
        arguments = {}
    payload = await execute_weather_tool(tool_name, arguments)
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": json.dumps(payload, ensure_ascii=False),
    }
