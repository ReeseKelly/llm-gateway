from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import httpx

from config import get_settings


WEATHER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "weather_query",
            "description": "Query current weather and near-term forecast.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "days": {"type": "integer", "minimum": 1, "maximum": 7},
                },
            },
        },
    }
]


@dataclass
class WeatherAPIClient:
    api_url: str
    api_key: str | None
    units: str

    async def fetch(self, *, location: str, days: int) -> dict[str, Any]:
        params: dict[str, Any] = {"q": location, "days": days, "units": self.units}
        if self.api_key:
            params["key"] = self.api_key
            params["api_key"] = self.api_key
            params["appid"] = self.api_key
        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
            resp = await client.get(self.api_url, params=params)
            resp.raise_for_status()
            payload = resp.json()
        if not isinstance(payload, dict):
            raise ValueError("weather API returned non-object JSON")
        return payload


def _pick(obj: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in obj:
            return obj.get(key)
    return None


def detect_extreme_conditions(forecast: list[dict[str, Any]]) -> list[str]:
    flags: set[str] = set()
    for day in forecast:
        max_temp = day.get("max")
        min_temp = day.get("min")
        wind = day.get("wind_speed")
        precip = day.get("precip_chance")
        condition = str(day.get("condition") or "").lower()

        if isinstance(max_temp, (int, float)) and max_temp >= 35:
            flags.add("hot")
        if isinstance(min_temp, (int, float)) and min_temp <= -5:
            flags.add("cold")
        if isinstance(wind, (int, float)) and wind >= 12:
            flags.add("strong_wind")
        if isinstance(precip, (int, float)) and precip >= 70:
            flags.add("heavy_precip")
        if "storm" in condition or "thunder" in condition:
            flags.add("storm")
        if "snow" in condition:
            flags.add("snow")
    return sorted(flags)


def _normalize_weather_payload(data: dict[str, Any], fallback_location: str) -> dict[str, Any]:
    current = data.get("current") if isinstance(data.get("current"), dict) else {}
    forecast_days = []

    # weatherapi.com style
    forecast_obj = data.get("forecast") if isinstance(data.get("forecast"), dict) else {}
    forecastday = forecast_obj.get("forecastday") if isinstance(forecast_obj.get("forecastday"), list) else []
    for day in forecastday:
        if not isinstance(day, dict):
            continue
        day_obj = day.get("day") if isinstance(day.get("day"), dict) else {}
        condition_obj = day_obj.get("condition") if isinstance(day_obj.get("condition"), dict) else {}
        forecast_days.append(
            {
                "date": str(day.get("date") or ""),
                "min": _pick(day_obj, "mintemp_c", "mintemp_f", "min"),
                "max": _pick(day_obj, "maxtemp_c", "maxtemp_f", "max"),
                "condition": _pick(condition_obj, "text", "description"),
                "precip_chance": _pick(day_obj, "daily_chance_of_rain", "daily_chance_of_snow"),
                "wind_speed": _pick(day_obj, "maxwind_kph", "maxwind_mph", "wind_speed"),
            }
        )

    # onecall style fallback
    if not forecast_days and isinstance(data.get("daily"), list):
        for row in data["daily"]:
            if not isinstance(row, dict):
                continue
            temp = row.get("temp") if isinstance(row.get("temp"), dict) else {}
            weather_arr = row.get("weather") if isinstance(row.get("weather"), list) else []
            first_weather = weather_arr[0] if weather_arr and isinstance(weather_arr[0], dict) else {}
            forecast_days.append(
                {
                    "date": str(row.get("dt") or ""),
                    "min": _pick(temp, "min"),
                    "max": _pick(temp, "max"),
                    "condition": _pick(first_weather, "description", "main"),
                    "precip_chance": row.get("pop"),
                    "wind_speed": _pick(row, "wind_speed"),
                }
            )

    current_condition = None
    if isinstance(current.get("condition"), dict):
        current_condition = _pick(current["condition"], "text")
    if not current_condition and isinstance(current.get("weather"), list) and current["weather"]:
        first = current["weather"][0]
        if isinstance(first, dict):
            current_condition = _pick(first, "description", "main")

    location_obj = data.get("location") if isinstance(data.get("location"), dict) else {}
    normalized = {
        "ok": True,
        "location": str(_pick(location_obj, "name") or data.get("timezone") or fallback_location),
        "current": {
            "temp": _pick(current, "temp_c", "temp_f", "temp"),
            "condition": current_condition,
            "precipitation": _pick(current, "precip_mm", "precip_in"),
            "wind_speed": _pick(current, "wind_kph", "wind_mph", "wind_speed"),
        },
        "forecast": forecast_days,
    }
    normalized["extreme_flags"] = detect_extreme_conditions(forecast_days)
    return normalized


async def get_daily_summary(location: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    target_location = location or settings.weather_default_location
    client = WeatherAPIClient(
        api_url=settings.weather_api_url,
        api_key=settings.weather_api_key,
        units=settings.weather_units,
    )
    payload = await client.fetch(location=target_location, days=1)
    return _normalize_weather_payload(payload, target_location)


async def execute_weather_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    try:
        if tool_name != "weather_query":
            return {"ok": False, "error": f"unsupported tool: {tool_name}"}
        if not settings.weather_api_url:
            return {"ok": False, "error": "WEATHER_API_URL is not configured"}

        location = str(arguments.get("location") or settings.weather_default_location).strip()
        if not location:
            return {"ok": False, "error": "location is required when WEATHER_DEFAULT_LOCATION is empty"}

        days = int(arguments.get("days", 1) or 1)
        days = max(1, min(days, 7))

        client = WeatherAPIClient(
            api_url=settings.weather_api_url,
            api_key=settings.weather_api_key,
            units=settings.weather_units,
        )
        data = await client.fetch(location=location, days=days)
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
