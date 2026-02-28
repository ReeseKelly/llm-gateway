from __future__ import annotations

from datetime import datetime
import json
import logging
from typing import Any

from caldav_client import CalDAVCalendarClient, CalendarEvent

logger = logging.getLogger(__name__)

CALENDAR_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calendar_query",
            "description": "List calendar events between start and end time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "ISO 8601 datetime"},
                    "end": {"type": "string", "description": "ISO 8601 datetime"},
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_create",
            "description": "Create a calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "start": {"type": "string", "description": "ISO 8601 datetime"},
                    "end": {"type": "string", "description": "ISO 8601 datetime"},
                    "description": {"type": "string"},
                },
                "required": ["summary", "start", "end"],
            },
        },
    },
]


def _parse_iso_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _event_to_dict(event: CalendarEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "summary": event.summary,
        "start": event.start.isoformat(),
        "end": event.end.isoformat(),
        "description": event.description,
    }


def execute_calendar_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        client = CalDAVCalendarClient()

        if tool_name == "calendar_query":
            start = _parse_iso_datetime(str(arguments["start"]))
            end = _parse_iso_datetime(str(arguments["end"]))
            events = client.list_events(start=start, end=end)
            logger.info("TOOL calendar_query start=%s end=%s count=%s", start.isoformat(), end.isoformat(), len(events))
            print(f"TOOL calendar_query start={start.isoformat()} end={end.isoformat()} count={len(events)}")
            return {"ok": True, "events": [_event_to_dict(e) for e in events]}

        if tool_name == "calendar_create":
            start = _parse_iso_datetime(str(arguments["start"]))
            end = _parse_iso_datetime(str(arguments["end"]))
            event = client.create_event(
                start=start,
                end=end,
                summary=str(arguments["summary"]),
                description=str(arguments.get("description")) if arguments.get("description") else None,
            )
            ok = bool(event.id)
            logger.info("TOOL calendar_create summary=%s start=%s end=%s ok=%s", str(arguments.get("summary", ""))[:80], start.isoformat(), end.isoformat(), ok)
            print(f"TOOL calendar_create summary={str(arguments.get('summary', ''))[:80]!r} start={start.isoformat()} end={end.isoformat()} ok={ok}")
            return {"ok": True, "event": _event_to_dict(event)}

        return {"ok": False, "error": f"unsupported tool: {tool_name}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def build_calendar_tool_message(tool_call_id: str, tool_name: str, arguments_json: str) -> dict[str, Any]:
    try:
        arguments = json.loads(arguments_json) if arguments_json else {}
        if not isinstance(arguments, dict):
            arguments = {}
    except Exception:
        arguments = {}
    payload = execute_calendar_tool(tool_name, arguments)
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": json.dumps(payload, ensure_ascii=False),
    }
