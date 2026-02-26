from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import caldav
from icalendar import Calendar

from config import get_settings


@dataclass
class CalendarEvent:
    id: str
    summary: str
    start: datetime
    end: datetime
    description: str | None = None


class CalDAVCalendarClient:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.caldav_url or not settings.caldav_username or not settings.caldav_password:
            raise ValueError("CalDAV settings are incomplete")

        self._tz = ZoneInfo(settings.timezone)
        self._client = caldav.DAVClient(
            url=settings.caldav_url,
            username=settings.caldav_username,
            password=settings.caldav_password,
        )

        principal = self._client.principal()
        if settings.caldav_principal:
            principal = caldav.Principal(client=self._client, url=settings.caldav_principal)
        self._principal = principal

        calendars = list(self._principal.calendars())
        if not calendars:
            raise ValueError("No calendars available from CalDAV principal")

        if settings.caldav_calendar_name:
            name = settings.caldav_calendar_name.strip().lower()
            selected = None
            for c in calendars:
                c_name = (getattr(c, "name", None) or "").strip().lower()
                if c_name == name:
                    selected = c
                    break
            if selected is None:
                raise ValueError(f"Calendar not found: {settings.caldav_calendar_name}")
            self._calendar = selected
        else:
            self._calendar = calendars[0]

    def list_events(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        events = self._calendar.search(start=start, end=end, event=True, expand=True)
        parsed: list[CalendarEvent] = []
        for e in events:
            parsed_event = self._parse_event(e)
            if parsed_event is not None:
                parsed.append(parsed_event)
        return parsed

    def create_event(
        self,
        start: datetime,
        end: datetime,
        summary: str,
        description: str | None,
    ) -> CalendarEvent:
        cal = Calendar()
        cal.add("prodid", "-//llm-gateway//calendar-tools//")
        cal.add("version", "2.0")

        from icalendar import Event

        event = Event()
        event.add("uid", f"llm-gateway-{datetime.now().timestamp()}@local")
        event.add("dtstart", self._to_tz(start))
        event.add("dtend", self._to_tz(end))
        event.add("summary", summary)
        if description:
            event.add("description", description)
        cal.add_component(event)

        saved = self._calendar.save_event(cal.to_ical())
        parsed = self._parse_event(saved)
        if parsed is None:
            raise ValueError("Failed to parse created event")
        return parsed

    def _to_tz(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=self._tz)
        return value.astimezone(self._tz)

    def _parse_event(self, raw: object) -> CalendarEvent | None:
        data = getattr(raw, "data", None)
        url = str(getattr(raw, "url", ""))
        if not data:
            return None
        cal = Calendar.from_ical(data)
        for component in cal.walk():
            if component.name != "VEVENT":
                continue
            dtstart = component.get("DTSTART").dt
            dtend = component.get("DTEND").dt
            if isinstance(dtstart, datetime):
                start = dtstart
            else:
                start = datetime.combine(dtstart, datetime.min.time(), self._tz)
            if isinstance(dtend, datetime):
                end = dtend
            else:
                end = datetime.combine(dtend, datetime.min.time(), self._tz)
            return CalendarEvent(
                id=str(component.get("UID") or url),
                summary=str(component.get("SUMMARY") or ""),
                start=start,
                end=end,
                description=str(component.get("DESCRIPTION")) if component.get("DESCRIPTION") else None,
            )
        return None
