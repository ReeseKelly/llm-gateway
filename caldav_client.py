from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import caldav
from icalendar import Calendar, Event

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
            raise ValueError("CalDAV is not configured: CALDAV_URL/CALDAV_USERNAME/CALDAV_PASSWORD required")

        self._default_tz = ZoneInfo(settings.default_tz)
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
            wanted = settings.caldav_calendar_name.strip().lower()
            selected = None
            for c in calendars:
                c_name = (getattr(c, "name", None) or "").strip().lower()
                if c_name == wanted:
                    selected = c
                    break
            if selected is None:
                raise ValueError(f"Calendar not found: {settings.caldav_calendar_name}")
            self._calendar = selected
        else:
            self._calendar = calendars[0]

    def list_events(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        start_utc = self._to_utc(start)
        end_utc = self._to_utc(end)
        events = self._calendar.search(start=start_utc, end=end_utc, event=True, expand=True)

        parsed: list[CalendarEvent] = []
        for event in events:
            parsed_event = self._parse_event(event)
            if parsed_event is not None:
                parsed.append(parsed_event)
        parsed.sort(key=lambda e: e.start)
        return parsed

    def create_event(
        self,
        start: datetime,
        end: datetime,
        summary: str,
        description: str | None,
    ) -> CalendarEvent:
        start_utc = self._to_utc(start)
        end_utc = self._to_utc(end)
        if end_utc <= start_utc:
            raise ValueError("end must be later than start")

        cal = Calendar()
        cal.add("prodid", "-//llm-gateway//calendar-tools//")
        cal.add("version", "2.0")

        event = Event()
        event.add("uid", f"llm-gateway-{datetime.now(timezone.utc).timestamp()}@local")
        event.add("dtstamp", datetime.now(timezone.utc))
        event.add("dtstart", start_utc)
        event.add("dtend", end_utc)
        event.add("summary", summary)
        if description:
            event.add("description", description)
        cal.add_component(event)

        saved = self._calendar.save_event(cal.to_ical())
        parsed = self._parse_event(saved)
        if parsed is None:
            raise ValueError("Failed to parse created event")
        return parsed

    def _to_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            value = value.replace(tzinfo=self._default_tz)
        return value.astimezone(timezone.utc)

    def _to_utc_from_obj(self, value: object) -> datetime:
        if isinstance(value, datetime):
            return self._to_utc(value)
        if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
            return datetime(value.year, value.month, value.day, tzinfo=self._default_tz).astimezone(timezone.utc)
        raise ValueError("unsupported calendar datetime value")

    def _parse_event(self, raw: object) -> CalendarEvent | None:
        data = getattr(raw, "data", None)
        url = str(getattr(raw, "url", ""))
        if not data:
            return None

        cal = Calendar.from_ical(data)
        for component in cal.walk():
            if component.name != "VEVENT":
                continue
            dtstart = component.get("DTSTART")
            dtend = component.get("DTEND")
            if dtstart is None or dtend is None:
                continue
            start = self._to_utc_from_obj(dtstart.dt)
            end = self._to_utc_from_obj(dtend.dt)
            return CalendarEvent(
                id=str(component.get("UID") or url),
                summary=str(component.get("SUMMARY") or ""),
                start=start,
                end=end,
                description=str(component.get("DESCRIPTION")) if component.get("DESCRIPTION") else None,
            )
        return None
