from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from config import get_settings

SUPPORTED_HEALTH_METRICS = {"sleep_hours", "steps", "heart_rate", "period"}


@dataclass
class HealthRecord:
    metric: str
    value: float | str
    start: datetime
    end: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "start": self.start.astimezone(timezone.utc).isoformat(),
            "end": self.end.astimezone(timezone.utc).isoformat(),
        }


class BaseHealthProvider:
    def append_records(self, records: list[HealthRecord]) -> int:
        raise NotImplementedError

    def list_metrics(
        self,
        start: datetime | None,
        end: datetime | None,
        metrics: list[str] | None,
        limit: int = 500,
    ) -> list[HealthRecord]:
        raise NotImplementedError


class FileHealthProvider(BaseHealthProvider):
    def __init__(self, data_path: str, log_path: str) -> None:
        self._data_path = Path(data_path)
        self._log_path = Path(log_path)

    def append_records(self, records: list[HealthRecord]) -> int:
        if not records:
            return 0
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return len(records)

    def list_metrics(
        self,
        start: datetime | None,
        end: datetime | None,
        metrics: list[str] | None,
        limit: int = 500,
    ) -> list[HealthRecord]:
        wanted = set(metrics or SUPPORTED_HEALTH_METRICS)
        raw_records = self._read_jsonl_records() + self._read_legacy_data_records()

        out: list[HealthRecord] = []
        for rec in raw_records:
            if rec.metric not in wanted:
                continue
            if start and rec.end < start:
                continue
            if end and rec.start > end:
                continue
            out.append(rec)

        out.sort(key=lambda r: r.start)
        if limit > 0:
            out = out[-limit:]
        return out

    def _read_jsonl_records(self) -> list[HealthRecord]:
        if not self._log_path.exists():
            return []
        out: list[HealthRecord] = []
        for line in self._log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rec = parse_health_record(obj)
            if rec is not None:
                out.append(rec)
        return out

    def _read_legacy_data_records(self) -> list[HealthRecord]:
        if not self._data_path.exists():
            return []
        try:
            raw = json.loads(self._data_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        out: list[HealthRecord] = []
        if not isinstance(raw, dict):
            return out
        for metric, points in raw.items():
            if metric not in SUPPORTED_HEALTH_METRICS or not isinstance(points, list):
                continue
            for point in points:
                if not isinstance(point, dict):
                    continue
                ts = point.get("timestamp") or point.get("start")
                rec = parse_health_record(
                    {
                        "metric": metric,
                        "value": point.get("value"),
                        "start": ts,
                        "end": point.get("end") or ts,
                    }
                )
                if rec is not None:
                    out.append(rec)
        return out


def parse_health_record(obj: dict[str, Any]) -> HealthRecord | None:
    metric = str(obj.get("metric") or "").strip()
    if metric not in SUPPORTED_HEALTH_METRICS:
        return None

    value = obj.get("value")
    if isinstance(value, bool):
        value = str(value)
    if not isinstance(value, (int, float, str)):
        return None

    start = _parse_datetime(obj.get("start"))
    end = _parse_datetime(obj.get("end"))
    if start is None:
        return None
    if end is None:
        end = start
    if end < start:
        start, end = end, start

    numeric_metrics = {"sleep_hours", "steps", "heart_rate"}
    if metric in numeric_metrics and isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None

    return HealthRecord(metric=metric, value=value, start=start, end=end)


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def get_health_provider() -> BaseHealthProvider:
    settings = get_settings()
    if settings.health_provider == "file":
        return FileHealthProvider(settings.health_data_path, settings.health_log_path)
    return FileHealthProvider(settings.health_data_path, settings.health_log_path)
