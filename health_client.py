from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from config import get_settings


class BaseHealthProvider:
    def list_metrics(
        self,
        start: datetime,
        end: datetime,
        metrics: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        raise NotImplementedError


class FileHealthProvider(BaseHealthProvider):
    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def list_metrics(
        self,
        start: datetime,
        end: datetime,
        metrics: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        if not self._path.exists():
            return {metric: [] for metric in metrics}

        raw = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {metric: [] for metric in metrics}

        result: dict[str, list[dict[str, Any]]] = {}
        for metric in metrics:
            points = raw.get(metric, [])
            kept: list[dict[str, Any]] = []
            if isinstance(points, list):
                for point in points:
                    if not isinstance(point, dict):
                        continue
                    ts = point.get("timestamp")
                    if not isinstance(ts, str):
                        continue
                    text = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
                    try:
                        dt = datetime.fromisoformat(text)
                    except ValueError:
                        continue
                    if start <= dt <= end:
                        kept.append(point)
            result[metric] = kept
        return result


def get_health_provider() -> BaseHealthProvider:
    settings = get_settings()
    if settings.health_provider == "file":
        return FileHealthProvider(settings.health_data_path)
    return FileHealthProvider(settings.health_data_path)
