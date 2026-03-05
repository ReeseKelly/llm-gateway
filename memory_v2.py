from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
from typing import Any
from telegram_adapter import send_debug_message
from uuid import uuid4

from config import Settings
from memories import load_all_ltm_memories, load_pinned_memories

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except Exception as exc:
        logger.warning("memory_v2 read_jsonl failed path=%s err=%r", str(path), exc)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class NoteRecord:
    id: str
    session_id: str | None
    scope: str
    title: str
    content: str
    created_at: str
    updated_at: str
    ttl_days: int
    tags: list[str] = field(default_factory=list)


@dataclass
class MidtermCard:
    id: str
    session_id: str | None
    topic: str
    keywords: list[str]
    summary: str
    emotional_undertone: str | None
    created_at: str
    updated_at: str
    ttl_days: int
    scope: str
    promoted_to_ltm: bool
    last_used_at: str | None = None

class NoteChangeEvent:
    id: str
    note_id: str
    kind: str
    actor: str
    action: str
    timestamp: str
    title_before: str | None
    title_after: str | None
    content_before: str | None
    content_after: str | None
    tags_before: list[str] | None
    tags_after: list[str] | None


def _change_log_path(settings: Settings) -> Path:
    return Path(settings.notes_change_log_path)


def _append_change_event(settings: Settings, event: NoteChangeEvent) -> None:
    path = _change_log_path(settings)
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")


def _list_change_events(settings: Settings, *, kind: str, note_id: str | None = None) -> list[dict[str, Any]]:
    rows = _read_jsonl(_change_log_path(settings))
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("kind") or "") != kind:
            continue
        if note_id is not None and str(row.get("note_id") or "") != note_id:
            continue
        out.append(row)
    out.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
    return out


def notify_memory_event(settings: Settings, kind: str, record: NoteRecord | MidtermCard | dict[str, Any], actor: str) -> None:
    if actor != "model":
        return
    if not settings.memory_event_enabled or not settings.memory_event_telegram_chat_id:
        return

    payload = asdict(record) if not isinstance(record, dict) else record
    if kind == "note_created":
        text = (
            "🌱 L1 note (model)\n"
            f"title: {str(payload.get('title') or '').strip() or '(untitled)'}\n"
            f"tags: {list(payload.get('tags') or [])}\n"
            f"ttl: {int(payload.get('ttl_days') or 0)}d"
        )
    elif kind == "note_updated":
        text = (
            "✏️ L1 note updated (model)\n"
            f"title: {str(payload.get('title') or '').strip() or '(untitled)'}\n"
            f"tags: {list(payload.get('tags') or [])}"
        )
    elif kind == "midterm_upserted":
        text = (
            "🧩 L2 midterm (model)\n"
            f"topic: {str(payload.get('topic') or '').strip() or '(untitled)'}\n"
            f"keywords: {list(payload.get('keywords') or [])}"
        )
    else:
        return

    send_debug_message(text=text, chat_id=settings.memory_event_telegram_chat_id)


def get_active_notes(settings: Settings, shared_session_id: str | None, scope: str = "default") -> list[dict[str, Any]]:
    now = _utc_now()
    rows = _read_jsonl(_note_path(settings))
    out: list[dict[str, Any]] = []
    for row in rows:
        row_scope = str(row.get("scope") or "")
        if not _is_active(row.get("created_at"), row.get("ttl_days"), now):
            continue
        if scope == "global" and row_scope != "global":
            continue
        if scope == "session" and row_scope != "session":
            continue
        if scope == "default":
            if row_scope == "session" and row.get("session_id") != shared_session_id:
                continue
            if row_scope not in {"global", "session"}:
                continue
        elif scope == "all":
            if row_scope == "session" and row.get("session_id") != shared_session_id:
                continue
        elif scope not in {"global", "session"}:
            if row_scope == "session" and row.get("session_id") != shared_session_id:
                continue
            if row_scope not in {"global", "session"}:
                continue
        if row_scope == "session" and row.get("session_id") != shared_session_id:
            continue
        out.append(row)
    out.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
    return out


def get_note_by_id(settings: Settings, note_id: str) -> dict[str, Any] | None:
    for row in _read_jsonl(_note_path(settings)):
        if str(row.get("id") or "") == note_id:
            return row
    return None


def update_note_by_id(
    settings: Settings,
    note_id: str,
    *,
    actor: str,
    title: str | None = None,
    content: str | None = None,
    tags: list[str] | None = None,
    ttl_days: int | None = None,
) -> dict[str, Any] | None:
    rows = _read_jsonl(_note_path(settings))
    now_iso = _to_iso(_utc_now())
    updated: dict[str, Any] | None = None
    for row in rows:
        if str(row.get("id") or "") != note_id:
            continue
        before = dict(row)
        if title is not None:
            row["title"] = title
        if content is not None:
            row["content"] = content
        if tags is not None:
            row["tags"] = [str(x) for x in tags]
        if ttl_days is not None:
            row["ttl_days"] = int(ttl_days)
        row["updated_at"] = now_iso
        updated = row
        event = NoteChangeEvent(
            id=uuid4().hex,
            note_id=note_id,
            kind="note",
            actor=actor,
            action="update",
            timestamp=now_iso,
            title_before=before.get("title"),
            title_after=row.get("title"),
            content_before=before.get("content"),
            content_after=row.get("content"),
            tags_before=list(before.get("tags") or []),
            tags_after=list(row.get("tags") or []),
        )
        _append_change_event(settings, event)
        break
    if updated is None:
        return None
    _write_jsonl(_note_path(settings), rows)
    return updated


def get_active_midterms(settings: Settings, shared_session_id: str | None) -> list[dict[str, Any]]:
    now = _utc_now()
    rows = _read_jsonl(_midterm_path(settings))
    out: list[dict[str, Any]] = []
    for row in rows:
        scope = str(row.get("scope") or "")
        if scope not in {"global", "session"}:
            continue
        if scope == "session" and row.get("session_id") != shared_session_id:
            continue
        if not _is_active(row.get("created_at"), row.get("ttl_days"), now):
            continue
        out.append(row)
    out.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
    return out


def get_note_changes(settings: Settings, note_id: str) -> list[dict[str, Any]]:
    return _list_change_events(settings, kind="note", note_id=note_id)


def _build_short_lines(items: list[str], limit: int) -> list[str]:
    out: list[str] = []
    used = 0
    for item in items:
        if used + len(item) > limit:
            break
        out.append(item)
        used += len(item)
    return out


def build_active_notes_snippet(settings: Settings, session_id: str | None, now: datetime) -> str | None:
    notes = get_active_notes(settings, session_id, scope="default")
    midterms = get_active_midterms(settings, session_id)

    lines: list[str] = ["[ACTIVE NOTES - SHORT]"]
    l1_lines: list[str] = []
    for row in notes[:3]:
        title = _compress_text(str(row.get("title") or "(untitled)"), 48)
        content = _compress_text(str(row.get("content") or ""), 72)
        l1_lines.append(f"- {title}: {content}")

    l2_lines: list[str] = []
    for row in midterms[:2]:
        topic = _compress_text(str(row.get("topic") or "(untitled)"), 48)
        summary = _compress_text(str(row.get("summary") or ""), 72)
        undertone = _compress_text(str(row.get("emotional_undertone") or ""), 32)
        suffix = f" ({undertone})" if undertone else ""
        l2_lines.append(f"- {topic}: {summary}{suffix}")

    if l1_lines:
        lines.append("L1:")
        lines.extend(_build_short_lines(l1_lines, 220))
    if l2_lines:
        lines.append("L2:")
        lines.extend(_build_short_lines(l2_lines, 160))

    if len(lines) <= 1:
        return None
    snippet = "\n".join(lines)
    return _compress_text(snippet, 400)


@dataclass
class LTMTopicIndex:
    topic: str
    tag: str | None
    note_ids: list[str]
    keywords: list[str]
    created_at: str
    updated_at: str


NOTE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "note_create",
            "description": "Create a short-lived note (L1 scratch note) for the current user/session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["session", "global"]},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "ttl_days": {"type": "integer", "description": "Optional TTL in days; if omitted, use default."},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["scope", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "note_list",
            "description": "List active L1 notes for this user/session, optionally filtered by scope and/or tag.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["session", "global"]},
                    "tag": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "note_delete",
            "description": "Delete a specific L1 note.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
]

MIDTERM_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "midterm_upsert",
            "description": "Create or update a mid-term memory card (L2). Use this for patterns or facts that matter across days to weeks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Optional existing card id; if omitted, create a new card."},
                    "scope": {"type": "string", "enum": ["session", "global"]},
                    "topic": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "summary": {"type": "string"},
                    "emotional_undertone": {"type": "string"},
                    "ttl_days": {"type": "integer", "description": "Optional TTL in days; if omitted, use default."},
                },
                "required": ["scope", "topic", "summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "midterm_list",
            "description": "List active mid-term memory cards for this user/session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["session", "global"]},
                    "topic_contains": {"type": "string"},
                    "keyword": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "midterm_mark_promoted",
            "description": "Mark a mid-term card as promoted to long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
]

LTM_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "ltm_register_topic",
            "description": "Register or update a long-term topic index entry, pointing to external notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "tag": {"type": "string"},
                    "note_ids": {"type": "array", "items": {"type": "string"}},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ltm_search",
            "description": "Search long-term topics by keyword; returns topic metadata and note ids, but not full note contents.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

MEMORY_TOOLS = NOTE_TOOLS + MIDTERM_TOOLS + LTM_TOOLS


def _note_path(settings: Settings) -> Path:
    return Path(settings.notes_log_path)


def _midterm_path(settings: Settings) -> Path:
    return Path(settings.midterm_memory_path)


def _ltm_index_path(settings: Settings) -> Path:
    return Path(settings.ltm_index_path)


def _is_active(created_at: str | None, ttl_days: int | None, now: datetime) -> bool:
    created = _parse_iso(created_at)
    if created is None:
        return False
    ttl = ttl_days if isinstance(ttl_days, int) and ttl_days > 0 else 1
    return created + timedelta(days=ttl) > now


def _load_ltm_topics(settings: Settings) -> list[dict[str, Any]]:
    path = _ltm_index_path(settings)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []
    except Exception as exc:
        logger.warning("memory_v2 load ltm index failed path=%s err=%r", str(path), exc)
        return []


def _save_ltm_topics(settings: Settings, topics: list[dict[str, Any]]) -> None:
    path = _ltm_index_path(settings)
    _ensure_parent(path)
    path.write_text(json.dumps(topics, ensure_ascii=False, indent=2), encoding="utf-8")


def _compat_midterm_from_pinned(settings: Settings) -> list[dict[str, Any]]:
    # TODO: future migration to unified schema
    out: list[dict[str, Any]] = []
    for mem in load_pinned_memories(settings).values():
        if not isinstance(mem, dict):
            continue
        out.append(
            {
                "id": str(mem.get("id") or f"legacy-pin-{uuid4().hex[:8]}"),
                "session_id": mem.get("logical_session_id"),
                "topic": str(mem.get("topic") or "legacy pinned memory"),
                "keywords": list(mem.get("keywords") or []),
                "summary": str(mem.get("summary") or mem.get("content") or ""),
                "created_at": str(mem.get("created_at") or _to_iso(_utc_now())),
                "updated_at": str(mem.get("updated_at") or mem.get("created_at") or _to_iso(_utc_now())),
                "ttl_days": 3650,
                "scope": str(mem.get("scope") or "session"),
                "emotional_undertone": None,
                "promoted_to_ltm": False,
                "last_used_at": None,
                "legacy_source": "pinned",
            }
        )
    return out


def _compat_ltm_topics_from_legacy(settings: Settings) -> list[dict[str, Any]]:
    # TODO: future migration to unified schema
    out: list[dict[str, Any]] = []
    for mem in load_all_ltm_memories(settings):
        topic = str(mem.get("topic") or "legacy topic")
        now_iso = _to_iso(_utc_now())
        out.append(
            {
                "topic": topic,
                "tag": None,
                "note_ids": [str(mem.get("id"))] if mem.get("id") else [],
                "keywords": list(mem.get("keywords") or []),
                "created_at": str(mem.get("created_at") or now_iso),
                "updated_at": str(mem.get("updated_at") or now_iso),
                "legacy_source": "ltm_memories",
            }
        )
    return out


def execute_memory_tool(tool_name: str, arguments: dict[str, Any], *, settings: Settings, shared_session_id: str | None) -> dict[str, Any]:
    now = _utc_now()

    if tool_name == "note_create":
        scope = str(arguments.get("scope") or "").strip()
        if scope not in {"session", "global"}:
            return {"ok": False, "error": "invalid scope"}
        content = str(arguments.get("content") or "").strip()
        if not content:
            return {"ok": False, "error": "content is required"}
        record = NoteRecord(
            id=uuid4().hex,
            session_id=shared_session_id if scope == "session" else None,
            scope=scope,
            title=str(arguments.get("title") or "").strip(),
            content=content,
            created_at=_to_iso(now),
            updated_at=_to_iso(now),
            ttl_days=int(arguments.get("ttl_days") or settings.notes_default_ttl_days),
            tags=[str(x) for x in (arguments.get("tags") or []) if isinstance(x, str)],
        )
        rows = _read_jsonl(_note_path(settings))
        rows.append(asdict(record))
        _write_jsonl(_note_path(settings), rows)
        _append_change_event(
            settings,
            NoteChangeEvent(
                id=uuid4().hex,
                note_id=record.id,
                kind="note",
                actor="model",
                action="create",
                timestamp=record.created_at,
                title_before=None,
                title_after=record.title,
                content_before=None,
                content_after=record.content,
                tags_before=None,
                tags_after=list(record.tags),
            ),
        )
        notify_memory_event(settings, "note_created", record, "model")
        logger.info("memory_v2 note_create id=%s scope=%s", record.id, scope)
        return {"ok": True, "note": asdict(record)}

    if tool_name == "note_list":
        rows = _read_jsonl(_note_path(settings))
        scope_filter = arguments.get("scope")
        tag_filter = str(arguments.get("tag") or "").strip().lower()
        notes: list[dict[str, Any]] = []
        for row in rows:
            scope = str(row.get("scope") or "")
            if scope_filter in {"session", "global"} and scope != scope_filter:
                continue
            if not _is_active(row.get("created_at"), row.get("ttl_days"), now):
                continue
            if scope == "session" and row.get("session_id") != shared_session_id:
                continue
            if tag_filter:
                tags = [str(t).lower() for t in (row.get("tags") or [])]
                if tag_filter not in tags:
                    continue
            notes.append(row)
        notes.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
        return {"ok": True, "count": len(notes), "notes": notes}

    if tool_name == "note_delete":
        target_id = str(arguments.get("id") or "").strip()
        if not target_id:
            return {"ok": False, "error": "id is required"}
        rows = _read_jsonl(_note_path(settings))
        kept = [r for r in rows if str(r.get("id") or "") != target_id]
        deleted = len(rows) - len(kept)
        if deleted:
            _write_jsonl(_note_path(settings), kept)
        logger.info("memory_v2 note_delete id=%s deleted=%s", target_id, deleted)
        return {"ok": True, "deleted": deleted, "id": target_id}

    if tool_name == "midterm_upsert":
        scope = str(arguments.get("scope") or "").strip()
        if scope not in {"session", "global"}:
            return {"ok": False, "error": "invalid scope"}
        topic = str(arguments.get("topic") or "").strip()
        summary = str(arguments.get("summary") or "").strip()
        if not topic or not summary:
            return {"ok": False, "error": "topic and summary are required"}

        rows = _read_jsonl(_midterm_path(settings))
        target_id = str(arguments.get("id") or "").strip()
        found = None
        for row in rows:
            if str(row.get("id") or "") == target_id and target_id:
                found = row
                break

        if found is None:
            card = MidtermCard(
                id=target_id or uuid4().hex,
                session_id=shared_session_id if scope == "session" else None,
                topic=topic,
                keywords=[str(x) for x in (arguments.get("keywords") or []) if isinstance(x, str)],
                summary=summary,
                emotional_undertone=str(arguments.get("emotional_undertone") or "").strip() or None,
                created_at=_to_iso(now),
                updated_at=_to_iso(now),
                ttl_days=int(arguments.get("ttl_days") or settings.midterm_default_ttl_days),
                scope=scope,
                promoted_to_ltm=False,
                last_used_at=None,
            )
            rows.append(asdict(card))
            result = asdict(card)
            _append_change_event(
                settings,
                NoteChangeEvent(
                    id=uuid4().hex,
                    note_id=card.id,
                    kind="midterm",
                    actor="model",
                    action="create",
                    timestamp=card.created_at,
                    title_before=None,
                    title_after=card.topic,
                    content_before=None,
                    content_after=card.summary,
                    tags_before=None,
                    tags_after=list(card.keywords),
                ),
            )
        else:
            before = dict(found)
            found["topic"] = topic
            found["summary"] = summary
            found["emotional_undertone"] = str(arguments.get("emotional_undertone") or "").strip() or found.get("emotional_undertone")
            found["keywords"] = [str(x) for x in (arguments.get("keywords") or []) if isinstance(x, str)]
            found["scope"] = scope
            found["session_id"] = shared_session_id if scope == "session" else None
            found["ttl_days"] = int(arguments.get("ttl_days") or settings.midterm_default_ttl_days)
            found["updated_at"] = _to_iso(now)
            result = found
            _append_change_event(
                settings,
                NoteChangeEvent(
                    id=uuid4().hex,
                    note_id=str(found.get("id") or ""),
                    kind="midterm",
                    actor="model",
                    action="update",
                    timestamp=str(found.get("updated_at") or _to_iso(now)),
                    title_before=str(before.get("topic") or "") or None,
                    title_after=str(found.get("topic") or "") or None,
                    content_before=(
                        f"summary: {str(before.get('summary') or '')}\nemotional_undertone: {str(before.get('emotional_undertone') or '')}"
                    ),
                    content_after=(
                        f"summary: {str(found.get('summary') or '')}\nemotional_undertone: {str(found.get('emotional_undertone') or '')}"
                    ),
                    tags_before=list(before.get("keywords") or []),
                    tags_after=list(found.get("keywords") or []),
                ),
            )

        _write_jsonl(_midterm_path(settings), rows)
        notify_memory_event(settings, "midterm_upserted", result, "model")
        logger.info("memory_v2 midterm_upsert id=%s scope=%s", result.get("id"), result.get("scope"))
        return {"ok": True, "card": result}

    if tool_name == "midterm_list":
        rows = _read_jsonl(_midterm_path(settings))
        # legacy compatibility read-only mapping
        rows.extend(_compat_midterm_from_pinned(settings))

        scope_filter = arguments.get("scope")
        topic_contains = str(arguments.get("topic_contains") or "").strip().lower()
        keyword_filter = str(arguments.get("keyword") or "").strip().lower()

        out: list[dict[str, Any]] = []
        for row in rows:
            scope = str(row.get("scope") or "")
            if scope_filter in {"session", "global"} and scope != scope_filter:
                continue
            if scope == "session" and row.get("session_id") != shared_session_id:
                continue
            if not _is_active(row.get("created_at"), row.get("ttl_days"), now):
                continue
            if topic_contains and topic_contains not in str(row.get("topic") or "").lower():
                continue
            if keyword_filter:
                kws = [str(x).lower() for x in (row.get("keywords") or [])]
                if keyword_filter not in kws:
                    continue
            row["last_used_at"] = _to_iso(now)
            out.append(row)

        # persist only non-legacy rows to update last_used_at
        persisted_rows = [r for r in rows if r.get("legacy_source") is None]
        _write_jsonl(_midterm_path(settings), persisted_rows)

        out.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
        return {"ok": True, "count": len(out), "cards": out}

    if tool_name == "midterm_mark_promoted":
        target_id = str(arguments.get("id") or "").strip()
        if not target_id:
            return {"ok": False, "error": "id is required"}
        rows = _read_jsonl(_midterm_path(settings))
        updated = False
        card: dict[str, Any] | None = None
        for row in rows:
            if str(row.get("id") or "") == target_id:
                row["promoted_to_ltm"] = True
                row["updated_at"] = _to_iso(now)
                updated = True
                card = row
                break
        if updated:
            _write_jsonl(_midterm_path(settings), rows)
        logger.info("memory_v2 midterm_mark_promoted id=%s updated=%s", target_id, updated)
        return {"ok": updated, "id": target_id, "card": card}

    if tool_name == "ltm_register_topic":
        topic = str(arguments.get("topic") or "").strip()
        if not topic:
            return {"ok": False, "error": "topic is required"}
        tag = str(arguments.get("tag") or "").strip() or None
        note_ids = [str(x) for x in (arguments.get("note_ids") or []) if isinstance(x, str)]
        keywords = [str(x) for x in (arguments.get("keywords") or []) if isinstance(x, str)]

        rows = _load_ltm_topics(settings)
        found = None
        for row in rows:
            if str(row.get("topic") or "").lower() == topic.lower():
                found = row
                break

        if found is None:
            rec = LTMTopicIndex(
                topic=topic,
                tag=tag,
                note_ids=note_ids,
                keywords=keywords,
                created_at=_to_iso(now),
                updated_at=_to_iso(now),
            )
            rows.append(asdict(rec))
            result = asdict(rec)
        else:
            found["tag"] = tag
            found["note_ids"] = note_ids
            found["keywords"] = keywords
            found["updated_at"] = _to_iso(now)
            result = found

        _save_ltm_topics(settings, rows)
        logger.info("memory_v2 ltm_register_topic topic=%s", topic[:80])
        return {"ok": True, "topic": result}

    if tool_name == "ltm_search":
        query = str(arguments.get("query") or "").strip().lower()
        if not query:
            return {"ok": False, "error": "query is required"}
        rows = _load_ltm_topics(settings)
        rows.extend(_compat_ltm_topics_from_legacy(settings))

        hits = []
        for row in rows:
            topic = str(row.get("topic") or "")
            tag = str(row.get("tag") or "")
            keywords = [str(x) for x in (row.get("keywords") or [])]
            haystack = " ".join([topic, tag, *keywords]).lower()
            if query in haystack:
                hits.append(row)
        return {"ok": True, "count": len(hits), "topics": hits}

    return {"ok": False, "error": f"unsupported tool: {tool_name}"}


def build_memory_tool_message(
    tool_call_id: str,
    tool_name: str,
    arguments_json: str,
    *,
    settings: Settings,
    shared_session_id: str | None,
) -> dict[str, Any]:
    try:
        arguments = json.loads(arguments_json) if arguments_json else {}
        if not isinstance(arguments, dict):
            arguments = {}
    except Exception:
        arguments = {}
    payload = execute_memory_tool(tool_name, arguments, settings=settings, shared_session_id=shared_session_id)
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": json.dumps(payload, ensure_ascii=False),
    }


def find_promotion_candidates(settings: Settings, now: datetime | None = None) -> list[dict[str, Any]]:
    now_dt = now or _utc_now()
    rows = _read_jsonl(_midterm_path(settings))
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if bool(row.get("promoted_to_ltm")):
            continue
        created = _parse_iso(str(row.get("created_at") or ""))
        last_used = _parse_iso(str(row.get("last_used_at") or ""))
        if not created:
            continue
        if now_dt - created <= timedelta(days=30):
            continue
        if not last_used or now_dt - last_used > timedelta(days=14):
            continue
        candidates.append(row)
    return candidates



def _compress_text(text: str, max_chars: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _select_recent_notes_by_tag(notes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}

    def has_tag(note: dict[str, Any], needle: str) -> bool:
        tags = [str(t).lower() for t in (note.get("tags") or []) if isinstance(t, str)]
        return needle in tags

    for note in notes:
        if "health" not in selected and has_tag(note, "health"):
            selected["health"] = note
        if "core_need" not in selected and has_tag(note, "core-need"):
            selected["core_need"] = note
        if "work" not in selected:
            tags = [str(t).lower() for t in (note.get("tags") or []) if isinstance(t, str)]
            if any(t in {"work", "gateway", "fibrosis"} for t in tags):
                selected["work"] = note
        if "schedule" not in selected and has_tag(note, "schedule"):
            selected["schedule"] = note
    return selected


def _select_active_midterm_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not cards:
        return []

    prioritized: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []
    for card in cards:
        keywords = [str(k).lower() for k in (card.get("keywords") or []) if isinstance(k, str)]
        if "sunset" in keywords:
            prioritized.append(card)
        else:
            others.append(card)

    def sort_key(card: dict[str, Any]) -> str:
        return str(card.get("updated_at") or card.get("created_at") or "")

    prioritized.sort(key=sort_key, reverse=True)
    others.sort(key=sort_key, reverse=True)

    out = prioritized[:1]
    if len(out) < 2 and others:
        out.extend(others[: 2 - len(out)])
    return out


def build_active_memory_snippet(
    settings: Settings,
    summary_session_id: str | None,
) -> tuple[str | None, dict[str, Any]]:
    now = _utc_now()

    l1_rows = [
        row for row in _read_jsonl(_note_path(settings))
        if str(row.get("scope") or "") == "global"
        and _is_active(row.get("created_at"), row.get("ttl_days"), now)
    ]
    l1_rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)

    selected_notes = _select_recent_notes_by_tag(l1_rows)

    l2_rows = [
        row for row in _read_jsonl(_midterm_path(settings))
        if str(row.get("scope") or "") == "global"
        and _is_active(row.get("created_at"), row.get("ttl_days"), now)
    ]
    l2_rows.sort(key=lambda row: str(row.get("updated_at") or row.get("created_at") or ""), reverse=True)
    selected_cards = _select_active_midterm_cards(l2_rows)

    debug_meta = {
        "summary_session_id": summary_session_id,
        "l1_counts": len(l1_rows),
        "l1_selected": list(selected_notes.keys()),
        "l2_counts": len(l2_rows),
        "l2_selected_topics": [str(card.get("topic") or "") for card in selected_cards],
    }

    if not selected_notes and not selected_cards:
        return None, debug_meta

    sections: list[str] = ["[ACTIVE NOTES]"]

    title_map = {
        "health": "Health (short-term)",
        "core_need": "Core need (short-term)",
        "work": "Work focus",
        "schedule": "Schedule constraints",
    }
    for key in ("health", "core_need", "work", "schedule"):
        note = selected_notes.get(key)
        if not note:
            continue
        title = _compress_text(str(note.get("title") or ""), 80)
        content = _compress_text(str(note.get("content") or ""), 180)
        body = f"{title}: {content}" if title else content
        sections.append(f"{title_map[key]}:\n- {body}")

    if selected_cards:
        sections.append("Current episode:")
        for card in selected_cards:
            topic = _compress_text(str(card.get("topic") or ""), 80)
            summary = _compress_text(str(card.get("summary") or ""), 220)
            if topic:
                sections.append(f"- {topic}: {summary}")
            else:
                sections.append(f"- {summary}")
            undertone = card.get("emotional_undertone")
            if isinstance(undertone, str) and undertone.strip():
                sections.append(f"- Emotional undertone: {_compress_text(undertone, 120)}")

    snippet = "\n\n".join(sections)
    snippet = _compress_text(snippet, 1000)
    return snippet, debug_meta
