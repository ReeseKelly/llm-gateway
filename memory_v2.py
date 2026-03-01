from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
from typing import Any
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
    created_at: str
    updated_at: str
    ttl_days: int
    scope: str
    promoted_to_ltm: bool
    last_used_at: str | None = None


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
                created_at=_to_iso(now),
                updated_at=_to_iso(now),
                ttl_days=int(arguments.get("ttl_days") or settings.midterm_default_ttl_days),
                scope=scope,
                promoted_to_ltm=False,
                last_used_at=None,
            )
            rows.append(asdict(card))
            result = asdict(card)
        else:
            found["topic"] = topic
            found["summary"] = summary
            found["keywords"] = [str(x) for x in (arguments.get("keywords") or []) if isinstance(x, str)]
            found["scope"] = scope
            found["session_id"] = shared_session_id if scope == "session" else None
            found["ttl_days"] = int(arguments.get("ttl_days") or settings.midterm_default_ttl_days)
            found["updated_at"] = _to_iso(now)
            result = found

        _write_jsonl(_midterm_path(settings), rows)
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
