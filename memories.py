from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from config import Settings


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_memory_id(memory: dict[str, Any], existing_ids: set[str]) -> str:
    mem_id = memory.get("id")
    if not mem_id:
        mem_id = f"mem-{_iso_now()}-{uuid4().hex[:8]}"

    while mem_id in existing_ids:
        mem_id = f"{mem_id}-{uuid4().hex[:4]}"
    return mem_id


def _normalize_memory(memory: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(memory)
    now_iso = _iso_now()
    normalized.setdefault("kind", "pin")
    normalized.setdefault("scope", "session")
    normalized.setdefault("logical_session_id", None)
    normalized.setdefault("topic", None)
    normalized.setdefault("keywords", [])
    normalized.setdefault("tags", [])
    normalized.setdefault("summary", "")
    normalized.setdefault("content", normalized.get("summary", ""))
    normalized.setdefault(
        "source",
        {
            "from_timestamp": None,
            "to_timestamp": None,
            "num_records": 0,
        },
    )
    normalized.setdefault("source_ref", None)
    normalized.setdefault("importance", "high")
    normalized.setdefault("created_at", now_iso)
    normalized.setdefault("updated_at", now_iso)
    return normalized


def _read_memory_store(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return {}
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        print(f"DEBUG memory store root is not dict: {path}")
        return {}
    except Exception as exc:
        print(f"DEBUG failed to load memory store {path}: {exc!r}")
        return {}


def _write_memory_store(path: Path, data: dict[str, dict]) -> None:
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"DEBUG failed to save memory store {path}: {exc!r}")


def get_pinned_memories_path(settings: Settings) -> Path:
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "pinned_memories.json"


def get_ltm_memories_path(settings: Settings) -> Path:
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "ltm_memories.json"


def load_pinned_memories(settings: Settings) -> dict[str, dict]:
    """
    Load the entire pinned_memories store as a dict[id -> memory].
    If the file doesn't exist, return {}.
    If parsing fails, print a debug message and return {}.
    """
    return _read_memory_store(get_pinned_memories_path(settings))


def save_pinned_memories(settings: Settings, data: dict[str, dict]) -> None:
    """
    Save the given pinned memories dict to logs/pinned_memories.json
    using UTF-8 encoding. Overwrite the file.
    """
    _write_memory_store(get_pinned_memories_path(settings), data)


def write_memory_to_obsidian(mem: dict[str, Any]) -> None:
    """
    把一条 memory 以 markdown 形式写入 Obsidian vault。
    目前路径固定为 D:\\llm-memory-ob\\_gateway_memories\\<id>.md
    如果目录不存在则自动创建。
    """
    try:
        base_dir = Path(r"D:\llm-memory-ob")  # 你现在的 vault 根目录
        target_dir = base_dir / "_gateway_memories"
        target_dir.mkdir(parents=True, exist_ok=True)

        mem_id = str(mem.get("id") or "")
        if not mem_id:
            return  # 没有 id 就不写

        filename = target_dir / f"{mem_id}.md"

        # --- YAML frontmatter ---
        lines: list[str] = []
        lines.append("---")
        lines.append(f"id: {mem_id}")
        lines.append(f"kind: {mem.get('kind', '')}")
        lines.append(f"scope: {mem.get('scope', '')}")
        logical_session_id = mem.get("logical_session_id")
        if logical_session_id is not None:
            lines.append(f"logical_session_id: {logical_session_id}")
        topic = mem.get("topic")
        if topic is not None:
            lines.append(f"topic: {topic}")

        # keywords
        keywords = mem.get("keywords") or []
        lines.append("keywords:")
        for kw in keywords:
            lines.append(f"  - {kw}")

        # tags
        tags = mem.get("tags") or []
        lines.append("tags:")
        for tag in tags:
            lines.append(f"  - {tag}")

        importance = mem.get("importance", "high")
        lines.append(f"importance: {importance}")
        created_at = mem.get("created_at")
        if created_at:
            lines.append(f"created_at: {created_at}")
        updated_at = mem.get("updated_at")
        if updated_at:
            lines.append(f"updated_at: {updated_at}")
        source_ref = mem.get("source_ref")
        if source_ref:
            lines.append(f"source_ref: {source_ref}")
        lines.append("---")
        lines.append("")  # 空行

        # --- 正文：用 summary 或 content ---
        body = str(mem.get("content") or mem.get("summary") or "").strip()
        if body:
            lines.append(body)
            lines.append("")

        text = "\n".join(lines)
        filename.write_text(text, encoding="utf-8")

    except Exception as exc:
        print(f"DEBUG write_memory_to_obsidian failed: {exc!r}")



def add_pinned_memory(settings: Settings, memory: dict[str, Any]) -> str:
    """
    - Load current pinned_memories.json.
    - Ensure the memory has a unique 'id' (generate if missing).
    - Insert it into the dict.
    - Save back to file.
    - Return the memory id.
    """
    data = load_pinned_memories(settings)
    normalized = _normalize_memory(memory)
    normalized["kind"] = "pin"
    normalized["scope"] = "session"

    mem_id = _ensure_memory_id(normalized, set(data.keys()))
    normalized["id"] = mem_id

    data[mem_id] = normalized
    save_pinned_memories(settings, data)
    write_memory_to_obsidian(normalized)
    return mem_id


def list_pinned_memories_by_session(
    settings: Settings,
    logical_session_id: str,
) -> list[dict]:
    """
    Return a list of pinned memories whose 'logical_session_id'
    matches the given value.
    The ordering can be by created_at ascending.
    """
    data = load_pinned_memories(settings)
    items = [
        mem
        for mem in data.values()
        if isinstance(mem, dict) and mem.get("logical_session_id") == logical_session_id
    ]
    return sorted(items, key=lambda item: str(item.get("created_at", "")))


def load_ltm_memories(settings: Settings) -> dict[str, dict]:
    return _read_memory_store(get_ltm_memories_path(settings))


def save_ltm_memories(settings: Settings, data: dict[str, dict]) -> None:
    _write_memory_store(get_ltm_memories_path(settings), data)


def load_all_ltm_memories(settings: Settings) -> list[dict]:
    return sorted(
        [mem for mem in load_ltm_memories(settings).values() if isinstance(mem, dict)],
        key=lambda item: str(item.get("created_at", "")),
    )


def add_ltm_memory(settings: Settings, mem: dict[str, Any]) -> str:
    data = load_ltm_memories(settings)
    normalized = _normalize_memory(mem)
    normalized["kind"] = "ltm"
    normalized["scope"] = "global"
    normalized["logical_session_id"] = None

    mem_id = _ensure_memory_id(normalized, set(data.keys()))
    normalized["id"] = mem_id

    data[mem_id] = normalized
    save_ltm_memories(settings, data)
    write_memory_to_obsidian(normalized)
    return mem_id
