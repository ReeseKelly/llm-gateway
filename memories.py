from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import re

from config import Settings


def get_pinned_memories_path(settings: Settings) -> Path:
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "pinned_memories.json"


def load_pinned_memories(settings: Settings) -> dict[str, dict]:
    """
    Load the entire pinned_memories store as a dict[id -> memory].
    If the file doesn't exist, return {}.
    If parsing fails, print a debug message and return {}.
    """
    path = get_pinned_memories_path(settings)
    if not path.exists():
        return {}

    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return {}
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        print("DEBUG pinned_memories has non-dict root, ignoring")
        return {}
    except Exception as exc:
        print(f"DEBUG failed to load pinned memories: {exc!r}")
        return {}


def save_pinned_memories(settings: Settings, data: dict[str, dict]) -> None:
    """
    Save the given pinned memories dict to logs/pinned_memories.json
    using UTF-8 encoding. Overwrite the file.
    """
    path = get_pinned_memories_path(settings)
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"DEBUG failed to save pinned memories: {exc!r}")


def add_pinned_memory(settings: Settings, memory: dict[str, Any]) -> str:
    """
    - Load current pinned_memories.json.
    - Ensure the memory has a unique 'id' (generate if missing).
    - Insert it into the dict.
    - Save back to file.
    - Return the memory id.
    """
    data = load_pinned_memories(settings)

    mem_id = memory.get("id")
    if not mem_id:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        mem_id = f"mem-{timestamp}-{uuid4().hex[:8]}"

    while mem_id in data:
        mem_id = f"{mem_id}-{uuid4().hex[:4]}"

    memory["id"] = mem_id
    data[mem_id] = memory
    save_pinned_memories(settings, data)
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

def _tokenize_text(text: str) -> set[str]:
    """
    非精确分词，只是粗略把英文/数字/下划线抓出来，转小写。
    中文场景下，keyword 建议用英文或较短的拼音/标签。
    """
    tokens = re.findall(r"\w+", text.lower())
    return set(tokens)

def select_relevant_pinned_memories(
    settings: Settings,
    logical_session_id: str,
    current_text: str,
    max_memories: int = 3,
) -> list[dict[str, Any]]:
    """
    粗筛：从某个 session 的 pinned 里选出“可能和当前对话相关”的几条。
    规则（第一版很朴素）：
    - scope == "session" 且 logical_session_id 匹配，或者 scope is None
    - keywords ∩ current_text_tokens 的数量作为 score
    - 如果没有 keywords，就 score=0，仅在完全没其他候选时作为兜底
    - 按 (score, importance, updated_at) 倒序取前 N 条
    """
    all_mems = list_pinned_memories_by_session(settings, logical_session_id)
    if not all_mems:
        return []

    tokens = _tokenize_text(current_text)
    selected: list[tuple[int, dict[str, Any]]] = []

    for mem in all_mems:
        if not isinstance(mem, dict):
            continue

        scope = mem.get("scope") or "session"
        if scope == "session" and mem.get("logical_session_id") != logical_session_id:
            continue

        kw_list = mem.get("keywords") or []
        topic = mem.get("topic")
        if topic:
            kw_list.append(str(topic)) # 把topic作为默认keyword
        kw_list = [str(k).lower() for k in kw_list if k]

        # 粗暴得很直接：统计有多少关键词出现在当前文本里
        score = 0
        if kw_list and tokens:
            for kw in kw_list:
                if kw in tokens:
                    score += 1

        # importance 简单映射一下
        imp_raw = str(mem.get("importance", "medium")).lower()
        importance = 2 if imp_raw == "high" else 1 if imp_raw == "medium" else 0

        # 把 updated_at 转成排序用的字符串（ISO 本身就能比较）
        updated_at = str(mem.get("updated_at", ""))

        # 暂时用 (score, importance, updated_at) 三元排序
        mem["_score_tuple"] = (score, importance, updated_at)
        selected.append((score, mem))

    # 如果所有 score 都是 0，可以选择：
    # - 要么直接返回 []（不注入 pinned）
    # - 要么兜底取最近更新的 1 条
    if not selected:
        return []

    # 找出有正分的
    positive = [mem for score, mem in selected if score > 0]
    if positive:
        candidates = positive
    else:
        # 兜底：按更新时间取 1 条
        candidates = [mem for _, mem in selected]
        candidates.sort(key=lambda m: m["_score_tuple"], reverse=True)
        return candidates[:1]

    # 有正分的情况下，按 _score_tuple 排序
    candidates.sort(key=lambda m: m["_score_tuple"], reverse=True)
    return candidates[:max_memories]