from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import httpx
from fastapi import Body, FastAPI, HTTPException, Request
from httpx_socks import AsyncProxyTransport

from config import Settings, get_settings
from memories import (
    add_ltm_memory,
    add_pinned_memory,
    list_pinned_memories_by_session,
    load_all_ltm_memories,
    load_ltm_memories,
    load_pinned_memories,
)

app = FastAPI()

MIN_RECORDS_FOR_SUMMARY = 8
UPDATE_EVERY_RECORDS = 10
MAX_PIN_RECORDS = 12

ACTIVE_MEMORIES: dict[str, dict[str, int]] = {}

SUMMARY_SYSTEM_PROMPT = """You are a summarization model that maintains a rolling memory for a long-running, high-context conversation between a single user and an assistant.

Your task:
- Given (1) the previous rolling summary for this logical session (if any), and (2) the latest segment of conversation,
  update the rolling summary so that it stays compact but faithful.

Very important rules:
- Treat the previous summary as the baseline. Update or refine it only where the new conversation clearly adds, clarifies, or contradicts it.
- Do NOT invent facts, events, or feelings that are not clearly supported by the conversation segment.
- Preserve concrete details that matter for future reasoning: decisions, hypotheses, preferences, constraints, open questions, plans, and TODOs.
- Capture emotional tone only at the level that is stable and recurring; do NOT psychoanalyze or speculate.
- Assume you cannot see the full history except for the previous summary plus this new segment. Be conservative when in doubt.

For pruning old details:
- If a detail appears *only* in the previous summary, does not appear at all in the latest conversation segment, and looks like a one-off daily-life remark (e.g. a single meal, a transient mood, small talk), you may remove it from the summary.
- Prefer to keep stable facts, long-term projects, enduring preferences, and recurring emotional patterns over transient details.

Output format (MUST be valid JSON, no extra text):
{
  "summary": "6~10 sentences capturing the current state of this logical session.",
  "key_points": [
    "bullet-style key facts, decisions, or hypotheses that the assistant should remember",
    "... more items as needed"
  ]
}
"""

PIN_MEMORY_SYSTEM_PROMPT = """You will see a short segment of conversation logs between a user and an assistant.
Please write 3–5 sentences summarizing only the information that would be long-term useful for this user and future assistants: facts, preferences, decisions, or recurring themes.
Do not restate all details; keep it compact and reusable."""

MEMORY_CONSOLIDATION_SYSTEM_PROMPT = """You will see several memory cards (id, kind, topic, summary).
Produce a compact, faithful consolidated summary that preserves key facts and preferences,
without inventing anything. Output plain text only."""


def strip_kelivo_autoprompt(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    清理 Kelivo 注入的自动提示段（Memory Tool 说明等），
    目前只处理第一条 system message 里的 '## Memory Tool' 段。
    """
    cleaned: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        if idx == 0 and msg.get("role") == "system":
            content = str(msg.get("content", ""))
            if "## Memory Tool" in content:
                content = re.sub(r"## Memory Tool[\s\S]*$", "", content)
            msg = {**msg, "content": content}
        cleaned.append(msg)
    return cleaned


def extract_new_turn(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    从完整的 messages 列表中抽取“最新的一轮对话”：
    - 最后一个 user message
    - 以及它前面最近的一个 assistant message（如果存在）

    如果没找到 user，就返回去掉 system 之后的原 messages。
    """
    if not isinstance(messages, list) or not messages:
        return messages

    last_user = None
    prev_assistant = None

    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user" and last_user is None:
            last_user = msg
            for j in range(idx - 1, -1, -1):
                m2 = messages[j]
                if isinstance(m2, dict) and m2.get("role") == "assistant":
                    prev_assistant = m2
                    break
            break

    if last_user is None:
        return [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]

    result: list[dict[str, Any]] = []
    if prev_assistant is not None:
        result.append(prev_assistant)
    result.append(last_user)

    result = [m for m in result if m.get("role") != "system"]
    return result


def build_httpx_client_kwargs(settings: Settings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "timeout": 60.0,
        "trust_env": False,
    }
    if settings.outbound_proxy_url:
        try:
            kwargs["transport"] = AsyncProxyTransport.from_url(settings.outbound_proxy_url)
        except Exception as exc:
            print(f"DEBUG failed to create proxy transport: {exc!r}")
    return kwargs


def get_chat_log_path(settings: Settings) -> Path:
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "chat_log.jsonl"


def load_session_records(settings: Settings, logical_session_id: str) -> list[dict[str, Any]]:
    path = get_chat_log_path(settings)
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"DEBUG failed to parse chat log line: {exc!r}")
                continue
            if obj.get("logical_session_id") == logical_session_id:
                records.append(obj)
    return records


def get_summary_store_path(settings: Settings) -> Path:
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / settings.summary_store_path


def load_session_summary(settings: Settings, logical_session_id: str) -> dict[str, Any] | None:
    path = get_summary_store_path(settings)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"DEBUG failed to load session summary: {exc!r}")
        return None
    if not isinstance(data, dict):
        print("DEBUG session_summaries has non-dict root, ignoring")
        return None
    item = data.get(logical_session_id)
    return item if isinstance(item, dict) else None


def load_session_summaries(settings: Settings) -> dict[str, Any]:
    path = get_summary_store_path(settings)
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return {}
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        print("DEBUG session_summaries has non-dict root, ignoring")
        return {}
    except Exception as exc:
        print(f"DEBUG failed to load session summaries: {exc!r}")
        return {}


def save_session_summary(settings: Settings, logical_session_id: str, summary_obj: dict[str, Any]) -> None:
    path = get_summary_store_path(settings)
    data = load_session_summaries(settings)
    data[logical_session_id] = summary_obj
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"DEBUG failed to save session summary: {exc!r}")


def append_chat_log(
    settings: Settings,
    logical_session_id: str,
    request_messages: list[dict[str, Any]],
    reply_message: dict[str, Any],
) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "logical_session_id": logical_session_id,
        "request_messages": request_messages,
        "reply_message": reply_message,
    }
    try:
        with get_chat_log_path(settings).open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"DEBUG failed to append chat log: {exc!r}")


def should_pin_from_messages(messages: list[dict[str, Any]]) -> bool:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return "/pin" in str(msg.get("content", ""))
    return False


def extract_recent_user_text(history_messages: list[dict[str, Any]], max_messages: int = 2) -> str:
    texts: list[str] = []
    for msg in reversed(history_messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            texts.append(str(msg.get("content", "")))
        if len(texts) >= max_messages:
            break
    texts.reverse()
    return "\n".join(texts)


def tokenize_text(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9_\-]{2,}", text)}


def _importance_score(value: Any) -> int:
    table = {"high": 2, "medium": 1, "low": 0}
    return table.get(str(value).lower(), 0)


def route_memories_with_small_model(
    current_text: str,
    candidate_mems: list[dict[str, Any]],
    max_memories: int,
) -> list[dict[str, Any]]:
    """
    Placeholder for small-model routing.
    """
    _ = current_text
    ranked = sorted(
        candidate_mems,
        key=lambda m: m.get("_score_tuple", (0, 0, "")),
        reverse=True,
    )
    return ranked[:max_memories]


def select_relevant_memories(
    settings: Settings,
    logical_session_id: str,
    current_text: str,
    max_memories: int,
    include_kinds: tuple[str, ...] = ("pin", "ltm"),
) -> list[dict[str, Any]]:
    """
    Coarse selection of candidate memories (pin + LTM).
    """
    _ = settings
    tokens = tokenize_text(current_text)

    candidates: list[dict[str, Any]] = []
    if "pin" in include_kinds:
        for mem in load_pinned_memories(settings).values():
            if not isinstance(mem, dict):
                continue
            if mem.get("kind", "pin") != "pin":
                continue
            if mem.get("scope", "session") == "session" and mem.get("logical_session_id") != logical_session_id:
                continue
            candidates.append(dict(mem))

    if "ltm" in include_kinds:
        for mem in load_ltm_memories(settings).values():
            if not isinstance(mem, dict):
                continue
            if mem.get("kind", "ltm") != "ltm":
                continue
            if mem.get("scope", "global") != "global":
                continue
            candidates.append(dict(mem))

    for mem in candidates:
        mem_keywords = {
            str(k).lower()
            for k in (mem.get("keywords") or [])
            if isinstance(k, str)
        }
        keyword_hits = len(tokens.intersection(mem_keywords)) if tokens else 0
        importance_score = _importance_score(mem.get("importance"))
        updated_at = str(mem.get("updated_at", ""))
        mem["_score_tuple"] = (keyword_hits, importance_score, updated_at)

    candidates = [m for m in candidates if m.get("_score_tuple", (0, 0, ""))[0] > 0 or not tokens]
    if not candidates:
        candidates = sorted(candidates, key=lambda m: m.get("updated_at", ""), reverse=True)

    return route_memories_with_small_model(current_text, candidates, max_memories)


def mark_memories_active(session_id: str, mem_ids: list[str], ttl: int = 3) -> None:
    if not session_id or not mem_ids:
        return
    bucket = ACTIVE_MEMORIES.setdefault(session_id, {})
    for mem_id in mem_ids:
        current = bucket.get(mem_id, 0)
        bucket[mem_id] = max(current, ttl)


def get_active_memory_ids(session_id: str) -> list[str]:
    if not session_id:
        return []
    bucket = ACTIVE_MEMORIES.get(session_id, {})
    return [mem_id for mem_id, ttl in bucket.items() if ttl > 0]


def decay_active_memories(session_id: str) -> None:
    if not session_id:
        return
    bucket = ACTIVE_MEMORIES.get(session_id)
    if not bucket:
        return
    to_del: list[str] = []
    for mem_id, ttl in bucket.items():
        new_ttl = ttl - 1
        if new_ttl <= 0:
            to_del.append(mem_id)
        else:
            bucket[mem_id] = new_ttl
    for mem_id in to_del:
        bucket.pop(mem_id, None)
    if not bucket:
        ACTIVE_MEMORIES.pop(session_id, None)


def resolve_memories_by_ids(settings: Settings, memory_ids: list[str]) -> list[dict[str, Any]]:
    pinned = load_pinned_memories(settings)
    ltms = load_ltm_memories(settings)
    resolved: list[dict[str, Any]] = []
    for mem_id in memory_ids:
        mem = pinned.get(mem_id) or ltms.get(mem_id)
        if isinstance(mem, dict):
            resolved.append(mem)
    return resolved


def build_memories_system_message(active_memories: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not active_memories:
        return None

    lines = [
        "[MEMORIES]",
        "These are pinned/LTM notes that appear relevant to the current conversation.",
        "If they conflict with the user's current message, trust the user and ask for clarification.",
    ]
    for idx, mem in enumerate(active_memories, start=1):
        lines.append(
            f"\n[{idx}] kind={mem.get('kind', 'unknown')} topic={mem.get('topic')}"
        )
        lines.append(str(mem.get("summary") or mem.get("content") or ""))

    return {"role": "system", "content": "\n".join(lines)}


async def maybe_consolidate_active_memories(
    settings: Settings,
    session_id: str,
    active_memories: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    If consolidation is enabled and there are enough active memories,
    call a small model to produce a single consolidated summary.
    """
    if not settings.memory_consolidation_enabled:
        return None
    if len(active_memories) < settings.memory_consolidation_min_count:
        return None

    model_name = settings.memory_consolidation_model or settings.summary_model
    if not model_name:
        return None

    mem_lines: list[str] = []
    for mem in active_memories:
        mem_lines.append(
            f"[{mem.get('id')}] kind={mem.get('kind')} topic={mem.get('topic')}\n"
            f"summary: {mem.get('summary') or mem.get('content') or ''}"
        )
    user_content = "\n\n".join(mem_lines)

    payload = {
        "model": model_name,
        "stream": False,
        "messages": [
            {"role": "system", "content": MEMORY_CONSOLIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    url = f"{settings.openrouter_base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(**build_httpx_client_kwargs(settings)) as client:
            response = await client.post(url, json=payload, headers=headers)
    except Exception as exc:
        print(f"DEBUG memory consolidation call failed: {exc!r}")
        return None

    if response.status_code >= 400:
        print(
            "DEBUG memory consolidation upstream error: "
            f"{response.status_code} {response.text[:200]!r}"
        )
        return None

    try:
        data = response.json()
        consolidated_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as exc:
        print(f"DEBUG failed to parse consolidation response: {exc!r}")
        return None

    if not str(consolidated_text).strip():
        return None

    return {
        "content": str(consolidated_text).strip(),
        "debug_ids": [str(mem.get("id")) for mem in active_memories if mem.get("id")],
        "session_id": session_id,
    }


def build_consolidated_memories_system_message(consolidated: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "system",
        "content": (
            "[MEMORIES - CONSOLIDATED]\n"
            "This is a consolidated view of several relevant memories. "
            "If needed, you may ask the user for clarification.\n\n"
            f"{consolidated.get('content', '')}"
        ),
    }


def estimate_tokens_for_messages(messages: list[dict[str, Any]]) -> int:
    total_chars = 0
    for msg in messages:
        if isinstance(msg, dict):
            total_chars += len(str(msg.get("content", "")))
    return total_chars // 2


def apply_context_budget(
    settings: Settings,
    system_messages: list[dict[str, Any]],
    history_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    initial_messages = system_messages + history_messages
    tokens_before = estimate_tokens_for_messages(initial_messages)
    print(f"DEBUG context tokens before trim: {tokens_before}")
    if tokens_before <= settings.context_max_tokens:
        return initial_messages

    # 找出 history 里面的 user / assistant / system 索引
    user_indexes = [
        idx for idx, msg in enumerate(history_messages) if msg.get("role") == "user"
    ]
    assistant_indexes = [
        idx for idx, msg in enumerate(history_messages) if msg.get("role") == "assistant"
    ]
    system_history_indexes = [
        idx for idx, msg in enumerate(history_messages) if msg.get("role") == "system"
    ]

    # 必须保留：
    # - 最近几条 user
    # - 最近几条 assistant
    # - 所有 history 里的 system（Kelivo <memories> 也在这里）
    keep_indexes: set[int] = set(
        user_indexes[-settings.context_keep_last_user_messages :]
    )
    keep_indexes.update(
        assistant_indexes[-settings.context_keep_last_assistant_messages :]
    )
    keep_indexes.update(system_history_indexes)

    current_history = list(history_messages)
    num_trimmed = 0

    def _build(messages_slice: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return system_messages + messages_slice

    # 从最老的 history 开始试着删掉非必须的 user/assistant
    for idx in range(len(current_history)):
        if idx in keep_indexes:
            continue

        if current_history[idx] is None:
            continue

        current_history[idx] = None
        candidate = [m for m in current_history if m is not None]
        candidate_tokens = estimate_tokens_for_messages(_build(candidate))
        num_trimmed += 1

        if candidate_tokens <= settings.context_max_tokens:
            print(f"DEBUG context tokens after trim: {candidate_tokens}")
            print(f"DEBUG trimmed {num_trimmed} history messages for context budget")
            return _build(candidate)

    # 如果怎么删都超限，只保留 must-keep，那就意味著 context_max_tokens 本身太小
    minimal_history = [
        msg
        for idx, msg in enumerate(history_messages)
        if idx in keep_indexes
    ]
    final_messages = _build(minimal_history)
    tokens_after = estimate_tokens_for_messages(final_messages)
    print(f"DEBUG context tokens after trim: {tokens_after}")
    print(f"DEBUG trimmed {num_trimmed} history messages for context budget")
    if tokens_after > settings.context_max_tokens:
        print("DEBUG context still above budget after mandatory keep set")
    return final_messages



def _build_pin_conversation_segment(records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for record in records:
        request_messages = record.get("request_messages", [])
        if isinstance(request_messages, list):
            for msg in request_messages:
                lines.append(f"{msg.get('role', 'unknown')}: {msg.get('content', '')}")
        reply = record.get("reply_message")
        if isinstance(reply, dict):
            lines.append(f"{reply.get('role', 'assistant')}: {reply.get('content', '')}")
    return "\n".join(lines)


async def create_pinned_memory_from_session(
    settings: Settings,
    logical_session_id: str,
    model_name: str | None = None,
) -> str | None:
    records = load_session_records(settings, logical_session_id)
    if not records:
        print(f"DEBUG no records available to pin for session={logical_session_id}")
        return None

    selected = records[-MAX_PIN_RECORDS:]
    segment = _build_pin_conversation_segment(selected)

    print(f"DEBUG [PIN] using summary_model = {settings.summary_model!r}")
    print("DEBUG pin segment (first 400 chars):", segment[:400].replace("\n","\\n"))

    model_to_use = model_name or settings.summary_model

    payload = {
        "model": model_to_use,
        "stream": False,
        "messages": [
            {"role": "system", "content": PIN_MEMORY_SYSTEM_PROMPT},
            {"role": "user", "content": segment},
        ],
    }
    url = f"{settings.openrouter_base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(**build_httpx_client_kwargs(settings)) as client:
            response = await client.post(url, json=payload, headers=headers)
        print("DEBUG [PIN] summarizer status:", response.status_code)
        print("DEBUG [PIN] summarizer body (first 200 chars):", response.text[:200].replace("\n", "\\n"))
    
    except Exception as exc:
        print(f"DEBUG pinned memory summarizer call failed: {exc!r}")
        return None

    if response.status_code >= 400:
        print(
            f"DEBUG pinned memory summarizer upstream error: "
            f"{response.status_code} {response.text[:200]!r}"
        )
        return None

    try:
        data = response.json()
        generated_summary = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print("DEBUG pinned summary raw content (first 200 chars):", str(generated_summary)[:200].replace("\n","\\n"))
    except Exception as exc:
        print(f"DEBUG failed to parse pinned memory response: {exc!r}")
        return None

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    memory = {
        "kind": "pin",
        "scope": "session",
        "logical_session_id": logical_session_id,
        "topic": None,
        "keywords": [],
        "tags": ["pinned"],
        "summary": str(generated_summary).strip(),
        "content": str(generated_summary).strip(),
        "source": {
            "from_timestamp": selected[0].get("timestamp") if selected else None,
            "to_timestamp": selected[-1].get("timestamp") if selected else None,
            "num_records": len(selected),
        },
        "source_ref": None,
        "importance": "high",
        "created_at": now_iso,
        "updated_at": now_iso,
    }

    mem_id = add_pinned_memory(settings, memory)
    print(
        f"DEBUG pinned memory created: id={mem_id}, "
        f"logical_session_id={logical_session_id}, "
        f"num_records={len(selected)}"
    )
    return mem_id


def create_ltm_from_pins(
    settings: Settings,
    pins: list[dict[str, Any]],
    topic: str,
    keywords: list[str],
    tags: list[str],
) -> str:
    """
    Create a new LTM card (kind="ltm", scope="global") from multiple pinned memories.
    """
    combined = "\n\n".join(
        str(pin.get("summary") or pin.get("content") or "")
        for pin in pins
        if isinstance(pin, dict)
    ).strip()
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    mem = {
        "kind": "ltm",
        "scope": "global",
        "logical_session_id": None,
        "topic": topic,
        "keywords": keywords,
        "tags": tags,
        "summary": combined,
        "content": combined,
        "source": {
            "from_timestamp": None,
            "to_timestamp": None,
            "num_records": len(pins),
        },
        "source_ref": None,
        "importance": "high",
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    return add_ltm_memory(settings, mem)


async def run_session_summarization(settings: Settings, logical_session_id: str) -> dict[str, Any]:
    records = load_session_records(settings, logical_session_id)
    total_records = len(records)
    if not records:
        return {"status": "no_records", "detail": "No records for this session."}
    if len(records) < MIN_RECORDS_FOR_SUMMARY:
        return {"status": "too_short", "detail": "Not enough records to summarize."}

    recent_records = records[-settings.summary_max_turns :]
    lines: list[str] = []
    for record in recent_records:
        for msg in record.get("request_messages", []) or []:
            lines.append(f"{msg.get('role', 'unknown')}: {msg.get('content', '')}")
        reply = record.get("reply_message")
        if isinstance(reply, dict):
            lines.append(f"{reply.get('role', 'assistant')}: {reply.get('content', '')}")

    summaries = load_session_summaries(settings)
    previous_summary = ""
    prev = summaries.get(logical_session_id)
    if isinstance(prev, dict):
        previous_summary = str(prev.get("summary", ""))

    messages: list[dict[str, Any]] = [{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}]
    if previous_summary:
        messages.append(
            {
                "role": "system",
                "content": "Previous rolling summary for this logical session:\n" + previous_summary,
            }
        )
    messages.append(
        {
            "role": "user",
            "content": (
                f"Here is the latest conversation segment for logical_session_id = {logical_session_id}.\n\n"
                "LATEST CONVERSATION SEGMENT:\n"
                + "\n".join(lines)
            ),
        }
    )

    payload = {"model": settings.summary_model, "stream": False, "messages": messages}
    url = f"{settings.openrouter_base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(**build_httpx_client_kwargs(settings)) as client:
            response = await client.post(url, json=payload, headers=headers)
    except Exception as exc:
        print(f"DEBUG Exception when calling OpenRouter summarizer: {exc!r}")
        return {"status": "error", "detail": "Error talking to OpenRouter"}

    if response.status_code >= 400:
        print(
            "DEBUG OpenRouter summarizer returned error: "
            f"{response.status_code} {response.text[:200]!r}"
        )
        return {"status": "error", "detail": f"OpenRouter error: {response.status_code}"}

    try:
        response_data = response.json()
        model_output = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as exc:
        print(f"DEBUG failed to parse OpenRouter response JSON: {exc!r}")
        return {"status": "error", "detail": "Invalid OpenRouter response"}

    try:
        core = json.loads(model_output)
        if not isinstance(core, dict):
            raise ValueError("summary output root is not dict")
    except Exception:
        core = {"summary": str(model_output), "key_points": [], "open_questions": []}

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "ok",
        "summary": core.get("summary", ""),
        "key_points": core.get("key_points", []),
        "open_questions": core.get("open_questions", []),
        "total_records": total_records,
    }
    save_session_summary(settings, logical_session_id, result)
    return result


async def maybe_auto_summarize_session(settings: Settings, logical_session_id: str) -> None:
    records = load_session_records(settings, logical_session_id)
    total = len(records)
    if total < MIN_RECORDS_FOR_SUMMARY:
        return

    summary = load_session_summary(settings, logical_session_id)
    if summary is None:
        print(
            f"DEBUG auto-summarize check: logical_session_id={logical_session_id}, "
            f"total={total}, seen=0, diff={total}"
        )
        await run_session_summarization(settings, logical_session_id)
        return

    try:
        seen = int(summary.get("total_records", 0))
    except Exception:
        seen = 0

    diff = total - seen
    print(
        f"DEBUG auto-summarize check: logical_session_id={logical_session_id}, "
        f"total={total}, seen={seen}, diff={diff}"
    )
    if diff >= UPDATE_EVERY_RECORDS:
        await run_session_summarization(settings, logical_session_id)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    body_bytes = await request.body()
    try:
        payload = json.loads(body_bytes)
    except json.JSONDecodeError as exc:
        print(f"DEBUG failed to parse request JSON: {exc!r}")
        print(f"DEBUG raw body (first 200 bytes): {body_bytes[:200]!r}")
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    print(f"DEBUG incoming payload keys: {list(payload.keys())}")
    print(f"DEBUG stream flag: {payload.get('stream')}")
    if payload.get("stream") is True:
        raise HTTPException(status_code=400, detail="Streaming is not supported yet.")

    settings = get_settings()
    headers = request.headers
    logical_session_id = headers.get("x-logical-session-id") or payload.get("logical_session_id")

    raw_messages = payload.get("messages")
    history_messages = raw_messages if isinstance(raw_messages, list) else []

    # 先清理 Kelivo 的自动 Memory Tool 等注入段
    if isinstance(history_messages, list):
        history_messages = strip_kelivo_autoprompt(history_messages)

    # 日志只写这一轮新增turn
    log_request_messages = extract_new_turn(history_messages)

    request_model_name = payload.get("model")

    system_messages: list[dict[str, Any]] = []
    if logical_session_id:
        summary_obj = load_session_summary(settings, logical_session_id)
        if isinstance(summary_obj, dict) and summary_obj.get("summary"):
            system_messages.append(
                {
                    "role": "system",
                    "content": f"[SESSION SUMMARY]\n{summary_obj.get('summary')}",
                }
            )

        current_text = extract_recent_user_text(history_messages, max_messages=2)
        candidate_memories = select_relevant_memories(
            settings=settings,
            logical_session_id=logical_session_id,
            current_text=current_text,
            max_memories=settings.memory_max_candidates,
            include_kinds=("pin", "ltm"),
        )
        mark_memories_active(
            logical_session_id,
            [str(m.get("id")) for m in candidate_memories if m.get("id")],
            ttl=settings.memory_activation_ttl,
        )
        active_ids = get_active_memory_ids(logical_session_id)
        active_memories = resolve_memories_by_ids(settings, active_ids)

        consolidation = await maybe_consolidate_active_memories(
            settings=settings,
            session_id=logical_session_id,
            active_memories=active_memories,
        )
        if consolidation is not None:
            system_messages.append(build_consolidated_memories_system_message(consolidation))
        else:
            memories_message = build_memories_system_message(active_memories)
            if memories_message:
                system_messages.append(memories_message)

    # === 这里是新加的 debug，缩进对齐上面的 if ===
    print("DEBUG system_messages before context budget:")
    for msg in system_messages:
        role = msg.get("role")
        content = str(msg.get("content", ""))
        snippet = content[:300].replace("\n", "\\n")
        print(f"  - {role}: {snippet}")
    # === debug 结束 ===    

    if isinstance(raw_messages, list):
        try:
            final_messages = apply_context_budget(
                settings=settings,
                system_messages=system_messages,
                history_messages=history_messages,
            )
            payload["messages"] = final_messages
            print(
                "DEBUG context budget applied, "
                f"history_len={len(history_messages)}, final_len={len(final_messages)}"
            )
        except Exception as exc:
            print(f"DEBUG apply_context_budget failed: {exc!r}")

    url = f"{settings.openrouter_base_url}/v1/chat/completions"
    out_headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(**build_httpx_client_kwargs(settings)) as client:
            response = await client.post(url, json=payload, headers=out_headers)
    except Exception as exc:
        print(f"DEBUG Exception when calling OpenRouter: {exc!r}")
        raise HTTPException(status_code=502, detail="Error talking to OpenRouter")

    if response.status_code >= 400:
        print(
            f"DEBUG OpenRouter returned error: "
            f"{response.status_code} {response.text[:200]!r}"
        )
        raise HTTPException(status_code=response.status_code, detail=response.text)

    response_data = response.json()

    append_ok = False
    if logical_session_id and isinstance(raw_messages, list):
        try:
            reply_message = response_data.get("choices", [{}])[0].get("message", {})
            if isinstance(reply_message, dict):
                append_chat_log(
                    settings=settings,
                    logical_session_id=logical_session_id,
                    request_messages=log_request_messages,
                    reply_message=reply_message,
                )
                append_ok = True
        except Exception as exc:
            print(f"DEBUG append_chat_log failed: {exc!r}")

    if logical_session_id and append_ok:
        try:
            asyncio.create_task(maybe_auto_summarize_session(settings, logical_session_id))
        except Exception as exc:
            print(f"DEBUG maybe_auto_summarize_session failed: {exc!r}")

        if should_pin_from_messages(log_request_messages):
            try:
                asyncio.create_task(
                    create_pinned_memory_from_session(
                        settings, 
                        logical_session_id, 
                        model_name=str(request_model_name) if request_model_name else None,
                        )
                )
            except Exception as exc:
                print(f"DEBUG create_pinned_memory_from_session failed: {exc!r}")

        decay_active_memories(logical_session_id)

    print(f"DEBUG OpenRouter status: {response.status_code}")
    print(f"DEBUG OpenRouter body (first 200 chars): {response.text[:200]!r}")
    print(f"DEBUG summary_model = {settings.summary_model!r}")
    return response_data


@app.post("/internal/list_pinned_memories")
async def list_pinned(payload: dict[str, str] = Body(...)) -> Any:
    logical_session_id = payload.get("logical_session_id")
    if not logical_session_id:
        raise HTTPException(status_code=400, detail="logical_session_id is required")

    settings = get_settings()
    return {
        "logical_session_id": logical_session_id,
        "memories": list_pinned_memories_by_session(settings, logical_session_id),
    }


@app.post("/internal/list_ltm_memories")
async def list_ltm() -> Any:
    settings = get_settings()
    return {"memories": load_all_ltm_memories(settings)}


@app.post("/internal/summarize_session")
async def summarize_session(payload: dict[str, str] = Body(...)) -> Any:
    logical_session_id = payload.get("logical_session_id")
    if not logical_session_id:
        raise HTTPException(status_code=400, detail="logical_session_id is required")

    settings = get_settings()
    
    result = await run_session_summarization(settings, logical_session_id)
    return {"logical_session_id": logical_session_id, "result": result}

@app.get("/internal/debug/chat-log-tail")
async def debug_chat_log_tail(limit: int = 50) -> Any:
    """
    返回云端 chat_log.jsonl 的最后 `limit` 行，用来确认网关到底在云上写了什么。
    """
    settings = get_settings()
    path = get_chat_log_path(settings)

    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "reason": "file_not_found",
            "lines": [],
        }

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return {
            "exists": False,
            "path": str(path),
            "reason": f"read_failed: {exc!r}",
            "lines": [],
        }

    all_lines = [ln for ln in text.splitlines() if ln.strip()]
    tail = all_lines[-limit:]

    return {
        "exists": True,
        "path": str(path),
        "total_lines": len(all_lines),
        "returned_lines": len(tail),
        "lines": tail,
    }

@app.get("/internal/debug/pinned-memories")
async def debug_pinned_memories() -> Any:
    """
    返回云端 pinned_memories.json 的内容（完整 dict）。
    仅用于自查：现在到底有哪些 pinned、它们长什么样。
    """
    settings = get_settings()

    data = load_pinned_memories(settings)
    return {
        "count": len(data),
        "memories": data,
    }

@app.get("/internal/debug/session-summary/{logical_session_id}")
async def debug_session_summary(logical_session_id: str) -> Any:
    """
    返回指定 logical_session_id 的滚动摘要对象。
    """
    settings = get_settings()
    summary = load_session_summary(settings, logical_session_id)
    if summary is None:
        return {
            "logical_session_id": logical_session_id,
            "exists": False,
        }
    return {
        "logical_session_id": logical_session_id,
        "exists": True,
        "summary": summary,
    }

