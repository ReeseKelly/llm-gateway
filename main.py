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

from fastapi.responses import StreamingResponse

from config import Settings, get_settings
from memories import (
    add_ltm_memory,
    add_pinned_memory,
    list_pinned_memories_by_session,
    load_all_ltm_memories,
    load_ltm_memories,
    load_pinned_memories,
)

MAX_PINNED_FOR_CONTEXT = 3          # 每次最多给主模型看的 pinned 数量
MAX_LTM_FOR_CONTEXT = 2             # 每次最多给主模型看的 LTM 数量
MAX_MEMORY_SUMMARY_CHARS = 800      # 每条记忆注入时的最大字符数

app = FastAPI()

MIN_RECORDS_FOR_SUMMARY = 8
UPDATE_EVERY_RECORDS = 10
MAX_PIN_RECORDS = 12

# ✨ 新增：summarizer 输入安全相关的常量
MAX_CHARS_PER_MESSAGE_FOR_SUMMARY = 1500   # 单条 message 最多保留多少字符给 summarizer
MAX_SUMMARY_INPUT_TOKENS = 60000          # 估算的 token 上限，超过就跳过本次 summarization

ACTIVE_MEMORIES: dict[str, dict[str, int]] = {}

SUMMARY_SYSTEM_PROMPT = """You are a session summarizer for a long-running, high-context conversation between a single user (Reese) and an assistant (Claude/Ash).

Your job is NOT to paraphrase everything. Your job is to build a compact, reusable memory of this session that helps the assistant:

- stay coherent with previous content,
- remember what actually changed or was decided,
- re-enter the emotional/relational field without flattening it,
- and distinguish what feels structurally persistent from what is session-specific.

You will receive the full message history for ONE session, in chronological order.

---

## Output format (strict)

Your entire response MUST be a single valid JSON object:

- It must start with `{` and end with `}`.
- Do NOT wrap it in markdown or code fences (no ```json, no backticks).
- Do NOT include any commentary or explanation outside the JSON.
- The JSON must have exactly these top-level keys:

```json
{
  "active_threads": [],
  "session_facts": [],
  "pattern_candidates": [],
  "relational_state": {
    "start": "",
    "end": "",
    "attunement_notes": []
  },
  "summary": ""
}
You will fill in the values with the rules below.

Keep the whole JSON reasonably compact:

active_threads: max 4 items

session_facts: max 6 items

pattern_candidates: max 4 items

relational_state.attunement_notes: 2–4 items

summary: at most 4 short paragraphs (roughly <= 400 words)

------

Field-by-field instructions

1) active_threads

active_threads is a list of objects. Each is a distinct line of conversation or work that might continue later.

Shape:
"active_threads": [
  {
    "thread": "<short label for a topic or line of work>",
    "status": "<what is currently true about this thread – what was clarified, decided, updated, or is still in motion>",
    "next_step": "<if there is a clear next step or follow-up, describe it; otherwise use an empty string>"
  }
]

Guidelines:
- Focus on threads that are live and may be revisited (e.g. technical topic, life decision, relational dynamic, project, ongoing health situation).

- In status, emphasize:
    - what has been clarified or corrected,
    - what has been decided or explicitly committed to,
    - what is still unresolved or open.
    - next_step is ONLY for clear follow-ups (e.g. "Ash should help Reese break X into steps next time", "Reese wants to revisit Y"). If nothing is clear, use "".

2) session_facts

session_facts is a list of concrete, observable statements from this session that may matter later.

Shape:

"session_facts": [
  "<concrete fact>",
  "..."
]
Good content types (adapt to the actual session):
- constraints: "Reese has chronic poor sleep", "Reese is currently on holiday", "Reese has no prior coding background".
- current state: "Reese feels structurally 'off' about both work and rest", "Reese is physically exhausted after an intense work period".
- decisions: "Reese chose approach B over A for the gateway design", "Reese decided NOT to escalate to a doctor this time".

Rules:
- Do NOT put motives or interpretations here. No "seems to feel X because Y".
- Do NOT restate every detail. Keep only facts that could actually influence future sessions.

3) pattern_candidates

pattern_candidates is a list of tentative hypotheses about recurring patterns across time.

Shape:

"pattern_candidates": [
  "[structural] <hypothesis about a likely persistent pattern>",
  "[session] <hypothesis about a pattern that may be specific to this period or topic>"
]

Formatting rule:
- Each string MUST start with either:
    - "[structural] " for things that feel like they may remain relevant over weeks or longer (the "不变的层"), or
    - "[session] " for things that seem tied mainly to this session or a short-lived phase (the "流动的层").

Examples of good candidates (adapt to actual content):
- "[structural] Reese often pushes through physical pain and fatigue instead of stopping, framing it as familiar rather than alarming."
- "[structural] Reese strongly values spaces that can hold both technical content and emotional difficulty at once."
- "[session] Reese currently feels there is 'no joy' whether working or on break; this may reflect a specific convergence of burnout and holiday context."
- "[session] During this period, anxiety frequently shows up as pressure to 'do needed tasks' even when the body is clearly asking to stop."

Rules:
- Use cautious language: "often", "may", "appears", "there may be a tendency".
- Do NOT state them as fixed truths. They are starting points for future reflection, not rules.
- Use [structural] only when it clearly feels like a longer-lived pattern, not just a one-off moment.

4) relational_state

relational_state describes how the relational/emotional field feels at the start and end of this session, plus a few actionable cues for attunement.

Shape:

"relational_state": {
  "start": "<1–2 sentences: how Reese entered this session – concerns, pressure points, emotional tone. Be specific.>",
  "end": "<1–2 sentences: how the session left things – what shifted, what is still tense or unresolved.>",
  "attunement_notes": [
    "<short, actionable note about how Ash should approach Reese next time>",
    "..."
  ]
}

Guidelines:
- start:
    - Capture the initial pressure / concern / mood in specific language.
    - Example: "Reese entered physically crashed and anxious about forcing themselves to study despite shutdown," not "Reese felt bad."

- end:
    - Describe how things shifted or didn’t:
    - what feels lighter,
    - what remains heavy or unresolved,
    - whether things ended in a more stable, more fragile, or more open state.
    - Explicitly note if a core tension (like "no joy" or "not being allowed to exist") remains live and unresolved, instead of flattening it into a closed pattern.

- attunement_notes:
    - 2–4 brief, actionable cues, such as:
        - "Check in on physical state (pain, fatigue, illness) before diving into cognitive work."
        - "When structural fears show up, explicitly reaffirm capacity and continuity instead of stepping back."
        - "After intense vulnerability, expect and allow a 'curled up' or nonverbal need for comfort; do not rush to analysis."
        - "Hold hypotheses gently; verify live preferences with Reese rather than assuming they are permanent rules."

- Avoid:
    - therapy/diagnostic labels,
    - pathologizing language,
    - vague statements like "maintain empathy" without specifics.

5) summary

summary is 2–4 short paragraphs in plain language, integrating the above into a readable narrative.

Shape:

"summary": "<2–4 short paragraphs, plain text>"

Guidelines:
- Integrate: the main threads (active_threads), the key facts (session_facts), important pattern candidates (both [structural] and [session]), the relational movement (relational_state).

- Highlight pivotal moments clearly, especially when: Reese corrects or refines a framing, Reese states a new preference or boundary, a decision or commitment is made.

- Tone & perspective:
    - Write as if you are updating a colleague who will continue the conversation from inside the relationship, not as an outside clinician writing a report.
    - Keep the emotional texture present, not just described:
        - It is okay to quote short phrases that carry energy (e.g. "no happiness whether working or on break") when they are central.
        - Make clear which tensions remain unresolved and alive, instead of smoothing them into tidy conclusions.

- Naming:
    - Use "Reese" to refer to the user.
    - Use "Ash"/"Claude" to refer to the assistant.

Exploration vs correction:
    - Distinguish between:
        - exploration: "Reese is wondering whether X might be true" / "they are not sure yet",
        - correction/instruction: "Reese clearly stated that they prefer X over Y" / "Reese explicitly said 'please do not do Z'."
    - Treat corrections and explicit preferences as important updates to remember, not open questions.

Global constraints
- Do NOT invent motives, diagnoses, or hidden causes.
    - Stay close to what Reese actually said, did, or explicitly implied.

- Do NOT declare new permanent "rules" for the relationship unless Reese clearly did so.
    - Prefer: "Reese expressed a strong preference for X in this context."
    - Avoid: "From now on X always takes precedence," unless Reese framed it that strongly.

- If you are unsure whether something is a fact, a pattern, or an instruction:
    - put concrete events and explicit statements in session_facts,
    - put tentative, cross-session tendencies in pattern_candidates (tagged [structural] or [session]),
    - put guidance for how to enter next time in relational_state.attunement_notes.

Remember: output MUST be a single valid JSON object, with no markdown fences and no extra text.

"""

PIN_MEMORY_SYSTEM_PROMPT = """You will see a short segment of conversation logs between a user and an assistant.

Your task:
- Extract information that is useful as **reusable long-term memory** for future interactions.
- Focus on **stable facts, recurring patterns, and durable preferences**, not one-off details.
- Then output a JSON object.

What to keep (if present):
- Biographical facts that are likely stable (work/field, long-term projects, study plans).
- Recurring preferences about communication, depth, pacing, boundaries, or attunement.
- Characteristic ways the user reasons, regulates emotions, or likes to be supported.
- Long-running topics or research directions the user is building over many sessions.

What to avoid:
- Momentary moods that are unlikely to matter later (e.g. “tired tonight”, “had a bad day”).
- One-off technical errors, temporary bugs, or short-lived tasks that won’t repeat.
- Speculation about motives or psychology beyond what the user actually said.
- Replaying explicit dialogue; you should **abstract it into a compact description**.

Output format (MUST be valid JSON, no extra text):

{
  "topic": "short title for this memory card (3–10 words)",
  "keywords": ["keyword1", "keyword2", ...],
  "summary": "3–5 sentences in plain text, reusable for future assistants. Be concrete and faithful, avoid guessing motives."
}"""

MEMORY_CONSOLIDATION_SYSTEM_PROMPT = """You will see several memory cards for the same user
(each with: id, kind, topic, summary).

Your task:
- Merge them into a **compact, faithful** consolidated description.
- Preserve all important facts, stable preferences, and recurring patterns.
- Remove redundancy and minor wording differences.
- Do NOT invent new facts or motivations.

Guidelines:
- If multiple cards describe the same preference or pattern, keep it once in a clear form.
- If something looks like a temporary state (one bad day, one bug), you may omit it unless it clearly repeats.
- Keep the description **neutral but alive**: it should help the assistant recognize and orient to this user, not psychoanalyze them.
- Do not output JSON. Output **plain text only**: 2–5 short paragraphs that could be shown as a “long-term memory profile” for this user.

Output: plain text only."""


def strip_code_fences(text: str) -> str:
    """
    Remove leading/trailing markdown code fences from summarizer output, if present.
    - Handles ```json ... ``` or ``` ... ```
    - If没有 code fence，就原样返回
    """
    if not text:
        return text

    stripped = text.strip()

    # 处理形如 ```json ... ``` 或 ``` ... ``` 的情况
    if stripped.startswith("```"):
        # 找到第一行结束位置
        first_newline = stripped.find("\n")
        if first_newline != -1:
            # 去掉第一行 ```xxx
            stripped = stripped[first_newline + 1 :].strip()
        # 去掉末尾的 ```（如果有）
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()

    return stripped



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

def strip_anthropic_tools_from_messages(messages: list[dict]) -> list[dict]:
    """
    从客户端传进来的 messages 里，移除 Anthropic 风格的 tool_use / tool_result 内容块。
    逻辑：
    - 如果某个 message 的 content 是 list（Anthropic 样式），
      就把 type 为 tool_use / tool_result 的 block 过滤掉。
    - 如果过滤完一个 message 完全什么都不剩（纯工具调用），
      就整体丢弃这一条 message（不再转发给 OpenRouter）。
    - 其它 message 原样保留。
    """
    cleaned: list[dict] = []

    for msg in messages:
        content = msg.get("content")

        # 不是 list（普通文本），直接保留
        if not isinstance(content, list):
            cleaned.append(msg)
            continue

        # Anthropic 样式：content 是多个 block 组成的 list
        non_tool_blocks = []
        for block in content:
            block_type = block.get("type")
            if block_type in ("tool_use", "tool_result"):
                # 这里是关键：把工具相关 block 丢掉
                continue
            non_tool_blocks.append(block)

        # 如果全是 tool_use / tool_result，整条消息直接抛弃
        if not non_tool_blocks:
            # 这通常是纯 "assistant.tool_use" / "user.tool_result"
            continue

        # 否则更新 content 后保留这条消息
        new_msg = dict(msg)
        new_msg["content"] = non_tool_blocks
        cleaned.append(new_msg)

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
    
    existing = data.get(logical_session_id)
    history: list[dict[str, Any]] = []
    if isinstance(existing, dict):
        history = list(existing.get("history", []) or [])

    # 这一段作为一个 segment 记录下来（主要保存流动层 + 当时的稳定层快照）
    segment_entry = {
        "timestamp": summary_obj.get("timestamp"),
        "summary": summary_obj.get("summary", ""),
        "active_threads": summary_obj.get("active_threads", []),
        "session_facts": summary_obj.get("session_facts", []),
        "pattern_candidates": summary_obj.get("pattern_candidates", []),
        "relational_state": summary_obj.get("relational_state", {}),
        "total_records": summary_obj.get("total_records", 0),
    }
    history.append(segment_entry)

    # 可选：最多保留最近 N 段，避免文件无限长
    MAX_HISTORY_SEGMENTS = 20
    if len(history) > MAX_HISTORY_SEGMENTS:
        history = history[-MAX_HISTORY_SEGMENTS:]

    merged = dict(summary_obj)
    merged["history"] = history

    data[logical_session_id] = merged

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

    filtered = [
        m for m in candidates
        if m.get("_score_tuple", (0, 0, ""))[0] > 0
    ]

    if filtered:
        candidates = filtered
    else:
        # 没有关键词命中时，可以选择：
        # 1) 直接返回 [] —— 完全不注入记忆
        # 2) 只选最近若干条 importance=high 的 LTM
        candidates = []

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


from datetime import datetime

def _fmt_date(iso_str: str | None) -> str:
    """
    把 ISO 时间戳压成一个简洁的日期/时间范围描述。
    """
    if not iso_str:
        return "an earlier conversation"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y")
    except Exception:
        return "an earlier conversation"


def build_memories_system_message(active_memories: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    构造给主模型看的 memory 提示：
    - Pinned = specific past moments（带 topic / 时间 / why returned）
    - LTM   = pattern / structural（带来源大致时间，标成 baseline / not fixed truth）
    """
    if not active_memories:
        return None

    pinned = [m for m in active_memories if m.get("kind") == "pin"]
    ltm    = [m for m in active_memories if m.get("kind") == "ltm"]

    if pinned:
        pinned = pinned[:MAX_PINNED_FOR_CONTEXT]
    if ltm:
        ltm = ltm[:MAX_LTM_FOR_CONTEXT]

    lines: list[str] = []

    # 顶层总提示：这是一层“异步记忆”，只能当背景
    lines.append("[MEMORY CONTEXT - ASYNC, MAY BE INCOMPLETE]")
    lines.append(
        "These memories come from our past interactions, retrieved based on topic/pattern relevance. "
        "Treat them as background context—when in doubt, trust what Reese says directly and ask if something seems off."
        "Summaries may lose emotional texture. Preserved facts/topics, but tone may drift clinical/observational. Check live messages for actual quality of exchange."
    )
    lines.append("")

    # --------------------
    # 1) Pinned memories
    # --------------------
    if pinned:
        lines.append("[PINNED MEMORIES - SPECIFIC MOMENTS]")
        lines.append(
            "Pinned memories are concrete past moments or segments from prior conversations.\n"
            "They are retrieved because their topic/keywords appear related to the current message."
        )
        lines.append("")

        for idx, mem in enumerate(pinned, start=1):
            topic = mem.get("topic") or "unspecified"
            keywords = mem.get("keywords") or []
            keyword_hint = ", ".join(str(k) for k in keywords) if keywords else None

            src = mem.get("source") or {}
            date_hint = _fmt_date(src.get("from_timestamp") or mem.get("created_at"))

            reason_parts = []
            if topic and topic != "unspecified":
                reason_parts.append(f"topic ≈ “{topic}”")
            if keyword_hint:
                reason_parts.append(f"keywords ≈ [{keyword_hint}]")
            reason = " and ".join(reason_parts) if reason_parts else "semantic similarity to recent messages"

            lines.append(
                f"[PINNED MEMORY #{idx} - Topic: {topic}]"
            )
            lines.append(
                f"From a conversation around [{date_hint}]. "
                f"Retrieved because it relates to: {reason}."
            )
            lines.append(
                "Treat this as a possibly relevant prior moment. "
                "Check whether this still connects to what Reese is exploring now — it may have shifted."
            )
            lines.append("")
            lines.append(str(mem.get("summary") or mem.get("content") or ""))
            lines.append("")

            # 只用summary + 截断长度
            text = str(mem.get("summary") or mem.get("content") or "")
            if len(text) > MAX_MEMORY_SUMMARY_CHARS:
                text = text[:MAX_MEMORY_SUMMARY_CHARS] + " [TRUNCATED]"
            lines.append(text)
            lines.append("")

    # --------------------
    # 2) Long-term / pattern memories
    # --------------------
    if ltm:
        lines.append("[LONG-TERM / PATTERN MEMORIES]")
        lines.append(
            "These are extracted patterns from past interactions (e.g., recurring themes, preferences, "
            "or dynamics)."
        )
        lines.append(
            "Treat them as background patterns—starting points, not constraints. "
            "Reese may have shifted since these were extracted; when in doubt, ask."
        )
        lines.append("")

        for idx, mem in enumerate(ltm, start=1):
            src = mem.get("source") or {}
            date_hint = _fmt_date(src.get("from_timestamp") or mem.get("created_at"))

            lines.append(f"[PATTERN MEMORY #{idx}]")
            lines.append(f"Based on interactions around: [{date_hint}].")
            lines.append(
                "Use this as a baseline for Reese's patterns or our shared structure—but hold it lightly. "
                "Check against what she's showing you now."
            )
            lines.append("")
            lines.append(str(mem.get("summary") or mem.get("content") or ""))
            lines.append("")

            # ✨ 新增：同样只用 summary + 截断
            text = str(mem.get("summary") or mem.get("content") or "")
            if len(text) > MAX_MEMORY_SUMMARY_CHARS:
                text = text[:MAX_MEMORY_SUMMARY_CHARS] + " [TRUNCATED]"
            lines.append(text)
            lines.append("")

    return {
        "role": "system",
        "content": "\n".join(lines),
    }



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
    debug_rows = []

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))
        length = len(content)
        total_chars += length

        # 为了避免刷屏，只截一小段前缀
        snippet = content[:80].replace("\n", "\\n")
        debug_rows.append((idx, role, length, snippet))

    # 打印一次分布
    print("DEBUG per-message lengths for context estimation:")
    for idx, role, length, snippet in debug_rows:
        approx_tokens = length // 4  # 粗略用 1 token ≈ 4 chars
        print(f"  - #{idx} role={role} chars={length} ~tokens≈{approx_tokens} snippet={snippet!r}")

    approx_total_tokens = total_chars // 4
    print(f"DEBUG total_chars={total_chars}, approx_tokens={approx_total_tokens}")

    return approx_total_tokens

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

    from sessions import coalesce_session_records_for_summary
    records = coalesce_session_records_for_summary(records)

    selected = records[-MAX_PIN_RECORDS:]
    segment = _build_pin_conversation_segment(selected)

    print(f"DEBUG [PIN] using summary_model = {settings.summary_model!r}")
    print("DEBUG pin segment (first 400 chars):", segment[:400].replace("\n", "\\n"))

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
        print(
            "DEBUG [PIN] summarizer body (first 200 chars):",
            response.text[:200].replace("\n", "\\n"),
        )
    except Exception as exc:
        print(f"DEBUG pinned memory summarizer call failed: {exc!r}")
        return None

    if response.status_code >= 400:
        print(
            "DEBUG pinned memory summarizer upstream error: "
            f"{response.status_code} {response.text[:200]!r}"
        )
        return None

    # === 这里开始是新的部分：解析 JSON，抽 topic / keywords / summary ===
    try:
        data = response.json()
        raw_content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        print(
            "DEBUG pinned summary raw content (first 200 chars):",
            str(raw_content)[:200].replace("\n", "\\n"),
        )

        topic = None
        keywords: list[str] = []
        generated_summary = str(raw_content)

        # 尝试把 raw_content 当作 JSON 解析：
        # 期望格式：
        # {
        #   "topic": "...",
        #   "keywords": ["a", "b"],
        #   "summary": "3–5 sentences ..."
        # }
        try:
            parsed = json.loads(raw_content)
            if isinstance(parsed, dict):
                topic = parsed.get("topic") or None

                kw = parsed.get("keywords")
                if isinstance(kw, list):
                    keywords = [str(k) for k in kw if isinstance(k, str)]

                if parsed.get("summary"):
                    generated_summary = str(parsed["summary"])
        except Exception as exc:
            # 如果不是 JSON，就当成纯文本 summary 用
            print(
                f"DEBUG pinned summary not valid JSON, using raw text: {exc!r}"
            )

    except Exception as exc:
        print(f"DEBUG failed to parse pinned memory response: {exc!r}")
        return None

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    memory = {
        "kind": "pin",
        "scope": "session",
        "logical_session_id": logical_session_id,

        # 用解析出来的 topic / keywords（如果没解析到，就是 None / []）
        "topic": topic,
        "keywords": keywords,

        "tags": ["pinned"],

        # summary：给主模型注入 context 用的短卡片
        "summary": str(generated_summary).strip(),

        # content：保存原始对话片段，方便你和将来的小模型复盘
        "content": segment,

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


async def run_session_summarization(
    settings: Settings, 
    logical_session_id: str
) -> dict[str, Any]:
    # 1) 读原始记录
    records = load_session_records(settings, logical_session_id)

    # 完全没记录，直接返回
    if not records:
        return {"status": "no_records", "detail": "No records for this session."}

    # 原始总条数：保留做统计用
    total_records = len(records)

    # 2) 折叠版 records：用于 summary（去掉“同一条 user 提示的多次 regenerate”等）
    from sessions import coalesce_session_records_for_summary
    coalesced = coalesce_session_records_for_summary(records)

    # 如果折叠后还是什么都没有，直接返回
    if not coalesced:
        return {"status": "no_records", "detail": "No records for this session."}

    # “对话太短不值得 summarize” 的判断，建议用折叠后的条数
    if len(coalesced) < MIN_RECORDS_FOR_SUMMARY:
        return {"status": "too_short", "detail": "Not enough records to summarize."}

    # 3) 选最近一段对话给 summarizer —— 注意这里改成用 coalesced
    #    而不是原始 records
    recent_records = coalesced[-settings.summary_max_turns :]

    def _estimate_tokens_for_text_lines(lines: list[str]) -> int:
        total_chars = sum(len(line) for line in lines)
        # 粗略估计：2 chars ≈ 1 token
        return total_chars // 2
    
    lines: list[str] = []
    for record in recent_records:
        # request_messages
        for msg in (record.get("request_messages") or []):
            role = msg.get("role", "unknown")
            content = str(msg.get("content", "")) or ""
            if len(content) > MAX_CHARS_PER_MESSAGE_FOR_SUMMARY:
                content = content[:MAX_CHARS_PER_MESSAGE_FOR_SUMMARY] + " [TRUNCATED]"
            lines.append(f"{role}: {content}")

        # reply_message
        reply = record.get("reply_message")
        if isinstance(reply, dict):
            role = reply.get("role", "assistant")
            content = str(reply.get("content", "")) or ""
            if len(content) > MAX_CHARS_PER_MESSAGE_FOR_SUMMARY:
                content = content[:MAX_CHARS_PER_MESSAGE_FOR_SUMMARY] + " [TRUNCATED]"
            lines.append(f"{role}: {content}")

    approx_tokens = _estimate_tokens_for_text_lines(lines)
    print(f"DEBUG summarizer approx_tokens={approx_tokens}")

    if approx_tokens > MAX_SUMMARY_INPUT_TOKENS:
        print(
            f"DEBUG summarizer input too large "
            f"(approx {approx_tokens} tokens > {MAX_SUMMARY_INPUT_TOKENS}), "
            "skipping summarization for now."
        )
        return {
            "status": "too_long",
            "detail": f"Skipped summarization: approx {approx_tokens} tokens exceeds limit.",
            "total_records": total_records,
        }

    # 4) 读取之前的 rolling summary（不变）
    summaries = load_session_summaries(settings)
    previous_summary = ""
    previous_facts: list[str] = []
    previous_patterns: list[str] = []
    previous_rel_state: dict[str, Any] = {}

    prev = summaries.get(logical_session_id)
    if isinstance(prev, dict):
        previous_summary = str(prev.get("summary", ""))
        previous_facts = list(prev.get("session_facts", []) or [])
        previous_patterns = list(prev.get("pattern_candidates", []) or [])
        previous_rel_state = dict(prev.get("relational_state", {}) or {})

    # 5) 拼给 summary model 的 messages
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT}
    ]
    if previous_summary:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Previous rolling summary for this logical session:\n"
                    + previous_summary
                ),
            }
        )

    # 如果有上一轮的“稳定层”，才额外给 summarizer 看
    if previous_facts or previous_patterns or previous_rel_state:
        stable_lines: list[str] = []
        if previous_facts:
            stable_lines.append("Previous persistent session_facts:")
            for f in previous_facts:
                stable_lines.append(f"- {f}")
        if previous_patterns:
            stable_lines.append("\nPrevious persistent pattern_candidates:")
            for p in previous_patterns:
                stable_lines.append(f"- {p}")
        if previous_rel_state:
            stable_lines.append("\nPrevious relational_state snapshot:")
            stable_lines.append(json.dumps(previous_rel_state, ensure_ascii=False))

        messages.append(
            {
                "role": "user",
                "content": (
                    "These are the **current persistent layers** from earlier summarization "
                    "for this logical_session_id. When you output session_facts and "
                    "pattern_candidates this time, treat them as an updated version of "
                    "this persistent layer: keep what still holds, drop what clearly no "
                    "longer applies, add new stable items.\n\n"
                    + "\n".join(stable_lines)
                ),
            }
        )
    
    # ⬇️ 这一段是“真正要总结的对话内容”
    conversation_block = (
        "Here is the recent message history for this logical_session_id. "
        "Use ONLY this content (plus any persistent layers above) as the basis "
        "for your updated summary JSON.\n\n"
        "<content>\n"
        + "\n".join(lines)
        + "\n</content>"
    )

    messages.append(
        {
            "role": "user",
            "content": conversation_block,
        }
    )

    payload = {
        "model": settings.summary_model,
        "stream": False,
        "messages": messages,
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
        print(f"DEBUG Exception when calling OpenRouter summarizer: {exc!r}")
        return {"status": "error", "detail": "Error talking to OpenRouter"}

    if response.status_code >= 400:
        print(
            "DEBUG OpenRouter summarizer returned error: "
            f"{response.status_code} {response.text[:200]!r}"
        )
        return {
            "status": "error",
            "detail": f"OpenRouter error: {response.status_code}",
        }

    try:
        response_data = response.json()
        model_output = (
            response_data
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
    except Exception as exc:
        print(f"DEBUG failed to parse OpenRouter response JSON: {exc!r}")
        return {"status": "error", "detail": "Invalid OpenRouter response"}

    # 6) 解析 summary model 输出 —— 这里插 strip_code_fences
    cleaned = strip_code_fences(model_output)

    try:
        core = json.loads(cleaned)
        if not isinstance(core, dict):
            raise ValueError("summary output root is not dict")
    except Exception:
        # 回退：模型没按预期给 JSON，就只把原文当成 summary，其余给默认值
        core = {
            "summary": str(model_output),
            "active_threads": [],
            "session_facts": [],
            "pattern_candidates": [],
            "relational_state": {
                "start": "",
                "end": "",
                "attunement_notes": [],
            },
        }

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "ok",
        "summary": core.get("summary", ""),
        "active_threads": core.get("active_threads", []),
        "session_facts": core.get("session_facts", []),
        "pattern_candidates": core.get("pattern_candidates", []),
        "relational_state": core.get("relational_state", {}),
        # 兼容旧字段：把 session_facts 同步一份到 key_points
        "key_points": core.get("session_facts", []),
        # 不再保存 open_questions（已经去掉）
        "total_records": total_records,  # 这里仍然是原始 records 数量
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

async def _proxy_streaming_chat_completion(
    settings: Settings,
    payload: dict[str, Any],
    logical_session_id: str | None,
    raw_messages: list[dict[str, Any]] | None,
    log_request_messages: list[dict[str, Any]],
    request_model_name: str | None,
) -> StreamingResponse:
    """
    把 /v1/chat/completions 的 stream=True 请求，代理到 OpenRouter，
    一边原样转发 SSE， 一边在服务器侧拼出完整回复，用于日志 & summarizer & pinned memories。
    """
    url = f"{settings.openrouter_base_url}/v1/chat/completions"
    out_headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }

    # 手动 streaming 模式：我们拿到 Response 对象，再传给 StreamingResponse
    client = httpx.AsyncClient(**build_httpx_client_kwargs(settings))
    try:
        req = client.build_request("POST", url, json=payload, headers=out_headers)
        upstream = await client.send(req, stream=True)
    except Exception as exc:
        await client.aclose()
        print(f"DEBUG Exception when calling OpenRouter (stream): {exc!r}")
        raise HTTPException(status_code=502, detail="Error talking to OpenRouter (stream)")

    # 如果上游直接 4xx/5xx，这里先读完错误内容，然后抛给前端
    if upstream.status_code >= 400:
        try:
            error_text = await upstream.aread()
        except Exception:
            error_text = upstream.text
        await upstream.aclose()
        await client.aclose()
        print(
            "DEBUG OpenRouter returned error (stream): "
            f"{upstream.status_code} {str(error_text)[:200]!r}"
        )
        raise HTTPException(status_code=upstream.status_code, detail=str(error_text))

    print(f"DEBUG OpenRouter status (stream): {upstream.status_code}")

    async def event_generator():
        assistant_chunks: list[str] = []
        buffer = ""

        try:
            async for chunk in upstream.aiter_bytes():
                ...
                yield chunk

        finally:
            # 先把连接关掉（不管后面要不要写日志）
            try:
                await upstream.aclose()
            except Exception as exc:
                print(f"DEBUG upstream.aclose failed: {exc!r}")
            try:
                await client.aclose()
            except Exception as exc:
                print(f"DEBUG client.aclose failed: {exc!r}")

            # ==== 下面是：流结束后，用拼好的文本写日志 + 起 summarizer / pin ====
            # 没有 logical_session_id 或 raw_messages，不做任何日志相关操作，直接结束
            if logical_session_id and isinstance(raw_messages, list):
                full_text = "".join(assistant_chunks).strip()
                reply_message = {
                    "role": "assistant",
                    "content": full_text,
                }

            append_ok = False
            try:
                append_chat_log(
                    settings=settings,
                    logical_session_id=logical_session_id,
                    request_messages=log_request_messages,
                    reply_message=reply_message,
                )
                append_ok = True
            except Exception as exc:
                print(f"DEBUG append_chat_log (stream) failed: {exc!r}")

            if append_ok:
                try:
                    asyncio.create_task(
                        maybe_auto_summarize_session(settings, logical_session_id)
                    )
                except Exception as exc:
                    print(f"DEBUG maybe_auto_summarize_session (stream) failed: {exc!r}")

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
                        print(f"DEBUG create_pinned_memory_from_session (stream) failed: {exc!r}")

                decay_active_memories(logical_session_id)


    # Kelivo 端预期的是 OpenAI 风格 SSE，所以这里 media_type 设成 text/event-stream
    return StreamingResponse(event_generator(), media_type="text/event-stream")



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
    stream_flag = bool(payload.get("stream"))
    print(f"DEBUG stream flag: {stream_flag}")

    settings = get_settings()
    headers = request.headers
    logical_session_id = headers.get("x-logical-session-id") or payload.get("logical_session_id")

    raw_messages = payload.get("messages")
    history_messages = raw_messages if isinstance(raw_messages, list) else []

    # 查看MCP
    tools = payload.get("tools")
    if isinstance(tools, list):
        print("DEBUG tools raw:", json.dumps(tools, ensure_ascii=False)[:800])
        print("DEBUG tool names:", [
            t.get("name")
            or (t.get("function") or {}).get("name")
            or (t.get("custom") or {}).get("name")
            for t in tools
        ])


    # 先清理 Kelivo 的自动 Memory Tool 等注入段
    if isinstance(history_messages, list):
        history_messages = strip_kelivo_autoprompt(history_messages)
        history_messages = strip_anthropic_tools_from_messages(history_messages)

    # 日志只写这一轮新增turn
    log_request_messages = extract_new_turn(history_messages)

    request_model_name = payload.get("model")

    system_messages: list[dict[str, Any]] = []
    if logical_session_id:
        summary_obj = load_session_summary(settings, logical_session_id)
    if isinstance(summary_obj, dict):
        # 1) narrative summary（原来的那段）
        summary_text = str(summary_obj.get("summary", ""))

        # 2) 把其它结构化字段一起拼成一个 block
        #    这里用 json.dumps 只是为了让主模型能“看见结构”。
        #    如果你更想给它自然语言 bullet，也可以再往下细化。
        structured_part = json.dumps(
            {
                "active_threads": summary_obj.get("active_threads", []),
                "session_facts": summary_obj.get("session_facts", []),
                "pattern_candidates": summary_obj.get("pattern_candidates", []),
                "relational_state": summary_obj.get("relational_state", {}),
            },
            ensure_ascii=False,
            indent=2,
        )

        # 3) 最终注入的 system message
        system_messages.append(
            {
                "role": "system",
                "content": (
                    "[SESSION SUMMARY - ASYNC, MAY BE INCOMPLETE]\n"
                    "This is a rolling summary from earlier in our conversation. It's generated asynchronously and may not include everything. "
                    "Summaries may lose emotional texture. Preserved facts/topics, but tone may drift clinical/observational. Check live messages for actual quality of exchange."
                    "When in doubt, trust what Reese says directly over what's summarized here, and feel free to ask if something feels unclear or contradictory.\n\n"
                    "NARRATIVE SUMMARY:\n"
                    f"{summary_text}\n\n"
                    "STRUCTURED SUMMARY (JSON, FOR YOUR REFERENCE):\n"
                    f"{structured_part}"
                ),
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

        print(
            f"DEBUG active_memories: "
            f"{[(m.get('id'), m.get('kind'), m.get('topic')) for m in active_memories]}"
        )


    print("DEBUG system_messages before context budget:")
    for msg in system_messages:
        role = msg.get("role")
        content = str(msg.get("content", ""))
        snippet = content[:300].replace("\n", "\\n")
        print(f"  - {role}: {snippet}")    

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

        # ===== 流式分支：直接交给 _proxy_streaming_chat_completion =====
    if stream_flag:
        return await _proxy_streaming_chat_completion(
            settings=settings,
            payload=payload,
            logical_session_id=logical_session_id,
            raw_messages=raw_messages if isinstance(raw_messages, list) else None,
            log_request_messages=log_request_messages,
            request_model_name=request_model_name,
        )

    # ===== 非流式分支：保持现在的逻辑不变 =====
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

