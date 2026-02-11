from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import httpx
from fastapi import Body, FastAPI, HTTPException, Request
from httpx_socks import AsyncProxyTransport

from config import Settings, get_settings

app = FastAPI()

MAX_RECENT_RECORDS = 5
MAX_RECENT_TEXT_CHARS = 200
MIN_RECORDS_FOR_SUMMARY = 4
UPDATE_EVERY_RECORDS = 5

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

Output format (MUST be valid JSON, no extra text):
{
  "summary": "3–8 sentences capturing the current state of this logical session.",
  "key_points": [
    "bullet-style key facts, decisions, or hypotheses that the assistant should remember",
    "... more items as needed"
  ],
  "open_questions": [
    "unresolved issues, user's pending questions, or threads that may come back later",
    "... more items as needed"
  ]
}
"""


def build_httpx_client_kwargs(settings: Settings) -> dict[str, Any]:
    """
    返回创建 httpx.AsyncClient 时需要的 kwargs：
    - timeout 固定为 60 秒
    - trust_env 固定为 False
    - 如果 settings.outbound_proxy_url 存在：
        使用 AsyncProxyTransport.from_url 构造 transport
        放入 kwargs["transport"]
    - 如果 outbound_proxy_url 为空，则不设置 transport
    """

    kwargs: dict[str, Any] = {
        "timeout": 60.0,
        "trust_env": False,
    }

    if settings.outbound_proxy_url:
        try:
            transport = AsyncProxyTransport.from_url(settings.outbound_proxy_url)
            kwargs["transport"] = transport
        except Exception as exc:
            print(f"DEBUG failed to create proxy transport: {exc!r}")

    return kwargs


def get_chat_log_path(settings: Settings) -> Path:
    """
    返回 chat_log.jsonl 的完整路径，确保目录存在。
    """
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "chat_log.jsonl"


def load_session_records(settings: Settings, logical_session_id: str) -> list[dict[str, Any]]:
    """
    从 chat_log.jsonl 中读取指定 logical_session_id 的所有记录。
    - 文件路径 = log_dir/chat_log.jsonl
    - 每一行是一个 JSON 对象
    - 如果文件不存在，返回 []
    - 如果某一行解析失败，跳过这一行并打印 debug
    - 只保留 record["logical_session_id"] == 传入参数 的记录
    - 返回顺序按照文件原始顺序（即时间顺序）
    """
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


def load_session_summary(
    settings: Settings, logical_session_id: str
) -> dict[str, Any] | None:
    """
    从 logs/session_summaries.json 读取指定 logical_session_id 的摘要。
    - 文件路径: settings.log_dir / "session_summaries.json"
    - 文件内容是一个 dict，顶层 key 是 logical_session_id
    - 如果文件不存在，返回 None
    - 如果没有这个 logical_session_id，返回 None
    - 解析失败时打印一行 debug 并返回 None
    """
    path = Path(settings.log_dir) / "session_summaries.json"
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

    summary_obj = data.get(logical_session_id)
    if not summary_obj:
        return None

    if isinstance(summary_obj, dict):
        return summary_obj
    return None


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    return text[:max_chars]


def build_messages_with_memory(
    settings: Settings,
    logical_session_id: str,
    original_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary_obj = load_session_summary(settings, logical_session_id)
    summary_text = None
    if summary_obj:
        summary_text = summary_obj.get("summary") if isinstance(summary_obj, dict) else None

    records = load_session_records(settings, logical_session_id)
    recent_records = records[-MAX_RECENT_RECORDS:] if records else []

    recent_pairs: list[str] = []
    for idx, record in enumerate(recent_records, start=1):
        request_messages = record.get("request_messages", [])
        user_text = ""
        if isinstance(request_messages, list):
            for msg in reversed(request_messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_text = msg.get("content", "")
                    break
        assistant_text = ""
        reply_message = record.get("reply_message")
        if isinstance(reply_message, dict):
            assistant_text = reply_message.get("content", "")

        user_text = _truncate_text(str(user_text), MAX_RECENT_TEXT_CHARS)
        assistant_text = _truncate_text(str(assistant_text), MAX_RECENT_TEXT_CHARS)

        recent_pairs.append(
            f"[{idx}] User: {user_text}\n    Assistant: {assistant_text}"
        )

    recent_pairs_text = "\n".join(recent_pairs).strip()

    memory_chunks: list[str] = []
    if summary_text:
        memory_chunks.append(
            "Summary of previous interactions for this logical session:\n"
            + summary_text
        )
    if recent_pairs_text:
        memory_chunks.append("Recent exchanges (truncated):\n" + recent_pairs_text)

    if not memory_chunks:
        return original_messages

    memory_system_message = {
        "role": "system",
        "content": "\n\n".join(memory_chunks),
    }
    return [memory_system_message] + original_messages


def build_recent_messages_for_session(
    settings: Settings,
    logical_session_id: str,
    max_turns: int,
) -> list[dict[str, Any]]:
    """
    构造某个 logical_session 的“最近 max_turns 条对话”的 messages 列表，
    用于将来注入到模型的上下文中。

    规则：
    - 使用 load_session_records 读出所有记录，取末尾 max_turns 条。
    - 按时间顺序遍历这些记录：
      - 对每一条 record：
        - 取 record["request_messages"]（已经是 OpenAI 风格的 message 列表），
          append 到结果里；
        - 再把 record["reply_message"]（role=assistant）也 append。
    - 最终返回一个 messages 列表，适合直接发给 /v1/chat/completions。

    这一步只是 helper，不要改变现有 /v1/chat/completions 的行为。
    """
    records = load_session_records(settings, logical_session_id)
    selected = records[-max_turns:] if max_turns > 0 else []

    messages: list[dict[str, Any]] = []
    for record in selected:
        request_messages = record.get("request_messages", [])
        if isinstance(request_messages, list):
            messages.extend(request_messages)
        reply_message = record.get("reply_message")
        if isinstance(reply_message, dict):
            messages.append(reply_message)

    return messages


def append_chat_log(
    settings: Settings,
    logical_session_id: str,
    request_messages: list[dict[str, Any]],
    reply_message: dict[str, Any],
) -> None:
    path = get_chat_log_path(settings)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "logical_session_id": logical_session_id,
        "request_messages": request_messages,
        "reply_message": reply_message,
    }
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"DEBUG failed to append chat log: {exc!r}")


def get_summary_store_path(settings: Settings) -> Path:
    """
    返回摘要存储文件路径：log_dir / summary_store_path。
    """
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / settings.summary_store_path


def load_session_summaries(settings: Settings) -> dict[str, Any]:
    """
    读取所有 logical_session_id 对应的摘要。
    - 文件如果不存在，返回 {}
    - 如果解析失败，打印 debug 并返回 {}
    - 文件格式是一个 JSON 对象：
      { "<logical_session_id>": { ...summary_object... }, ... }
    """
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


def save_session_summary(
    settings: Settings,
    logical_session_id: str,
    summary_obj: dict[str, Any],
) -> None:
    """
    更新某个 logical_session_id 的摘要，并写回文件。
    - load_session_summaries 得到全量 dict
    - 更新 data[logical_session_id] = summary_obj
    - 写回 JSON（utf-8, ensure_ascii=False, indent=2）
    - 写入失败时打印 debug，不抛异常
    """
    path = get_summary_store_path(settings)
    data = load_session_summaries(settings)
    data[logical_session_id] = summary_obj
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"DEBUG failed to save session summary: {exc!r}")


async def run_session_summarization(
    settings: Settings, logical_session_id: str
) -> dict[str, Any]:
    """
    对指定 logical_session_id 跑一次 summarization，
    更新 session_summaries.json，并返回本次 summary 对象。
    """
    records = load_session_records(settings, logical_session_id)
    total_records = len(records)
    if not records:
        return {"status": "no_records", "detail": "No records for this session."}

    if len(records) < 4:
        return {
            "status": "too_short",
            "detail": "Not enough records to summarize.",
        }

    recent_records = records[-settings.summary_max_turns :]

    conversation_lines: list[str] = []
    for record in recent_records:
        request_messages = record.get("request_messages", [])
        if isinstance(request_messages, list):
            for msg in request_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                conversation_lines.append(f"{role}: {content}")

        reply_message = record.get("reply_message")
        if isinstance(reply_message, dict):
            role = reply_message.get("role", "assistant")
            content = reply_message.get("content", "")
            conversation_lines.append(f"{role}: {content}")

    conversation_segment = "\n".join(conversation_lines)

    summaries = load_session_summaries(settings)
    previous_summary_obj = summaries.get(logical_session_id, {})
    previous_summary_text = ""
    if isinstance(previous_summary_obj, dict):
        previous_summary_text = previous_summary_obj.get("summary", "")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT}
    ]

    if previous_summary_text:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Previous rolling summary for this logical session:\n"
                    f"{previous_summary_text}"
                ),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": (
                "Here is the latest conversation segment for logical_session_id = "
                f"{logical_session_id}.\n\n"
                "LATEST CONVERSATION SEGMENT:\n"
                f"{conversation_segment}"
            ),
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

    client_kwargs = build_httpx_client_kwargs(settings)

    try:
        async with httpx.AsyncClient(**client_kwargs) as client:
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
            response_data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
    except Exception as exc:
        print(f"DEBUG failed to parse OpenRouter response JSON: {exc!r}")
        return {"status": "error", "detail": "Invalid OpenRouter response"}

    summary_core: dict[str, Any]
    try:
        summary_core = json.loads(model_output)
        if not isinstance(summary_core, dict):
            summary_core = {
                "summary": model_output,
                "key_points": [],
                "open_questions": [],
            }
    except json.JSONDecodeError:
        summary_core = {
            "summary": model_output,
            "key_points": [],
            "open_questions": [],
        }

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    result = {
        "timestamp": timestamp,
        "status": "ok",
        **summary_core,
        "total_records": total_records,
    }
    save_session_summary(settings, logical_session_id, result)
    return result


async def maybe_auto_summarize_session(
    settings: Settings,
    logical_session_id: str,
) -> None:
    """
    在 chat_completions 成功返回后调用。
    根据当前记录条数和上次 summary 的 total_records 决定是否重新 summarization。
    """
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

    seen_raw = summary.get("total_records", 0) if isinstance(summary, dict) else 0
    try:
        seen = int(seen_raw)
    except (TypeError, ValueError):
        seen = 0

    diff = total - seen
    print(
        f"DEBUG auto-summarize check: logical_session_id={logical_session_id}, "
        f"total={total}, seen={seen}, diff={diff}"
    )

    if diff < UPDATE_EVERY_RECORDS:
        return

    await run_session_summarization(settings, logical_session_id)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    body_bytes = await request.body()
    try:
        payload = json.loads(body_bytes)
    except json.JSONDecodeError as e:
        print(f"DEBUG failed to parse request JSON: {e!r}")
        print(f"DEBUG raw body (first 200 bytes): {body_bytes[:200]!r}")
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    print(f"DEBUG incoming payload keys: {list(payload.keys())}")
    print(f"DEBUG stream flag: {payload.get('stream')}")

    if payload.get("stream") is True:
        raise HTTPException(status_code=400, detail="Streaming is not supported yet.")

    settings = get_settings()
    headers = request.headers
    logical_session_id = headers.get("x-logical-session-id") or payload.get(
        "logical_session_id"
    )

    messages = payload.get("messages")
    log_request_messages = messages if isinstance(messages, list) else []

    if logical_session_id and isinstance(messages, list):
        try:
            new_messages = build_messages_with_memory(
                settings=settings,
                logical_session_id=logical_session_id,
                original_messages=messages,
            )
            payload["messages"] = new_messages
            print(
                "DEBUG injected memory for logical_session_id="
                f"{logical_session_id}, orig_len={len(messages)}, "
                f"new_len={len(new_messages)}"
            )
        except Exception as exc:
            print(f"DEBUG build_messages_with_memory failed: {exc!r}")

    url = f"{settings.openrouter_base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }

    client_kwargs = build_httpx_client_kwargs(settings)

    print(f"DEBUG openrouter_base_url: {settings.openrouter_base_url}")
    print(f"DEBUG final URL: {url}")
    print(f"DEBUG httpx.AsyncClient kwargs: {client_kwargs}")

    try:
        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.post(url, json=payload, headers=headers)
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
    if logical_session_id and isinstance(messages, list):
        try:
            reply_message = (
                response_data.get("choices", [{}])[0]
                .get("message", {})
            )
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
            asyncio.create_task(
                maybe_auto_summarize_session(settings, logical_session_id)
            )
        except Exception as exc:
            print(f"DEBUG maybe_auto_summarize_session failed: {exc!r}")

    print(f"DEBUG OpenRouter status: {response.status_code}")
    print(f"DEBUG OpenRouter body (first 200 chars): {response.text[:200]!r}")
    return response_data


@app.post("/internal/summarize_session")
async def summarize_session(payload: dict[str, str] = Body(...)) -> Any:
    """
    手动触发某个 logical_session 的滚动摘要更新。

    请求体 JSON:
    {
      "logical_session_id": "xxx"
    }
    """
    logical_session_id = payload.get("logical_session_id")
    if not logical_session_id:
        raise HTTPException(status_code=400, detail="logical_session_id is required")

    settings = get_settings()

    result = await run_session_summarization(settings, logical_session_id)

    return {
        "logical_session_id": logical_session_id,
        "result": result,
    }
