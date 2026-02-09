from typing import Any
import json
import os
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException, Request

from config import get_settings

try:
    # 如果安装了 httpx_socks，就用它来支持 socks5 代理
    from httpx_socks import AsyncProxyTransport
except ImportError:
    AsyncProxyTransport = None  # 后面做兼容处理

app = FastAPI()

def append_chat_log(settings, session_id: str, logical_session_id: str | None, request_payload: dict[str, Any], response_payload: dict[str, Any]) -> None:
    """
    把每次 chat 调用的请求/响应，追加到一个 JSONL 文件中。
    logical_session_id: 逻辑会话 ID
    """
    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        log_path = os.path.join(settings.log_dir, "chat_log.jsonl")

        # 尝试从请求中抓一些关键信息
        model = request_payload.get("model")
        messages = request_payload.get("messages", [])

        # 尝试从返回中抓出第一条回复
        choices = response_payload.get("choices", [])
        reply_message: dict[str, Any] | None = None
        if choices:
            reply_message = choices[0].get("message")

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "session_id": session_id,
            "logical_session_id": logical_session_id,  # 👈 注意这里：字符串 key + 变量名都带下划线、没有空格
            "model": model,
            "request_messages": messages,
            "reply_message": reply_message,
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
    except Exception as exc:
        # 日志写失败不能影响正常请求；先打印一行 DEBUG 就好
        print(f"DEBUG failed to write chat log: {exc!r}")


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
    
    # 技术通路/窗口 ID（channel）
    session_id = (
        request.headers.get("x-session-id")
        or payload.get("session_id")
        or "default"
    )

    # 逻辑会话 ID（可以跨窗口的长线 project）
    logical_session_id = (
        request.headers.get("x-logical-session-id")
        or payload.get("logical_session_id")
        or None
    )    

    print(f"DEBUG incoming payload keys: {list(payload.keys())}")
    print(f"DEBUG stream flag: {payload.get('stream')}")

    if payload.get("stream") is True:
        raise HTTPException(status_code=400, detail="Streaming is not supported yet.")

    settings = get_settings()
    url = f"{settings.openrouter_base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }

    # 统一的 AsyncClient 配置
    client_kwargs: dict[str, Any] = {
        "timeout": 60.0,
        "trust_env": False,  # 禁止从环境变量自动读取 socks4 等代理
    }

    # 如果配置了 OUTBOUND_PROXY_URL，且安装了 httpx_socks，就通过 transport 使用 socks5
    transport = None
    if settings.outbound_proxy_url:
        print(f"DEBUG outbound_proxy_url: {settings.outbound_proxy_url}")
        if AsyncProxyTransport is None:
            print("DEBUG httpx_socks is not installed, cannot use socks proxy.")
        else:
            transport = AsyncProxyTransport.from_url(settings.outbound_proxy_url)

    if transport is not None:
        client_kwargs["transport"] = transport

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
    
    response_payload = response.json()

    try:
        append_chat_log(settings, session_id, logical_session_id, payload, response_payload)
    except Exception as exc:
        print(f"DEBUG append_chat_log raised: {exc!r}")

    print(f"DEBUG OpenRouter status: {response.status_code}")
    print(f"DEBUG OpenRouter body (first 200 chars): {response.text[:200]!r}")
    return response.json()


