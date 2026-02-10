import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api"
    # 新增：可选的出站代理地址（例如 socks5://127.0.0.1:1080）
    outbound_proxy_url: str | None = None
    log_dir: str = "logs"

    # 新增：摘要相关配置
    summary_model: str = "openai/gpt-4o-mini"
    summary_max_turns: int = 40
    summary_store_path: str = "session_summaries.json"


def get_settings() -> Settings:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api")
    outbound_proxy_url = os.getenv("OUTBOUND_PROXY_URL")

    summary_model = os.getenv("SUMMARY_MODEL", "openai/gpt-4o-mini")
    summary_store_path = os.getenv("SUMMARY_STORE_PATH", "session_summaries.json")

    summary_max_turns = 40
    summary_max_turns_raw = os.getenv("SUMMARY_MAX_TURNS")
    if summary_max_turns_raw:
        try:
            summary_max_turns = int(summary_max_turns_raw)
        except ValueError:
            summary_max_turns = 40

    return Settings(
        openrouter_api_key=api_key,
        openrouter_base_url=base_url,
        outbound_proxy_url=outbound_proxy_url,
        summary_model=summary_model,
        summary_max_turns=summary_max_turns,
        summary_store_path=summary_store_path,
    )
