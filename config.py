import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api"
    outbound_proxy_url: str | None = None
    log_dir: str = "logs"

    summary_model: str = "deepseek/deepseek-v3.2"
    summary_max_turns: int = 40
    summary_store_path: str = "session_summaries.json"

    context_max_tokens: int = 9000
    context_keep_last_user_messages: int = 4
    context_keep_last_assistant_messages: int = 4

    memory_max_candidates: int = 5
    memory_activation_ttl: int = 3
    memory_consolidation_enabled: bool = False
    memory_consolidation_min_count: int = 3
    memory_consolidation_model: str = ""


def _safe_int(value: str | None, default: int) -> int:
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _safe_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    return Settings(
        openrouter_api_key=api_key,
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api"),
        outbound_proxy_url=os.getenv("OUTBOUND_PROXY_URL"),
        log_dir=os.getenv("LOG_DIR", "logs"),
        summary_model=os.getenv("SUMMARY_MODEL", "deepseek/deepseek-v3.2"),
        summary_max_turns=_safe_int(os.getenv("SUMMARY_MAX_TURNS"), 40),
        summary_store_path=os.getenv("SUMMARY_STORE_PATH", "session_summaries.json"),
        context_max_tokens=_safe_int(os.getenv("CONTEXT_MAX_TOKENS"), 3500),
        context_keep_last_user_messages=_safe_int(
            os.getenv("CONTEXT_KEEP_LAST_USER_MESSAGES"), 4
        ),
        context_keep_last_assistant_messages=_safe_int(
            os.getenv("CONTEXT_KEEP_LAST_ASSISTANT_MESSAGES"), 4
        ),
        memory_max_candidates=_safe_int(os.getenv("MEMORY_MAX_CANDIDATES"), 5),
        memory_activation_ttl=_safe_int(os.getenv("MEMORY_ACTIVATION_TTL"), 3),
        memory_consolidation_enabled=_safe_bool(
            os.getenv("MEMORY_CONSOLIDATION_ENABLED"), False
        ),
        memory_consolidation_min_count=_safe_int(
            os.getenv("MEMORY_CONSOLIDATION_MIN_COUNT"), 3
        ),
        memory_consolidation_model=os.getenv("MEMORY_CONSOLIDATION_MODEL", ""),
    )
