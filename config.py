import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


load_dotenv()


class Settings(BaseSettings):
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api"
    outbound_proxy_url: str | None = None
    log_dir: str = Field(default="logs", env="LOG_DIR")

    summary_model: str = "deepseek/deepseek-v3.2"
    summary_max_turns: int = 40
    summary_store_path: str = Field(
        default="session_summaries.json", env="SUMMARY_STORE_PATH"
    )

    # tg
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_target_chat_id: str = os.getenv("TELEGRAM_TARGET_CHAT_ID", "")
    gateway_base_url: str = os.getenv(
        "GATEWAY_BASE_URL", "https://llm-gateway-sser.onrender.com"
    )
    telegram_default_model: str = os.getenv(
        "TELEGRAM_DEFAULT_MODEL", "anthropic/claude-sonnet-4.5"
    )

    telegram_webhook_secret: str = "changeme-super-secret"
    shared_session_id: str | None = None
    gateway_api_key: str = ""

    context_max_tokens: int = 8000
    context_keep_last_user_messages: int = 4
    context_keep_last_assistant_messages: int = 4

    memory_max_candidates: int = 5
    memory_activation_ttl: int = 3
    memory_consolidation_enabled: bool = False
    memory_consolidation_min_count: int = 3
    memory_consolidation_model: str = ""

    caldav_url: str = ""
    caldav_username: str = ""
    caldav_password: str = ""
    caldav_principal: str | None = None
    caldav_calendar_name: str | None = None

    health_provider: str = "file"
    health_data_path: str = "/data/health.json"
    health_log_path: str = "/data/health_log.jsonl"
    health_webhook_token: str = ""

    timezone: str = "Asia/Shanghai"
    default_tz: str = "Asia/Shanghai"

    weather_api_url: str = ""
    weather_api_key: str | None = None
    weather_default_location: str = ""
    weather_units: str = "metric"

    notes_log_path: str = "/data/notes_log.jsonl"
    notes_default_ttl_days: int = 7
    midterm_memory_path: str = "/data/midterm_memory.jsonl"
    midterm_default_ttl_days: int = 14

    ltm_index_path: str = "/data/ltm_index.json"

    tasks_log_path: str = "/data/tasks.jsonl"
    task_check_interval_minutes: int = 2

@lru_cache
def get_settings() -> Settings:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    return Settings(
        openrouter_api_key=api_key,
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api"),
        outbound_proxy_url=os.getenv("OUTBOUND_PROXY_URL"),
        summary_model=os.getenv("SUMMARY_MODEL", "deepseek/deepseek-v3.2"),
        summary_max_turns=_safe_int(os.getenv("SUMMARY_MAX_TURNS"), 40),
        context_max_tokens=_safe_int(os.getenv("CONTEXT_MAX_TOKENS"), 8000),
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
        shared_session_id=os.getenv("SHARED_SESSION_ID") or None,
        gateway_api_key=os.getenv("GATEWAY_API_KEY", ""),
        caldav_url=os.getenv("CALDAV_URL", ""),
        caldav_username=os.getenv("CALDAV_USERNAME", ""),
        caldav_password=os.getenv("CALDAV_PASSWORD", ""),
        caldav_principal=os.getenv("CALDAV_PRINCIPAL") or None,
        caldav_calendar_name=os.getenv("CALDAV_CALENDAR_NAME") or None,
        health_provider=os.getenv("HEALTH_PROVIDER", "file"),
        health_data_path=os.getenv("HEALTH_DATA_PATH", "/data/health.json"),
        health_log_path=os.getenv("HEALTH_LOG_PATH", "/data/health_log.jsonl"),
        health_webhook_token=os.getenv("HEALTH_WEBHOOK_TOKEN", ""),
        timezone=os.getenv("TIMEZONE", "Asia/Shanghai"),
        default_tz=os.getenv("DEFAULT_TZ", os.getenv("TIMEZONE", "Asia/Shanghai")),
        weather_api_url=os.getenv("WEATHER_API_URL", ""),
        weather_api_key=os.getenv("WEATHER_API_KEY") or None,
        weather_default_location=os.getenv("WEATHER_DEFAULT_LOCATION", ""),
        weather_units=os.getenv("WEATHER_UNITS", "metric"),
        notes_log_path=os.getenv("NOTES_LOG_PATH", "/data/notes_log.jsonl"),
        notes_default_ttl_days=_safe_int(os.getenv("NOTES_DEFAULT_TTL_DAYS"), 7),
        midterm_memory_path=os.getenv("MIDTERM_MEMORY_PATH", "/data/midterm_memory.jsonl"),
        midterm_default_ttl_days=_safe_int(os.getenv("MIDTERM_DEFAULT_TTL_DAYS"), 14),
        ltm_index_path=os.getenv("LTM_INDEX_PATH", "/data/ltm_index.json"),
        tasks_log_path=os.getenv("TASKS_LOG_PATH", "/data/tasks.jsonl"),
        task_check_interval_minutes=_safe_int(os.getenv("TASK_CHECK_INTERVAL_MINUTES"), 2),
    )


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
