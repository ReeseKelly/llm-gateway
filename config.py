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
    log_dir: str = "logs" # 新增：日志目录


def get_settings() -> Settings:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api")
    outbound_proxy_url = os.getenv("OUTBOUND_PROXY_URL")

    log_dir = os.getenv("GATEWAY_LOG_DIR", "logs")

    return Settings(
        openrouter_api_key=api_key,
        openrouter_base_url=base_url,
        outbound_proxy_url=outbound_proxy_url,
        log_dir = log_dir
    )
