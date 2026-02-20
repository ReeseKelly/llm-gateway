import asyncio
import httpx


async def main():
    url = "https://llm-gateway-sser.onrender.com"
#    url = "http://127.0.0.1:8000/telegram/webhook"

    # 这就是我们想发给 webhook 的 JSON
    payload = {
        "message": {
            "chat": {"id": 123456},
            "text": "你好吗~",
        }
    }

    async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
        resp = await client.post(url, json=payload)
        print("Status:", resp.status_code)
        print("Body:", resp.text)


if __name__ == "__main__":
    asyncio.run(main())
