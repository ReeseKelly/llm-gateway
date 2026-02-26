import asyncio
import httpx

TELEGRAM_BOT_TOKEN = "8219238737:AAHjrVTXnGis1_0R4jfR8q0bOoReh2n0QuY"  
GATEWAY_URL = "https://llm-gateway-sser.onrender.com/telegram/webhook&secret_token=changeme-super-secret"

async def main():
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook&secret_token=changeme-telegram-super-secret"
    payload = {
        "url": GATEWAY_URL,
    }

    proxies = {
        "https://": "socks5://127.0.0.1:1080",
        "http://": "socks5://127.0.0.1:1080",        
    }

    async with httpx.AsyncClient(timeout=30.0, proxies = proxies) as client:
        resp = await client.post(url, data=payload)
        print("Status:", resp.status_code)
        print("Body:", resp.text)

if __name__ == "__main__":
    asyncio.run(main())
