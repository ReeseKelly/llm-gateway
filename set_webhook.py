import asyncio
import httpx

TELEGRAM_BOT_TOKEN = "8509028841:AAHSlEPm9O762PeIHUd6m8R9sSSFonp77as"  
GATEWAY_URL = "https://llm-gateway-sser.onrender.com/telegram/webhook"

async def main():
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook"
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
