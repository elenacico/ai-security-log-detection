import os
from google import genai

key = os.environ.get("GEMINI_API_KEY", "")
print(f"Loaded key prefix: '{key[:6]}' (Length: {len(key)})")

try:
    client = genai.Client(api_key=key)
    res = client.models.generate_content(
        model="gemini-3.6-flash", contents="Hello"
    )
    print("✅ Connection successful:", res.text)
except Exception as e:
    print("❌ API Key failed:", e)
