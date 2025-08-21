# main.py
from __future__ import annotations
import os,re
import json
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal

# ---------------------------
# Config
# ---------------------------
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment or .env file")

app = FastAPI(title="AI Server (Simplified)", version="1.0.1")

# ---------------------------
# Pydantic models
# ---------------------------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gpt-5"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None

class PrefillRequest(BaseModel):
    email_text: str

class PrefillResponse(BaseModel):
    success: bool
    message: str
    fields: Dict[str, Optional[str]]

# ---------------------------
# OpenAI helper
# ---------------------------
async def call_openai(payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(OPENAI_URL, json=payload, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()

# ---------------------------
# Routes
# ---------------------------

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    payload = {
        "model": req.model,
        "messages": [m.dict() for m in req.messages],
    }
    if req.max_tokens:
        payload["max_completion_tokens"] = req.max_tokens
    return await call_openai(payload)

@app.post("/v1/prefill", response_model=PrefillResponse)
async def prefill(req: PrefillRequest):
    system_prompt = (
        "You are an invoice field extraction assistant. "
        "Extract exactly these fields from the email: amount, currency, due_date, "
        "description, company, contact. Always respond in pure JSON with string values. "
        "If a field is missing, use null."
    )

    payload = {
        "model": "gpt-5",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.email_text},
        ],
        "response_format": {"type": "json_object"},
    }

    data = await call_openai(payload)
    content = data["choices"][0]["message"]["content"]

    try:
        fields_raw = json.loads(content)
    except Exception:
        fields = regex_extract(req.email_text)
        return PrefillResponse(success=True, message="data extracted using regex", fields=fields)
        #raise HTTPException(status_code=500, detail="LLM did not return valid JSON")
        

    # 🔧 Normalize everything to string (or None)
    fields = {k: (str(v) if v is not None else None) for k, v in fields_raw.items()}

    return PrefillResponse(success=True, message="data extracted using LLM", fields=fields)


# =========================
# Regex helpers (fallback only)
# =========================
def regex_extract(email_text: str) -> Dict[str, Optional[str]]:
    """Minimal regex extraction as fallback."""
    amount = None
    currency = None
    due_date = None

    amt_match = re.search(r"\$([\d,]+\.\d{2})", email_text)
    if amt_match:
        amount = amt_match.group(1)
        currency = "USD"

    date_match = re.search(r"Due\s+Date[:\s]+([A-Za-z]+\s+\d{1,2},\s*\d{4})", email_text)
    if date_match:
        due_date = date_match.group(1)

    return {
        "amount": amount,
        "currency": currency,
        "due_date": due_date,
        "description": "Regex fallback extraction",
        "company": None,
        "contact": None,
    }


@app.get("/health")
def health():
    return {"status": "ok", "openai_key": bool(OPENAI_API_KEY)}

# ---------------------------
# Local run
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8090, reload=True)
