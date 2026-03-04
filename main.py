# ============================================================
# CodeX Master AI API - FastAPI
# Author: CodeX_Network (Aman Kumar)
# Version: 3.0.0 - Multi-Key Rotation + Auto Fallback
# Deploy: Render
# ============================================================

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union, AsyncGenerator
import httpx
import itertools
import time
import os
import secrets
import uuid
import sqlite3
from datetime import datetime

# ============================================================
# APP
# ============================================================
app = FastAPI(
    title="CodeX Master AI API",
    description="""🚀 Master AI API v3.0
    ✅ Multi-Key Rotation (no rate limits!)
    ✅ Auto Fallback (if one provider fails, auto switches)
    ✅ 15+ Models (Free + Premium)
    ✅ Intelligent Auto-Routing
    ✅ Multimodal (Vision/Image)
    ✅ HTML Builder (like Lovable)
    ✅ API Key Generator
    ✅ OpenAI-Compatible /v1/chat/completions
    """,
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ============================================================
# MULTI-KEY ROTATION SETUP
# Env mein comma-separated keys daalo:
# OPENROUTER_API_KEY=key1,key2,key3,key4,key5
# GROQ_API_KEY=gsk_key1,gsk_key2,gsk_key3
# GOOGLE_API_KEY=AIza_key1,AIza_key2,AIza_key3
# ============================================================

def _load_keys(env_name: str) -> list:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]

# Load all keys on startup
_OR_KEYS   = _load_keys("OPENROUTER_API_KEY")
_GROQ_KEYS = _load_keys("GROQ_API_KEY")
_GGL_KEYS  = _load_keys("GOOGLE_API_KEY")

# Create infinite cyclers for each provider
_or_cycle   = itertools.cycle(_OR_KEYS)   if _OR_KEYS   else iter([""])
_groq_cycle = itertools.cycle(_GROQ_KEYS) if _GROQ_KEYS else iter([""])
_ggl_cycle  = itertools.cycle(_GGL_KEYS)  if _GGL_KEYS  else iter([""])

def next_or_key()   -> str: return next(_or_cycle)
def next_groq_key() -> str: return next(_groq_cycle)
def next_ggl_key()  -> str: return next(_ggl_cycle)

# ============================================================
# DATABASE - SQLite API Key Manager
# ============================================================
DB_PATH = "api_keys.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS api_keys (
            key TEXT PRIMARY KEY, name TEXT, email TEXT,
            created_at TEXT, requests_count INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1, tier TEXT DEFAULT 'free', last_used TEXT
        )
    ''')
    conn.commit(); conn.close()

init_db()

# ============================================================
# MODELS REGISTRY
# ============================================================
MODELS: Dict[str, Dict[str, Any]] = {
    # ── FREE: OpenRouter ──────────────────────────────────
    "llama-3.3-70b": {
        "id": "meta-llama/llama-3.3-70b-instruct:free", "provider": "openrouter",
        "tier": "free", "context_window": 131072, "multimodal": False,
        "best_for": ["general", "coding", "analysis", "writing", "chat"],
        "speed": "fast", "developer": "Meta",
        "description": "Meta Llama 3.3 70B - Best free general model. 128K context, multilingual, strong coding."
    },
    "gemini-2.0-flash-free": {
        "id": "google/gemini-2.0-flash-exp:free", "provider": "openrouter",
        "tier": "free", "context_window": 1048576, "multimodal": True,
        "best_for": ["multimodal", "vision", "long-context", "general"],
        "speed": "very-fast", "developer": "Google",
        "description": "Gemini 2.0 Flash (free via OpenRouter) - 1M context, vision, multimodal."
    },
    "deepseek-r1-free": {
        "id": "deepseek/deepseek-r1:free", "provider": "openrouter",
        "tier": "free", "context_window": 163840, "multimodal": False,
        "best_for": ["math", "reasoning", "science", "logic", "complex-problems"],
        "speed": "medium", "developer": "DeepSeek",
        "description": "DeepSeek R1 - Chain-of-thought. Best free model for math/science/logic."
    },
    "qwen-2.5-72b-free": {
        "id": "qwen/qwen-2.5-72b-instruct:free", "provider": "openrouter",
        "tier": "free", "context_window": 131072, "multimodal": False,
        "best_for": ["coding", "math", "multilingual", "analysis"],
        "speed": "fast", "developer": "Alibaba",
        "description": "Qwen 2.5 72B - Excellent coding (88% HumanEval), math, multilingual."
    },
    "mistral-7b-free": {
        "id": "mistralai/mistral-7b-instruct:free", "provider": "openrouter",
        "tier": "free", "context_window": 32768, "multimodal": False,
        "best_for": ["quick-tasks", "writing", "lightweight"],
        "speed": "fast", "developer": "Mistral AI",
        "description": "Mistral 7B - Lightweight, fast, efficient for quick tasks."
    },
    "phi-3-mini-free": {
        "id": "microsoft/phi-3-mini-128k-instruct:free", "provider": "openrouter",
        "tier": "free", "context_window": 128000, "multimodal": False,
        "best_for": ["lightweight", "simple-tasks", "edge"],
        "speed": "ultra-fast", "developer": "Microsoft",
        "description": "Phi-3 Mini - Tiny but powerful. 128K context. Ultra lightweight."
    },
    # ── FREE: Groq (Ultra-Fast LPU) ───────────────────────
    "llama-groq-fast": {
        "id": "llama-3.3-70b-versatile", "provider": "groq",
        "tier": "free", "context_window": 128000, "multimodal": False,
        "best_for": ["speed", "real-time", "chat", "quick-tasks"],
        "speed": "ultra-fast (~1200 tok/s)", "developer": "Meta + Groq",
        "description": "Llama 3.3 on Groq LPU - FASTEST. ~1200 tok/s. Best for real-time chat."
    },
    "mixtral-groq-fast": {
        "id": "mixtral-8x7b-32768", "provider": "groq",
        "tier": "free", "context_window": 32768, "multimodal": False,
        "best_for": ["speed", "general", "moe-reasoning"],
        "speed": "ultra-fast", "developer": "Mistral + Groq",
        "description": "Mixtral MoE on Groq - Fast mixture-of-experts."
    },
    "gemma2-groq": {
        "id": "gemma2-9b-it", "provider": "groq",
        "tier": "free", "context_window": 8192, "multimodal": False,
        "best_for": ["lightweight", "quick-chat", "summarization"],
        "speed": "ultra-fast", "developer": "Google + Groq",
        "description": "Gemma 2 9B on Groq - Ultra lightweight and fast."
    },
    # ── FREE: Google AI Studio ────────────────────────────
    "gemini-2.0-flash": {
        "id": "gemini-2.0-flash", "provider": "google",
        "tier": "free", "context_window": 1048576, "multimodal": True,
        "best_for": ["vision", "multimodal", "documents", "long-context", "general"],
        "speed": "fast", "developer": "Google",
        "description": "Gemini 2.0 Flash - Google AI Studio free. 1M context, vision/audio/video."
    },
    "gemini-1.5-pro": {
        "id": "gemini-1.5-pro", "provider": "google",
        "tier": "free", "context_window": 2097152, "multimodal": True,
        "best_for": ["2M-context", "large-docs", "video-analysis"],
        "speed": "medium", "developer": "Google",
        "description": "Gemini 1.5 Pro - 2M token context. Analyze entire codebases/books."
    },
    # ── PREMIUM ────────────────────────────────────────────
    "claude-opus-4": {
        "id": "anthropic/claude-opus-4", "provider": "openrouter",
        "tier": "premium", "context_window": 200000, "multimodal": True,
        "best_for": ["complex-reasoning", "research", "advanced-coding", "creative"],
        "speed": "slow", "developer": "Anthropic",
        "description": "Claude Opus 4 - Most powerful. Deep reasoning, extended thinking, 200K ctx."
    },
    "claude-sonnet-4": {
        "id": "anthropic/claude-sonnet-4-5", "provider": "openrouter",
        "tier": "premium", "context_window": 200000, "multimodal": True,
        "best_for": ["coding", "analysis", "balanced"],
        "speed": "medium", "developer": "Anthropic",
        "description": "Claude Sonnet 4.5 - Best balance of speed and intelligence."
    },
    "gpt-4o": {
        "id": "openai/gpt-4o", "provider": "openrouter",
        "tier": "premium", "context_window": 128000, "multimodal": True,
        "best_for": ["vision", "json", "function-calling", "general"],
        "speed": "fast", "developer": "OpenAI",
        "description": "GPT-4o - OpenAI flagship. Vision, JSON mode, function calling."
    },
    "o3-mini": {
        "id": "openai/o3-mini", "provider": "openrouter",
        "tier": "premium", "context_window": 200000, "multimodal": False,
        "best_for": ["math", "science", "STEM", "reasoning"],
        "speed": "medium", "developer": "OpenAI",
        "description": "o3-mini - Extended thinking. Best for STEM, competitive math."
    },
    "gemini-2.5-pro": {
        "id": "google/gemini-2.5-pro-preview", "provider": "openrouter",
        "tier": "premium", "context_window": 2097152, "multimodal": True,
        "best_for": ["2M-context", "multimodal", "video"],
        "speed": "medium", "developer": "Google",
        "description": "Gemini 2.5 Pro - 2M ctx, extended thinking, video analysis."
    },
    "perplexity-sonar-pro": {
        "id": "perplexity/sonar-pro", "provider": "openrouter",
        "tier": "premium", "context_window": 127072, "multimodal": False,
        "best_for": ["web-search", "real-time", "news", "current-events"],
        "speed": "fast", "developer": "Perplexity",
        "description": "Sonar Pro - Real-time web search with citations."
    },
    "codestral-2501": {
        "id": "mistralai/codestral-2501", "provider": "openrouter",
        "tier": "premium", "context_window": 256000, "multimodal": False,
        "best_for": ["code-generation", "code-completion", "FIM", "debugging"],
        "speed": "fast", "developer": "Mistral AI",
        "description": "Codestral 2501 - Dedicated code model. FIM, 256K context."
    },
    "deepseek-v3": {
        "id": "deepseek/deepseek-chat-v3-0324", "provider": "openrouter",
        "tier": "premium", "context_window": 163840, "multimodal": False,
        "best_for": ["coding", "math", "analysis"],
        "speed": "fast", "developer": "DeepSeek",
        "description": "DeepSeek V3 - Rivals GPT-4 at fraction of cost."
    },
    "nova-pro": {
        "id": "amazon/nova-pro-v1", "provider": "openrouter",
        "tier": "premium", "context_window": 300000, "multimodal": True,
        "best_for": ["document-analysis", "enterprise", "long-context"],
        "speed": "fast", "developer": "Amazon",
        "description": "Amazon Nova Pro - 300K context, enterprise-grade."
    },
}

# ============================================================
# FALLBACK CHAIN - Order of preference when rate-limited
# ============================================================
FALLBACK_FREE = [
    "llama-groq-fast",       # Ultra-fast (Groq)
    "llama-3.3-70b",         # OpenRouter free
    "qwen-2.5-72b-free",     # OpenRouter free
    "gemini-2.0-flash",      # Google free
    "deepseek-r1-free",      # OpenRouter free
    "mistral-7b-free",       # Lightweight fallback
    "phi-3-mini-free",       # Last resort
]

# ============================================================
# INTELLIGENT ROUTER
# ============================================================
def intelligent_route(prompt: str, task_type: str = "auto", prefer_free: bool = True) -> str:
    p = prompt.lower()
    signals = {
        "code":      ["code","program","function","debug","html","css","javascript","python",
                      "api","fix","error","build","website","react","nodejs","sql","deploy"],
        "math":      ["calculate","solve","equation","integral","derivative","math",
                      "algebra","geometry","physics","chemistry","formula"],
        "vision":    ["image","picture","photo","vision","see","screenshot","diagram","chart"],
        "search":    ["latest","recent","today","news","current","2025","2026","live","update"],
        "reasoning": ["analyze","think","reason","explain why","step by step","complex","deduce"],
        "creative":  ["write","story","poem","creative","novel","script","blog","essay"],
        "speed":     ["quickly","fast","quick","instant","real-time"],
    }
    detected = {k: sum(1 for w in v if w in p) for k, v in signals.items() if any(w in p for w in v)}
    if task_type != "auto":
        detected[task_type] = 99

    if prefer_free:
        if "vision"    in detected: return "gemini-2.0-flash"
        if "math"      in detected: return "deepseek-r1-free"
        if "code"      in detected: return "qwen-2.5-72b-free"
        if "speed"     in detected: return "llama-groq-fast"
        if "search"    in detected: return "gemini-2.0-flash"
        if "reasoning" in detected: return "deepseek-r1-free"
        if len(prompt) > 10000:     return "gemini-1.5-pro"
        return "llama-3.3-70b"
    else:
        if "vision"    in detected: return "gpt-4o"
        if "math"      in detected: return "o3-mini"
        if "code"      in detected: return "codestral-2501"
        if "search"    in detected: return "perplexity-sonar-pro"
        if "reasoning" in detected: return "claude-opus-4"
        if "creative"  in detected: return "claude-opus-4"
        if len(prompt) > 50000:     return "gemini-2.5-pro"
        return "claude-sonnet-4"

# ============================================================
# PYDANTIC SCHEMAS
# ============================================================
class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatRequest(BaseModel):
    model: str = Field(default="auto")
    messages: List[Message]
    max_tokens: Optional[int] = Field(default=4096, ge=1, le=32000)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    stream: Optional[bool] = False
    system: Optional[str] = None
    task_type: Optional[str] = "auto"
    prefer_free: Optional[bool] = True
    image_url: Optional[str] = None
    image_base64: Optional[str] = None

class HTMLBuilderRequest(BaseModel):
    prompt: str
    project_type: Optional[str] = "webapp"
    style: Optional[str] = "modern"
    include_animations: Optional[bool] = True
    responsive: Optional[bool] = True
    preferred_model: Optional[str] = "auto"

class APIKeyRequest(BaseModel):
    name: str
    email: Optional[str] = None
    tier: Optional[str] = "free"

# ============================================================
# PROVIDER CALLERS (with rotating keys)
# ============================================================
OR_URL   = "https://openrouter.ai/api/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GGL_URL  = "https://generativelanguage.googleapis.com/v1beta/models"

async def _call_openai_style(url: str, key: str, model_id: str,
                              messages: list, max_tokens: int, temperature: float,
                              extra_headers: dict = {}) -> dict:
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        **extra_headers
    }
    payload = {"model": model_id, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.text[:500])
        return r.json()

async def _stream_openrouter(model_id: str, key: str, messages: list,
                              max_tokens: int, temperature: float) -> AsyncGenerator[str, None]:
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("SITE_URL", "https://codex-master-api.onrender.com"),
        "X-Title": "CodeX Master AI API"
    }
    payload = {"model": model_id, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature, "stream": True}
    async with httpx.AsyncClient(timeout=120.0) as c:
        async with c.stream("POST", OR_URL, json=payload, headers=headers) as r:
            async for line in r.aiter_lines():
                if line:
                    yield line + "\n"

async def call_openrouter(model_id: str, messages: list, max_tokens: int, temperature: float) -> dict:
    key = next_or_key()
    if not key:
        raise HTTPException(503, "OPENROUTER_API_KEY not set in environment")
    return await _call_openai_style(
        OR_URL, key, model_id, messages, max_tokens, temperature,
        extra_headers={
            "HTTP-Referer": os.getenv("SITE_URL", "https://codex-master-api.onrender.com"),
            "X-Title": "CodeX Master AI API"
        }
    )

async def call_groq(model_id: str, messages: list, max_tokens: int, temperature: float) -> dict:
    key = next_groq_key()
    if not key:
        raise HTTPException(503, "GROQ_API_KEY not set in environment")
    return await _call_openai_style(GROQ_URL, key, model_id, messages, min(max_tokens, 8000), temperature)

async def call_google(model_id: str, messages: list, max_tokens: int, temperature: float) -> dict:
    key = next_ggl_key()
    if not key:
        raise HTTPException(503, "GOOGLE_API_KEY not set in environment")
    contents = []
    sys_inst = None
    for m in messages:
        if m["role"] == "system":
            sys_inst = {"parts": [{"text": m["content"]}]}
            continue
        role = "user" if m["role"] == "user" else "model"
        parts = []
        c = m["content"]
        if isinstance(c, str):
            parts.append({"text": c})
        elif isinstance(c, list):
            for p in c:
                if p.get("type") == "text":
                    parts.append({"text": p.get("text", "")})
                elif p.get("type") == "image_url":
                    url = (p.get("image_url") or {}).get("url", "")
                    if url.startswith("data:"):
                        mime = url.split(";")[0].split(":")[1]
                        b64  = url.split(",")[1]
                        parts.append({"inline_data": {"mime_type": mime, "data": b64}})
                    else:
                        parts.append({"text": f"[image]{url}"})
        contents.append({"role": role, "parts": parts})
    payload = {
        "contents": contents,
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}
    }
    if sys_inst:
        payload["systemInstruction"] = sys_inst
    endpoint = f"{GGL_URL}/{model_id}:generateContent?key={key}"
    async with httpx.AsyncClient(timeout=120.0) as cl:
        r = await cl.post(endpoint, json=payload)
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.text[:500])
        data = r.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            text = "No response generated"
        return {"choices": [{"message": {"role": "assistant", "content": text}}], "usage": {}}

async def _raw_call(alias: str, messages: list, max_tokens: int, temperature: float) -> dict:
    """Single model call without fallback"""
    info = MODELS[alias]
    prov = info["provider"]
    mid  = info["id"]
    if prov == "groq":       return await call_groq(mid, messages, max_tokens, temperature)
    if prov == "google":     return await call_google(mid, messages, max_tokens, temperature)
    return await call_openrouter(mid, messages, max_tokens, temperature)

async def call_provider_with_fallback(
    alias: str, messages: list, max_tokens: int, temperature: float
) -> tuple[dict, str]:
    """
    Smart call with auto-fallback:
    1. Try requested model
    2. On 429/503/500 -> auto switch to next in fallback chain
    Returns (result_dict, actual_model_used)
    """
    chain = [alias] + [m for m in FALLBACK_FREE if m != alias]
    last_error = None
    for model_alias in chain:
        try:
            result = await _raw_call(model_alias, messages, max_tokens, temperature)
            return result, model_alias
        except HTTPException as e:
            if e.status_code in (429, 503, 500, 502, 504):
                last_error = e
                continue  # Try next model
            raise  # Other errors (400, 401) - don't fallback
        except Exception as e:
            last_error = e
            continue
    raise HTTPException(503, f"All providers failed/rate-limited. Last error: {str(last_error)}")

# ============================================================
# API KEY AUTH
# ============================================================
def validate_api_key(key: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT is_active FROM api_keys WHERE key = ?", (key,))
    row = c.fetchone()
    if row and row[0] == 1:
        c.execute("UPDATE api_keys SET requests_count=requests_count+1, last_used=? WHERE key=?",
                  (datetime.now().isoformat(), key))
        conn.commit(); conn.close(); return True
    conn.close(); return False

async def get_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    if os.getenv("REQUIRE_AUTH", "false").lower() != "true":
        return "no-auth"
    if not x_api_key or not validate_api_key(x_api_key):
        raise HTTPException(401, "Invalid/missing X-API-Key. Get one: POST /generate-key")
    return x_api_key

# ============================================================
# ROUTES
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def root():
    or_count  = len(_OR_KEYS)
    groq_count = len(_GROQ_KEYS)
    ggl_count  = len(_GGL_KEYS)
    total_keys = or_count + groq_count + ggl_count
    html = f"""
    <!DOCTYPE html><html><head><title>CodeX Master AI API v3</title><meta charset="utf-8">
    <style>
      *{{margin:0;padding:0;box-sizing:border-box}}
      body{{font-family:'Segoe UI',sans-serif;background:#08080f;color:#e0e0ff;min-height:100vh}}
      .hero{{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);padding:60px 40px;text-align:center}}
      h1{{font-size:2.8em;background:linear-gradient(90deg,#00d2ff,#7b2ff7,#ff6b6b);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px}}
      .sub{{color:#aaa;margin:10px 0}}
      .badge{{display:inline-block;padding:4px 12px;border-radius:20px;font-size:.75em;margin:4px}}
      .free{{background:#0d3b1c;color:#4ade80;border:1px solid #166534}}
      .hot{{background:#3b1a0d;color:#fb923c;border:1px solid #9a3412}}
      .blue{{background:#1a2a3b;color:#60a5fa;border:1px solid #1e40af}}
      .stats{{display:flex;justify-content:center;gap:30px;padding:20px;flex-wrap:wrap}}
      .stat{{text-align:center;background:#111122;border:1px solid #2a2a4a;border-radius:12px;padding:16px 24px}}
      .stat-num{{font-size:2em;font-weight:bold;color:#00d2ff}}
      .stat-label{{color:#888;font-size:.85em;margin-top:4px}}
      .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;padding:30px;max-width:1100px;margin:0 auto}}
      .card{{background:#111122;border:1px solid #2a2a4a;border-radius:10px;padding:20px}}
      .card:hover{{border-color:#7b2ff7;transform:translateY(-3px);transition:.2s}}
      .card h3{{color:#00d2ff;margin-bottom:8px;font-size:.95em}}
      .card p{{color:#aaa;font-size:.82em;line-height:1.5}}
      .links{{text-align:center;padding:16px}}
      .btn{{display:inline-block;padding:11px 22px;margin:6px;border-radius:8px;text-decoration:none;font-weight:bold;transition:.2s}}
      .p{{background:#7b2ff7;color:#fff}}.s{{background:#111;border:1px solid #444;color:#aaa}}
      .key-status{{background:#0a1a0a;border:1px solid #1a4a1a;border-radius:8px;padding:16px;margin:10px auto;max-width:500px;font-family:monospace;font-size:.85em}}
    </style></head><body>
    <div class="hero">
      <h1>⚡ CodeX Master AI API</h1>
      <div class="sub">v3.0 — Multi-Key Rotation · Auto Fallback · 15+ Models</div>
      <div style="margin-top:14px">
        <span class="badge free">🔄 Key Rotation ON</span>
        <span class="badge hot">🛡️ Auto Fallback ON</span>
        <span class="badge blue">📡 {len(MODELS)} Models</span>
        <span class="badge free">👁️ Multimodal</span>
      </div>
    </div>
    <div class="stats">
      <div class="stat"><div class="stat-num">{or_count}</div><div class="stat-label">OpenRouter Keys</div></div>
      <div class="stat"><div class="stat-num">{groq_count}</div><div class="stat-label">Groq Keys</div></div>
      <div class="stat"><div class="stat-num">{ggl_count}</div><div class="stat-label">Google Keys</div></div>
      <div class="stat"><div class="stat-num">{total_keys}</div><div class="stat-label">Total Keys Active</div></div>
      <div class="stat"><div class="stat-num">{len(MODELS)}</div><div class="stat-label">Models Available</div></div>
    </div>
    <div class="links">
      <a href="/docs" class="btn p">📚 API Docs</a>
      <a href="/models" class="btn s">🤖 Models</a>
      <a href="/health" class="btn s">💚 Health</a>
      <a href="/key-stats" class="btn s">🔑 Key Stats</a>
    </div>
    <div class="grid">
      <div class="card"><h3>🤖 POST /chat</h3><p>Chat with any model. model=auto for intelligent routing.</p></div>
      <div class="card"><h3>🏗️ POST /build-html</h3><p>Build production HTML projects (Lovable-style).</p></div>
      <div class="card"><h3>👁️ POST /analyze-image</h3><p>Analyze images with Gemini multimodal.</p></div>
      <div class="card"><h3>🔄 POST /v1/chat/completions</h3><p>OpenAI-compatible endpoint.</p></div>
      <div class="card"><h3>🔑 POST /generate-key</h3><p>Generate API keys for this master API.</p></div>
      <div class="card"><h3>📊 GET /key-stats</h3><p>See how many keys are loaded per provider.</p></div>
    </div>
    </body></html>"""
    return HTMLResponse(html)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "total_models": len(MODELS),
        "key_rotation": "enabled",
        "auto_fallback": "enabled",
        "providers_configured": {
            "openrouter": len(_OR_KEYS) > 0,
            "groq": len(_GROQ_KEYS) > 0,
            "google": len(_GGL_KEYS) > 0,
        }
    }

@app.get("/key-stats")
async def key_stats():
    """See how many keys are loaded per provider"""
    return {
        "openrouter_keys": len(_OR_KEYS),
        "groq_keys": len(_GROQ_KEYS),
        "google_keys": len(_GGL_KEYS),
        "total_keys": len(_OR_KEYS) + len(_GROQ_KEYS) + len(_GGL_KEYS),
        "rotation_strategy": "round-robin",
        "fallback_chain": FALLBACK_FREE,
        "tip": "Add more keys in env as comma-separated: OPENROUTER_API_KEY=key1,key2,key3"
    }

@app.get("/models")
async def list_models():
    return {
        "total": len(MODELS),
        "free": sum(1 for m in MODELS.values() if m["tier"] == "free"),
        "premium": sum(1 for m in MODELS.values() if m["tier"] == "premium"),
        "models": MODELS
    }

@app.post("/chat")
async def chat(req: ChatRequest, _k: str = Depends(get_api_key)):
    """Chat endpoint with key rotation + auto fallback"""
    if not req.messages:
        raise HTTPException(400, "messages required")

    # Resolve model
    alias = req.model
    routed = False
    if alias == "auto":
        last = req.messages[-1].content
        last_text = last if isinstance(last, str) else " ".join(
            p.get("text", "") for p in last if isinstance(p, dict) and p.get("type") == "text"
        )
        alias = intelligent_route(str(last_text), req.task_type or "auto", req.prefer_free)
        routed = True

    if alias not in MODELS:
        raise HTTPException(400, f"Unknown model: {alias}. See /models")

    # Build messages
    msgs = []
    if req.system:
        msgs.append({"role": "system", "content": req.system})
    for m in req.messages:
        content = m.content
        if m.role == "user" and (req.image_url or req.image_base64):
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            if req.image_url:
                content.append({"type": "image_url", "image_url": {"url": req.image_url}})
            elif req.image_base64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{req.image_base64}"}})
        msgs.append({"role": m.role, "content": content})

    # Streaming (no fallback for streams)
    if req.stream and MODELS[alias]["provider"] == "openrouter":
        key = next_or_key()
        return StreamingResponse(
            _stream_openrouter(MODELS[alias]["id"], key, msgs, req.max_tokens, req.temperature),
            media_type="text/event-stream"
        )

    # Call with fallback
    result, used_alias = await call_provider_with_fallback(alias, msgs, req.max_tokens, req.temperature)
    return {
        "success": True,
        "model_requested": req.model,
        "model_used": used_alias,
        "model_id": MODELS[used_alias]["id"],
        "provider": MODELS[used_alias]["provider"],
        "tier": MODELS[used_alias]["tier"],
        "routing": "intelligent" if routed else "manual",
        "fallback_used": used_alias != alias,
        "response": result["choices"][0]["message"]["content"],
        "usage": result.get("usage", {})
    }

@app.post("/v1/chat/completions")
async def openai_compat(body: dict, _k: str = Depends(get_api_key)):
    """OpenAI-compatible — works with any OpenAI SDK"""
    req = ChatRequest(
        model=body.get("model", "auto"),
        messages=[Message(**m) for m in body.get("messages", [])],
        max_tokens=body.get("max_tokens", 4096),
        temperature=body.get("temperature", 0.7),
        stream=body.get("stream", False)
    )
    r = await chat(req, _k)
    if req.stream:
        return r
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": r["model_id"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": r["response"]}, "finish_reason": "stop"}],
        "usage": r.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    }

@app.post("/build-html")
async def build_html(req: HTMLBuilderRequest, _k: str = Depends(get_api_key)):
    """Build production-ready HTML projects like Lovable"""
    styles = {
        "modern":        "Clean, CSS variables, subtle shadows, smooth animations",
        "glassmorphism": "Frosted glass, backdrop-filter, gradient backgrounds",
        "dark":          "Dark cyberpunk, neon accents (#00ff88, #ff006e), glow effects",
        "minimal":       "Ultra minimal, whitespace, perfect typography",
        "colorful":      "Vibrant gradients, playful, energetic",
        "corporate":     "Professional, navy/white, clean grid",
        "neon":          "Neon on dark, retro-futuristic, vivid colors"
    }
    system = (
        "You are an elite web developer. Return ONLY pure HTML — no markdown, no explanations, "
        "no code blocks. ALL CSS in <style>, ALL JS in <script>. "
        "No placeholders, every interaction must work, mobile-responsive, production-ready."
    )
    user = (
        f"Build a complete {req.project_type}: {req.prompt}\n"
        f"Style: {styles.get(req.style, styles['modern'])}\n"
        f"{'Rich animations, CSS keyframes, micro-interactions.' if req.include_animations else 'Minimal animations.'}\n"
        f"{'Fully responsive mobile-first.' if req.responsive else 'Desktop only.'}"
    )
    alias = req.preferred_model if req.preferred_model != "auto" and req.preferred_model in MODELS else "qwen-2.5-72b-free"
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    result, used = await call_provider_with_fallback(alias, msgs, 8192, 0.75)
    html = result["choices"][0]["message"]["content"].strip()
    if "```html" in html: html = html.split("```html")[1].split("```")[0].strip()
    elif "```" in html: html = html.split("```")[1].split("```")[0].strip()
    if not html.lower().startswith(("<!doctype", "<html")):
        html = "<!DOCTYPE html>\n" + html
    return {"success": True, "html": html, "model_used": used,
            "chars": len(html), "project_type": req.project_type, "style": req.style}

@app.post("/analyze-image")
async def analyze_image(
    prompt: str, image_url: Optional[str] = None,
    image_base64: Optional[str] = None, model: Optional[str] = "gemini-2.0-flash",
    _k: str = Depends(get_api_key)
):
    """Analyze images with multimodal models"""
    if not image_url and not image_base64:
        raise HTTPException(400, "Provide image_url or image_base64")
    content = [{"type": "text", "text": prompt}]
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    else:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
    alias = model if (model in MODELS and MODELS[model].get("multimodal")) else "gemini-2.0-flash"
    result, used = await call_provider_with_fallback(alias, [{"role": "user", "content": content}], 2048, 0.4)
    return {"analysis": result["choices"][0]["message"]["content"], "model_used": used}

@app.post("/route-test")
async def route_test(body: dict):
    """Test which model would be selected for a prompt"""
    prompt     = body.get("prompt", "")
    task_type  = body.get("task_type", "auto")
    prefer_free = bool(body.get("prefer_free", True))
    chosen = intelligent_route(prompt, task_type, prefer_free)
    return {"chosen": chosen, "profile": MODELS[chosen], "fallback_chain": FALLBACK_FREE}

@app.post("/generate-key")
async def generate_key(req: APIKeyRequest):
    """Generate API key for this master API"""
    k = "cx-" + secrets.token_urlsafe(32)
    conn = sqlite3.connect(DB_PATH)
    conn.cursor().execute(
        "INSERT INTO api_keys(key,name,email,created_at,tier) VALUES(?,?,?,?,?)",
        (k, req.name, req.email or "", datetime.now().isoformat(), req.tier)
    )
    conn.commit(); conn.close()
    return {"api_key": k, "name": req.name, "tier": req.tier,
            "header": "X-API-Key", "note": "Set REQUIRE_AUTH=true to enforce auth"}

@app.get("/key-info/{api_key}")
async def key_info(api_key: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name,email,created_at,requests_count,tier,is_active,last_used FROM api_keys WHERE key=?", (api_key,))
    row = c.fetchone(); conn.close()
    if not row: raise HTTPException(404, "Key not found")
    return {"name": row[0], "email": row[1], "created_at": row[2],
            "requests": row[3], "tier": row[4], "active": bool(row[5]), "last_used": row[6]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
