# ============================================================
# CodeX Master AI API - FastAPI
# Author: CodeX_Network (Aman Kumar)
# Version: 2.0.0
# Deploy: Render / Vercel
# ============================================================

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union, AsyncGenerator
import httpx
import asyncio
import json
import secrets
import hashlib
import time
import os
import base64
import uuid
import sqlite3
from datetime import datetime

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(
    title="CodeX Master AI API",
    description="""🚀 Universal Master AI API
    
    **Features:**
    - 15+ AI Models (Free + Premium)
    - Intelligent Auto-Routing (picks best model for task)
    - Multimodal Support (Text + Images)
    - HTML Project Builder (like Lovable)
    - API Key Generator & Manager
    - OpenAI-Compatible Endpoint
    - Streaming Support
    - Deploy on Render/Vercel
    
    **Free Models:** Llama 3.3, Gemini Flash, DeepSeek R1, Mistral, Qwen 2.5, Groq (ultra-fast)
    **Premium Models:** Claude Opus/Sonnet, GPT-4o, o3-mini, Gemini Pro, Perplexity Sonar, Codestral
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DATABASE SETUP - SQLite for API Key Management
# ============================================================
DB_PATH = "api_keys.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS api_keys (
            key TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            created_at TEXT,
            requests_count INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            tier TEXT DEFAULT 'free',
            last_used TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ============================================================
# MODELS REGISTRY - Complete Analysis of Each Model
# ============================================================
MODELS = {
    # ─────────────────────────────────────
    # FREE TIER - OpenRouter Free Models
    # ─────────────────────────────────────
    "llama-3.3-70b": {
        "id": "meta-llama/llama-3.3-70b-instruct:free",
        "provider": "openrouter",
        "tier": "free",
        "context_window": 131072,
        "multimodal": False,
        "supports_streaming": True,
        "supports_tools": True,
        "best_for": ["general", "coding", "analysis", "writing", "chat"],
        "speed": "fast",
        "language_support": "multilingual",
        "developer": "Meta",
        "description": "Meta Llama 3.3 70B - Best free general-purpose model. Excellent at instruction following, coding, analysis. 128K context.",
        "strengths": ["instruction-following", "multilingual", "coding", "reasoning"]
    },
    "gemini-2.0-flash-free": {
        "id": "google/gemini-2.0-flash-exp:free",
        "provider": "openrouter",
        "tier": "free",
        "context_window": 1048576,
        "multimodal": True,
        "supports_vision": True,
        "supports_streaming": True,
        "best_for": ["multimodal", "long-context", "general", "vision", "documents"],
        "speed": "very-fast",
        "language_support": "multilingual",
        "developer": "Google",
        "description": "Gemini 2.0 Flash - Google's fastest multimodal model. 1M token context, vision, audio. Best free multimodal.",
        "strengths": ["vision", "1M-context", "speed", "multimodal"]
    },
    "deepseek-r1-free": {
        "id": "deepseek/deepseek-r1:free",
        "provider": "openrouter",
        "tier": "free",
        "context_window": 163840,
        "multimodal": False,
        "supports_streaming": True,
        "supports_thinking": True,
        "best_for": ["reasoning", "math", "science", "complex-problems", "logic"],
        "speed": "medium",
        "language_support": "multilingual",
        "developer": "DeepSeek",
        "description": "DeepSeek R1 - Chain-of-thought reasoning model. Best free model for math/science/logic. Extended thinking mode.",
        "strengths": ["chain-of-thought", "math", "scientific-reasoning", "extended-thinking"]
    },
    "mistral-7b-free": {
        "id": "mistralai/mistral-7b-instruct:free",
        "provider": "openrouter",
        "tier": "free",
        "context_window": 32768,
        "multimodal": False,
        "supports_streaming": True,
        "best_for": ["quick-tasks", "coding", "writing", "lightweight"],
        "speed": "fast",
        "language_support": "multilingual",
        "developer": "Mistral AI",
        "description": "Mistral 7B - Lightweight, fast, efficient. Good for quick coding tasks and writing assistance.",
        "strengths": ["efficiency", "speed", "coding"]
    },
    "qwen-2.5-72b-free": {
        "id": "qwen/qwen-2.5-72b-instruct:free",
        "provider": "openrouter",
        "tier": "free",
        "context_window": 131072,
        "multimodal": False,
        "supports_streaming": True,
        "best_for": ["coding", "analysis", "math", "chinese", "multilingual"],
        "speed": "fast",
        "language_support": "multilingual (excellent Chinese/English)",
        "developer": "Alibaba",
        "description": "Qwen 2.5 72B - Alibaba's flagship. Outstanding at coding (HumanEval 88%), math, and multilingual tasks.",
        "strengths": ["coding", "math", "multilingual", "instruction-following"]
    },
    "phi-3-mini-free": {
        "id": "microsoft/phi-3-mini-128k-instruct:free",
        "provider": "openrouter",
        "tier": "free",
        "context_window": 128000,
        "multimodal": False,
        "supports_streaming": True,
        "best_for": ["lightweight", "edge", "simple-tasks"],
        "speed": "ultra-fast",
        "developer": "Microsoft",
        "description": "Microsoft Phi-3 Mini - Tiny but powerful. 128K context. Great for simple tasks with minimal compute.",
        "strengths": ["efficiency", "128K-context", "edge-deployment"]
    },

    # ─────────────────────────────────────
    # FREE TIER - Groq (Ultra-Fast LPU)
    # ─────────────────────────────────────
    "llama-groq-fast": {
        "id": "llama-3.3-70b-versatile",
        "provider": "groq",
        "tier": "free",
        "context_window": 128000,
        "multimodal": False,
        "supports_streaming": True,
        "best_for": ["speed", "real-time", "quick-tasks", "chat", "general"],
        "speed": "ultra-fast (LPU)",
        "tokens_per_second": "~1200 tok/s",
        "developer": "Meta + Groq",
        "description": "Llama 3.3 70B on Groq LPU - FASTEST available. ~1200 tokens/sec. Use when speed is critical. Real-time chat.",
        "strengths": ["speed", "low-latency", "real-time"]
    },
    "mixtral-groq-fast": {
        "id": "mixtral-8x7b-32768",
        "provider": "groq",
        "tier": "free",
        "context_window": 32768,
        "multimodal": False,
        "supports_streaming": True,
        "best_for": ["speed", "general", "moe-reasoning"],
        "speed": "ultra-fast (LPU)",
        "developer": "Mistral AI + Groq",
        "description": "Mixtral 8x7B MoE on Groq - Very fast mixture-of-experts model. Great balance of quality and speed.",
        "strengths": ["MoE-efficiency", "speed", "reasoning"]
    },
    "llama-guard-groq": {
        "id": "llama-guard-3-8b",
        "provider": "groq",
        "tier": "free",
        "context_window": 8192,
        "multimodal": False,
        "best_for": ["content-moderation", "safety-check"],
        "speed": "ultra-fast",
        "developer": "Meta + Groq",
        "description": "Llama Guard 3 - Specialized for content safety and moderation.",
        "strengths": ["safety", "moderation"]
    },
    "gemma2-groq": {
        "id": "gemma2-9b-it",
        "provider": "groq",
        "tier": "free",
        "context_window": 8192,
        "multimodal": False,
        "supports_streaming": True,
        "best_for": ["lightweight", "quick-chat", "summarization"],
        "speed": "ultra-fast",
        "developer": "Google + Groq",
        "description": "Gemma 2 9B on Groq - Google's lightweight model on fastest hardware. Ultra-fast for simple tasks.",
        "strengths": ["speed", "lightweight", "summarization"]
    },

    # ─────────────────────────────────────
    # FREE TIER - Google AI Studio
    # ─────────────────────────────────────
    "gemini-2.0-flash": {
        "id": "gemini-2.0-flash",
        "provider": "google",
        "tier": "free",
        "context_window": 1048576,
        "multimodal": True,
        "supports_vision": True,
        "supports_audio": True,
        "supports_video": True,
        "supports_streaming": True,
        "supports_tools": True,
        "best_for": ["multimodal", "vision", "documents", "long-context", "general"],
        "speed": "fast",
        "developer": "Google",
        "description": "Gemini 2.0 Flash via Google AI Studio - Free 1M context, image/audio/video analysis, tool calling. Google's best free model.",
        "strengths": ["1M-context", "vision", "audio", "video", "tool-use", "free"]
    },
    "gemini-1.5-pro": {
        "id": "gemini-1.5-pro",
        "provider": "google",
        "tier": "free",
        "context_window": 2097152,
        "multimodal": True,
        "supports_vision": True,
        "supports_audio": True,
        "supports_video": True,
        "best_for": ["2M-context", "large-documents", "video-analysis", "vision"],
        "speed": "medium",
        "developer": "Google",
        "description": "Gemini 1.5 Pro - 2M token context window. Analyze entire codebases, books, hour-long videos.",
        "strengths": ["2M-context", "long-document", "video", "largest-context"]
    },

    # ─────────────────────────────────────
    # PREMIUM MODELS - Require API Keys
    # ─────────────────────────────────────
    "claude-opus-4": {
        "id": "anthropic/claude-opus-4",
        "provider": "openrouter",
        "tier": "premium",
        "context_window": 200000,
        "multimodal": True,
        "supports_vision": True,
        "supports_streaming": True,
        "supports_tools": True,
        "supports_extended_thinking": True,
        "best_for": ["complex-reasoning", "research", "analysis", "advanced-coding", "creative-writing"],
        "speed": "slow",
        "developer": "Anthropic",
        "description": "Claude Opus 4 - Anthropic's most powerful model. Best at complex reasoning, nuanced analysis, and creative tasks. Extended thinking mode.",
        "strengths": ["deep-reasoning", "nuanced-writing", "safety", "extended-thinking", "200K-context"]
    },
    "claude-sonnet-4": {
        "id": "anthropic/claude-sonnet-4-5",
        "provider": "openrouter",
        "tier": "premium",
        "context_window": 200000,
        "multimodal": True,
        "supports_vision": True,
        "supports_streaming": True,
        "supports_tools": True,
        "supports_extended_thinking": True,
        "best_for": ["coding", "analysis", "balanced-tasks", "writing"],
        "speed": "medium",
        "developer": "Anthropic",
        "description": "Claude Sonnet 4.5 - Best balance of intelligence and speed. Excellent coder, writer, analyst.",
        "strengths": ["balance", "coding", "analysis", "speed-vs-quality"]
    },
    "gpt-4o": {
        "id": "openai/gpt-4o",
        "provider": "openrouter",
        "tier": "premium",
        "context_window": 128000,
        "multimodal": True,
        "supports_vision": True,
        "supports_streaming": True,
        "supports_tools": True,
        "supports_json_mode": True,
        "best_for": ["general", "vision", "coding", "json", "function-calling"],
        "speed": "fast",
        "developer": "OpenAI",
        "description": "GPT-4o - OpenAI flagship multimodal. Excellent at vision, JSON structured output, function calling, and general tasks.",
        "strengths": ["vision", "json-mode", "function-calling", "speed", "versatility"]
    },
    "gpt-4-turbo": {
        "id": "openai/gpt-4-turbo",
        "provider": "openrouter",
        "tier": "premium",
        "context_window": 128000,
        "multimodal": True,
        "supports_vision": True,
        "best_for": ["advanced-coding", "analysis", "complex-tasks"],
        "speed": "medium",
        "developer": "OpenAI",
        "description": "GPT-4 Turbo - Powerful with vision. Great for complex coding, deep analysis, and long context tasks.",
        "strengths": ["coding", "analysis", "vision", "128K-context"]
    },
    "o3-mini": {
        "id": "openai/o3-mini",
        "provider": "openrouter",
        "tier": "premium",
        "context_window": 200000,
        "multimodal": False,
        "supports_extended_thinking": True,
        "best_for": ["math", "science", "reasoning", "coding", "STEM"],
        "speed": "medium",
        "developer": "OpenAI",
        "description": "OpenAI o3-mini - Extended thinking reasoning model. Best for STEM, competitive math, scientific problems. Reasoning chains.",
        "strengths": ["extended-thinking", "math", "science", "reasoning-chains", "STEM"]
    },
    "gemini-2.5-pro": {
        "id": "google/gemini-2.5-pro-preview",
        "provider": "openrouter",
        "tier": "premium",
        "context_window": 2097152,
        "multimodal": True,
        "supports_vision": True,
        "supports_audio": True,
        "supports_video": True,
        "supports_extended_thinking": True,
        "best_for": ["2M-context", "multimodal", "analysis", "research", "video"],
        "speed": "medium",
        "developer": "Google",
        "description": "Gemini 2.5 Pro - Google's flagship with 2M context. Extended thinking, video analysis, multimodal.",
        "strengths": ["2M-context", "video-understanding", "extended-thinking", "multimodal"]
    },
    "perplexity-sonar-pro": {
        "id": "perplexity/sonar-pro",
        "provider": "openrouter",
        "tier": "premium",
        "context_window": 127072,
        "multimodal": False,
        "supports_web_search": True,
        "supports_citations": True,
        "best_for": ["web-search", "real-time-info", "current-events", "research", "news"],
        "speed": "fast",
        "developer": "Perplexity",
        "description": "Perplexity Sonar Pro - Real-time web search with citations. Best for current info, news, research with sources.",
        "strengths": ["web-search", "real-time", "citations", "current-events"]
    },
    "codestral-2501": {
        "id": "mistralai/codestral-2501",
        "provider": "openrouter",
        "tier": "premium",
        "context_window": 256000,
        "multimodal": False,
        "supports_fim": True,
        "supports_streaming": True,
        "best_for": ["code-generation", "code-completion", "fill-in-middle", "debugging", "refactoring"],
        "speed": "fast",
        "developer": "Mistral AI",
        "description": "Codestral 2501 - Dedicated code model. Fill-in-the-Middle (FIM), 256K context. Best for code completion, IDE integration.",
        "strengths": ["code-completion", "FIM", "256K-context", "all-programming-languages"]
    },
    "deepseek-v3": {
        "id": "deepseek/deepseek-chat-v3-0324",
        "provider": "openrouter",
        "tier": "premium",
        "context_window": 163840,
        "multimodal": False,
        "supports_streaming": True,
        "supports_tools": True,
        "best_for": ["coding", "math", "analysis", "chinese", "science"],
        "speed": "fast",
        "developer": "DeepSeek",
        "description": "DeepSeek V3 - Extremely capable coding + reasoning model. Rivals GPT-4 at fraction of cost.",
        "strengths": ["coding", "math", "analysis", "cost-effective"]
    },
    "nova-pro": {
        "id": "amazon/nova-pro-v1",
        "provider": "openrouter",
        "tier": "premium",
        "context_window": 300000,
        "multimodal": True,
        "supports_vision": True,
        "best_for": ["document-analysis", "vision", "enterprise", "long-context"],
        "speed": "fast",
        "developer": "Amazon",
        "description": "Amazon Nova Pro - AWS flagship model. 300K context, vision, enterprise-grade reliability.",
        "strengths": ["300K-context", "vision", "enterprise", "reliability"]
    },
}

# ============================================================
# INTELLIGENT ROUTER - Analyzes request and picks best model
# ============================================================
def intelligent_route(prompt: str, task_type: str = "auto", prefer_free: bool = True) -> str:
    """Smart routing: analyzes prompt content and picks optimal model"""
    prompt_lower = prompt.lower()
    
    # Task signal detection
    signals = {
        "code": ["code", "program", "function", "debug", "html", "css", "javascript", "python",
                 "api", "fix", "error", "build", "create app", "website", "react", "nodejs",
                 "database", "sql", "backend", "frontend", "deploy", "dockerfile"],
        "math": ["calculate", "solve", "equation", "integral", "derivative", "math",
                 "algebra", "geometry", "physics", "chemistry", "formula", "compute"],
        "vision": ["image", "picture", "photo", "vision", "see", "look at", "analyze this image",
                   "what is in", "screenshot", "diagram", "chart", "graph"],
        "search": ["latest", "recent", "today", "news", "current", "2025", "2026",
                   "now", "live", "real-time", "happening", "update"],
        "reasoning": ["analyze", "think", "reason", "explain why", "step by step",
                      "logical", "deduce", "infer", "complex", "deep dive"],
        "creative": ["write", "story", "poem", "create", "design", "creative",
                     "novel", "script", "blog", "article", "essay"],
        "speed": ["quickly", "fast", "real-time", "quick", "instant", "rapid"],
    }
    
    detected = {}
    for category, keywords in signals.items():
        score = sum(1 for kw in keywords if kw in prompt_lower)
        if score > 0:
            detected[category] = score
    
    # Override with explicit task_type
    if task_type != "auto":
        detected[task_type] = 10
    
    # Routing logic (free vs premium)
    if prefer_free:
        if "vision" in detected:
            return "gemini-2.0-flash"
        elif "math" in detected or "reasoning" in detected:
            return "deepseek-r1-free"
        elif "code" in detected:
            return "qwen-2.5-72b-free"
        elif "speed" in detected:
            return "llama-groq-fast"
        elif "search" in detected:
            return "gemini-2.0-flash"
        elif len(prompt) > 10000:
            return "gemini-1.5-pro"
        else:
            return "llama-3.3-70b"
    else:
        if "vision" in detected:
            return "gpt-4o"
        elif "math" in detected:
            return "o3-mini"
        elif "reasoning" in detected:
            return "claude-opus-4"
        elif "code" in detected:
            return "codestral-2501"
        elif "search" in detected:
            return "perplexity-sonar-pro"
        elif "creative" in detected:
            return "claude-opus-4"
        elif len(prompt) > 50000:
            return "gemini-2.5-pro"
        else:
            return "claude-sonnet-4"

# ============================================================
# PYDANTIC SCHEMAS
# ============================================================
class MessageContent(BaseModel):
    type: str  # "text" | "image_url"
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatRequest(BaseModel):
    model: str = Field(default="auto", description="Model name or 'auto' for intelligent routing")
    messages: List[Message]
    max_tokens: Optional[int] = Field(default=4096, ge=1, le=32000)
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    stream: Optional[bool] = False
    system: Optional[str] = None
    task_type: Optional[str] = Field(default="auto", description="code|math|vision|search|reasoning|creative|speed|auto")
    prefer_free: Optional[bool] = True
    image_url: Optional[str] = None
    image_base64: Optional[str] = None

class HTMLBuilderRequest(BaseModel):
    prompt: str = Field(..., description="Describe what you want to build")
    project_type: Optional[str] = Field(default="webapp", description="webapp|landing|dashboard|game|portfolio|ecommerce|blog")
    style: Optional[str] = Field(default="modern", description="modern|glassmorphism|dark|minimal|colorful|corporate|neon")
    include_animations: Optional[bool] = True
    responsive: Optional[bool] = True
    preferred_model: Optional[str] = "auto"

class APIKeyRequest(BaseModel):
    name: str = Field(..., description="Name/label for this API key")
    email: Optional[str] = None
    tier: Optional[str] = Field(default="free", description="free|premium")

class RouteTestRequest(BaseModel):
    prompt: str
    task_type: Optional[str] = "auto"
    prefer_free: Optional[bool] = True

# ============================================================
# PROVIDER CLIENTS
# ============================================================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

async def call_openrouter(
    model_id: str, messages: list, max_tokens: int,
    temperature: float, stream: bool = False
):
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY not configured in environment")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("SITE_URL", "https://codex-master-api.onrender.com"),
        "X-Title": "CodeX Master AI API"
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(OPENROUTER_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

async def stream_openrouter(
    model_id: str, messages: list, max_tokens: int, temperature: float
) -> AsyncGenerator[str, None]:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY not configured")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("SITE_URL", "https://codex-master-api.onrender.com"),
        "X-Title": "CodeX Master AI API"
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", OPENROUTER_URL, json=payload, headers=headers) as response:
            async for line in response.aiter_lines():
                if line:
                    yield line + "\n"

async def call_groq(model_id: str, messages: list, max_tokens: int, temperature: float):
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured in environment")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": min(max_tokens, 8000),
        "temperature": temperature
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(GROQ_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

async def call_google(model_id: str, messages: list, max_tokens: int, temperature: float):
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="GOOGLE_API_KEY not configured in environment")
    
    # Convert to Google format
    contents = []
    system_instruction = None
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = {"parts": [{"text": msg["content"]}]}
            continue
        role = "user" if msg["role"] == "user" else "model"
        parts = []
        if isinstance(msg["content"], str):
            parts.append({"text": msg["content"]})
        elif isinstance(msg["content"], list):
            for part in msg["content"]:
                if part.get("type") == "text":
                    parts.append({"text": part["text"]})
                elif part.get("type") == "image_url":
                    img_url = part["image_url"]["url"]
                    if img_url.startswith("data:"):
                        media_type = img_url.split(";")[0].split(":")[1]
                        b64 = img_url.split(",")[1]
                        parts.append({"inline_data": {"mime_type": media_type, "data": b64}})
                    else:
                        parts.append({"text": f"[Image: {img_url}]"})
        contents.append({"role": role, "parts": parts})
    
    url = f"{GOOGLE_URL}/{model_id}:generateContent?key={api_key}"
    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature
        }
    }
    if system_instruction:
        payload["systemInstruction"] = system_instruction
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            text = "No response generated"
        return {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "model": model_id,
            "usage": {}
        }

async def call_provider(model_name: str, messages: list, max_tokens: int, temperature: float):
    model = MODELS[model_name]
    if model["provider"] == "groq":
        return await call_groq(model["id"], messages, max_tokens, temperature)
    elif model["provider"] == "google":
        return await call_google(model["id"], messages, max_tokens, temperature)
    else:  # openrouter
        return await call_openrouter(model["id"], messages, max_tokens, temperature)

# ============================================================
# API KEY AUTH
# ============================================================
def validate_api_key(key: str) -> bool:
    if not key:
        return False
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT is_active FROM api_keys WHERE key = ?", (key,))
    result = c.fetchone()
    if result and result[0] == 1:
        c.execute(
            "UPDATE api_keys SET requests_count = requests_count + 1, last_used = ? WHERE key = ?",
            (datetime.now().isoformat(), key)
        )
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

async def get_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    require_auth = os.getenv("REQUIRE_AUTH", "false").lower() == "true"
    if not require_auth:
        return "dev-mode-no-auth"
    if not x_api_key or not validate_api_key(x_api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Generate one at POST /generate-key"
        )
    return x_api_key

# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    html = """
    <!DOCTYPE html><html><head>
    <title>CodeX Master AI API</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #0a0a0f; color: #e0e0ff; min-height: 100vh; }
        .hero { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                padding: 60px 40px; text-align: center; border-bottom: 1px solid #333; }
        h1 { font-size: 3em; background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6b6b);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; }
        .subtitle { color: #aaa; font-size: 1.1em; margin-top: 10px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px; padding: 40px; max-width: 1200px; margin: 0 auto; }
        .card { background: #111122; border: 1px solid #2a2a4a; border-radius: 12px;
                padding: 24px; transition: transform 0.2s; }
        .card:hover { transform: translateY(-4px); border-color: #7b2ff7; }
        .card h3 { color: #00d2ff; margin-bottom: 10px; }
        .card p { color: #aaa; font-size: 0.9em; line-height: 1.6; }
        .badge { display: inline-block; padding: 3px 10px; border-radius: 20px;
                 font-size: 0.75em; margin: 3px; }
        .free { background: #0d3b1c; color: #4ade80; border: 1px solid #166534; }
        .premium { background: #3b1a0d; color: #fb923c; border: 1px solid #9a3412; }
        .links { text-align: center; padding: 20px; }
        .btn { display: inline-block; padding: 12px 24px; margin: 8px; border-radius: 8px;
               text-decoration: none; font-weight: bold; transition: all 0.2s; }
        .btn-primary { background: #7b2ff7; color: white; }
        .btn-secondary { background: #111; border: 1px solid #444; color: #aaa; }
        .btn:hover { opacity: 0.8; transform: translateY(-2px); }
    </style>
    </head><body>
    <div class="hero">
        <h1>⚡ CodeX Master AI API</h1>
        <div class="subtitle">15+ Models · Multimodal · HTML Builder · Auto-Routing · API Keys</div>
        <div style="margin-top:20px">
            <span class="badge free">Free Models</span>
            <span class="badge premium">Premium Models</span>
            <span class="badge" style="background:#1a2a3b;color:#60a5fa;border:1px solid #1e40af">Streaming</span>
            <span class="badge" style="background:#2a1a3b;color:#c084fc;border:1px solid #6b21a8">Vision</span>
        </div>
    </div>
    <div class="links">
        <a href="/docs" class="btn btn-primary">📚 API Docs</a>
        <a href="/models" class="btn btn-secondary">🤖 All Models</a>
        <a href="/health" class="btn btn-secondary">💚 Health</a>
    </div>
    <div class="grid">
        <div class="card"><h3>🤖 Chat API</h3><p>POST /chat — Send messages to any model. Set model=auto for intelligent routing.</p></div>
        <div class="card"><h3>🏗️ HTML Builder</h3><p>POST /build-html — Build complete production-ready HTML projects like Lovable.</p></div>
        <div class="card"><h3>🔑 API Keys</h3><p>POST /generate-key — Generate your own API keys to use with this master API.</p></div>
        <div class="card"><h3>👁️ Vision/Image</h3><p>POST /analyze-image — Analyze images using Gemini multimodal.</p></div>
        <div class="card"><h3>⚡ Streaming</h3><p>POST /chat with stream=true — Real-time streaming responses.</p></div>
        <div class="card"><h3>🔄 OpenAI Compatible</h3><p>POST /v1/chat/completions — Drop-in replacement for OpenAI SDK.</p></div>
    </div>
    </body></html>
    """
    return HTMLResponse(content=html)

@app.get("/models")
async def list_models():
    """Get all available models with their capabilities"""
    return {
        "total": len(MODELS),
        "free_count": sum(1 for m in MODELS.values() if m["tier"] == "free"),
        "premium_count": sum(1 for m in MODELS.values() if m["tier"] == "premium"),
        "models": MODELS
    }

@app.post("/chat")
async def chat(request: ChatRequest, api_key: str = Depends(get_api_key)):
    """Main chat endpoint with intelligent routing"""
    
    # Determine model
    model_name = request.model
    routed = False
    if model_name == "auto":
        last_content = request.messages[-1].content
        if isinstance(last_content, list):
            last_content = " ".join(p.get("text", "") for p in last_content if isinstance(p, dict))
        model_name = intelligent_route(str(last_content), request.task_type or "auto", request.prefer_free)
        routed = True
    
    if model_name not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}"
        )
    
    model = MODELS[model_name]
    
    # Build message list
    msgs = []
    if request.system:
        msgs.append({"role": "system", "content": request.system})
    
    for msg in request.messages:
        content = msg.content
        # Inject image if provided on last user message
        if msg.role == "user" and (request.image_url or request.image_base64):
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            if request.image_url:
                content.append({"type": "image_url", "image_url": {"url": request.image_url}})
            elif request.image_base64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{request.image_base64}"}})
        msgs.append({"role": msg.role, "content": content})
    
    # Streaming
    if request.stream and model["provider"] == "openrouter":
        return StreamingResponse(
            stream_openrouter(model["id"], msgs, request.max_tokens, request.temperature),
            media_type="text/event-stream"
        )
    
    try:
        result = await call_provider(model_name, msgs, request.max_tokens, request.temperature)
        return {
            "success": True,
            "model_used": model_name,
            "model_id": model["id"],
            "provider": model["provider"],
            "tier": model["tier"],
            "routing": "intelligent" if routed else "manual",
            "response": result["choices"][0]["message"]["content"],
            "usage": result.get("usage", {})
        }
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Provider error: {e.response.text[:500]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def openai_compatible(body: dict, api_key: str = Depends(get_api_key)):
    """OpenAI-compatible endpoint — works with any OpenAI SDK"""
    model = body.get("model", "auto")
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 4096)
    temperature = body.get("temperature", 0.7)
    stream = body.get("stream", False)
    
    chat_req = ChatRequest(
        model=model,
        messages=[Message(**m) for m in messages],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream
    )
    result = await chat(chat_req, api_key)
    
    if stream:
        return result  # Already StreamingResponse
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result["model_id"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["response"]},
            "finish_reason": "stop"
        }],
        "usage": result.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    }

@app.post("/build-html")
async def build_html_project(request: HTMLBuilderRequest, api_key: str = Depends(get_api_key)):
    """Build production-ready HTML projects (like Lovable) with live-preview support"""
    
    style_guides = {
        "modern": "Clean professional design with CSS variables, subtle glassmorphism, smooth animations",
        "glassmorphism": "Frosted glass effect, backdrop-filter blur, gradient backgrounds, translucent cards",
        "dark": "Dark cyberpunk theme, neon accent colors (#00ff88, #ff006e), gradient text, glowing effects",
        "minimal": "Ultra minimal, generous whitespace, perfect typography, no decoration",
        "colorful": "Vibrant gradient backgrounds, playful animations, bright colors, energetic feel",
        "corporate": "Professional business style, navy/white palette, clean grid layout",
        "neon": "Neon on dark background, glow effects, retro-futuristic, bright vivid colors"
    }
    
    system_prompt = """You are an elite web developer. Build COMPLETE, PRODUCTION-READY single-file HTML applications.

STRICT RULES:
1. Return PURE HTML only — no markdown, no explanations, no ```html``` blocks
2. ALL CSS inside <style> tags — no external CDN dependencies (embed fonts via @import if needed)
3. ALL JavaScript inside <script> tags — fully functional, no placeholder logic
4. ZERO placeholder content — no 'Lorem ipsum', no 'TODO', no dummy data
5. Every button, link, form, modal MUST work with real JavaScript logic
6. Include meta viewport, SEO meta tags, favicon (emoji in <link>)
7. Use CSS custom properties (variables), flexbox/grid, smooth transitions
8. Add meaningful micro-interactions and hover effects
9. Mobile-first responsive design
10. The output must be deployable as-is to production
11. Include realistic sample data/content relevant to the project
12. Beautiful typography with proper hierarchy"""
    
    user_prompt = f"""Build a complete, production-ready {request.project_type} for: {request.prompt}

Style System: {style_guides.get(request.style, style_guides['modern'])}
Animations: {'Rich animations — CSS keyframes, scroll-triggered effects, loading states, transitions' if request.include_animations else 'Minimal, only essential transitions'}
Responsive: {'Fully responsive mobile-first, breakpoints for mobile/tablet/desktop' if request.responsive else 'Desktop optimized'}

Requirements:
- 100% functional with working JavaScript interactions
- Professional, polished UI that could be sold commercially
- Relevant, realistic content (not placeholder text)
- Smooth user experience throughout
- All links, buttons, and forms must work"""
    
    # Choose best coding model
    if request.preferred_model == "auto":
        model_name = "qwen-2.5-72b-free" if os.getenv("OPENROUTER_API_KEY") else "gemini-2.0-flash"
    else:
        model_name = request.preferred_model if request.preferred_model in MODELS else "qwen-2.5-72b-free"
    
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        result = await call_provider(model_name, msgs, 8192, 0.75)
        html_content = result["choices"][0]["message"]["content"]
        
        # Clean up any markdown wrapping
        if "```html" in html_content:
            html_content = html_content.split("```html")[1].split("```")[0].strip()
        elif "```" in html_content:
            parts = html_content.split("```")
            if len(parts) >= 2:
                html_content = parts[1].strip()
        
        # Ensure valid HTML
        if not html_content.strip().startswith(("<!DOCTYPE", "<html", "<!")):
            html_content = "<!DOCTYPE html>\n" + html_content
        
        return {
            "success": True,
            "html": html_content,
            "model_used": model_name,
            "project_type": request.project_type,
            "style": request.style,
            "char_count": len(html_content),
            "preview_tip": "Save as .html and open in browser for live preview"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-image")
async def analyze_image(
    prompt: str,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    model: Optional[str] = "gemini-2.0-flash",
    api_key: str = Depends(get_api_key)
):
    """Analyze images using multimodal models"""
    if not image_url and not image_base64:
        raise HTTPException(status_code=400, detail="Provide either image_url or image_base64")
    
    content = [{"type": "text", "text": prompt}]
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    elif image_base64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
    
    msgs = [{"role": "user", "content": content}]
    model_name = model if model in MODELS and MODELS[model].get("multimodal") else "gemini-2.0-flash"
    
    result = await call_provider(model_name, msgs, 2048, 0.5)
    return {
        "analysis": result["choices"][0]["message"]["content"],
        "model_used": model_name
    }

@app.post("/route-test")
async def test_routing(request: RouteTestRequest):
    """Test intelligent routing — see which model would be selected"""
    selected = intelligent_route(request.prompt, request.task_type, request.prefer_free)
    model_info = MODELS[selected]
    return {
        "prompt_preview": request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt,
        "selected_model": selected,
        "model_info": model_info,
        "routing_mode": "free_preference" if request.prefer_free else "premium_preference",
        "task_type": request.task_type
    }

@app.post("/generate-key")
async def generate_key(request: APIKeyRequest):
    """Generate a new API key for this master API"""
    key = "cx-" + secrets.token_urlsafe(32)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO api_keys (key, name, email, created_at, tier) VALUES (?, ?, ?, ?, ?)",
        (key, request.name, request.email or "", datetime.now().isoformat(), request.tier)
    )
    conn.commit()
    conn.close()
    return {
        "success": True,
        "api_key": key,
        "name": request.name,
        "tier": request.tier,
        "created_at": datetime.now().isoformat(),
        "usage": "Add header: X-API-Key: <your-key>",
        "note": "Set REQUIRE_AUTH=true in environment to enforce key validation"
    }

@app.get("/key-info/{api_key}")
async def key_info(api_key: str):
    """Get API key usage statistics"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT name, email, created_at, requests_count, tier, is_active, last_used FROM api_keys WHERE key = ?",
        (api_key,)
    )
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="API key not found")
    return {
        "name": row[0], "email": row[1], "created_at": row[2],
        "requests_count": row[3], "tier": row[4],
        "is_active": bool(row[5]), "last_used": row[6]
    }

@app.get("/health")
async def health():
    """Health check"""
    configured_providers = []
    if os.getenv("OPENROUTER_API_KEY"): configured_providers.append("openrouter")
    if os.getenv("GROQ_API_KEY"): configured_providers.append("groq")
    if os.getenv("GOOGLE_API_KEY"): configured_providers.append("google")
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "total_models": len(MODELS),
        "configured_providers": configured_providers,
        "require_auth": os.getenv("REQUIRE_AUTH", "false")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
