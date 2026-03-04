# ⚡ CodeX Master AI API

> Universal AI API with 15+ models, intelligent routing, multimodal support, HTML builder, and API key management.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 🤖 **15+ AI Models** | Free + Premium (Llama, Gemini, Claude, GPT, Groq, DeepSeek...) |
| 🧠 **Intelligent Routing** | Auto-selects best model based on your prompt type |
| 👁️ **Multimodal** | Image/vision analysis via Gemini + GPT-4o |
| 🏗️ **HTML Builder** | Build production-ready HTML projects (like Lovable) |
| 🔑 **API Key Manager** | Generate & manage unlimited API keys |
| ⚡ **Streaming** | Real-time streaming responses |
| 🔄 **OpenAI Compatible** | Drop-in `/v1/chat/completions` endpoint |
| 🆓 **Free Tier** | Works without spending a rupee (free model keys) |

## 📋 Models Available

### Free Models (No cost)
| Model | Provider | Speed | Best For |
|-------|----------|-------|----------|
| `llama-3.3-70b` | OpenRouter | Fast | General, Chat |
| `gemini-2.0-flash-free` | OpenRouter | Very Fast | Multimodal, Long Context |
| `deepseek-r1-free` | OpenRouter | Medium | Math, Reasoning |
| `qwen-2.5-72b-free` | OpenRouter | Fast | Coding, Analysis |
| `mistral-7b-free` | OpenRouter | Fast | Quick Tasks |
| `phi-3-mini-free` | OpenRouter | Ultra Fast | Lightweight |
| `llama-groq-fast` | Groq LPU | ⚡ Ultra Fast | Real-time, Speed |
| `mixtral-groq-fast` | Groq LPU | ⚡ Ultra Fast | MoE Reasoning |
| `gemma2-groq` | Groq LPU | ⚡ Ultra Fast | Simple Tasks |
| `gemini-2.0-flash` | Google | Fast | Vision, 1M Context |
| `gemini-1.5-pro` | Google | Medium | 2M Context, Video |

### Premium Models (Require paid API keys)
| Model | Provider | Best For |
|-------|----------|----------|
| `claude-opus-4` | Anthropic | Complex Reasoning |
| `claude-sonnet-4` | Anthropic | Balanced Coding |
| `gpt-4o` | OpenAI | Vision, JSON |
| `o3-mini` | OpenAI | STEM, Math |
| `gemini-2.5-pro` | Google | 2M Context |
| `perplexity-sonar-pro` | Perplexity | Web Search |
| `codestral-2501` | Mistral | Code Completion |
| `deepseek-v3` | DeepSeek | Coding, Math |
| `nova-pro` | Amazon | Documents |

## 🔧 Setup & Deploy

### Step 1: Get Free API Keys

1. **OpenRouter** (Most important - free models): https://openrouter.ai  
2. **Groq** (Ultra-fast, free): https://console.groq.com  
3. **Google AI Studio** (Gemini, free): https://aistudio.google.com/app/apikey  

### Step 2: Deploy to Render

1. Fork this repo
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Add environment variables:
   - `OPENROUTER_API_KEY` = your key
   - `GROQ_API_KEY` = your key  
   - `GOOGLE_API_KEY` = your key
5. Deploy! ✅

### Step 3: Local Development

```bash
git clone https://github.com/Aman262626/codex-master-ai-api
cd codex-master-ai-api
pip install -r requirements.txt
cp .env.example .env  # Fill in your API keys
python main.py
```

Open http://localhost:8000

## 📡 API Usage

### Chat (Auto-routing)
```python
import requests

response = requests.post("https://your-app.onrender.com/chat", json={
    "model": "auto",           # Auto-picks best free model
    "prefer_free": True,       # Use free models only
    "task_type": "code",       # code|math|vision|reasoning|speed|auto
    "messages": [
        {"role": "user", "content": "Build a Python Flask API"}
    ]
})
print(response.json()["response"])
```

### HTML Builder
```python
response = requests.post("https://your-app.onrender.com/build-html", json={
    "prompt": "Portfolio website for a full-stack developer",
    "project_type": "portfolio",
    "style": "dark",
    "include_animations": True
})
html = response.json()["html"]
with open("project.html", "w") as f:
    f.write(html)
```

### Generate API Key
```python
response = requests.post("https://your-app.onrender.com/generate-key", json={
    "name": "My App Key",
    "tier": "free"
})
print(response.json()["api_key"])  # cx-xxxxxxxxxx
```

### Image Analysis
```python
response = requests.post("https://your-app.onrender.com/analyze-image", params={
    "prompt": "What is in this image? Describe in detail.",
    "image_url": "https://example.com/image.jpg"
})
print(response.json()["analysis"])
```

### OpenAI SDK Compatible
```python
from openai import OpenAI

client = OpenAI(
    api_key="cx-your-key",  # Your generated key
    base_url="https://your-app.onrender.com/v1"
)

response = client.chat.completions.create(
    model="llama-groq-fast",  # or "auto"
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## 🛠️ Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Recommended | Access to 100+ models including free ones |
| `GROQ_API_KEY` | Recommended | Ultra-fast LPU inference (free) |
| `GOOGLE_API_KEY` | Recommended | Gemini models with 1M-2M context (free) |
| `REQUIRE_AUTH` | Optional | Set `true` to require X-API-Key header |
| `SITE_URL` | Optional | Your deployed URL |

## 📚 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/models` | List all models |
| POST | `/chat` | Main chat endpoint |
| POST | `/v1/chat/completions` | OpenAI-compatible |
| POST | `/build-html` | HTML project builder |
| POST | `/analyze-image` | Image analysis |
| POST | `/route-test` | Test intelligent routing |
| POST | `/generate-key` | Create API key |
| GET | `/key-info/{key}` | Key usage stats |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

---
Built by **CodeX_Network** 🚀
