# backend/main.py
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ⬇️ Tu agente ya preparado (usa LangChain, Neo4jVector, Tavily, etc.)
# Debe exponer una variable `agent` (AgentExecutor o Runnable) con .astream_events
from backend.src.agent import AGENT

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Chatbot API (FastAPI + LangChain)", version="1.0.0")

# CORS
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8501")
origins = [FRONTEND_URL]
# Para desarrollo puedes abrirlo más:
# origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Modelos (solo si NO tienes ya backend/pydantic_models.py)
# Si ya lo tienes, elimina esta sección y usa tu import.
# ──────────────────────────────────────────────────────────────────────────────
try:
    from backend.pydantic_models import ChatRequest  # type: ignore
except Exception:
    from pydantic import BaseModel

    class Message(BaseModel):
        role: str  # "user" | "assistant" | "system"
        content: str

    class ChatRequest(BaseModel):
        messages: List[Message]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: extracción de fuentes
# ──────────────────────────────────────────────────────────────────────────────
def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc or url
    except Exception:
        return url

def _find_urls(text: str) -> List[str]:
    return re.findall(r"https?://[^\s\)\]\}\>]+", text or "")

def _dedup_sources(sources: List[Dict[str, str]]) -> List[Dict[str, str]]:
    dedup = {}
    for s in sources:
        url = s.get("url")
        if url:
            dedup[url] = s
    return list(dedup.values())

def _extract_tavily_sources(tool_output) -> List[Dict[str, str]]:
    """
    Devuelve [{title, url}] a partir de la salida de Tavily.
    Soporta dict, lista, JSON string o texto con URLs.
    """
    sources: List[Dict[str, str]] = []
    try:
        data = tool_output
        if isinstance(tool_output, str):
            try:
                data = json.loads(tool_output)
            except Exception:
                # Texto plano: rascar URLs
                for u in _find_urls(tool_output):
                    sources.append({"title": _domain(u), "url": u})
                return _dedup_sources(sources)

        if isinstance(data, dict):
            if isinstance(data.get("results"), list):
                for r in data["results"]:
                    if not isinstance(r, dict):
                        continue
                    url = r.get("url") or r.get("source") or ""
                    title = r.get("title") or _domain(url)
                    if url:
                        sources.append({"title": title, "url": url})
            if isinstance(data.get("sources"), list):
                for s in data["sources"]:
                    if isinstance(s, dict):
                        url = s.get("url") or s.get("source") or ""
                        title = s.get("title") or _domain(url)
                        if url:
                            sources.append({"title": title, "url": url})
                    elif isinstance(s, str):
                        sources.append({"title": _domain(s), "url": s})

        elif isinstance(data, list):
            for r in data:
                if isinstance(r, dict):
                    url = r.get("url") or r.get("source") or ""
                    title = r.get("title") or _domain(url)
                    if url:
                        sources.append({"title": title, "url": url})
    except Exception:
        pass
    return _dedup_sources(sources)

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "The chat app is running"}

@app.get("/health")
def health_check():
    logger.info("Health check endpoint called")
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Streaming JSONL (una línea JSON por chunk):
      - {"type":"delta","text":"..."}                 ← tokens del modelo
      - {"type":"tool_start","name":"..."}            ← inicio de tool
      - {"type":"sources", ...}                       ← resumen final con fuentes
      - {"type":"error","message":"..."}              ← en caso de fallo
    """
    try:
        # 1) Obtener último mensaje de usuario
        last_user_text: Optional[str] = None
        for m in reversed(req.messages):
            if m.role.lower() == "user":
                last_user_text = m.content
                break
        if not last_user_text:
            raise HTTPException(status_code=400, detail="Falta mensaje de usuario.")

        # 2) Acumuladores
        tools_used: Dict[str, int] = {}
        sources: List[Dict[str, str]] = []
        fragments: List[Dict[str, str]] = []

        async def gen():
            # Headers anti-buffering (algunos proxies ignoran, pero ayuda)
            yield ""  # kick-off

            try:
                # 3) Recorrer eventos del agente (v1 = eventos detallados)
                async for ev in AGENT.astream_events({"input": last_user_text}, version="v1"):
                    et = ev.get("event")
                    data = ev.get("data", {}) or {}

                    # ---- Texto del modelo (token a token) ----
                    if et == "on_chat_model_stream":
                        chunk = data.get("chunk")
                        delta = getattr(chunk, "content", None) if chunk is not None else None
                        if delta:
                            yield json.dumps({"type": "delta", "text": delta}) + "\n"

                    # ---- Tool start ----
                    elif et == "on_tool_start":
                        # name puede venir en ev["name"] o en data["name"]/serialized
                        name = ev.get("name") or data.get("name") or "tool"
                        tools_used[name] = tools_used.get(name, 0) + 1
                        yield json.dumps({"type": "tool_start", "name": name}) + "\n"

                    # ---- Tool end -> intentar recolectar fuentes ----
                    elif et == "on_tool_end":
                        name = ev.get("name") or data.get("name") or "tool"
                        out = data.get("output")
                        # Tavily suele llamarse "tavily_search" o similar
                        if str(name).lower().startswith("tavily"):
                            sources.extend(_extract_tavily_sources(out))
                        else:
                            if isinstance(out, str):
                                for u in _find_urls(out):
                                    sources.append({"title": _domain(u), "url": u})

                    # ---- Retriever (RAG) ----
                    elif et == "on_retriever_end":
                        docs = data.get("documents") or []
                        for d in docs:
                            meta = getattr(d, "metadata", {}) or {}
                            link = meta.get("source") or meta.get("url") or meta.get("link")
                            title = (
                                meta.get("title")
                                or meta.get("file_name")
                                or (link and _domain(link))
                                or "Documento"
                            )
                            snippet = (getattr(d, "page_content", "") or "")[:280]
                            if link:
                                sources.append({"title": title, "url": link})
                            fragments.append({"title": title, "snippet": snippet})

            except Exception as e:
                # Enviar un error como último frame y terminar
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
                return

            # 4) Al terminar, resumen de herramientas y fuentes
            summary = {
                "type": "sources",
                "question": last_user_text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tools_used": tools_used,                    # dict {tool: count}
                "sources": _dedup_sources(sources),          # [{title,url}]
                "fragments": fragments[:5],                  # opcional: snippets
            }
            yield json.dumps(summary) + "\n"

        # JSONL por compatibilidad con la mayoría de front-ends (Streamlit/httpx)
        return StreamingResponse(
            gen(),
            media_type="text/plain; charset=utf-8",
            headers={
                # Sugerencias para evitar buffering en proxies/CDN
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Fallo inesperado en /chat")
        raise HTTPException(status_code=500, detail=f"Error inesperado: {e}") from e

# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    BACKEND_PORT = int(os.getenv("BACKEND_PORT", 8000))
    uvicorn.run(
        "backend.src.main:app",
        host="0.0.0.0",
        port=BACKEND_PORT,
        reload=bool(int(os.getenv("UVICORN_RELOAD", "1"))),
        # http="h11",  # En algunos entornos h11 va más fino que httptools
    )
