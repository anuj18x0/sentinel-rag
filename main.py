import time
import hashlib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from vector_store import vector_store
from llm_client import llm_client

app = FastAPI(title="SentinelAI RAG Service")


class QueryRequest(BaseModel):
    session_id: str
    question: str
    context: Optional[str] = ""

class QueryResponse(BaseModel):
    answer: str
    cached: bool = False

# ── In-memory caches ─────────────────────────────────────────────
sessions: dict = {}                     # session_id → list of messages
response_cache: dict = {}               # hash → (answer, timestamp)
RESPONSE_CACHE_TTL = 60                 # seconds

def _cache_key(question: str, context: str) -> str:
    """Generate a cache key from question + context hash."""
    combined = question.strip().lower() + hashlib.md5(context.encode()).hexdigest()
    return hashlib.sha256(combined.encode()).hexdigest()

def _evict_stale_cache():
    """Remove expired entries from response cache."""
    now = time.time()
    stale_keys = [k for k, (_, ts) in response_cache.items() if now - ts > RESPONSE_CACHE_TTL]
    for k in stale_keys:
        del response_cache[k]

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        # ── 1. Check response cache ────────────────────────────────
        cache_key = _cache_key(request.question, request.context or "")
        if cache_key in response_cache:
            answer, ts = response_cache[cache_key]
            if time.time() - ts < RESPONSE_CACHE_TTL:
                return QueryResponse(answer=answer, cached=True)

        # ── 2. Retrieve conversation history ───────────────────────
        history = sessions.get(request.session_id, [])

        # ── 3. Semantic search in vector store ─────────────────────
        try:
            docs = vector_store.search(request.question, limit=3)
            vector_context = "\n".join([doc.get("text", "") for doc in docs]) if docs else ""
        except Exception as e:
            print(f"Vector search skipped: {e}")
            vector_context = ""

        # ── 4. Combine contexts ────────────────────────────────────
        full_context = f"{request.context or ''}\n\n{vector_context}".strip()

        # ── 5. Generate response ───────────────────────────────────
        answer = await llm_client.generate_response(request.question, full_context, history)

        # ── 6. Update session history (keep last 10 messages) ──────
        history.append({"role": "user", "content": request.question})
        history.append({"role": "assistant", "content": answer})
        sessions[request.session_id] = history[-10:]

        # ── 7. Cache the response ──────────────────────────────────
        response_cache[cache_key] = (answer, time.time())
        _evict_stale_cache()

        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_insight(insight_id: str, text: str, metadata: dict = {}):
    try:
        vector_store.upsert_insight(insight_id, text, metadata)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
