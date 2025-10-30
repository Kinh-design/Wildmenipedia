from fastapi import FastAPI

from .agents.orchestrator import run_pipeline
from .settings import get_settings

app = FastAPI(title="Wildmenipedia API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "ok": True,
        "neo4j": bool(settings.NEO4J_URL),
        "qdrant": bool(settings.QDRANT_HOST),
        "env": settings.ENV,
    }


@app.get("/ask")
def ask(q: str) -> dict:
    out = run_pipeline(q)
    return {"ok": True, **out}
