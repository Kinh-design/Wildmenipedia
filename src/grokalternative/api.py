from fastapi import FastAPI

from .agents.orchestrator import run_pipeline
from .ingest import ingest_from_dbpedia, ingest_from_wikidata
from .rag import graphrag_answer
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


@app.post("/ingest")
def ingest(q: str, sources: str = "wikidata,dbpedia") -> dict:
    count = 0
    if "wikidata" in sources:
        try:
            count += ingest_from_wikidata(q)
        except Exception:
            pass
    if "dbpedia" in sources:
        try:
            count += ingest_from_dbpedia(q)
        except Exception:
            pass
    return {"ok": True, "inserted": count}


@app.get("/graphrag")
def graphrag(q: str) -> dict:
    out = graphrag_answer(q)
    return {"ok": True, **out}
