import hashlib

from fastapi import FastAPI

from .agents.orchestrator import run_pipeline
from .embeddings import Embedder
from .ingest import ingest_from_dbpedia, ingest_from_wikidata
from .rag import graphrag_answer, hybrid_answer
from .settings import get_settings
from .stores import KG, VS

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
    hybrid = hybrid_answer(q)
    return {"ok": True, **out, "retrieval": hybrid}


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


@app.post("/index")
def index_entity(id: str, include_aliases: bool = True) -> dict:
    """Backfill embeddings in Qdrant for a given entity id based on its label and aliases.

    Best-effort: if vector store is unreachable or the entity is missing, returns indexed=0.
    """
    kg = KG.from_env()
    props = {}
    try:
        props = kg.get_entity_props(id)
    except Exception:
        props = {}
    if not props:
        return {"ok": True, "indexed": 0, "id": id, "message": "entity not found"}

    # Init vector backend
    embedder = None
    vs = None
    try:
        embedder = Embedder(dim=256)
        vs = VS.from_env()
        vs.ensure_collection(name="entities", dim=embedder.dim)
    except Exception:
        embedder = None
        vs = None

    if not (embedder and vs):
        return {"ok": True, "indexed": 0, "id": id, "message": "vector backend unavailable"}

    indexed = 0
    label = props.get("name")
    if isinstance(label, str) and label.strip():
        try:
            vec = embedder.embed(label)
            src = "wikidata" if "wikidata.org" in id else ("dbpedia" if "dbpedia.org" in id else "unknown")
            vs.upsert_point(
                collection="entities",
                point_id=id,
                vector=vec,
                payload={"id": id, "label": label, "source": src},
            )
            indexed += 1
        except Exception:
            pass

    if include_aliases:
        aliases = props.get("aliases") or []
        if isinstance(aliases, list):
            for a in aliases:
                if not isinstance(a, str) or not a.strip():
                    continue
                try:
                    vec = embedder.embed(a)
                    ah = hashlib.sha1(a.encode("utf-8")).hexdigest()[:12]
                    point_id = f"{id}#alias:{ah}"
                    vs.upsert_point(
                        collection="entities",
                        point_id=point_id,
                        vector=vec,
                        payload={"id": id, "alias": a, "is_alias": True},
                    )
                    indexed += 1
                except Exception:
                    continue

    return {"ok": True, "indexed": indexed, "id": id}
