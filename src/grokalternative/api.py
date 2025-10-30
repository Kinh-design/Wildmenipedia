import hashlib
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlencode

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .agents.orchestrator import run_pipeline
from .embeddings import Embedder
from .ingest import ingest_from_dbpedia, ingest_from_wikidata
from .rag import graphrag_answer, hybrid_answer
from .settings import get_settings
from .stores import KG, VS

app = FastAPI(title="Wildmenipedia API", version="0.1.0")

templates = Jinja2Templates(directory=str((Path(__file__).parent / "templates").resolve()))


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


# ---------- UI (Grokipedia-like) ----------
@app.get("/", response_class=HTMLResponse)
def ui_home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "q": "",
            "summary": None,
            "facts": [],
            "vector_hits": [],
            "entities": [],
        },
    )


@app.get("/search", response_class=HTMLResponse)
def ui_search(request: Request, q: str = "", page: int = 1, page_size: int = 15) -> HTMLResponse:
    q = (q or "").strip()
    if not q:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "q": "",
                "summary": None,
                "facts": [],
                "vector_hits": [],
                "entities": [],
                "facts_total": 0,
                "facts_page": 1,
                "facts_pages": 1,
                "facts_page_size": page_size,
                "facts_prev": None,
                "facts_next": None,
            },
        )
    pipe = run_pipeline(q)
    hybrid = hybrid_answer(q)
    facts = (hybrid or {}).get("facts", [])
    total = len(facts)
    p = max(1, int(page or 1))
    ps = max(1, min(50, int(page_size or 15)))
    start = (p - 1) * ps
    end = start + ps
    page_count = max(1, (total + ps - 1) // ps)
    facts_slice = facts[start:end]

    def qp(new_page: int) -> str:
        return f"/search?{urlencode({'q': q, 'page': new_page, 'page_size': ps})}"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "q": q,
            "summary": (hybrid or {}).get("answer"),
            "facts": facts_slice,
            "vector_hits": (hybrid or {}).get("vector_hits", [])[:10],
            "entities": (hybrid or {}).get("selected_entities", []),
            "pipeline": pipe,
            "facts_total": total,
            "facts_page": p,
            "facts_pages": page_count,
            "facts_page_size": ps,
            "facts_prev": qp(p-1) if p > 1 else None,
            "facts_next": qp(p+1) if p < page_count else None,
        },
    )


@app.get("/entity", response_class=HTMLResponse)
def ui_entity(request: Request, id: str, page: int = 1, page_size: int = 15) -> HTMLResponse:
    kg = KG.from_env()
    props: Dict[str, Any] = {}
    neighbors: list[dict[str, Any]] = []
    total = 0
    try:
        props = kg.get_entity_props(id)
    except Exception:
        props = {}
    try:
        total = kg.count_neighbors(id)
        p = max(1, int(page or 1))
        ps = max(1, min(50, int(page_size or 15)))
        offset = (p - 1) * ps
        neighbors = kg.neighbors(id, limit=ps, offset=offset)
    except Exception:
        neighbors = []
        total = 0

    # Related entities (top unique objects/subjects from neighbors)
    related: list[dict[str, Any]] = []
    seen: set[str] = set()
    for n in neighbors:
        for eid in [n.get("o"), n.get("s")]:
            if not isinstance(eid, str) or eid == id:
                continue
            if eid in seen:
                continue
            seen.add(eid)
            # best-effort name lookup
            try:
                eprops = kg.get_entity_props(eid)
            except Exception:
                eprops = {}
            related.append({"id": eid, "name": eprops.get("name") or eid})
            if len(related) >= 10:
                break
        if len(related) >= 10:
            break

    # Recommended queries
    name = props.get("name") or id
    rec_q: list[str] = [
        f"Who is {name}?",
        f"What is {name}?",
        f"What type is {name}?",
    ]
    for n in neighbors[:3]:
        m = (n.get("meta") or {})
        tgt_label = m.get("object_label") or n.get("o")
        try:
            if isinstance(tgt_label, str):
                rec_q.append(f"How is {name} related to {tgt_label}?")
        except Exception:
            pass

    # Pagination helpers
    p = max(1, int(page or 1))
    ps = max(1, min(50, int(page_size or 15)))
    page_count = max(1, (total + ps - 1) // ps) if total else 1
    def qp(new_page: int) -> str:
        return f"/entity?{urlencode({'id': id, 'page': new_page, 'page_size': ps})}"

    return templates.TemplateResponse(
        "entity.html",
        {
            "request": request,
            "id": id,
            "props": props,
            "neighbors": neighbors,
            "canonical": str(request.url),
            "neighbors_total": total,
            "neighbors_page": p,
            "neighbors_pages": page_count,
            "neighbors_page_size": ps,
            "neighbors_prev": qp(p-1) if p > 1 else None,
            "neighbors_next": qp(p+1) if p < page_count else None,
            "related": related,
            "recommended": rec_q,
        },
    )


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


def _index_single(id: str, include_label: bool, include_aliases: bool) -> Dict[str, Any]:
    kg = KG.from_env()
    props: Dict[str, Any] = {}
    try:
        props = kg.get_entity_props(id)
    except Exception:
        props = {}
    if not props:
        return {"id": id, "indexed": 0, "message": "entity not found"}

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
        return {"id": id, "indexed": 0, "message": "vector backend unavailable"}

    indexed = 0
    label = props.get("name")
    if include_label and isinstance(label, str) and label.strip():
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

    return {"id": id, "indexed": indexed}


@app.post("/index")
def index_entity(id: str, include_label: bool = True, include_aliases: bool = True) -> dict:
    """Backfill embeddings for a single entity id.

    Set include_label=False for aliases-only indexing.
    """
    res = _index_single(id=id, include_label=include_label, include_aliases=include_aliases)
    return {"ok": True, **res}


@app.post("/index/batch")
def index_batch(ids: str, include_label: bool = True, include_aliases: bool = True) -> dict:
    """Batch backfill. ids is a comma-separated list of entity IDs.

    Example: /index/batch?ids=http://www.wikidata.org/entity/Q42,http://dbpedia.org/resource/Neo4j
    """
    raw = [x.strip() for x in (ids or "").split(",") if x.strip()]
    seen = []
    for x in raw:
        if x not in seen:
            seen.append(x)
    results = []
    total = 0
    for eid in seen:
        r = _index_single(id=eid, include_label=include_label, include_aliases=include_aliases)
        results.append(r)
        try:
            total += int(r.get("indexed", 0))
        except Exception:
            pass
    return {"ok": True, "count": len(seen), "indexed_total": total, "results": results}
