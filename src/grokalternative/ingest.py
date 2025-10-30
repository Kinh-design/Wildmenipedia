from __future__ import annotations

import hashlib
from typing import Dict, List

from .connectors import sparql
from .embeddings import Embedder
from .stores import KG, VS


def ingest_labels(kg: KG, items: List[Dict[str, str]]) -> int:
    count = 0
    # Prepare vector indexing (best-effort)
    embedder: Embedder | None = None
    vs: VS | None = None
    try:
        embedder = Embedder(dim=256)
        vs = VS.from_env()
        vs.ensure_collection(name="entities", dim=embedder.dim)
    except Exception:
        embedder = None
        vs = None
    for it in items:
        sid = it.get("id", "").strip()
        label = it.get("label", "").strip()
        if not sid or not label:
            continue
        # Store node label property and an auxiliary edge
        kg.set_label(sid, label)
        # Keep an edge to preserve prior behavior/minimal provenance
        kg.upsert_triple(sid, "has_label", sid, {"label": label})
        # Index into Qdrant (if available)
        if embedder and vs:
            try:
                vec = embedder.embed(label)
                src = "wikidata" if "wikidata.org" in sid else ("dbpedia" if "dbpedia.org" in sid else "unknown")
                vs.upsert_point(
                    collection="entities",
                    point_id=sid,
                    vector=vec,
                    payload={"id": sid, "label": label, "source": src},
                )
            except Exception:
                pass
        count += 1
    return count


def ingest_from_wikidata(term: str, kg: KG | None = None, limit: int = 5) -> int:
    backend = kg or KG.from_env()
    try:
        backend.ensure_schema()
    except Exception:
        pass
    # Labels
    data = sparql.wikidata_search(term, limit)
    items = sparql.extract_labels_from_results(data)
    total = ingest_labels(backend, items)
    # Entity resolution (top hit) + triples
    top = sparql.wikidata_resolve(term)
    if top:
        triples_raw = sparql.wikidata_triples(top["id"], limit=200)
        triples = sparql.extract_wikidata_triples(top["id"], triples_raw)
        total += ingest_triples(backend, triples)
        # with qualifiers
        stmts_raw = sparql.wikidata_statements_with_qualifiers(top["id"], limit=300)
        stmts = sparql.extract_wikidata_triples_with_qualifiers(top["id"], stmts_raw)
        stmts = sparql.select_preferred_statements(stmts)
        total += ingest_triples(backend, stmts)
        # aliases
        try:
            aliases_raw = sparql.wikidata_aliases(top["id"], limit=100)
            aliases = sparql.extract_aliases(aliases_raw)
            # Prepare vector backend
            embedder: Embedder | None = None
            vs: VS | None = None
            try:
                embedder = Embedder(dim=256)
                vs = VS.from_env()
                vs.ensure_collection(name="entities", dim=embedder.dim)
            except Exception:
                embedder = None
                vs = None
            for a in aliases:
                backend.add_alias(top["id"], a)
                if embedder and vs:
                    try:
                        vec = embedder.embed(a)
                        ah = hashlib.sha1(a.encode("utf-8")).hexdigest()[:12]
                        point_id = f"{top['id']}#alias:{ah}"
                        vs.upsert_point(
                            collection="entities",
                            point_id=point_id,
                            vector=vec,
                            payload={
                                "id": top["id"],
                                "alias": a,
                                "is_alias": True,
                                "source": "wikidata",
                            },
                        )
                    except Exception:
                        pass
        except Exception:
            pass
    return total


def ingest_from_dbpedia(term: str, kg: KG | None = None, limit: int = 5) -> int:
    backend = kg or KG.from_env()
    try:
        backend.ensure_schema()
    except Exception:
        pass
    data = sparql.dbpedia_search(term, limit)
    items = sparql.extract_labels_from_results(data)
    total = ingest_labels(backend, items)
    # top-hit triples
    top = sparql.dbpedia_resolve(term) or (items[0] if items else None)
    if top:
        triples_raw = sparql.dbpedia_triples(top["id"], limit=200)
        triples = sparql.extract_dbpedia_triples(top["id"], triples_raw)
        total += ingest_triples(backend, triples)
    return total


def ingest_triples(kg: KG, triples: List[Dict[str, str]]) -> int:
    count = 0
    for t in triples:
        s = t.get("subject", "").strip()
        p = t.get("predicate", "").strip()
        o = t.get("object", "").strip()
        if not s or not p or not o:
            continue
        meta = {}
        if "predicate_label" in t and t["predicate_label"]:
            meta["predicate_label"] = t["predicate_label"]
        if "object_label" in t and t["object_label"]:
            meta["object_label"] = t["object_label"]
        if "rank" in t and t["rank"]:
            meta["rank"] = t["rank"]
        if "qualifiers" in t and t["qualifiers"]:
            # store qualifiers as JSON string to keep schema simple
            try:
                import json

                meta["qualifiers"] = json.dumps(t["qualifiers"]) 
            except Exception:
                pass
        # short predicate code
        try:
            meta["pred_code"] = sparql.predicate_short_code(p)
        except Exception:
            pass
        kg.upsert_triple(s, p, o, meta)
        # Derived node properties
        try:
            code = meta.get("pred_code", "")
            if code == "P31":
                kg.add_type(s, o)
        except Exception:
            pass
        count += 1
    return count
