from __future__ import annotations

from typing import Dict, List

from .connectors import sparql
from .stores import KG


def ingest_labels(kg: KG, items: List[Dict[str, str]]) -> int:
    count = 0
    for it in items:
        sid = it.get("id", "").strip()
        label = it.get("label", "").strip()
        if not sid or not label:
            continue
        # Store as a self edge with label metadata, as minimal representation
        kg.upsert_triple(sid, "has_label", sid, {"label": label})
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
        kg.upsert_triple(s, p, o, meta)
        count += 1
    return count
