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
    data = sparql.wikidata_search(term, limit)
    items = sparql.extract_labels_from_results(data)
    return ingest_labels(kg or KG.from_env(), items)


def ingest_from_dbpedia(term: str, kg: KG | None = None, limit: int = 5) -> int:
    data = sparql.dbpedia_search(term, limit)
    items = sparql.extract_labels_from_results(data)
    return ingest_labels(kg or KG.from_env(), items)
