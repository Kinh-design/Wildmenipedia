from __future__ import annotations

from typing import Any, Dict, List

from .embeddings import Embedder
from .stores import KG, VS


def graphrag_answer(question: str, kg: KG | None = None) -> Dict[str, Any]:
    kg = kg or KG.from_env()
    # naive heuristic: use question as node_id directly; real impl will resolve entities
    neighbors = []
    try:
        neighbors = kg.neighbors(question, limit=10)
    except Exception:
        # Graph may not be reachable; return empty facts
        neighbors = []
    facts = [
        {"subject": n.get("s"), "predicate": n.get("p"), "object": n.get("o"), "meta": n.get("meta")}
        for n in neighbors
    ]
    return {
        "answer": f"GraphRAG summary for '{question}': {len(facts)} facts found.",
        "facts": facts,
    }


def hybrid_answer(question: str, kg: KG | None = None, vs: VS | None = None, top_k: int = 5) -> Dict[str, Any]:
    kg = kg or KG.from_env()
    vs = vs or VS.from_env()
    embedder = Embedder(dim=256)

    # Vector side
    vector_hits: List[Dict[str, Any]] = []
    try:
        q_vec = embedder.embed(question)
        vector_hits = vs.search("entities", q_vec, top_k=top_k)
    except Exception:
        vector_hits = []

    # Graph side: prefer neighbors of top vector-hit entity ids; fallback to raw question node id
    facts: List[Dict[str, Any]] = []
    candidate_ids: List[str] = []
    for hit in vector_hits:
        payload = hit.get("payload") or {}
        cand = payload.get("id") or payload.get("entity_id")
        if isinstance(cand, str):
            candidate_ids.append(cand)
    if not candidate_ids:
        candidate_ids = [question]
    for node_id in candidate_ids[:3]:
        try:
            neighbors = kg.neighbors(node_id, limit=10)
        except Exception:
            neighbors = []
        for n in neighbors:
            facts.append({
                "subject": n.get("s"),
                "predicate": n.get("p"),
                "object": n.get("o"),
                "meta": n.get("meta"),
            })

    return {
        "answer": f"Hybrid summary for '{question}': graph_facts={len(facts)}, vector_hits={len(vector_hits)}.",
        "facts": facts,
        "vector_hits": vector_hits,
        "selected_entities": candidate_ids,
    }
