from __future__ import annotations

from typing import Any, Dict, List, Optional

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


def hybrid_answer(
    question: str,
    kg: KG | None = None,
    vs: VS | None = None,
    top_k: int = 5,
    web_docs: Optional[List[Dict[str, Any]]] = None,
    *,
    tone: str = "neutral",
    length: int = 300,
    audience: str = "general",
    timeframe: int = 90,
) -> Dict[str, Any]:
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
    id_to_score: Dict[str, float] = {}
    if vector_hits:
        best = max((h.get("score") or 0.0) for h in vector_hits) or 1.0
        for h in vector_hits:
            pid = ((h.get("payload") or {}).get("id") or "")
            try:
                id_to_score[str(pid)] = float(h.get("score") or 0.0) / float(best)
            except Exception:
                pass

    for node_id in candidate_ids[:3]:
        try:
            neighbors = kg.neighbors(node_id, limit=10)
        except Exception:
            neighbors = []
        for n in neighbors:
            rec = {
                "subject": n.get("s"),
                "predicate": n.get("p"),
                "object": n.get("o"),
                "meta": n.get("meta"),
            }
            # Simple fusion: normalized vector score + small bonuses
            score = float(id_to_score.get(node_id, 0.0))
            meta = rec.get("meta") or {}
            try:
                if (meta.get("rank") or "") == "preferred":
                    score += 0.1
                if (meta.get("pred_code") or "") in ("P31", "P279"):
                    score += 0.05
            except Exception:
                pass
            rec["score"] = round(score, 4)
            facts.append(rec)

    # Sort facts by score desc
    facts_sorted = sorted(facts, key=lambda x: x.get("score", 0.0), reverse=True)

    # Simple citation aggregation from web docs
    sources: List[Dict[str, Any]] = []
    if web_docs:
        for d in web_docs[:10]:
            try:
                sources.append({
                    "url": d.get("url"),
                    "title": d.get("title") or d.get("url"),
                    "engine": d.get("engine"),
                })
            except Exception:
                continue
    confidence = "low"
    if len(sources) >= 3:
        confidence = "high"
    elif len(sources) == 2:
        confidence = "medium"

    # Build style-aware summary with inline footnote markers tied to sources order
    if tone == "executive":
        lead = "Executive summary"
    elif tone == "humorous":
        lead = "Quick take (with a wink)"
    else:
        lead = "Summary"
    aud = "for a general audience" if audience == "general" else (
        "for non-technical readers" if audience == "non-technical" else "for experts"
    )
    tf = "all time" if int(timeframe or 0) == 0 else f"last {int(timeframe)} days"
    fn = ""  # footnotes like [1][2]
    if sources:
        lim = min(3, len(sources))
        fn = " " + "".join([f"[{i}]" for i in range(1, lim + 1)])
    extra = "" if length < 400 else " Additional detail included per requested length."
    answer_text = (
        f"{lead} {aud} ({tf}): {len(facts_sorted)} graph facts fused with {len(vector_hits)} vector hits.{extra}{fn}"
    )

    return {
        "answer": answer_text,
        "facts": facts_sorted,
        "vector_hits": vector_hits,
        "selected_entities": candidate_ids,
        "sources": sources,
        "confidence": confidence,
    }
