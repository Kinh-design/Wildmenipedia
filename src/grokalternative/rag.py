from __future__ import annotations

import math
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
    docs: List[Dict[str, Any]] = []
    if web_docs:
        for d in web_docs[:10]:
            try:
                docs.append(d)
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

    # Per-claim citation mapping with embedding-based relevance; cap to top-1
    if sources and facts_sorted and docs:
        # Precompute doc embeddings once
        doc_texts: List[str] = [
            " ".join(str(d.get(k) or "") for k in ("title", "summary", "text")) for d in docs
        ]
        doc_vecs: List[List[float]] = []
        for t in doc_texts:
            try:
                doc_vecs.append(embedder.embed(t))
            except Exception:
                doc_vecs.append([0.0] * embedder.dim)

        def _cos(a: List[float], b: List[float]) -> float:
            try:
                da = math.sqrt(sum(x * x for x in a)) or 1.0
                db = math.sqrt(sum(x * x for x in b)) or 1.0
                return sum(x * y for x, y in zip(a, b)) / (da * db)
            except Exception:
                return 0.0

        for rec in facts_sorted:
            s = rec.get("subject")
            p = rec.get("predicate")
            o = rec.get("object")
            m = (rec.get("meta") or {})
            ol = m.get("object_label")
            claim = " | ".join(
                [str(x) for x in [s, p, o, ol] if isinstance(x, str) and x]
            )
            try:
                qv = embedder.embed(claim)
            except Exception:
                qv = [0.0] * embedder.dim
            best_idx = None
            best_sc = -1.0
            for idx, dv in enumerate(doc_vecs):
                sc = _cos(qv, dv)
                if sc > best_sc:
                    best_sc = sc
                    best_idx = idx
            if best_idx is not None:
                rec["citations"] = [best_idx + 1]
            elif sources:
                rec["citations"] = [1]

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
