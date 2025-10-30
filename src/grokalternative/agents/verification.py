from __future__ import annotations

from typing import Dict, List, TypedDict


class VerifiedTriple(TypedDict):
    subject: str
    predicate: str
    object: str
    confidence: float


class VerifiedDraft(TypedDict):
    triples: List[VerifiedTriple]
    sources: List[str]


def run(draft: Dict) -> VerifiedDraft:
    triples: List[VerifiedTriple] = []
    for t in draft.get("triples", []):
        triples.append({
            "subject": t["subject"],
            "predicate": t["predicate"],
            "object": t["object"],
            "confidence": 0.85,
        })
    return {"triples": triples, "sources": draft.get("sources", [])}
