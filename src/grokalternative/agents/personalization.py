from __future__ import annotations

from typing import Dict, TypedDict


class Personalized(TypedDict):
    answer: str
    citations: list[str]


def run(curated: Dict) -> Personalized:
    # Simple renderer
    cnt = len(curated.get("triples", []))
    citations = curated.get("citations", [])
    return {
        "answer": f"Synthesized {cnt} verified facts.",
        "citations": citations,
    }
