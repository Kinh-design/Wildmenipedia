from __future__ import annotations

from typing import Dict, List, TypedDict


class Draft(TypedDict):
    triples: List[Dict]
    sources: List[str]


def run(query: str) -> Draft:
    # Minimal placeholder; replace with real retrieval & extraction
    return {
        "triples": [
            {"subject": "wd:Quantum_entanglement", "predicate": "is_a", "object": "Concept"}
        ],
        "sources": ["https://www.wikidata.org/", "https://en.wikipedia.org/"]
    }
