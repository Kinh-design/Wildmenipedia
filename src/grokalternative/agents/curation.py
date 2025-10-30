from __future__ import annotations

from typing import Dict, List, TypedDict


class Curated(TypedDict):
    triples: List[Dict]
    citations: List[str]


def run(verified: Dict) -> Curated:
    # Minimal normalization pass
    citations = verified.get("sources", [])
    triples = verified.get("triples", [])
    return {"triples": triples, "citations": citations}
