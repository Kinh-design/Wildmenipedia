from __future__ import annotations

from typing import Dict, TypedDict

from ..llm import generate_answer


class Personalized(TypedDict):
    answer: str
    citations: list[str]

def run(
    curated: Dict,
    *,
    tone: str = "neutral",
    length: int = 300,
    audience: str = "general",
    timeframe: int = 90,
) -> Personalized:
    # Try LLM-backed rendering; fallback handled inside generate_answer
    facts = curated.get("triples", [])
    answer = generate_answer(
        query=curated.get("query") or "",
        facts=facts,
        tone=tone,
        length=length,
        audience=audience,
        timeframe=timeframe,
    )
    citations = curated.get("citations", [])
    return {"answer": answer, "citations": citations}
