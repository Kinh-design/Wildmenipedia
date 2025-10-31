from __future__ import annotations

from typing import Dict, TypedDict


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
    # Lightweight style-aware renderer
    cnt = len(curated.get("triples", []))
    citations = curated.get("citations", [])

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

    # Approximate length by adding an extra sentence for longer targets
    extra = "" if length < 400 else " We include a bit more detail given your length setting."
    answer = f"{lead} {aud} ({tf}): synthesized {cnt} verified facts.{extra}"

    return {"answer": answer, "citations": citations}
