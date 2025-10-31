from __future__ import annotations

from typing import Any, Dict, cast

from . import curation, personalization, prediction, research, verification


def run_pipeline(
    query: str,
    *,
    tone: str = "neutral",
    length: int = 300,
    audience: str = "general",
    timeframe: int = 90,
) -> Dict[str, Any]:
    draft = research.run(query)
    verified = verification.run(cast(Dict[str, Any], draft))
    curated = curation.run(cast(Dict[str, Any], verified))
    personalized = personalization.run(
        cast(Dict[str, Any], curated),
        tone=tone,
        length=length,
        audience=audience,
        timeframe=timeframe,
    )
    pred = prediction.run({"query": query, "triples": curated.get("triples", [])})

    confs = [t.get("confidence", 0.0) for t in verified.get("triples", [])]
    avg_conf = round(sum(confs) / len(confs), 3) if confs else 0.0

    return {
        "answer": personalized["answer"],
        "citations": personalized.get("citations", []),
        "facts": curated.get("triples", []),
        "prediction": pred,
        "confidence_avg": avg_conf,
    }
