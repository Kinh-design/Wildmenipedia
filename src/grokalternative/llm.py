from __future__ import annotations

from typing import Any, Dict, List

import httpx

from .settings import get_settings


def _clip_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text
    parts = (text or "").split()
    if len(parts) <= max_words:
        return text
    return " ".join(parts[:max_words])


def generate_answer(
    query: str,
    facts: List[Dict[str, Any]] | None = None,
    *,
    tone: str = "neutral",
    length: int = 300,
    audience: str = "general",
    timeframe: int = 90,
) -> str:
    """Generate a styled answer using the configured LLM provider.

    If no remote LLM is configured, returns a lightweight locally rendered answer.
    """
    s = get_settings()
    provider = (s.LLM_PROVIDER or "local").lower()
    model = s.LLM_MODEL or "grok-2-latest"

    # Build compact facts list for prompt
    facts = facts or []
    bullets: List[str] = []
    for f in facts[:8]:
        try:
            subj = str(f.get("subject") or "")
            pred = str(f.get("predicate") or "")
            obj = str(f.get("object") or "")
            bullets.append(f"- {subj} — {pred} → {obj}")
        except Exception:
            continue
    facts_block = "\n".join(bullets)

    sys = (
        "You are an expert knowledge assistant for a Grokipedia-like site. "
        "Write concise, factual answers grounded in the provided facts. "
        "Respect the requested tone, audience, and timeframe."
    )
    user = (
        f"Question: {query}\n\n"
        f"Known facts (may be partial):\n{facts_block}\n\n"
        f"Style: tone={tone}, audience={audience}, timeframe={timeframe} days, target_length≈{length} words.\n"
        "Compose a single-paragraph answer."
    )

    # Provider: xAI Grok (OpenAI-compatible)
    if provider in ("xai", "grok") and (s.__dict__.get("XAI_API_KEY") or ""):
        api_key = str(getattr(s, "XAI_API_KEY"))
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": sys},
                            {"role": "user", "content": user},
                        ],
                        "temperature": 0.3,
                    },
                )
            resp.raise_for_status()
            data = resp.json()
            msg = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
            if isinstance(msg, str) and msg.strip():
                return _clip_words(msg.strip(), max(50, int(length)))
        except Exception:
            pass

    # Fallback local render
    lead = "Executive summary" if tone == "executive" else ("Quick take (with a wink)" if tone == "humorous" else "Summary")
    aud = "for a general audience" if audience == "general" else ("for non-technical readers" if audience == "non-technical" else "for experts")
    tf = "all time" if int(timeframe or 0) == 0 else f"last {int(timeframe)} days"
    base = f"{lead} {aud} ({tf}): synthesized {len(facts)} verified facts."
    return base
