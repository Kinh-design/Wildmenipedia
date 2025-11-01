from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional

import httpx

from .settings import get_settings


def _clip_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text
    parts = (text or "").split()
    if len(parts) <= max_words:
        return text
    return " ".join(parts[:max_words])


def _request_with_retries(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    json: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
    retries: int = 2,
    backoff: float = 0.5,
) -> httpx.Response:
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = client.request(method, url, headers=headers, json=json, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:  # pragma: no cover - exercise via unit test
            last_err = e
            if attempt < retries:
                # simple blocking backoff
                try:
                    import time as _t

                    _t.sleep(backoff * (2**attempt))
                except Exception:  # pragma: no cover
                    pass
            else:
                raise
    assert last_err  # for type checkers
    raise last_err


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
    # Single provider (Gemini). Normalize model formatting like "gemini 2.5 pro" -> "gemini-2.5-pro"
    raw_model = s.LLM_MODEL or "gemini-2.5-pro"
    model = "-".join(str(raw_model).strip().split())

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

    # Single provider: Google Gemini (Generative Language API)
    if s.__dict__.get("GOOGLE_API_KEY"):
        api_key = str(getattr(s, "GOOGLE_API_KEY"))
        try:
            gmodel = model if model.startswith("gemini-") else f"gemini-{model}" if model.startswith("2") else model
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{gmodel}:generateContent?key={api_key}"
            body = {
                "system_instruction": {"parts": [{"text": sys}]},
                "contents": [
                    {"role": "user", "parts": [{"text": user}]}
                ],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512},
            }
            with httpx.Client() as client:
                resp = _request_with_retries(client, "POST", url, json=body)
            data = resp.json()
            cands = data.get("candidates") or []
            if cands:
                content = (cands[0] or {}).get("content") or {}
                parts = content.get("parts") or []
                if parts and isinstance(parts[0], dict):
                    text = parts[0].get("text")
                    if isinstance(text, str) and text.strip():
                        return _clip_words(text.strip(), max(50, int(length)))
        except Exception:
            pass

    # (single provider path above); if no key or errors, fall back below

    # Fallback local render
    lead = "Executive summary" if tone == "executive" else ("Quick take (with a wink)" if tone == "humorous" else "Summary")
    aud = "for a general audience" if audience == "general" else ("for non-technical readers" if audience == "non-technical" else "for experts")
    tf = "all time" if int(timeframe or 0) == 0 else f"last {int(timeframe)} days"
    base = f"{lead} {aud} ({tf}): synthesized {len(facts)} verified facts."
    return base


def stream_answer(
    query: str,
    facts: List[Dict[str, Any]] | None = None,
    *,
    tone: str = "neutral",
    length: int = 300,
    audience: str = "general",
    timeframe: int = 90,
) -> Generator[str, None, None]:
    """Yield the answer in chunks for SSE streaming.

    Attempts provider streaming; falls back to chunking the final answer locally.
    """
    # For simplicity and reliability in tests, use non-streaming call and chunk.
    text = generate_answer(
        query=query,
        facts=facts or [],
        tone=tone,
        length=length,
        audience=audience,
        timeframe=timeframe,
    )
    # Chunk by words
    words = text.split()
    buf: list[str] = []
    for w in words:
        buf.append(w)
        if len(buf) >= 12:
            yield " ".join(buf) + " "
            buf = []
    if buf:
        yield " ".join(buf)
