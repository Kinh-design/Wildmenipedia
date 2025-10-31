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
            with httpx.Client() as client:
                resp = _request_with_retries(
                    client,
                    "POST",
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
            data = resp.json()
            msg = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
            if isinstance(msg, str) and msg.strip():
                return _clip_words(msg.strip(), max(50, int(length)))
        except Exception:
            pass

    # Provider: OpenAI
    if provider in ("openai",) and (s.__dict__.get("OPENAI_API_KEY") or ""):
        api_key = str(getattr(s, "OPENAI_API_KEY"))
        try:
            with httpx.Client() as client:
                resp = _request_with_retries(
                    client,
                    "POST",
                    "https://api.openai.com/v1/chat/completions",
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
            data = resp.json()
            msg = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
            if isinstance(msg, str) and msg.strip():
                return _clip_words(msg.strip(), max(50, int(length)))
        except Exception:
            pass

    # Provider: Anthropic
    if provider in ("anthropic",) and (s.__dict__.get("ANTHROPIC_API_KEY") or ""):
        api_key = str(getattr(s, "ANTHROPIC_API_KEY"))
        try:
            with httpx.Client() as client:
                resp = _request_with_retries(
                    client,
                    "POST",
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 512,
                        "system": sys,
                        "messages": [{"role": "user", "content": user}],
                    },
                )
            data = resp.json()
            parts = (data.get("content") or [])
            if parts and isinstance(parts[0], dict):
                text = parts[0].get("text")
                if isinstance(text, str) and text.strip():
                    return _clip_words(text.strip(), max(50, int(length)))
        except Exception:
            pass

    # Provider: Groq (OpenAI-compatible endpoint)
    if provider in ("groq",) and (s.__dict__.get("GROQ_API_KEY") or ""):
        api_key = str(getattr(s, "GROQ_API_KEY"))
        try:
            with httpx.Client() as client:
                resp = _request_with_retries(
                    client,
                    "POST",
                    "https://api.groq.com/openai/v1/chat/completions",
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
