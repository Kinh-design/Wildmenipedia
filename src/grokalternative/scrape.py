from __future__ import annotations

import re
import time
from dataclasses import dataclass
from html import unescape
from typing import Any, Dict, List, Optional

import httpx

UA_LIST = [
    # A few common desktop/mobile UAs to avoid trivial blocks
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1",
]


def _pick_ua(i: int = 0) -> str:
    try:
        return UA_LIST[i % len(UA_LIST)]
    except Exception:
        return UA_LIST[0]


def _extract_title_html(html: str) -> Optional[str]:
    m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    title = re.sub(r"\s+", " ", m.group(1)).strip()
    return unescape(title) if title else None


def _strip_tags(html: str) -> str:
    # quick-and-dirty removal of scripts/styles and tags
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return unescape(text).strip()


def _summarize(text: str, max_sentences: int = 5) -> str:
    # naive: take the first N non-empty sentences
    parts = re.split(r"(?<=[\.!?])\s+", text)
    out: List[str] = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        out.append(s)
        if len(out) >= max_sentences:
            break
    return " ".join(out)


@dataclass
class FetchResult:
    url: str
    ok: bool
    status_code: Optional[int] = None
    engine: str = "requests"
    duration_ms: int = 0
    title: Optional[str] = None
    html_length: int = 0
    text: Optional[str] = None
    summary: Optional[str] = None
    error: Optional[str] = None


def _fetch_requests(url: str, timeout: float = 15.0) -> FetchResult:
    t0 = time.time()
    try:
        headers = {"User-Agent": _pick_ua(0), "Accept-Language": "en-US,en;q=0.9"}
        with httpx.Client(follow_redirects=True, headers=headers, timeout=timeout) as client:
            r = client.get(url)
            html = r.text or ""
            title = _extract_title_html(html)
            text = _strip_tags(html)
            return FetchResult(
                url=url,
                ok=(r.status_code >= 200 and r.status_code < 400),
                status_code=r.status_code,
                engine="httpx",
                duration_ms=int((time.time() - t0) * 1000),
                title=title,
                html_length=len(html),
                text=text,
            )
    except Exception as e:
        return FetchResult(
            url=url,
            ok=False,
            status_code=None,
            engine="httpx",
            duration_ms=int((time.time() - t0) * 1000),
            error=str(e),
        )


def _fetch_cloudscraper(url: str, timeout: float = 15.0) -> Optional[FetchResult]:
    try:
        import cloudscraper  # type: ignore
    except Exception:
        return None

    t0 = time.time()
    try:
        scraper = cloudscraper.create_scraper()
        r = scraper.get(url, timeout=timeout)
        html = r.text or ""
        title = _extract_title_html(html)
        text = _strip_tags(html)
        return FetchResult(
            url=url,
            ok=(r.status_code >= 200 and r.status_code < 400),
            status_code=r.status_code,
            engine="cloudscraper",
            duration_ms=int((time.time() - t0) * 1000),
            title=title,
            html_length=len(html),
            text=text,
        )
    except Exception as e:
        return FetchResult(
            url=url,
            ok=False,
            status_code=None,
            engine="cloudscraper",
            duration_ms=int((time.time() - t0) * 1000),
            error=str(e),
        )


def realtime_fetch(
    urls: List[str],
    strategy: str = "auto",
    timeout: float = 15.0,
    summarize: bool = True,
    max_sentences: int = 5,
) -> List[Dict[str, Any]]:
    """Fetch pages in near real-time with multiple strategies.

    Strategies supported now (auto, httpx, cloudscraper). Other frameworks are
    pluggable via future adapters (playwright, selenium, pyppeteer, scrapy,
    nutch, crawlee, crawl4ai, firecrawl, botasaurus). When unavailable, we fall
    back to httpx.
    """
    out: List[Dict[str, Any]] = []
    for u in urls:
        u = (u or "").strip()
        if not u:
            continue
        res: Optional[FetchResult] = None
        # Choose engine
        if strategy == "cloudscraper":
            res = _fetch_cloudscraper(u, timeout=timeout)
            if res is None:
                res = _fetch_requests(u, timeout=timeout)
        elif strategy in ("httpx", "requests"):
            res = _fetch_requests(u, timeout=timeout)
        else:  # auto
            # Try cloudscraper if available, then fallback
            res = _fetch_cloudscraper(u, timeout=timeout) or _fetch_requests(u, timeout=timeout)

        if summarize and res and res.ok and (res.text or "").strip():
            try:
                res.summary = _summarize(res.text or "", max_sentences=max_sentences)
            except Exception:
                res.summary = None

        out.append({
            "url": res.url if res else u,
            "ok": bool(res and res.ok),
            "status_code": res.status_code if res else None,
            "engine": res.engine if res else strategy,
            "duration_ms": res.duration_ms if res else 0,
            "title": res.title if res else None,
            "html_length": res.html_length if res else 0,
            "text": res.text if res else None,
            "summary": res.summary if res else None,
            "error": res.error if res else "unavailable",
        })
    return out
