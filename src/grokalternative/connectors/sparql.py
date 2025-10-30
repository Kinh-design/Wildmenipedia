from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import httpx

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
DBPEDIA_SPARQL = "https://dbpedia.org/sparql"


def wikidata_search_query(term: str, limit: int = 5) -> str:
    term_esc = term.replace("\"", "\\\"")
    return f"""
    SELECT ?item ?itemLabel WHERE {{
      ?item rdfs:label ?itemLabel .
      FILTER(LANG(?itemLabel) = "en")
      FILTER(CONTAINS(LCASE(?itemLabel), LCASE("{term_esc}")))
    }} LIMIT {limit}
    """.strip()


def dbpedia_search_query(term: str, limit: int = 5) -> str:
    term_esc = term.replace("\"", "\\\"")
    return f"""
    SELECT ?resource ?label WHERE {{
      ?resource rdfs:label ?label .
      FILTER(LANG(?label) = "en")
      FILTER(CONTAINS(LCASE(?label), LCASE("{term_esc}")))
    }} LIMIT {limit}
    """.strip()


def query_sparql(endpoint: str, query: str, timeout: float = 20.0) -> Dict[str, Any]:
    headers = {"Accept": "application/sparql-results+json"}
    params = {"query": query}
    with httpx.Client(timeout=timeout, headers=headers) as client:
        r = client.get(endpoint, params=params)
        r.raise_for_status()
        return r.json()


def wikidata_search(term: str, limit: int = 5) -> Dict[str, Any]:
    return query_sparql(WIKIDATA_SPARQL, wikidata_search_query(term, limit))


def dbpedia_search(term: str, limit: int = 5) -> Dict[str, Any]:
    return query_sparql(DBPEDIA_SPARQL, dbpedia_search_query(term, limit))


def extract_labels_from_results(data: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for b in data.get("results", {}).get("bindings", []):
        id_val = b.get("item", b.get("resource"))
        label_val = b.get("itemLabel", b.get("label"))
        if id_val and label_val:
            out.append({
                "id": id_val.get("value", ""),
                "label": label_val.get("value", ""),
            })
    return out


# -------------------------
# Entity resolution helpers
# -------------------------

def wikidata_top_hit(term: str) -> Optional[Dict[str, str]]:
    """Return top candidate {id,label} from Wikidata search, if any."""
    items = extract_labels_from_results(wikidata_search(term, limit=1))
    return items[0] if items else None


def _score_candidate(term: str, label: str, id_value: str) -> Tuple[int, int]:
    """Heuristic score for candidate selection; higher is better.

    Returns (score, tie_breaker) where tie_breaker lower is better.
    """
    t = term.strip().lower()
    label_norm = label.strip().lower()
    score = 0
    if label_norm == t:
        score += 100
    elif label_norm.startswith(t):
        score += 60
    elif t in label_norm:
        score += 30
    # Prefer canonical Wikidata/DBpedia URIs
    if "wikidata.org/entity/" in id_value or "dbpedia.org/resource/" in id_value:
        score += 5
    # Shorter labels tend to be more canonical
    tie = abs(len(label) - len(term))
    return score, tie


def select_best_candidate(term: str, items: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    best: Optional[Dict[str, str]] = None
    best_score: Tuple[int, int] = (-1, 10**9)
    for it in items:
        lab = it.get("label", "")
        idv = it.get("id", "")
        sc = _score_candidate(term, lab, idv)
        if sc > best_score:
            best_score = sc
            best = it
    return best


def wikidata_resolve(term: str, limit: int = 10) -> Optional[Dict[str, str]]:
    items = extract_labels_from_results(wikidata_search(term, limit=limit))
    return select_best_candidate(term, items)


def dbpedia_resolve(term: str, limit: int = 10) -> Optional[Dict[str, str]]:
    items = extract_labels_from_results(dbpedia_search(term, limit=limit))
    return select_best_candidate(term, items)


# ---------------------------------------
# Rich triple extraction (Wikidata/DBpedia)
# ---------------------------------------

def _qid_from_uri(uri_or_qid: str) -> str:
    if uri_or_qid.startswith("http"):
        return uri_or_qid.rstrip("/").split("/")[-1]
    return uri_or_qid


def wikidata_triples_query(qid_or_uri: str, limit: int = 200) -> str:
    qid = _qid_from_uri(qid_or_uri)
    return f"""
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?p ?pLabel ?o ?oLabel WHERE {{
      wd:{qid} ?p ?o .
      FILTER(STRSTARTS(STR(?p), STR(wdt:)))
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }} LIMIT {limit}
    """.strip()


def wikidata_triples(qid_or_uri: str, limit: int = 200) -> Dict[str, Any]:
    return query_sparql(WIKIDATA_SPARQL, wikidata_triples_query(qid_or_uri, limit))


def extract_wikidata_triples(subject_qid_or_uri: str, data: Dict[str, Any]) -> List[Dict[str, str]]:
    subject = subject_qid_or_uri if subject_qid_or_uri.startswith("http") else f"http://www.wikidata.org/entity/{_qid_from_uri(subject_qid_or_uri)}"
    out: List[Dict[str, str]] = []
    for b in data.get("results", {}).get("bindings", []):
        p = b.get("p", {}).get("value", "")
        o = b.get("o", {}).get("value", "")
        p_label = b.get("pLabel", {}).get("value", "")
        o_label = b.get("oLabel", {}).get("value", "")
        if p and o:
            out.append({
                "subject": subject,
                "predicate": p,
                "object": o,
                "predicate_label": p_label,
                "object_label": o_label,
            })
    return out


def dbpedia_triples_query(resource_uri: str, limit: int = 200) -> str:
    # Query directly with full resource URI. Restrict to dbo: ontology predicates.
    return f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?p ?o ?oLabel WHERE {{
      <{resource_uri}> ?p ?o .
      OPTIONAL {{ ?o rdfs:label ?oLabel FILTER (LANG(?oLabel) = 'en') }}
      FILTER(STRSTARTS(STR(?p), 'http://dbpedia.org/ontology/'))
    }} LIMIT {limit}
    """.strip()


def dbpedia_triples(resource_uri: str, limit: int = 200) -> Dict[str, Any]:
    return query_sparql(DBPEDIA_SPARQL, dbpedia_triples_query(resource_uri, limit))


def extract_dbpedia_triples(subject_uri: str, data: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for b in data.get("results", {}).get("bindings", []):
        p = b.get("p", {}).get("value", "")
        o = b.get("o", {}).get("value", "")
        o_label = b.get("oLabel", {}).get("value", "")
        if p and o:
            out.append({
                "subject": subject_uri,
                "predicate": p,
                "object": o,
                "object_label": o_label,
            })
    return out
