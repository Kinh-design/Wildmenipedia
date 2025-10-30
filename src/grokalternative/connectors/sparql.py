from __future__ import annotations

from typing import Any, Dict, List

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
