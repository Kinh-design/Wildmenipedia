from grokalternative.connectors.sparql import (
    extract_dbpedia_triples,
    extract_labels_from_results,
    extract_wikidata_triples,
    extract_wikidata_triples_with_qualifiers,
    predicate_short_code,
    select_best_candidate,
    select_preferred_statements,
)


def test_extract_labels_from_results_minimal():
    data = {
        "results": {
            "bindings": [
                {
                    "item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q123"},
                    "itemLabel": {"type": "literal", "value": "Sample"},
                },
                {
                    "resource": {"type": "uri", "value": "http://dbpedia.org/resource/Sample"},
                    "label": {"type": "literal", "value": "Sample DBpedia"},
                },
            ]
        }
    }
    rows = extract_labels_from_results(data)
    assert {"id": "http://www.wikidata.org/entity/Q123", "label": "Sample"} in rows
    assert {"id": "http://dbpedia.org/resource/Sample", "label": "Sample DBpedia"} in rows


def test_extract_wikidata_triples_minimal():
    subject = "http://www.wikidata.org/entity/Q42"
    data = {
        "results": {
            "bindings": [
                {
                    "p": {"type": "uri", "value": "http://www.wikidata.org/prop/direct/P31"},
                    "pLabel": {"type": "literal", "value": "instance of"},
                    "o": {"type": "uri", "value": "http://www.wikidata.org/entity/Q5"},
                    "oLabel": {"type": "literal", "value": "human"},
                }
            ]
        }
    }
    triples = extract_wikidata_triples(subject, data)
    assert len(triples) == 1
    assert triples[0]["subject"] == subject
    assert triples[0]["predicate"].endswith("/P31")
    assert triples[0]["object"].endswith("/Q5")
    assert triples[0]["predicate_label"] == "instance of"
    assert triples[0]["object_label"] == "human"


def test_extract_dbpedia_triples_minimal():
    subject = "http://dbpedia.org/resource/Albert_Einstein"
    data = {
        "results": {
            "bindings": [
                {
                    "p": {"type": "uri", "value": "http://dbpedia.org/ontology/birthPlace"},
                    "o": {"type": "uri", "value": "http://dbpedia.org/resource/Ulm"},
                    "oLabel": {"type": "literal", "value": "Ulm"},
                }
            ]
        }
    }
    triples = extract_dbpedia_triples(subject, data)
    assert len(triples) == 1
    assert triples[0]["subject"] == subject
    assert triples[0]["predicate"].endswith("/birthPlace")
    assert triples[0]["object"].endswith("/Ulm")
    assert triples[0]["object_label"] == "Ulm"


def test_select_best_candidate_prefers_exact_match():
    term = "Albert Einstein"
    items = [
        {"id": "http://dbpedia.org/resource/Einstein_(crater)", "label": "Einstein (crater)"},
        {"id": "http://dbpedia.org/resource/Albert_Einstein", "label": "Albert Einstein"},
        {"id": "http://dbpedia.org/resource/Einstein", "label": "Einstein"},
    ]
    best = select_best_candidate(term, items)
    assert best and best["id"].endswith("/Albert_Einstein")


def test_extract_wikidata_triples_with_qualifiers_minimal():
    subject = "http://www.wikidata.org/entity/Q42"
    data = {
        "results": {
            "bindings": [
                {
                    "p": {"type": "uri", "value": "http://www.wikidata.org/prop/P26"},
                    "pLabel": {"type": "literal", "value": "spouse"},
                    "ps": {"type": "uri", "value": "http://www.wikidata.org/entity/Q146236"},
                    "psLabel": {"type": "literal", "value": "Jane Belson"},
                    "qualPred": {"type": "uri", "value": "http://www.wikidata.org/prop/qualifier/P580"},
                    "qualPredLabel": {"type": "literal", "value": "start time"},
                    "qual": {"type": "literal", "value": "1991-11-25"},
                    "qualLabel": {"type": "literal", "value": "1991-11-25"},
                }
            ]
        }
    }
    triples = extract_wikidata_triples_with_qualifiers(subject, data)
    assert len(triples) == 1
    t = triples[0]
    assert t["predicate"].endswith("/P26")
    assert t["object"].endswith("/Q146236")
    assert t["predicate_label"] == "spouse"
    assert t["object_label"] == "Jane Belson"
    assert t["qualifiers"] and t["qualifiers"][0]["predicate"].endswith("/P580")


def test_predicate_short_code_mappings():
    assert predicate_short_code("http://www.wikidata.org/prop/direct/P31") == "P31"
    assert predicate_short_code("http://www.wikidata.org/prop/qualifier/P580").startswith("pq:")
    assert predicate_short_code("http://dbpedia.org/ontology/birthPlace").startswith("dbo:")


def test_select_preferred_statements():
    triples = [
        {"rank": "normal"},
        {"rank": "preferred"},
        {"rank": "deprecated"},
    ]
    out = select_preferred_statements(triples)  # type: ignore[arg-type]
    assert all(t.get("rank") == "preferred" for t in out)
