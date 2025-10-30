from grokalternative.connectors.sparql import extract_labels_from_results


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
