from grokalternative.embeddings import Embedder


def test_embedder_fallback_is_deterministic_and_dimensionality():
    emb = Embedder(dim=64)
    v1 = emb.embed("Hello world")
    v2 = emb.embed("Hello world")
    assert len(v1) == 64
    assert len(v2) == 64
    assert v1 == v2
