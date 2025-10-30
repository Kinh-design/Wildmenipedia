from __future__ import annotations

import hashlib
from typing import List


class Embedder:
    """Lightweight embedder with optional Sentence-Transformers backend.

    Falls back to a deterministic hash-based embedding that is fast and
    requires no external downloads, suitable for tests and CI.
    """

    def __init__(self, dim: int = 256, model: str | None = None) -> None:
        self.dim = dim
        self.model_name = model
        self._model = None
        if model:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(model)
                # If model exists, set dim from model
                try:
                    self.dim = int(self._model.get_sentence_embedding_dimension())
                except Exception:
                    pass
            except Exception:
                self._model = None

    def embed(self, text: str) -> List[float]:
        if self._model is not None:
            vec = self._model.encode(text)
            return [float(x) for x in vec]
        # Fallback: deterministic hash -> vector in [0,1)
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # expand hash deterministically
        values = list(h) * ((self.dim // len(h)) + 1)
        arr = [v / 255.0 for v in values[: self.dim]]
        # L2-normalize
        import math

        norm = math.sqrt(sum(x * x for x in arr)) or 1.0
        return [x / norm for x in arr]
