"""
Embedding wrapper for sentence-transformers/all-MiniLM-L6-v2.

Loads the model once and exposes an encode() method. The model produces
384-dimensional float vectors, matching VECTOR_DIM in services/shared/qdrant_init.py.
"""

import logging
from typing import Union

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Thin wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        logger.info("Loading embedding model %r …", model_name)
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")

    def encode(self, texts: Union[str, list[str]]) -> list[list[float]]:
        """
        Encode one or more texts to 384-dim float vectors.

        Returns a list of vectors (one per input text).
        """
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]
