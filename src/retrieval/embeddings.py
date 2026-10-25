"""Sentence Transformers Embedding Service."""

import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from ..config import config

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Manages local embedding generation using SentenceTransformers."""

    _instance: Optional["EmbeddingService"] = None
    _model: Optional[SentenceTransformer] = None

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.embedding_model_name

    def _get_model(self) -> SentenceTransformer:
        """Lazy load the embedding model onto CPU or GPU."""
        if self._model is None:
            logger.info(f"Loading SentenceTransformer embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Compute normalized vector embeddings for a list of text strings."""
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Compute normalized vector embedding for a single search query."""
        model = self._get_model()
        embedding = model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embedding.tolist()

