"""Optional Cross-Encoder Reranker Service."""

import logging
from typing import List, Optional
from .vector_store import SearchResult

logger = logging.getLogger(__name__)

class RerankerService:
    """Optional cross-encoder reranking stage for high-precision retrieval refinement."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2", enabled: bool = False):
        self.model_name = model_name
        self.enabled = enabled
        self._model = None

    def _load_model(self):
        """Lazy load cross encoder model if enabled."""
        if self._model is None and self.enabled:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder: {self.model_name}...")
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder ({e}). Falling back to passthrough.")
                self.enabled = False

    def rerank(self, query: str, candidates: List[SearchResult], top_k: int = 4) -> List[SearchResult]:
        """Rerank candidates using cross-encoder scores or return unchanged if disabled."""
        if not self.enabled or not candidates:
            return candidates[:top_k]

        self._load_model()
        if not self._model:
            return candidates[:top_k]

        try:
            pairs = [[query, c.text] for c in candidates]
            scores = self._model.predict(pairs)

            ranked_pairs = list(zip(scores, candidates))
            ranked_pairs.sort(key=lambda x: x[0], reverse=True)

            reranked_results: List[SearchResult] = []
            for score, chunk in ranked_pairs[:top_k]:
                reranked_results.append(
                    SearchResult(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        filename=chunk.filename,
                        page_number=chunk.page_number,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        score=float(score),
                        retrieval_method=f"reranked({chunk.retrieval_method})"
                    )
                )
            return reranked_results
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Returning original candidates.")
            return candidates[:top_k]

