"""BM25 Sparse Keyword Search Implementation."""

import re
import logging
from typing import List, Optional
from rank_bm25 import BM25Okapi

from .vector_store import SearchResult

logger = logging.getLogger(__name__)

class BM25Index:
    """In-memory BM25 Okapi keyword search index."""

    def __init__(self):
        self.chunks: List[SearchResult] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple, deterministic tokenization for keyword matching."""
        # Convert to lower case and find all alphanumeric word tokens
        return re.findall(r"\b\w{2,}\b", text.lower())

    def index_chunks(self, chunks: List[SearchResult]) -> None:
        """Build or refresh BM25 inverted index from a list of chunks."""
        self.chunks = chunks
        self.corpus_tokens = [self.tokenize(c.text) for c in chunks]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
            logger.info(f"Built BM25 index with {len(self.chunks)} passages.")
        else:
            self.bm25 = None

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_doc_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Retrieve top passages matching query keywords."""
        if not self.bm25 or not self.chunks:
            return []

        tokens = self.tokenize(query_text)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked: List[tuple] = []

        if not any(score > 0 for score in scores):
            token_set = set(tokens)
            for idx, chunk in enumerate(self.chunks):
                if filter_doc_ids and chunk.doc_id not in filter_doc_ids:
                    continue

                chunk_tokens = self.corpus_tokens[idx]
                if not chunk_tokens:
                    continue

                overlap = sum(chunk_tokens.count(token) for token in token_set)
                unique_overlap = len(set(chunk_tokens).intersection(token_set))
                if overlap == 0:
                    continue

                fallback_score = (unique_overlap / len(token_set)) + (overlap / max(len(chunk_tokens), 1))
                ranked.append((fallback_score, chunk))
        else:
            for idx, score in enumerate(scores):
                chunk = self.chunks[idx]
                if filter_doc_ids and chunk.doc_id not in filter_doc_ids:
                    continue
                if score > 0:
                    ranked.append((score, chunk))

        ranked.sort(key=lambda x: x[0], reverse=True)
        top_items = ranked[:top_k]

        results: List[SearchResult] = []
        for raw_score, chunk in top_items:
            # Normalize BM25 score to 0..1 scale using soft saturation
            norm_score = float(raw_score / (raw_score + 10.0)) if raw_score > 0 else 0.0
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=norm_score,
                    retrieval_method="bm25"
                )
            )

        return results

