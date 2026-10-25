"""Hybrid Retrieval using Reciprocal Rank Fusion (RRF)."""

import logging
from typing import List, Dict, Optional

from .vector_store import ChromaVectorStore, SearchResult
from .embeddings import EmbeddingService
from .bm25_search import BM25Index
from ..config import config

logger = logging.getLogger(__name__)


def resolve_document_scope(
    selected_doc_ids: List[str],
    attached_doc_ids: List[str],
    indexed_doc_ids: List[str],
) -> Optional[List[str]]:
    """Resolve selected filters plus this chat's attachments; empty means all indexed PDFs."""
    available = set(indexed_doc_ids)
    selected = [doc_id for doc_id in selected_doc_ids if doc_id in available]
    attached = [doc_id for doc_id in attached_doc_ids if doc_id in available]
    if not selected:
        return None
    return list(dict.fromkeys([*selected, *attached]))


def _merge_adjacent_passages(passages: List[SearchResult]) -> List[SearchResult]:
    """Join retrieved neighboring chunks from the same page to restore split evidence."""
    ordered = sorted(
        passages,
        key=lambda result: (result.doc_id, result.page_number, result.chunk_index),
    )
    groups: List[List[SearchResult]] = []
    for passage in ordered:
        if (
            groups
            and groups[-1][-1].doc_id == passage.doc_id
            and groups[-1][-1].page_number == passage.page_number
            and groups[-1][-1].chunk_index + 1 == passage.chunk_index
        ):
            groups[-1].append(passage)
        else:
            groups.append([passage])

    merged: List[SearchResult] = []
    for group in groups:
        representative = max(group, key=lambda result: result.score)
        text = group[0].text
        for passage in group[1:]:
            max_overlap = min(len(text), len(passage.text), 200)
            overlap = next(
                (
                    length
                    for length in range(max_overlap, 0, -1)
                    if text.endswith(passage.text[:length])
                ),
                0,
            )
            if overlap:
                text += passage.text[overlap:]
            else:
                text = f"{text.rstrip()} {passage.text.lstrip()}"
        merged.append(
            representative.model_copy(
                update={
                    "chunk_id": group[0].chunk_id,
                    "chunk_index": group[0].chunk_index,
                    "text": text,
                }
            )
        )

    return sorted(merged, key=lambda result: result.score, reverse=True)


class HybridRetriever:
    """Combines Dense Vector Retrieval and Sparse BM25 Keyword Search via RRF."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_service: EmbeddingService,
        bm25_index: Optional[BM25Index] = None,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.bm25_index = bm25_index or BM25Index()
        self.rrf_k = rrf_k
        self._ensure_bm25_synced()

    def _ensure_bm25_synced(self) -> None:
        """Synchronize BM25 index with current chunks in vector store."""
        all_chunks = self.vector_store.get_all_chunks()
        self.bm25_index.index_chunks(all_chunks)

    def refresh_bm25(self) -> None:
        """Call whenever new documents are ingested."""
        self._ensure_bm25_synced()

    def search_dense(
        self,
        query: str,
        top_k: int = 5,
        filter_doc_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Perform purely dense semantic search."""
        query_vec = self.embedding_service.embed_query(query)
        return self.vector_store.query(query_vec, top_k=top_k, filter_doc_ids=filter_doc_ids)

    def search_sparse(
        self,
        query: str,
        top_k: int = 5,
        filter_doc_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Perform purely sparse keyword search."""
        return self.bm25_index.query(query, top_k=top_k, filter_doc_ids=filter_doc_ids)

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        filter_doc_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Execute Reciprocal Rank Fusion (RRF) across Dense and Sparse retrieval."""
        candidate_multiplier = 3
        fetch_k = max(top_k * candidate_multiplier, 10)

        dense_results = self.search_dense(query, top_k=fetch_k, filter_doc_ids=filter_doc_ids)
        sparse_results = self.search_sparse(query, top_k=fetch_k, filter_doc_ids=filter_doc_ids)

        if not dense_results and not sparse_results:
            return []
        if not sparse_results:
            return dense_results[:top_k]
        if not dense_results:
            return sparse_results[:top_k]

        # Calculate RRF scores: RRF(d) = sum(1 / (k + rank))
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, SearchResult] = {}
        provenance_map: Dict[str, List[str]] = {}

        # 1. Dense rankings (1-indexed)
        for rank, res in enumerate(dense_results, start=1):
            cid = res.chunk_id
            chunk_map[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            provenance_map.setdefault(cid, []).append(f"dense_rank_{rank}")

        # 2. Sparse rankings (1-indexed)
        for rank, res in enumerate(sparse_results, start=1):
            cid = res.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            provenance_map.setdefault(cid, []).append(f"sparse_rank_{rank}")

        # Sort candidate chunks by combined RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)

        final_results: List[SearchResult] = []
        for cid in sorted_cids[:top_k]:
            orig = chunk_map[cid]
            prov = provenance_map.get(cid, [])
            method_desc = "hybrid(" + "+".join(prov) + ")"
            final_results.append(
                SearchResult(
                    chunk_id=orig.chunk_id,
                    doc_id=orig.doc_id,
                    filename=orig.filename,
                    page_number=orig.page_number,
                    chunk_index=orig.chunk_index,
                    text=orig.text,
                    score=float(rrf_scores[cid]),
                    retrieval_method=method_desc
                )
            )

        return _merge_adjacent_passages(final_results)
