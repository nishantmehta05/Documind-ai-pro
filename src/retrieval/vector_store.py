"""ChromaDB Persistent Vector Store Manager."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import chromadb
from chromadb.config import Settings

from ..config import config
from ..ingestion.chunking import DocumentChunk

logger = logging.getLogger(__name__)

class SearchResult(BaseModel):
    """Normalized search result across dense, sparse, and hybrid retrievers."""
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    score: float = Field(default=0.0, description="Higher score indicates stronger relevance")
    retrieval_method: str = Field(default="vector", description="vector, bm25, hybrid, or reranked")

class ChromaVectorStore:
    """Persistent vector store implementation powered by ChromaDB."""

    COLLECTION_NAME = "documind_research_corpus"

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = persist_dir or config.chroma_persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "DocuMind Research Assistant Document Store"}
        )

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> int:
        """Upsert document chunks and pre-computed embeddings into ChromaDB."""
        if not chunks:
            return 0

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.to_metadata() for chunk in chunks]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Upserted {len(chunks)} chunks into ChromaDB collection '{self.COLLECTION_NAME}'.")
        return len(chunks)

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_doc_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Perform cosine/dense vector similarity search."""
        count = self.collection.count()
        if count == 0:
            return []

        limit = min(top_k, count)
        where_filter = None
        if filter_doc_ids:
            if len(filter_doc_ids) == 1:
                where_filter = {"doc_id": filter_doc_ids[0]}
            elif len(filter_doc_ids) > 1:
                where_filter = {"doc_id": {"$in": filter_doc_ids}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        search_results: List[SearchResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return search_results

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else [""] * len(ids)
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
        distances = results["distances"][0] if results["distances"] else [1.0] * len(ids)

        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
            # Chroma returns L2/cosine distance (smaller is better). Convert to similarity score (higher is better)
            similarity = 1.0 / (1.0 + float(dist))
            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    doc_id=meta.get("doc_id", ""),
                    filename=meta.get("filename", "unknown"),
                    page_number=int(meta.get("page_number", 1)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=text,
                    score=similarity,
                    retrieval_method="vector"
                )
            )

        return search_results

    def get_all_chunks(self, filter_doc_ids: Optional[List[str]] = None) -> List[SearchResult]:
        """Retrieve stored chunks for keyword indexing (BM25) and evaluation."""
        count = self.collection.count()
        if count == 0:
            return []

        where_filter = None
        if filter_doc_ids:
            if len(filter_doc_ids) == 1:
                where_filter = {"doc_id": filter_doc_ids[0]}
            elif len(filter_doc_ids) > 1:
                where_filter = {"doc_id": {"$in": filter_doc_ids}}

        all_data = self.collection.get(
            where=where_filter,
            include=["documents", "metadatas"]
        )

        results: List[SearchResult] = []
        for chunk_id, text, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    doc_id=meta.get("doc_id", ""),
                    filename=meta.get("filename", "unknown"),
                    page_number=int(meta.get("page_number", 1)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=text,
                    score=1.0,
                    retrieval_method="store"
                )
            )
        return results

    def get_indexed_documents_stats(self) -> List[Dict[str, Any]]:
        """Get summary statistics for all ingested documents."""
        doc_stats: Dict[str, Dict[str, Any]] = {}
        stored_data = self.collection.get(include=["metadatas"])
        for metadata in stored_data["metadatas"] or []:
            did = metadata.get("doc_id", "")
            if not did:
                continue
            if did not in doc_stats:
                doc_stats[did] = {
                    "doc_id": did,
                    "filename": metadata.get("filename", "unknown"),
                    "chunk_count": 0,
                    "pages": set(),
                    "document_page_count": 0,
                }
            doc_stats[did]["chunk_count"] += 1
            doc_stats[did]["pages"].add(int(metadata.get("page_number", 1)))
            doc_stats[did]["document_page_count"] = max(
                doc_stats[did]["document_page_count"],
                int(metadata.get("document_page_count", 0)),
            )

        result_list = []
        for s in doc_stats.values():
            result_list.append({
                "doc_id": s["doc_id"],
                "filename": s["filename"],
                "chunk_count": s["chunk_count"],
                "page_count": s["document_page_count"] or len(s["pages"]),
                "page_count_is_total": bool(s["document_page_count"]),
            })
        return result_list

    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks associated with a specific document ID."""
        self.collection.delete(where={"doc_id": doc_id})
        logger.info(f"Deleted all records for doc_id: {doc_id}")

    def clear(self) -> None:
        """Clear entire collection."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(self.COLLECTION_NAME)
