"""Unit tests for BM25 and Hybrid retrieval algorithms."""

from src.retrieval.bm25_search import BM25Index
from src.ingestion.chunking import DocumentChunk
from src.retrieval.vector_store import SearchResult
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.reranker import RerankerService
from src.retrieval.hybrid_search import _merge_adjacent_passages, resolve_document_scope

def test_bm25_tokenization_and_keyword_retrieval():
    """Verify BM25 indexing and keyword scoring."""
    chunks = [
        SearchResult(
            chunk_id="chunk_alpha", doc_id="d1", filename="ml.pdf", page_number=1, chunk_index=0,
            text="Convolutional neural networks are specialized architectures for computer vision and image processing.",
            score=1.0
        ),
        SearchResult(
            chunk_id="chunk_beta", doc_id="d2", filename="nlp.pdf", page_number=3, chunk_index=0,
            text="Transformer models and self-attention mechanisms revolutionized natural language processing.",
            score=1.0
        )
    ]

    index = BM25Index()
    index.index_chunks(chunks)

    # Search for NLP keywords
    nlp_results = index.query("natural language self-attention transformer", top_k=1)
    assert len(nlp_results) == 1
    assert nlp_results[0].chunk_id == "chunk_beta"
    assert nlp_results[0].score > 0

    # Search for vision keywords
    vision_results = index.query("computer vision convolutional", top_k=1)
    assert len(vision_results) == 1
    assert vision_results[0].chunk_id == "chunk_alpha"

def test_reranker_disabled_passthrough():
    """Ensure reranker operates safely as a passthrough when disabled."""
    reranker = RerankerService(enabled=False)
    candidates = [
        SearchResult(chunk_id="c1", doc_id="d1", filename="a.pdf", page_number=1, chunk_index=0, text="First", score=0.9),
        SearchResult(chunk_id="c2", doc_id="d1", filename="a.pdf", page_number=2, chunk_index=0, text="Second", score=0.8),
    ]
    reranked = reranker.rerank("test query", candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c1"


def test_document_stats_keep_original_pdf_page_count(tmp_path):
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma")
    chunk = DocumentChunk(
        chunk_id="doc1_p1_c0",
        doc_id="doc1",
        filename="paper.pdf",
        page_number=1,
        chunk_index=0,
        text="Evidence from the first page.",
        char_length=29,
        document_page_count=7,
    )
    store.add_chunks([chunk], [[0.1, 0.2, 0.3]])

    stats = store.get_indexed_documents_stats()
    assert stats == [
        {
            "doc_id": "doc1",
            "filename": "paper.pdf",
            "chunk_count": 1,
            "page_count": 7,
            "page_count_is_total": True,
        }
    ]


def test_empty_selection_searches_all_documents():
    assert resolve_document_scope([], [], ["doc1", "doc2"]) is None
    assert resolve_document_scope([], ["doc2"], ["doc1", "doc2"]) is None


def test_attached_documents_are_included_with_explicit_scope():
    assert resolve_document_scope(
        ["doc1", "removed-doc"],
        ["doc2", "removed-doc"],
        ["doc1", "doc2"],
    ) == ["doc1", "doc2"]


def test_stale_selection_does_not_create_an_empty_filter():
    assert resolve_document_scope(["removed-doc"], [], ["doc1"]) is None


def test_hybrid_results_merge_adjacent_chunks_to_restore_split_evidence():
    passages = [
        SearchResult(
            chunk_id="doc_p4_c5",
            doc_id="doc",
            filename="paper.pdf",
            page_number=4,
            chunk_index=5,
            text="reporting using chatbots.",
            score=0.8,
        ),
        SearchResult(
            chunk_id="doc_p4_c4",
            doc_id="doc",
            filename="paper.pdf",
            page_number=4,
            chunk_index=4,
            text="31% of respondents",
            score=0.9,
        ),
    ]

    merged = _merge_adjacent_passages(passages)

    assert len(merged) == 1
    assert merged[0].chunk_id == "doc_p4_c4"
    assert merged[0].text == "31% of respondents reporting using chatbots."
    assert merged[0].page_number == 4
