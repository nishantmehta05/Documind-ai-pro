"""Tests for file-specific retrieval evaluation."""

from src.evaluation.benchmark import BenchmarkItem, BenchmarkRunner
from src.retrieval.vector_store import SearchResult


class FakeRetriever:
    def search_dense(self, query, top_k, filter_doc_ids=None):
        return [
            SearchResult(
                chunk_id="wrong-file",
                doc_id="other",
                filename="other.pdf",
                page_number=2,
                chunk_index=0,
                text="A matching page number in a different file.",
            ),
            SearchResult(
                chunk_id="expected-file",
                doc_id="expected",
                filename="paper.pdf",
                page_number=2,
                chunk_index=0,
                text="The expected page in the selected file.",
            ),
        ]

    def search_hybrid(self, query, top_k, filter_doc_ids=None):
        return [
            SearchResult(
                chunk_id="expected-file",
                doc_id="expected",
                filename="paper.pdf",
                page_number=2,
                chunk_index=0,
                text="The expected page in the selected file.",
            )
        ]


def test_benchmark_ignores_matching_pages_from_other_files():
    result = BenchmarkRunner(FakeRetriever()).run_benchmark(
        [
            BenchmarkItem(
                query="What is the finding?",
                expected_filename="paper.pdf",
                expected_pages=[2],
            )
        ],
        top_k=2,
    )

    assert result.dense_metrics.mean_recall_at_1 == 0.0
    assert result.dense_metrics.mean_recall_at_3 == 1.0
    assert result.hybrid_metrics.mean_recall_at_1 == 1.0
    assert result.query_details[0]["dense_retrieved_pages"] == [2, 2]
    assert result.query_details[0]["dense_retrieved_sources"] == [
        "other.pdf: page 2",
        "paper.pdf: page 2",
    ]
