"""Empirical RAG Benchmark Runner comparing Dense vs. Hybrid Retrieval."""

import time
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from .metrics import calculate_recall_at_k, calculate_mrr, EvaluationMetrics
from ..retrieval.hybrid_search import HybridRetriever

logger = logging.getLogger(__name__)

class BenchmarkItem(BaseModel):
    """Ground truth benchmark query definition."""
    query: str
    expected_filename: Optional[str] = None
    expected_doc_id: Optional[str] = None
    expected_pages: List[int] = Field(..., description="1-indexed ground truth pages containing answer")
    category: str = Field(default="factual")

class BenchmarkResult(BaseModel):
    """Result of running benchmark across methods."""
    dense_metrics: EvaluationMetrics
    hybrid_metrics: EvaluationMetrics
    query_details: List[Dict[str, Any]] = Field(default_factory=list)

class BenchmarkRunner:
    """Executes empirical evaluations on the local dataset without fabricating data."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def run_benchmark(
        self,
        items: List[BenchmarkItem],
        top_k: int = 5,
        filter_doc_ids: Optional[List[str]] = None,
    ) -> BenchmarkResult:
        """Run dense vs hybrid retrieval on all benchmark items and compute metrics."""
        if not items:
            raise ValueError("No benchmark items provided for evaluation.")

        dense_r1, dense_r3, dense_r5, dense_mrrs, dense_latencies = [], [], [], [], []
        hybrid_r1, hybrid_r3, hybrid_r5, hybrid_mrrs, hybrid_latencies = [], [], [], [], []
        query_details = []

        for item in items:
            # 1. Semantic (Dense) Retrieval
            t0 = time.perf_counter()
            dense_results = self.retriever.search_dense(
                item.query,
                top_k=top_k,
                filter_doc_ids=filter_doc_ids,
            )
            t_dense = (time.perf_counter() - t0) * 1000

            dense_scored_pages = [
                result.page_number
                if (
                    (item.expected_doc_id is None or result.doc_id == item.expected_doc_id)
                    and (item.expected_filename is None or result.filename == item.expected_filename)
                )
                else None
                for result in dense_results
            ]
            dense_pages = [result.page_number for result in dense_results]
            dense_r1.append(calculate_recall_at_k(dense_scored_pages, item.expected_pages, k=1))
            dense_r3.append(calculate_recall_at_k(dense_scored_pages, item.expected_pages, k=3))
            dense_r5.append(calculate_recall_at_k(dense_scored_pages, item.expected_pages, k=5))
            dense_mrrs.append(calculate_mrr(dense_scored_pages, item.expected_pages))
            dense_latencies.append(t_dense)

            # 2. Hybrid (Dense + BM25 RRF) Retrieval
            t0 = time.perf_counter()
            hybrid_results = self.retriever.search_hybrid(
                item.query,
                top_k=top_k,
                filter_doc_ids=filter_doc_ids,
            )
            t_hybrid = (time.perf_counter() - t0) * 1000

            hybrid_scored_pages = [
                result.page_number
                if (
                    (item.expected_doc_id is None or result.doc_id == item.expected_doc_id)
                    and (item.expected_filename is None or result.filename == item.expected_filename)
                )
                else None
                for result in hybrid_results
            ]
            hybrid_pages = [result.page_number for result in hybrid_results]
            hybrid_r1.append(calculate_recall_at_k(hybrid_scored_pages, item.expected_pages, k=1))
            hybrid_r3.append(calculate_recall_at_k(hybrid_scored_pages, item.expected_pages, k=3))
            hybrid_r5.append(calculate_recall_at_k(hybrid_scored_pages, item.expected_pages, k=5))
            hybrid_mrrs.append(calculate_mrr(hybrid_scored_pages, item.expected_pages))
            hybrid_latencies.append(t_hybrid)

            query_details.append({
                "query": item.query,
                "expected_filename": item.expected_filename,
                "expected_pages": item.expected_pages,
                "dense_retrieved_pages": dense_pages,
                "dense_retrieved_sources": [
                    f"{result.filename}: page {result.page_number}" for result in dense_results
                ],
                "dense_mrr": dense_mrrs[-1],
                "hybrid_retrieved_pages": hybrid_pages,
                "hybrid_retrieved_sources": [
                    f"{result.filename}: page {result.page_number}" for result in hybrid_results
                ],
                "hybrid_mrr": hybrid_mrrs[-1],
            })

        n = len(items)
        dense_eval = EvaluationMetrics(
            method_name="Dense Semantic Search",
            num_queries=n,
            mean_recall_at_1=sum(dense_r1) / n,
            mean_recall_at_3=sum(dense_r3) / n,
            mean_recall_at_5=sum(dense_r5) / n,
            mean_mrr=sum(dense_mrrs) / n,
            mean_latency_ms=sum(dense_latencies) / n,
        )

        hybrid_eval = EvaluationMetrics(
            method_name="Hybrid Search (Dense + BM25 RRF)",
            num_queries=n,
            mean_recall_at_1=sum(hybrid_r1) / n,
            mean_recall_at_3=sum(hybrid_r3) / n,
            mean_recall_at_5=sum(hybrid_r5) / n,
            mean_mrr=sum(hybrid_mrrs) / n,
            mean_latency_ms=sum(hybrid_latencies) / n,
        )

        return BenchmarkResult(
            dense_metrics=dense_eval,
            hybrid_metrics=hybrid_eval,
            query_details=query_details
        )
