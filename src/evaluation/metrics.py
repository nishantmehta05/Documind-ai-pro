"""Retrieval and Citation Evaluation Metrics."""

from typing import List, Optional
from pydantic import BaseModel, Field

class EvaluationMetrics(BaseModel):
    """Aggregate evaluation scores for a retrieval experiment."""
    method_name: str
    num_queries: int = 0
    mean_recall_at_1: float = 0.0
    mean_recall_at_3: float = 0.0
    mean_recall_at_5: float = 0.0
    mean_mrr: float = 0.0
    mean_latency_ms: float = 0.0
    citation_precision: float = 1.0

def calculate_recall_at_k(
    retrieved_pages: List[Optional[int]],
    expected_pages: List[int],
    k: int,
) -> float:
    """Calculate Recall@k: fraction of expected ground-truth pages found in top-k results."""
    if not expected_pages:
        return 1.0
    top_k_pages = {page for page in retrieved_pages[:k] if page is not None}
    expected_set = set(expected_pages)
    hits = top_k_pages.intersection(expected_set)
    return len(hits) / len(expected_set)

def calculate_mrr(
    retrieved_pages: List[Optional[int]],
    expected_pages: List[int],
) -> float:
    """Calculate Mean Reciprocal Rank (MRR): 1 / rank of first relevant page."""
    if not expected_pages:
        return 1.0
    expected_set = set(expected_pages)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected_set:
            return 1.0 / rank
    return 0.0

def calculate_citation_precision(cited_indices: List[int], valid_indices: List[int]) -> float:
    """Fraction of citations made by the LLM that correspond to actual valid sources."""
    if not cited_indices:
        return 1.0
    valid_set = set(valid_indices)
    valid_cited = [c for c in cited_indices if c in valid_set]
    return len(valid_cited) / len(cited_indices)
