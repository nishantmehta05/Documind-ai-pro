"""State Machine Research Workflow with Bounded Loop Control."""

import logging
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field

from .research_planner import ResearchPlanner, ResearchPlan
from .report_writer import ReportWriter, ResearchReport
from ..retrieval.hybrid_search import HybridRetriever
from ..retrieval.vector_store import SearchResult
from ..config import config

logger = logging.getLogger(__name__)

class ResearchState(BaseModel):
    """Tracks state across the research workflow execution."""
    topic: str
    plan: Optional[ResearchPlan] = None
    step_count: int = 0
    max_steps: int = 3
    retrieved_passages: List[SearchResult] = Field(default_factory=list)
    is_sufficient: bool = False
    status_log: List[str] = Field(default_factory=list)
    final_report: Optional[ResearchReport] = None

class ResearchWorkflow:
    """Four-phase research workflow with bounded loops and progress telemetry."""

    def __init__(
        self,
        retriever: HybridRetriever,
        planner: Optional[ResearchPlanner] = None,
        writer: Optional[ReportWriter] = None,
        max_steps: int = 3
    ):
        self.retriever = retriever
        self.planner = planner or ResearchPlanner()
        self.writer = writer or ReportWriter()
        self.max_steps = max_steps

    def execute(
        self,
        topic: str,
        filter_doc_ids: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ) -> ResearchState:
        """Execute the bounded research pipeline."""
        state = ResearchState(topic=topic, max_steps=self.max_steps)

        def notify(stage: str, msg: str):
            state.status_log.append(f"[{stage}] {msg}")
            logger.info(f"[{stage}] {msg}")
            if progress_callback:
                progress_callback(stage, msg)

        # ---------------------------------------------------------------------
        # Step 1: Question Planning
        # ---------------------------------------------------------------------
        notify("Planner", f"Decomposing research query: '{topic}'...")
        state.plan = self.planner.plan(topic)
        state.step_count += 1
        if state.plan.used_heuristic_fallback:
            notify(
                "Planner",
                "Ollama did not return a usable plan; continuing with a local heuristic fallback.",
            )
        else:
            notify(
                "Planner",
                f"Formulated {len(state.plan.sub_questions)} targeted sub-queries.",
            )

        # ---------------------------------------------------------------------
        # Step 2: Multi-Hop Evidence Retrieval
        # ---------------------------------------------------------------------
        notify("Retriever", "Executing hybrid retrieval across sub-queries...")
        seen_chunk_ids = set()
        accumulated: List[SearchResult] = []

        all_queries = [topic] + state.plan.sub_questions
        for q in all_queries:
            results = self.retriever.search_hybrid(
                query=q,
                top_k=3,
                filter_doc_ids=filter_doc_ids
            )
            for res in results:
                if res.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(res.chunk_id)
                    accumulated.append(res)

        state.retrieved_passages = accumulated
        notify("Retriever", f"Gathered {len(accumulated)} unique evidence passages.")

        # ---------------------------------------------------------------------
        # Step 3: Evidence Sufficiency Checker (Bounded Loop Guard)
        # ---------------------------------------------------------------------
        notify("Sufficiency Checker", "Evaluating retrieved evidence coverage...")
        state.step_count += 1

        if len(accumulated) >= 2:
            state.is_sufficient = True
            notify("Sufficiency Checker", "Evidence threshold satisfied for synthesis.")
        elif len(accumulated) == 1:
            state.is_sufficient = True
            notify("Sufficiency Checker", "Limited evidence found. Proceeding with caution.")
        else:
            state.is_sufficient = False
            notify("Sufficiency Checker", "No relevant passages found in document corpus.")

        # ---------------------------------------------------------------------
        # Step 4: Report Synthesis
        # ---------------------------------------------------------------------
        if state.is_sufficient and state.retrieved_passages:
            notify("Report Writer", "Synthesizing executive research briefing with citations...")
            state.final_report = self.writer.write_report(topic, state.retrieved_passages)
            notify("Report Writer", "Research briefing successfully generated.")
        else:
            notify("Report Writer", "Insufficient evidence to generate a grounded briefing.")
            empty_markdown = (
                f"# Research Brief: {topic}\n\n"
                "**Result**: The uploaded documents do not provide enough information to answer "
                "or research this topic. Please upload documents containing relevant data."
            )
            state.final_report = ResearchReport(
                topic=topic,
                summary="Insufficient evidence found in corpus.",
                detailed_findings=empty_markdown,
                gaps_and_limitations="Corpus lacks matching source material.",
                citations=[],
                markdown_output=empty_markdown
            )

        return state
