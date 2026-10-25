"""Citation Extraction, Tracking, and Verification."""

import re
from html import escape
from typing import Any, Dict, List, Set
from pydantic import BaseModel, Field

from ..retrieval.vector_store import SearchResult

class Citation(BaseModel):
    """Specific page-grounded citation."""
    source_index: int = Field(..., description="Citation index number, e.g. 1 for [1]")
    doc_id: str
    filename: str
    page_number: int
    chunk_id: str
    excerpt: str = Field(..., description="Short supporting excerpt snippet")

    def display_tag(self) -> str:
        """Formatted citation tag."""
        return f"[{self.source_index}]"

    def markdown_reference(self) -> str:
        """Markdown formatted bibliography reference."""
        return f"**[{self.source_index}]** *{self.filename}* (Page {self.page_number})\n> \"{self.excerpt[:220]}...\""

class CitationTracker:
    """Manages passage evidence numbering and citation validation."""

    def __init__(self, passages: List[SearchResult]):
        self.passages = passages
        self.citations: List[Citation] = []
        self._build_citations()

    def _build_citations(self) -> None:
        """Assign sequential indices to retrieved evidence passages."""
        for idx, p in enumerate(self.passages, start=1):
            # Take representative excerpt
            clean_excerpt = " ".join(p.text.split())
            if len(clean_excerpt) > 240:
                clean_excerpt = clean_excerpt[:240].rsplit(" ", 1)[0] + "..."

            self.citations.append(
                Citation(
                    source_index=idx,
                    doc_id=p.doc_id,
                    filename=p.filename,
                    page_number=p.page_number,
                    chunk_id=p.chunk_id,
                    excerpt=clean_excerpt
                )
            )

    def format_context_for_prompt(self) -> str:
        """Format evidence passages for inclusion in LLM prompt with strict source markers."""
        if not self.passages:
            return "No relevant passages found."

        formatted_blocks = []
        for c, p in zip(self.citations, self.passages):
            safe_filename = escape(" ".join(p.filename.split()))
            safe_text = escape(p.text.strip())
            block = (
                f"[Source {c.source_index} | Document: {safe_filename} | Page: {c.page_number}]\n"
                f"{safe_text}\n"
                f"[End of Source {c.source_index}]"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)

    def verify_answer_citations(self, answer_text: str) -> Dict[str, Any]:
        """Verify whether generated citations correspond to provided sources."""
        # Find all [Source X] or [X] patterns
        found_indices = set(map(int, re.findall(r"\[(?:Source\s*)?(\d+)\]", answer_text)))
        valid_indices = {c.source_index for c in self.citations}

        unsupported_indices = found_indices - valid_indices
        used_citations = [c for c in self.citations if c.source_index in found_indices]

        has_hallucinated_citations = len(unsupported_indices) > 0
        return {
            "cited_source_indices": sorted(list(found_indices)),
            "valid_indices": sorted(list(valid_indices)),
            "hallucinated_indices": sorted(list(unsupported_indices)),
            "is_valid": not has_hallucinated_citations,
            "used_citations": used_citations
        }
