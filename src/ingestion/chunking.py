"""Document Chunking with Semantic Boundaries and Stable IDs."""

import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from .pdf_processor import ProcessedDocument

class DocumentChunk(BaseModel):
    """A discrete passage of text with page citation metadata."""
    chunk_id: str = Field(..., description="Unique deterministic identifier: {doc_id}_p{page}_c{index}")
    doc_id: str = Field(..., description="Source document hash")
    filename: str = Field(..., description="Source document filename")
    page_number: int = Field(..., description="Page number of the chunk (1-indexed)")
    chunk_index: int = Field(..., description="Sequential index within the page")
    text: str = Field(..., description="Cleaned chunk text content")
    char_length: int = Field(..., description="Length of chunk text")
    document_page_count: int = Field(default=0, description="Total pages in the source PDF")

    def to_metadata(self) -> Dict[str, Any]:
        """Convert metadata to flat dictionary for vector database storage."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page_number": int(self.page_number),
            "chunk_index": int(self.chunk_index),
            "char_length": int(self.char_length),
            "document_page_count": int(self.document_page_count),
        }

class TextChunker:
    """Configurable text chunker with sliding window overlap and deterministic IDs."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text_with_overlap(self, text: str) -> List[str]:
        """Split raw text into chunks respecting chunk_size and chunk_overlap."""
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # If not at the very end of the text, try to snap to nearest sentence or space boundary
            if end < len(text):
                # Search for sentence break in the last 20% of the chunk
                tail_window = chunk[int(len(chunk) * 0.7):]
                break_match = re.search(r"(\. |\n\n|\n)", tail_window)
                if break_match:
                    snap_offset = int(len(chunk) * 0.7) + break_match.end()
                    chunk = text[start:start + snap_offset]
                    start = start + snap_offset
                else:
                    start += step
            else:
                start += step

            cleaned_chunk = chunk.strip()
            if cleaned_chunk:
                chunks.append(cleaned_chunk)

        return chunks

    def chunk_document(self, doc: ProcessedDocument) -> List[DocumentChunk]:
        """Chunk all pages of a processed document and assign stable chunk IDs."""
        all_chunks: List[DocumentChunk] = []

        for page in doc.pages:
            if not page.has_text or not page.text.strip():
                continue

            page_chunks = self._split_text_with_overlap(page.text)
            for idx, text_block in enumerate(page_chunks):
                chunk_id = f"{doc.doc_id}_p{page.page_number}_c{idx}"
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        filename=doc.filename,
                        page_number=page.page_number,
                        chunk_index=idx,
                        text=text_block,
                        char_length=len(text_block),
                        document_page_count=doc.total_pages,
                    )
                )

        return all_chunks
