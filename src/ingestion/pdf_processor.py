"""PDF Ingestion and Text Extraction with Quality Validation."""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

import pymupdf as fitz

logger = logging.getLogger(__name__)

class PageContent(BaseModel):
    """Extracted text and metadata for a single PDF page."""
    page_number: int = Field(..., description="1-indexed page number")
    text: str = Field(..., description="Cleaned textual content of the page")
    char_count: int = Field(default=0)
    has_text: bool = Field(default=True)

class ProcessedDocument(BaseModel):
    """Complete document representation with metadata and extraction status."""
    doc_id: str = Field(..., description="SHA-256 fingerprint of file contents")
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Absolute path on disk")
    total_pages: int = Field(..., description="Total number of pages")
    pages: List[PageContent] = Field(default_factory=list)
    total_chars: int = Field(default=0)
    is_scanned_or_empty: bool = Field(default=False)
    warning_message: Optional[str] = Field(default=None)

class PDFProcessor:
    """Robust PDF processor that handles corrupted files, scanned pages, and duplicates."""

    def __init__(self, min_chars_per_doc: int = 40):
        self.min_chars_per_doc = min_chars_per_doc

    @staticmethod
    def validate_upload(filename: str, file_bytes: bytes, max_size_bytes: int) -> None:
        """Reject unsupported, empty, oversized, or non-PDF upload content."""
        if Path(filename).suffix.lower() != ".pdf":
            raise ValueError("Only PDF files can be uploaded.")
        if not file_bytes:
            raise ValueError("The uploaded file is empty.")
        if len(file_bytes) > max_size_bytes:
            limit_mb = max_size_bytes / (1024 * 1024)
            raise ValueError(f"The PDF exceeds the {limit_mb:g} MB upload limit.")
        if b"%PDF-" not in file_bytes[:1024]:
            raise ValueError("The file does not have a valid PDF signature.")

    @staticmethod
    def compute_file_hash(file_bytes: bytes) -> str:
        """Calculate deterministic SHA-256 hash of document bytes."""
        return hashlib.sha256(file_bytes).hexdigest()[:16]

    def process_file(self, file_path: Path, original_filename: Optional[str] = None) -> ProcessedDocument:
        """Extract text page-by-page from a PDF file."""
        file_path = Path(file_path).resolve()
        filename = original_filename or file_path.name

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        file_size = file_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"File '{filename}' is empty (0 bytes).")

        # Read bytes to compute deterministic fingerprint
        with open(file_path, "rb") as f:
            content_bytes = f.read()

        doc_id = self.compute_file_hash(content_bytes)

        # Open PDF with PyMuPDF
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.error(f"Failed to parse PDF {filename}: {e}")
            raise ValueError(f"File '{filename}' is corrupted or not a valid PDF: {str(e)}")

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise ValueError(f"Document '{filename}' has no pages.")

        extracted_pages: List[PageContent] = []
        total_chars = 0

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            try:
                page = doc.load_page(page_idx)
                # Extract text using PyMuPDF's layout-preserving extractor
                page_text = page.get_text("text").strip()
                char_count = len(page_text)
                total_chars += char_count

                extracted_pages.append(
                    PageContent(
                        page_number=page_num,
                        text=page_text,
                        char_count=char_count,
                        has_text=char_count > 0
                    )
                )
            except Exception as e:
                logger.warning(f"Error reading page {page_num} of {filename}: {e}")
                extracted_pages.append(
                    PageContent(
                        page_number=page_num,
                        text="",
                        char_count=0,
                        has_text=False
                    )
                )

        doc.close()

        # Check for scanned or text-less PDF
        is_scanned = total_chars < self.min_chars_per_doc
        warning = None
        if is_scanned:
            warning = (
                f"Document '{filename}' contains almost no extractable text ({total_chars} characters across {total_pages} pages). "
                "It appears to be a scanned document or image-only PDF. Optical Character Recognition (OCR) is required to process this file."
            )
            logger.warning(warning)

        return ProcessedDocument(
            doc_id=doc_id,
            filename=filename,
            file_path=str(file_path),
            total_pages=total_pages,
            pages=extracted_pages,
            total_chars=total_chars,
            is_scanned_or_empty=is_scanned,
            warning_message=warning
        )
