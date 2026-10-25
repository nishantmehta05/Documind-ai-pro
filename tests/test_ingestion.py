"""Unit tests for PDF ingestion and chunking."""

import pytest
from pathlib import Path
from src.ingestion.pdf_processor import PDFProcessor, ProcessedDocument, PageContent
from src.ingestion.chunking import TextChunker, DocumentChunk

def test_empty_file_detection(tmp_path: Path):
    """Ensure 0-byte file raises ValueError."""
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")

    processor = PDFProcessor()
    with pytest.raises(ValueError, match="empty"):
        processor.process_file(empty_file)

def test_missing_file_detection(tmp_path: Path):
    """Ensure non-existent file raises FileNotFoundError."""
    missing = tmp_path / "does_not_exist.pdf"
    processor = PDFProcessor()
    with pytest.raises(FileNotFoundError):
        processor.process_file(missing)

def test_pdf_upload_validation():
    processor = PDFProcessor()
    valid_pdf = b"%PDF-1.7\nvalid pdf bytes"

    processor.validate_upload("research.pdf", valid_pdf, max_size_bytes=len(valid_pdf))

    with pytest.raises(ValueError, match="Only PDF"):
        processor.validate_upload("research.txt", valid_pdf, max_size_bytes=100)
    with pytest.raises(ValueError, match="empty"):
        processor.validate_upload("research.pdf", b"", max_size_bytes=100)
    with pytest.raises(ValueError, match="upload limit"):
        processor.validate_upload("research.pdf", valid_pdf, max_size_bytes=1)
    with pytest.raises(ValueError, match="signature"):
        processor.validate_upload("research.pdf", b"not a PDF", max_size_bytes=100)

def test_scanned_pdf_heuristic():
    """Verify scanned/image-only PDF detection when text is negligible."""
    doc = ProcessedDocument(
        doc_id="test1234",
        filename="scanned_receipt.pdf",
        file_path="/tmp/scanned_receipt.pdf",
        total_pages=3,
        pages=[
            PageContent(page_number=1, text="", char_count=0, has_text=False),
            PageContent(page_number=2, text="Scan", char_count=4, has_text=True),
            PageContent(page_number=3, text="", char_count=0, has_text=False),
        ],
        total_chars=4,
        is_scanned_or_empty=True,
        warning_message="Optical Character Recognition (OCR) is required"
    )
    assert doc.is_scanned_or_empty is True
    assert "OCR" in doc.warning_message

def test_chunking_with_overlap_and_stable_ids():
    """Verify chunker creates sliding window chunks and assigns deterministic IDs."""
    sample_text = (
        "Quantum computing utilizes qubits for computation. "
        "Unlike classical bits which can be either 0 or 1, qubits exist in superposition. "
        "Entanglement allows qubits to be correlated instantaneously across distances. "
        "Fault-tolerant quantum error correction is vital for scaling practical quantum systems."
    )
    doc = ProcessedDocument(
        doc_id="doc_abc",
        filename="quantum_intro.pdf",
        file_path="/data/quantum_intro.pdf",
        total_pages=1,
        pages=[
            PageContent(page_number=1, text=sample_text, char_count=len(sample_text), has_text=True)
        ],
        total_chars=len(sample_text),
        is_scanned_or_empty=False
    )

    chunker = TextChunker(chunk_size=120, chunk_overlap=30)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    for idx, c in enumerate(chunks):
        assert c.chunk_id == f"doc_abc_p1_c{idx}"
        assert c.filename == "quantum_intro.pdf"
        assert c.page_number == 1
        assert len(c.text) > 0
