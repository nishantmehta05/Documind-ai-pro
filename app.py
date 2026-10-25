"""DocuMind AI Pro - private, evidence-grounded document research."""

import logging
import time
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from src.agents.workflow import ResearchState, ResearchWorkflow
from src.config import config
from src.evaluation.benchmark import BenchmarkItem, BenchmarkRunner
from src.ingestion.chunking import DocumentChunk, TextChunker
from src.ingestion.pdf_processor import PDFProcessor, ProcessedDocument
from src.rag.answer_generator import GroundedAnswerGenerator, RAGResponse
from src.retrieval.bm25_search import BM25Index
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.hybrid_search import HybridRetriever, resolve_document_scope
from src.retrieval.vector_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DocuMindApp")

st.set_page_config(
    page_title="DocuMind AI Pro",
    page_icon=":material/auto_awesome:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.html(f"<style>{(Path(__file__).parent / 'src' / 'liquid_glass.css').read_text(encoding='utf-8')}</style>")


@st.cache_resource
def get_pdf_processor() -> PDFProcessor:
    return PDFProcessor()


@st.cache_resource
def get_text_chunker() -> TextChunker:
    return TextChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)


@st.cache_resource
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@st.cache_resource
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore()


@st.cache_resource
def get_bm25_index() -> BM25Index:
    return BM25Index()


@st.cache_resource
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(
        vector_store=get_vector_store(),
        embedding_service=get_embedding_service(),
        bm25_index=get_bm25_index(),
        rrf_k=config.rrf_k,
    )


@st.cache_resource
def get_answer_generator() -> GroundedAnswerGenerator:
    return GroundedAnswerGenerator()


@st.cache_resource
def get_research_workflow() -> ResearchWorkflow:
    return ResearchWorkflow(retriever=get_hybrid_retriever())


@st.cache_data(ttl=20)
def get_ollama_status() -> Tuple[bool, bool]:
    return get_answer_generator().check_status()


def remove_indexed_document(doc_id: str, filename: str) -> None:
    get_vector_store().delete_document(doc_id)
    get_hybrid_retriever().refresh_bm25()
    st.session_state["selected_doc_ids"] = [
        selected_id
        for selected_id in st.session_state.get("selected_doc_ids", [])
        if selected_id != doc_id
    ]
    st.session_state["last_ingestion_filenames"] = [
        stored_filename
        for stored_filename in st.session_state.get("last_ingestion_filenames", [])
        if stored_filename != filename
    ]
    st.session_state["ingestion_status"].pop(filename, None)
    st.toast(f"Removed {filename} from the index.")


def reset_document_index() -> None:
    get_vector_store().clear()
    get_hybrid_retriever().refresh_bm25()
    st.session_state["ingestion_status"] = {}
    st.session_state["chat_history"] = []
    st.session_state["selected_doc_ids"] = []
    st.session_state["chat_attached_doc_ids"] = []
    st.session_state["last_ingestion_filenames"] = []
    st.toast("Local document index reset.")


def queue_suggested_prompt(prompts: Dict[str, str]) -> None:
    selected_prompt = st.session_state.get("suggested_prompt")
    if selected_prompt in prompts:
        st.session_state["pending_chat_prompt"] = prompts[selected_prompt]
        st.session_state["suggested_prompt"] = None


def _select_view(view: str) -> None:
    st.session_state["active_view"] = view
    st.session_state["top_navigation"] = {
        "Documents": "Sources",
        "Evaluation": "Insights",
    }.get(view, "Home")


def _friendly_upload_name(filename: str) -> str:
    return Path(filename).name.strip() or "uploaded PDF"


def _index_uploaded_files(uploaded_files: List[Any]) -> List[str]:
    if not uploaded_files:
        return []

    processor = get_pdf_processor()
    chunker = get_text_chunker()
    embedder = get_embedding_service()
    known_documents = {
        document["doc_id"]: document["filename"]
        for document in get_vector_store().get_indexed_documents_stats()
    }
    attached_doc_ids: List[str] = []
    indexed_new_document = False
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    batch_start = time.perf_counter()

    for index, uploaded_file in enumerate(uploaded_files):
        filename = _friendly_upload_name(uploaded_file.name)
        document_start = time.perf_counter()
        save_path: Optional[Path] = None
        created_file = False
        index_completed = False
        try:
            content = uploaded_file.getvalue()
            processor.validate_upload(
                filename,
                content,
                int(st.get_option("server.maxUploadSize")) * 1024 * 1024,
            )
            doc_id = processor.compute_file_hash(content)
            if doc_id in known_documents:
                existing_name = known_documents[doc_id]
                st.session_state.ingestion_status[filename] = {
                    "status": "duplicate",
                    "message": f"Already indexed as {existing_name}; reusing its index.",
                    "doc_id": doc_id,
                }
                attached_doc_ids.append(doc_id)
                status_text.write(f"Already indexed · {filename}")
                progress_bar.progress((index + 1) / len(uploaded_files))
                continue

            status_text.write(f"Processing {index + 1} of {len(uploaded_files)} · {filename}")
            save_path = config.upload_dir / f"{doc_id}.pdf"
            if not save_path.exists():
                created_file = True
                save_path.write_bytes(content)

            document: ProcessedDocument = processor.process_file(
                save_path,
                original_filename=filename,
            )
            extraction_ms = (time.perf_counter() - document_start) * 1000
            if document.is_scanned_or_empty:
                message = document.warning_message or "No searchable PDF text was found."
                st.session_state.ingestion_status[filename] = {
                    "status": "warning",
                    "message": message,
                    "pages": document.total_pages,
                    "chunks": 0,
                }
            else:
                chunking_start = time.perf_counter()
                chunks: List[DocumentChunk] = chunker.chunk_document(document)
                chunking_ms = (time.perf_counter() - chunking_start) * 1000
                if not chunks:
                    st.session_state.ingestion_status[filename] = {
                        "status": "warning",
                        "message": "No indexable text was found.",
                        "pages": document.total_pages,
                        "chunks": 0,
                    }
                else:
                    chunking_ms = (time.perf_counter() - chunking_start) * 1000
                    embedding_start = time.perf_counter()
                    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
                    embedding_ms = (time.perf_counter() - embedding_start) * 1000
                    indexing_start = time.perf_counter()
                    written_chunks = get_vector_store().add_chunks(chunks, embeddings)
                    if written_chunks != len(chunks):
                        raise RuntimeError(
                            f"Expected to index {len(chunks)} chunks, wrote {written_chunks}."
                        )
                    index_completed = True
                    indexed_doc_ids = {
                        item["doc_id"]
                        for item in get_vector_store().get_indexed_documents_stats()
                    }
                    if doc_id not in indexed_doc_ids:
                        raise RuntimeError(
                            "The PDF chunks were written, but their document ID was not "
                            "visible in the index."
                        )
                    indexing_ms = (time.perf_counter() - indexing_start) * 1000
                    indexed_new_document = True
                    known_documents[doc_id] = filename
                    attached_doc_ids.append(doc_id)
                    st.session_state.ingestion_status[filename] = {
                        "status": "success",
                        "pages": document.total_pages,
                        "chunks": len(chunks),
                        "doc_id": document.doc_id,
                    }
                    st.toast(f"Indexed {filename} · {len(chunks)} chunks.")
                    logger.info(
                        "Indexed PDF %s (doc_id=%s, chunks=%d): extraction=%.0fms chunking=%.0fms "
                        "embedding=%.0fms vector_write=%.0fms",
                        filename,
                        doc_id,
                        written_chunks,
                        extraction_ms,
                        chunking_ms,
                        embedding_ms,
                        indexing_ms,
                    )
        except Exception as error:
            logger.exception("Failed to process uploaded PDF %s", filename)
            if created_file and not index_completed and save_path is not None:
                try:
                    save_path.unlink(missing_ok=True)
                except OSError:
                    logger.exception("Could not remove failed upload %s", save_path)
            message = (
                str(error)
                if isinstance(error, (ValueError, FileNotFoundError))
                else "Could not validate, read, or index this PDF. Check the application log."
            )
            st.session_state.ingestion_status[filename] = {
                "status": "error",
                "message": message,
            }
            st.error(f"{filename}: {message}")

        progress_bar.progress((index + 1) / len(uploaded_files))

    if indexed_new_document:
        get_hybrid_retriever().refresh_bm25()
    st.session_state["last_ingestion_filenames"] = [
        _friendly_upload_name(uploaded_file.name) for uploaded_file in uploaded_files
    ]
    status_text.write("PDF processing finished.")
    logger.info(
        "Processed %d uploaded PDF(s) in %.0fms",
        len(uploaded_files),
        (time.perf_counter() - batch_start) * 1000,
    )
    return attached_doc_ids


def _page_count_label(document: Dict[str, Any]) -> str:
    count = document["page_count"]
    suffix = "page" if count == 1 else "pages"
    if document.get("page_count_is_total", False):
        return f"{count} {suffix}"
    return f"{count} indexed {suffix}"


def _render_citations(citations: List[Dict[str, Any]], expanded: bool = False) -> None:
    if not citations:
        return
    with st.expander(
        f"Sources · {len(citations)} cited passage(s)",
        expanded=expanded,
        icon=":material/format_quote:",
    ):
        for citation in citations:
            st.markdown(
                f"**[{citation['source_index']}] Source · Page {citation['page_number']}**"
            )
            st.text(citation["filename"])
            st.caption(citation["excerpt"])
            st.divider()


for state_key, initial_value in (
    ("chat_history", []),
    ("ingestion_status", {}),
    ("active_view", "Research Chat"),
    ("selected_doc_ids", []),
    ("chat_attached_doc_ids", []),
):
    st.session_state.setdefault(state_key, initial_value)

views = ["Research Chat", "Documents", "Evaluation", "Advanced Tools"]
if st.session_state.active_view not in views:
    st.session_state.active_view = "Research Chat"

generator = get_answer_generator()
ollama_online, configured_model_available = get_ollama_status()
try:
    vector_store = get_vector_store()
    retriever = get_hybrid_retriever()
    indexed_docs = vector_store.get_indexed_documents_stats()
except Exception:
    logger.exception("Could not initialize the local document index")
    st.error(
        "The local document index could not be opened. Check the application log; "
        "your stored PDFs and index have not been reset."
    )
    st.stop()

doc_names = {doc["doc_id"]: doc["filename"] for doc in indexed_docs}
available_doc_ids = set(doc_names)
st.session_state["chat_attached_doc_ids"] = [
    doc_id
    for doc_id in st.session_state["chat_attached_doc_ids"]
    if doc_id in available_doc_ids
]
pending_selected_doc_ids = st.session_state.pop("pending_selected_doc_ids", [])
if pending_selected_doc_ids:
    current_selection = st.session_state.get("selected_doc_ids", [])
    available_doc_ids = set(doc_names)
    st.session_state["selected_doc_ids"] = list(
        dict.fromkeys(
            doc_id
            for doc_id in [*current_selection, *pending_selected_doc_ids]
            if doc_id in available_doc_ids
        )
    )

nav_by_view = {
    "Research Chat": "Home",
    "Documents": "Sources",
    "Evaluation": "Insights",
    "Advanced Tools": "Home",
}
if "top_navigation" not in st.session_state:
    st.session_state["top_navigation"] = nav_by_view.get(
        st.session_state.active_view,
        "Home",
    )
with st.container(key="app-header"):
    brand_column, navigation_column, status_column, menu_column = st.columns(
        [2.15, 4.8, 1.75, 1.05],
        vertical_alignment="center",
        gap="small",
    )
    with brand_column:
        st.markdown(
            """
            <div class="brand-lockup">
                <span class="brand-mark" aria-hidden="true">D</span>
                <span><strong>DocuMind <small class="brand-ai">AI</small></strong><small>Your private document workspace</small></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with navigation_column:
        selected_navigation = st.pills(
            "Workspace navigation",
            options=["Home", "Sources", "Library", "Chat", "Insights"],
            key="top_navigation",
            label_visibility="collapsed",
            width="stretch",
        )
    with status_column:
        if configured_model_available:
            ready_label = "Local AI ready" if config.llm_provider == "ollama" else "Hosted AI ready"
            st.markdown(
                '<div class="model-status model-status--ready" role="status">'
                f'<span class="model-status-dot" aria-hidden="true"></span>{ready_label}</div>',
                unsafe_allow_html=True,
            )
    with menu_column:
        selected_view = st.menu_button(
            "Menu",
            options=views,
            key="workspace_menu",
            format_func=lambda view: {
                "Research Chat": "Research chat",
                "Documents": "Documents",
                "Evaluation": "Evaluation",
                "Advanced Tools": "Advanced tools",
            }[view],
            icon=":material/menu:",
            width="content",
        )
    if selected_view:
        st.session_state.active_view = selected_view
        st.session_state.top_navigation = nav_by_view.get(selected_view, "Home")
        st.rerun()
    nav_to_view = {
        "Home": "Research Chat",
        "Sources": "Documents",
        "Library": "Documents",
        "Chat": "Research Chat",
        "Insights": "Evaluation",
    }
    if selected_navigation and nav_to_view[selected_navigation] != st.session_state.active_view:
        st.session_state.active_view = nav_to_view[selected_navigation]
        st.rerun()

active_view = st.session_state.active_view
selected_doc_ids: List[str] = st.session_state.get("selected_doc_ids", [])
if active_view == "Documents":
    st.markdown('<div class="eyebrow">YOUR SOURCE LIBRARY</div>', unsafe_allow_html=True)
    st.title("Documents")
    st.markdown("Upload PDFs, index their text locally, and choose which sources to search.")

    with st.container(border=True):
        st.markdown("#### Add PDFs")
        st.caption("Choose one or more readable PDFs, then index them for cited research.")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="PDF text is extracted and indexed on this device.",
            key="document_upload",
        )
        index_clicked = st.button(
            "Index selected PDFs",
            icon=":material/upload_file:",
            type="primary",
            width="stretch",
            disabled=not uploaded_files,
            key="index-selected-pdfs",
        )

    if index_clicked and uploaded_files:
        attached_doc_ids = _index_uploaded_files(uploaded_files)
        if selected_doc_ids:
            st.session_state["pending_selected_doc_ids"] = list(
                dict.fromkeys([*selected_doc_ids, *attached_doc_ids])
            )
        st.rerun()

    for filename, status in st.session_state.ingestion_status.items():
        if status["status"] == "success":
            st.success(
                f"{filename} indexed · {status['pages']} "
                f"{'page' if status['pages'] == 1 else 'pages'} · {status['chunks']} chunks."
            )
        elif status["status"] == "warning":
            st.warning(f"{filename}: {status['message']}")
        elif status["status"] == "duplicate":
            st.info(f"{filename}: {status['message']}")
        else:
            st.error(f"{filename}: {status['message']}")

    st.markdown('<div class="section-heading">Indexed library</div>', unsafe_allow_html=True)
    indexed_docs = vector_store.get_indexed_documents_stats()
    if indexed_docs:
        for doc in indexed_docs:
            with st.container(border=True):
                status_column, action_column = st.columns([4, 1], vertical_alignment="center")
                with status_column:
                    if doc["doc_id"] in selected_doc_ids:
                        st.markdown(
                            '<span class="scope-status">In current scope</span>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.badge("Indexed", icon=":material/check_circle:", color="gray")
                    st.text(doc["filename"])
                    st.caption(
                        f"Indexed · {_page_count_label(doc)} · {doc['chunk_count']} chunks"
                    )
                with action_column:
                    if st.button(
                        "Remove",
                        type="secondary",
                        key=f"remove-{doc['doc_id']}",
                        on_click=remove_indexed_document,
                        args=(doc["doc_id"], doc["filename"]),
                    ):
                        st.rerun()
    else:
        st.info("No documents are indexed yet. Upload a PDF above to create your local library.")

    with st.expander("PDF preview"):
        st.caption(
            "An in-app PDF page preview is not implemented. Open the original PDF on your device; "
            "answers include cited excerpts and page numbers."
        )

elif active_view == "Research Chat":
    if not st.session_state.chat_history:
        with st.container(key="welcome-hero"):
            st.markdown('<div class="hero-kicker">AI FOR YOUR DOCUMENTS</div>', unsafe_allow_html=True)
            st.markdown(
                '<h1><span class="hero-title-line hero-title-line--primary">Your documents.</span>'
                '<br><span class="hero-title-line hero-title-line--accent">Better understood.</span></h1>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="hero-subtitle">Ask questions, explore ideas, and discover insights from your PDFs'
                + (
                    '<br>with the power of local AI — private, on-device and evidence-grounded.'
                    if config.llm_provider == "ollama"
                    else '<br>with evidence-grounded answers from your configured hosted model.'
                )
                + "</div>",
                unsafe_allow_html=True,
            )

    recent_uploads = st.session_state.get("last_ingestion_filenames", [])
    if recent_uploads:
        with st.expander("Recent PDF uploads", expanded=False):
            for filename in recent_uploads:
                status = st.session_state.ingestion_status.get(filename, {})
                if status.get("status") == "success":
                    st.success(
                        f"{filename} indexed · {status['pages']} "
                        f"{'page' if status['pages'] == 1 else 'pages'} · "
                        f"{status['chunks']} chunks."
                    )
                elif status.get("status") == "duplicate":
                    st.info(f"{filename}: {status['message']}")
                elif status.get("status") == "warning":
                    st.warning(f"{filename}: {status['message']}")
                elif status.get("status") == "error":
                    st.error(f"{filename}: {status['message']}")

    for message in st.session_state.chat_history:
        with st.chat_message(
            message["role"],
            avatar=":material/auto_awesome:" if message["role"] == "assistant" else None,
        ):
            if message["role"] == "assistant" and message.get("is_error"):
                st.error(message["content"])
            elif message["role"] == "assistant" and message.get("insufficient_evidence"):
                st.info(message["content"])
            else:
                st.markdown(message["content"])
            for attachment in message.get("attachments", []):
                st.caption(
                    f"Attached · {attachment['filename']} · {attachment['size_mb']:.2f} MB"
                )
            if message["role"] == "assistant" and message.get("retrieval_ms") is not None:
                with st.expander("Response diagnostics"):
                    st.caption(f"Retrieval · {message['retrieval_ms']:.0f} ms")
                    st.caption(f"Generation · {message['generation_ms'] / 1000:.1f} s")
            _render_citations(message.get("citations", []))

    with st.container(key="chat-composer"):
        submission = st.chat_input(
            "Ask about your PDFs or attach one to get started...",
            accept_file="multiple",
            file_type=["pdf"],
            key="research_chat_input",
        )

    suggestion_prompts = {
        "Summarize a document": "Summarize the key ideas across my sources.",
        "Explain a concept": "Explain the main ideas in my sources in simple terms.",
        "Find supporting evidence": "Find the strongest evidence in my sources for their main claims.",
        "Analyze key insights": "Identify and analyze the key insights across my sources.",
    }
    if not st.session_state.chat_history:
        with st.container(key="suggestion-row"):
            st.pills(
                "Suggested questions",
                options=list(suggestion_prompts),
                key="suggested_prompt",
                on_change=queue_suggested_prompt,
                args=(suggestion_prompts,),
                label_visibility="collapsed",
                width="stretch",
            )

        with st.container(key="recent-documents-panel"):
            recent_column, view_all_column = st.columns([5, 1], vertical_alignment="center")
            with recent_column:
                st.markdown(
                    '<div class="recent-heading"><span class="recent-heading-icon">◷</span>'
                    '<span><strong>Recent documents</strong>'
                    '<small>Pick up where you left off</small></span></div>',
                    unsafe_allow_html=True,
                )
            with view_all_column:
                st.button(
                    "View all →",
                    key="recent-documents-view-all",
                    on_click=_select_view,
                    args=("Documents",),
                    width="stretch",
                )

            recent_names = st.session_state.get("last_ingestion_filenames", [])
            recent_documents = [
                document
                for filename in reversed(recent_names)
                for document in indexed_docs
                if document["filename"] == filename
            ]
            recent_ids = {document["doc_id"] for document in recent_documents}
            recent_documents.extend(
                document for document in indexed_docs if document["doc_id"] not in recent_ids
            )
            if recent_documents:
                for start in range(0, min(len(recent_documents), 4), 4):
                    document_columns = st.columns(4, gap="small")
                    for column, document in zip(document_columns, recent_documents[start:start + 4]):
                        with column:
                            with st.container(key=f"recent-document-{document['doc_id']}"):
                                st.markdown(
                                    '<div class="recent-document-icon">'
                                    '<span aria-hidden="true"></span>'
                                    '</div>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    f"<div class='recent-document-name'>{escape(document['filename'])}</div>",
                                    unsafe_allow_html=True,
                                )
                                st.caption(
                                    f"{_page_count_label(document)} · "
                                    f"{document['chunk_count']} indexed sections"
                                )
            else:
                st.markdown(
                    '<div class="recent-empty">'
                    '<strong>Your document library is ready</strong>'
                    '<span>Attach a PDF in the composer below to begin.</span></div>',
                    unsafe_allow_html=True,
                )

        ai_feature_title = "Local AI answers" if config.llm_provider == "ollama" else "Hosted AI answers"
        ai_feature_detail = (
            "Powered by your configured model"
            if config.llm_provider == "ollama"
            else "Using your configured endpoint"
        )
        st.markdown(
            f"""
            <div class="feature-highlights">
              <div class="feature-item"><span class="feature-icon feature-icon--shield" aria-hidden="true"></span><span><strong>Local document processing</strong><small>Indexed on this device</small></span></div>
              <div class="feature-item"><span class="feature-icon feature-icon--bolt" aria-hidden="true"></span><span><strong>{ai_feature_title}</strong><small>{ai_feature_detail}</small></span></div>
              <div class="feature-item"><span class="feature-icon feature-icon--quote" aria-hidden="true"></span><span><strong>Evidence linked</strong><small>Sources and pages included</small></span></div>
              <div class="feature-item"><span class="feature-icon feature-icon--sparkles" aria-hidden="true"></span><span><strong>Research tools</strong><small>Explore and compare your sources</small></span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    sources_panel = st.expander(
        "Sources",
        expanded=False,
        icon=":material/filter_list:",
        key="source_scope_panel",
        on_change="rerun",
    )
    with sources_panel:
        if sources_panel.open:
            st.caption("Choose PDFs to search. Leave empty to use all sources.")
            if indexed_docs:
                selected_doc_ids = st.multiselect(
                    "Research scope",
                    options=list(doc_names),
                    format_func=lambda doc_id: doc_names[doc_id],
                    key="selected_doc_ids",
                    label_visibility="collapsed",
                )
                st.caption(
                    f"Searching {len(selected_doc_ids)} selected document(s)."
                    if selected_doc_ids
                    else f"Searching all {len(indexed_docs)} indexed document(s)."
                )
            else:
                st.caption("Attach a PDF in the chat composer to create your source library.")

            with st.expander("Advanced settings", expanded=False, icon=":material/settings:"):
                st.markdown("**Model provider**")
                st.caption(
                    "Local Ollama"
                    if config.llm_provider == "ollama"
                    else "Hosted OpenAI-compatible endpoint"
                )
                st.caption(f"Configured model · `{generator.model_name}`")
                if configured_model_available:
                    st.caption("The configured provider and model are available.")
                elif config.llm_provider == "ollama" and ollama_online:
                    st.warning(f"Model not installed. Run `ollama pull {config.ollama_model}`.")
                else:
                    st.warning(generator.status_error or "The configured model provider is unavailable.")
                st.caption(f"Embedding model · `{config.embedding_model_name}`")
                st.divider()
                st.caption("Remove every indexed chunk and clear this session's conversation.")
                st.button(
                    "Reset document index",
                    type="secondary",
                    width="stretch",
                    on_click=reset_document_index,
                )

    queued_question = st.session_state.pop("pending_chat_prompt", None)
    if submission is not None or queued_question:
        if isinstance(submission, str):
            question = submission
            uploaded_files = []
        else:
            question = submission.text if submission is not None else ""
            uploaded_files = submission.files if submission is not None else []

        attached_doc_ids = _index_uploaded_files(uploaded_files)
        if selected_doc_ids and attached_doc_ids:
            st.session_state["pending_selected_doc_ids"] = list(
                dict.fromkeys([*selected_doc_ids, *attached_doc_ids])
            )

        question = queued_question or question
        cleaned_question = question.strip()
        indexed_docs = vector_store.get_indexed_documents_stats()
        doc_names = {document["doc_id"]: document["filename"] for document in indexed_docs}
        available_doc_ids = set(doc_names)
        if attached_doc_ids:
            st.session_state["chat_attached_doc_ids"] = list(
                dict.fromkeys(
                    [
                        *st.session_state["chat_attached_doc_ids"],
                        *attached_doc_ids,
                    ]
                )
            )
        st.session_state["chat_attached_doc_ids"] = [
            doc_id
            for doc_id in st.session_state["chat_attached_doc_ids"]
            if doc_id in available_doc_ids
        ]
        attachment_details = [
            {
                "filename": _friendly_upload_name(upload.name),
                "size_mb": len(upload.getvalue()) / (1024 * 1024),
            }
            for upload in uploaded_files
        ]
        if not cleaned_question:
            if uploaded_files:
                st.toast("PDF processing finished. Ask a question whenever you're ready.")
                st.rerun()
            else:
                st.warning("Enter a question or attach a PDF before sending.")
        elif uploaded_files and not attached_doc_ids:
            st.session_state.chat_history.extend(
                [
                    {
                        "role": "user",
                        "content": cleaned_question,
                        "attachments": attachment_details,
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "None of the attached PDFs could be indexed, so I did not search "
                            "other documents or generate an answer. Review the upload status; "
                            "scanned PDFs require OCR before they can be searched."
                        ),
                        "citations": [],
                        "is_grounded": False,
                        "is_error": True,
                    },
                ]
            )
            st.rerun()
        elif not indexed_docs:
            st.session_state.chat_history.extend(
                [
                    {"role": "user", "content": cleaned_question, "attachments": []},
                    {
                        "role": "assistant",
                        "content": (
                            "Attach a searchable PDF or index one from Documents before asking "
                            "a question. Open Sources to choose from your existing library."
                        ),
                        "citations": [],
                        "is_grounded": False,
                        "is_error": True,
                    },
                ]
            )
            st.rerun()
        elif not configured_model_available:
            st.session_state.chat_history.extend(
                [
                    {
                        "role": "user",
                        "content": cleaned_question,
                        "attachments": attachment_details,
                    },
                    {
                        "role": "assistant",
                        "content": (
                            generator.status_error
                            or "The configured model provider is unavailable. Check the Sources menu for setup details."
                        ),
                        "citations": [],
                        "is_grounded": False,
                        "is_error": True,
                    },
                ]
            )
            st.rerun()
        else:
            prior_history = st.session_state.chat_history[-4:]
            prior_questions = [
                message["content"]
                for message in reversed(prior_history)
                if message["role"] == "user" and message.get("content")
            ]
            retrieval_query = cleaned_question
            if prior_questions and len(cleaned_question) <= 100:
                retrieval_query = f"{prior_questions[0]}\n{cleaned_question}"

            selected_scope = (
                resolve_document_scope(
                    selected_doc_ids,
                    st.session_state["chat_attached_doc_ids"],
                    list(available_doc_ids),
                )
            )
            logger.info(
                "Research scope: indexed=%d selected=%d attached=%d filter=%s",
                len(indexed_docs),
                len(selected_doc_ids),
                len(st.session_state["chat_attached_doc_ids"]),
                selected_scope if selected_scope is not None else "all indexed documents",
            )
            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": cleaned_question,
                    "attachments": attachment_details,
                }
            )
            try:
                question_start = time.perf_counter()
                retrieval_start = time.perf_counter()
                with st.status(
                    "Searching your documents...",
                    expanded=True,
                ) as status:
                    passages = (
                        retriever.search_hybrid(
                            query=retrieval_query,
                            top_k=config.top_k,
                            filter_doc_ids=selected_scope,
                        )
                        if indexed_docs or attached_doc_ids
                        else []
                    )
                    logger.info(
                        "Retrieved %d evidence passage(s): %s",
                        len(passages),
                        [
                            (passage.filename, passage.page_number, passage.chunk_id)
                            for passage in passages
                        ],
                    )
                    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
                    status.write("Generating a cited answer from the retrieved evidence.")
                    response: RAGResponse = generator.generate_answer(
                        question=cleaned_question,
                        retrieved_passages=passages,
                        conversation_history=prior_history,
                    )
                    status.update(
                        label=(
                            "Answer unavailable"
                            if response.is_error
                            else "Evidence check complete"
                            if response.insufficient_evidence
                            else "Answer ready"
                        ),
                        state="complete",
                    )
                total_ms = (time.perf_counter() - question_start) * 1000
                logger.info(
                    "Answered research question: retrieval=%.0fms generation=%.0fms total=%.0fms",
                    retrieval_ms,
                    response.latency_ms,
                    total_ms,
                )
                assistant_message = {
                    "role": "assistant",
                    "content": response.answer,
                    "citations": [citation.model_dump() for citation in response.citations],
                    "is_grounded": response.is_grounded,
                    "is_error": response.is_error,
                    "insufficient_evidence": response.insufficient_evidence,
                    "retrieval_ms": retrieval_ms,
                    "generation_ms": response.latency_ms,
                    "total_ms": total_ms,
                }
            except Exception:
                logger.exception("Question answering failed")
                assistant_message = {
                    "role": "assistant",
                    "content": (
                        "I couldn't search the local document index or generate an answer. "
                        "Check the local embedding model and document index, then try again."
                    ),
                    "citations": [],
                    "is_grounded": False,
                    "is_error": True,
                }
            st.session_state.chat_history.append(assistant_message)
            st.rerun()

elif active_view == "Evaluation":
    st.markdown('<div class="eyebrow">MEASURED ON YOUR DOCUMENTS</div>', unsafe_allow_html=True)
    st.title("Retrieval evaluation")
    st.markdown(
        "Compare dense semantic retrieval with hybrid BM25 and semantic search using "
        "pages you identify as ground truth."
    )
    with st.container(border=True):
        st.text_input(
            "Benchmark question",
            placeholder="Enter a question with a known answer in your PDFs.",
            key="benchmark_query",
        )
        st.text_input(
            "Expected source pages",
            placeholder="Comma-separated, 1-indexed pages (for example: 1, 4)",
            key="benchmark_pages",
        )
        expected_doc_id = st.selectbox(
            "Expected source document",
            options=selected_doc_ids or list(doc_names),
            index=None,
            placeholder="Select the PDF containing the answer",
            format_func=lambda doc_id: doc_names[doc_id],
            key="benchmark_doc_id",
        )
        run_benchmark = st.button(
            "Run retrieval benchmark",
            icon=":material/query_stats:",
            type="primary",
            disabled=not indexed_docs,
        )

    if run_benchmark:
        expected_pages: List[int] = []
        try:
            expected_pages = [
                int(page.strip())
                for page in st.session_state.benchmark_pages.split(",")
                if page.strip()
            ]
        except ValueError:
            st.error("Enter positive page numbers separated by commas.")

        if not st.session_state.benchmark_query.strip():
            st.error("Enter a benchmark question.")
        elif expected_doc_id is None:
            st.error("Select the source PDF that contains the expected answer.")
        elif not expected_pages or any(page < 1 for page in expected_pages):
            st.error("Enter one or more positive, 1-indexed source page numbers.")
        else:
            item = BenchmarkItem(
                query=st.session_state.benchmark_query.strip(),
                expected_filename=doc_names[expected_doc_id],
                expected_doc_id=expected_doc_id,
                expected_pages=expected_pages,
                category="manual_test",
            )
            try:
                with st.spinner("Measuring dense and hybrid retrieval..."):
                    result = BenchmarkRunner(retriever=retriever).run_benchmark(
                        [item],
                        top_k=5,
                        filter_doc_ids=selected_doc_ids or None,
                    )
                st.markdown('<div class="section-heading">Measured results</div>', unsafe_allow_html=True)
                dense_column, hybrid_column = st.columns(2, gap="medium")
                for column, metrics, title in (
                    (dense_column, result.dense_metrics, "Dense semantic"),
                    (hybrid_column, result.hybrid_metrics, "Hybrid · BM25 + dense"),
                ):
                    with column:
                        with st.container(border=True):
                            st.markdown(f"#### {title}")
                            metric_columns = st.columns(2)
                            metric_columns[0].metric("Recall@1", f"{metrics.mean_recall_at_1:.2f}")
                            metric_columns[1].metric("Recall@3", f"{metrics.mean_recall_at_3:.2f}")
                            metric_columns[0].metric("Mean reciprocal rank", f"{metrics.mean_mrr:.2f}")
                            metric_columns[1].metric("Average latency", f"{metrics.mean_latency_ms:.1f} ms")
                with st.expander("Query ranking details"):
                    for details in result.query_details:
                        st.markdown(f"**Question:** {details['query']}")
                        st.caption(
                            f"Expected source · {details['expected_filename']} "
                            f"pages {details['expected_pages']}"
                        )
                        st.caption(
                            f"Dense retrieved · {details['dense_retrieved_sources']} "
                            f"(MRR {details['dense_mrr']:.2f})"
                        )
                        st.caption(
                            f"Hybrid retrieved · {details['hybrid_retrieved_sources']} "
                            f"(MRR {details['hybrid_mrr']:.2f})"
                        )
            except Exception:
                logger.exception("Retrieval evaluation failed")
                st.error("The benchmark could not run. Check the local embedding model and index.")
    elif not indexed_docs:
        st.info("Index one or more PDFs before running a retrieval benchmark.")

elif active_view == "Advanced Tools":
    st.markdown('<div class="eyebrow">OPTIONAL WORKFLOWS</div>', unsafe_allow_html=True)
    st.title("Advanced tools")
    st.markdown(
        "Plan a multi-step inquiry across the selected sources and export a cited briefing."
    )
    with st.container(border=True):
        topic = st.text_area(
            "Research topic",
            placeholder="For example: Compare the risks and mitigations discussed in these papers.",
            key="research_topic_input",
            height=90,
        )
        start_research = st.button(
            "Generate research report",
            icon=":material/description:",
            type="primary",
            disabled=not indexed_docs or not configured_model_available,
        )

    if not configured_model_available:
        st.warning(generator.status_error or "Configure an available model provider to generate a report.")
    elif not indexed_docs:
        st.info("Index at least one PDF before generating a research report.")

    if start_research:
        if not topic.strip():
            st.warning("Enter a research topic before continuing.")
        else:
            try:
                with st.status("Running the research workflow...", expanded=True) as status:
                    def update_progress(stage: str, message: str) -> None:
                        status.write(f"**{stage}** · {message}")

                    state: ResearchState = get_research_workflow().execute(
                        topic=topic.strip(),
                        filter_doc_ids=selected_doc_ids or None,
                        progress_callback=update_progress,
                    )
                    status.update(
                        label="Research workflow complete",
                        state="complete",
                        expanded=False,
                    )
                if state.plan and state.plan.used_heuristic_fallback:
                    st.warning(
                        "The planner used a local heuristic fallback because Ollama did not "
                        "return a usable plan. The report still uses retrieved document evidence."
                    )
                if state.final_report:
                    if state.final_report.is_error:
                        st.error(state.final_report.markdown_output)
                    else:
                        st.markdown('<div class="section-heading">Research briefing</div>', unsafe_allow_html=True)
                        with st.container(border=True):
                            st.markdown(state.final_report.markdown_output)
                        st.download_button(
                            "Export report as Markdown",
                            data=state.final_report.markdown_output,
                            file_name=f"research_report_{int(time.time())}.md",
                            mime="text/markdown",
                            icon=":material/download:",
                            type="secondary",
                        )
            except Exception:
                logger.exception("Advanced research workflow failed")
                st.error("The research report could not be generated. Check the local model and index.")
