# DocuMind AI Pro — Evidence-Grounded Research Assistant

> **A Local-First, Evidence-Grounded Research Assistant & Empirical Citation Engine**  
> Powered by local **Qwen 2.5 (7B)** via Ollama, **Sentence Transformers (`all-MiniLM-L6-v2`)**, **ChromaDB**, and **Reciprocal Rank Fusion (BM25 + Semantic Search)**.

---

## 🌟 Key Features

- **🔒 Configurable inference**: Use local Ollama by default or select a hosted OpenAI-compatible chat endpoint for cloud deployment. Embeddings and document indexing continue to run locally. Hosted credentials are read from Streamlit secrets or environment variables and are never included in prompts or logs.
- **📑 Multi-PDF Ingestion & Smart Filtering**:
  - Page-by-page text extraction with layout preservation via PyMuPDF.
  - Automatic detection of scanned / image-only PDFs with clear OCR notifications.
  - Deterministic document deduplication (SHA-256 fingerprinting) and stable chunk IDs (`{doc_id}_p{page}_c{chunk}`).
- **💬 Unified research chat**:
  - Attach one or more PDFs directly in the chat composer, see their names and sizes, and ask about them in the same turn.
  - Uploads are validated, limited to 25 MB each by default, and indexed only once; follow-up questions retain conversation context without treating prior answers as evidence.
- **⚡ Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**:
  - Dense Semantic Vector Search (Sentence Transformers + ChromaDB).
  - Sparse BM25 Keyword Search (Rank-BM25).
  - Reciprocal Rank Fusion ($k=60$) combining keyword precision with conceptual recall.
- **🛡️ Strict Grounding & Anti-Hallucination Guardrails**:
  - External PDF text is isolated inside `<untrusted_document_context>` tags with active prompt injection defenses.
  - Explicit instruction refusal when documents contain insufficient information.
  - Automated citation verification flagging hallucinated source indices.
- **🔬 Deep Research Multi-Agent Workflow**:
  - Bounded 4-stage state machine: **Planner → Hybrid Multi-Hop Retriever → Sufficiency Checker → Synthesis Report Writer**.
  - Strict loop boundaries preventing endless agent cycles.
- **📊 Real Empirical Benchmarking Dashboard**:
  - Evaluates Recall@1, Recall@3, Recall@5, MRR (Mean Reciprocal Rank), and latency between Dense and Hybrid retrieval without fabricated scores.
- **✨ Liquid Glass research workspace**:
  - Responsive dark Streamlit interface with translucent surfaces, hot-pink actions, chartreuse highlights, live progress, expandable citations, and exportable Markdown reports.

---

## 🏗️ Architecture Overview

```
                                ┌──────────────────────────────────────┐
                                │             User Query               │
                                └──────────────────┬───────────────────┘
                                                   │
                         ┌─────────────────────────┴────────────────────────┐
                         ▼                                                  ▼
             ┌───────────────────────┐                          ┌───────────────────────┐
             │ Dense Semantic Search │                          │   Sparse BM25 Search  │
             │ (SentenceTransformers)│                          │      (Rank-BM25)      │
             └───────────┬───────────┘                          └───────────┬───────────┘
                         │                                                  │
                         └─────────────────────────┬────────────────────────┘
                                                   ▼
                                ┌──────────────────────────────────────┐
                                │ Reciprocal Rank Fusion (RRF k=60)    │
                                └──────────────────┬───────────────────┘
                                                   ▼
                                ┌──────────────────────────────────────┐
                                │ Prompt Injection Defense Boundary    │
                                │   <untrusted_document_context>       │
                                └──────────────────┬───────────────────┘
                                                   ▼
                                ┌──────────────────────────────────────┐
                                │ Grounded Generation: Qwen 2.5 (7B)   │
                                └──────────────────┬───────────────────┘
                                                   ▼
                                ┌──────────────────────────────────────┐
                                │ Citation Verification & UI Rendering │
                                └──────────────────────────────────────┘
```

---

## 📋 System Requirements

- **Operating System**: Windows 10/11
- **Python**: Python 3.12 (Installed locally via Windows Python Launcher `py install 3.12`)
- **RAM**: 16 GB Recommended
- **Local LLM**: Ollama with `qwen2.5:7b` downloaded (`ollama pull qwen2.5:7b`)

---

## 🚀 Quickstart Guide

### 1. Verify Ollama and Model
Ensure Ollama is running and `qwen2.5:7b` is present:
```powershell
ollama list
```

### 2. Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Configure local settings (optional)
Copy the example configuration if you want to change the defaults:
```powershell
Copy-Item .env.example .env
```
The app reads `.env` from the project directory. Local development defaults to Ollama and the `qwen2.5:7b` model.

### Configure the LLM provider

The main chat, research planner, and report writer use the same provider setting. Both providers use the existing prompts and citation checks.

**Local Ollama (default)** — add to `.env`:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```

Install the model with `ollama pull qwen2.5:7b`. The local Ollama service must be reachable from the machine running Streamlit.

**Streamlit Community Cloud / hosted endpoint** — in the app's Streamlit Cloud **Settings → Secrets**, add:

```toml
[llm]
provider = "hosted"
api_base_url = "https://your-provider.example/v1"
model = "your-model-name"
api_key = "your-provider-api-key"
```

The endpoint must implement OpenAI-compatible `GET /models` and `POST /chat/completions`. Set `api_base_url` to the API base (for example, ending in `/v1`) or to the full `/chat/completions` URL. The model ID must appear in the endpoint's `/models` response. Keep the key in Streamlit secrets; do not commit it or put it in a URL.

The equivalent environment variables are `LLM_PROVIDER=hosted`, `LLM_API_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY`. Environment variables take precedence over Streamlit secrets. Flat uppercase Streamlit secret keys are also supported. If the endpoint, model, or key is missing, the app reports the specific missing setting and does not generate a substitute answer.

Hosted mode changes only LLM transport. PDF extraction, embeddings, ChromaDB, retrieval, citation verification, and upload storage remain as configured by this project. In hosted mode, the user's question and retrieved evidence excerpts are sent to the configured model provider for generation; keep this in mind when choosing an endpoint.
`server.maxUploadSize` in `.streamlit/config.toml` sets the per-file upload limit (default `25` MB). `OLLAMA_NUM_PREDICT`, `OLLAMA_PLAN_NUM_PREDICT`, and `OLLAMA_REPORT_NUM_PREDICT` cap answer, planning, and report generation lengths; the corresponding timeout settings are listed in `.env.example`.

### 4. Run the Streamlit Application
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 🧪 Running Unit Tests

The test suite covers PDF ingestion, chunking with sliding window overlap, BM25 indexing, citation tracking, anti-hallucination checks, and prompt injection defense:

```powershell
.\.venv\Scripts\pytest -v
```

## 💬 Using the research chat

Start from the centered **Research chat** workspace. Attach one or more PDFs with the paperclip, optionally enter a question in the same message, and submit. PDFs are validated and indexed locally before the question is answered. Expand **Sources** to limit research to selected PDFs; leaving the scope empty searches the full local index. **Advanced settings** in that section contains local model details and index reset controls. Choose **Documents**, **Evaluation**, or **Advanced tools** from the compact **Menu** in the header. The three suggested prompts can be used once searchable PDFs are available. Per-answer retrieval and generation durations are kept under **Response diagnostics**; ingestion stage timings are recorded in the application log.

---

## 📁 Project Directory Structure

```
gen ai project/
├── app.py                     # Streamlit research workspace and navigation
├── requirements.txt           # Project dependencies
├── .streamlit/
│   └── config.toml            # Dark palette and native Streamlit theme
├── .env.example               # Configuration template
├── .gitignore                 # Secure Git rules ignoring uploads, venv, vector store
├── README.md                  # Comprehensive portfolio documentation
├── src/
│   ├── liquid_glass.css       # Responsive Liquid Glass interface styling
│   ├── config.py              # Centralized application settings & paths
│   ├── ingestion/
│   │   ├── pdf_processor.py   # Page-by-page PDF extraction & scanned file detection
│   │   └── chunking.py        # Sliding-window semantic chunker with stable IDs
│   ├── retrieval/
│   │   ├── embeddings.py      # Sentence Transformers embedding service
│   │   ├── vector_store.py    # Persistent ChromaDB storage & metadata queries
│   │   ├── bm25_search.py     # In-memory BM25 Okapi sparse search
│   │   ├── hybrid_search.py   # Reciprocal Rank Fusion (RRF) retriever
│   │   └── reranker.py        # Cross-encoder reranker with passthrough fallback
│   ├── rag/
│   │   ├── citations.py       # Page-specific citation numbering & hallucination checker
│   │   └── answer_generator.py# Grounded provider-backed LLM client with security boundary
│   ├── llm/
│   │   └── provider.py        # Local Ollama and hosted API transports
│   ├── agents/
│   │   ├── research_planner.py# Inquiry decomposition into targeted sub-queries
│   │   ├── report_writer.py   # Synthesis of multi-hop evidence into Markdown report
│   │   └── workflow.py        # Bounded 4-stage research loop state machine
│   └── evaluation/
│       ├── metrics.py         # Recall@k, MRR, latency, citation precision formulas
│       └── benchmark.py       # Empirical benchmark runner comparing Dense vs. Hybrid
├── tests/
│   ├── test_ingestion.py      # Tests for PDF loading, 0-byte detection, scanned heuristics
│   ├── test_retrieval.py      # Tests for BM25 and Hybrid retrieval
│   ├── test_citations.py      # Tests for citation verification & hallucination checks
│   └── test_security.py       # Tests for untrusted prompt injection containment
└── data/
    └── .gitkeep               # Kept in Git while ignoring uploads and vector embeddings
```

---

## 🛡️ Security & Privacy Architecture

1. **Untrusted Data Isolation**: Document content extracted from user PDFs is escaped before being placed inside the untrusted context boundary. Ollama receives grounding rules as its system prompt.
2. **Deterministic Attribution**: Chunks carry deterministic source hashes, filenames, and 1-indexed page numbers.
3. **Citation Verification**: Unsupported or missing source references are rejected rather than displayed as verified answers.
4. **No Secret Leakage**: Vector databases, uploaded PDF files, and local logs are excluded by `.gitignore`.

---

## 📤 Publishing to GitHub with GitHub Desktop

1. Open **GitHub Desktop**.
2. Click **File > Add Local Repository...** (or press `Ctrl+O`).
3. Browse to `D:\computer science\gen ai\gen ai project` and click **Add Repository**.
4. Before staging, run `git status --short --ignored` and confirm `.env`, `.venv/`, private PDFs, and `data/chroma/` are ignored. Never stage local data, secrets, or model caches.
5. Enter a commit summary: `Initial commit: DocuMind AI Pro complete portfolio project`.
6. Click **Commit to main**.
7. Click **Publish repository** to create a public or private GitHub repository.
