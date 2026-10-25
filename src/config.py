"""DocuMind AI Pro - Centralized Configuration."""

import os
from pathlib import Path

from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Base workspace directory
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _path_from_env(name: str, default: Path) -> Path:
    value = Path(os.getenv(name, str(default))).expanduser()
    return value if value.is_absolute() else BASE_DIR / value

class AppConfig(BaseModel):
    """Application settings with environment variable fallbacks."""
    
    # Model settings
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    )
    embedding_model_name: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    )
    
    # Storage settings
    data_dir: Path = Field(default_factory=lambda: BASE_DIR / "data")
    upload_dir: Path = Field(
        default_factory=lambda: _path_from_env("UPLOAD_DIR", BASE_DIR / "data" / "uploads")
    )
    chroma_persist_dir: Path = Field(
        default_factory=lambda: _path_from_env(
            "CHROMA_PERSIST_DIR", BASE_DIR / "data" / "chroma"
        )
    )
    
    # Chunking & Retrieval parameters
    chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")),
        description="Target character count per chunk",
    )
    chunk_overlap: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100")),
        description="Character overlap between consecutive chunks",
    )
    top_k: int = Field(
        default_factory=lambda: int(os.getenv("TOP_K", "4")),
        description="Number of context passages to return",
    )
    rrf_k: int = Field(
        default_factory=lambda: int(os.getenv("RRF_K", "60")),
        description="Reciprocal Rank Fusion smoothing parameter",
    )
    
    # Agent Workflow constraints
    max_research_steps: int = Field(default=3, description="Bounded retry limit for research agent")
    
    # LLM generation parameters
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
        description="Low temperature for factual, grounded answers",
    )
    timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60")),
        description="Ollama API timeout",
    )
    planner_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("OLLAMA_PLANNER_TIMEOUT_SECONDS", "25")),
        gt=0,
        description="Research planner Ollama API timeout",
    )
    report_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("OLLAMA_REPORT_TIMEOUT_SECONDS", "90")),
        gt=0,
        description="Research report Ollama API timeout",
    )
    max_generation_tokens: int = Field(
        default_factory=lambda: int(os.getenv("OLLAMA_NUM_PREDICT", "512")),
        gt=0,
        description="Maximum number of tokens generated for an answer",
    )
    max_planning_tokens: int = Field(
        default_factory=lambda: int(os.getenv("OLLAMA_PLAN_NUM_PREDICT", "256")),
        gt=0,
        description="Maximum number of tokens generated for a research plan",
    )
    max_report_tokens: int = Field(
        default_factory=lambda: int(os.getenv("OLLAMA_REPORT_NUM_PREDICT", "2048")),
        gt=0,
        description="Maximum number of tokens generated for a research report",
    )

    def ensure_directories(self) -> None:
        """Create necessary data storage directories safely."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

# Global default configuration instance
config = AppConfig()
config.ensure_directories()
