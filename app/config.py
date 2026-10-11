from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SEMANTIC_INDEX_PATH = DATA_DIR / "semantic_index.json"


# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# Ollama
import os

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_EMBEDDING_URL = f"{OLLAMA_BASE_URL}/api/embeddings"

LLM_MODEL = "qwen2.5:3b"
EMBEDDING_MODEL = "nomic-embed-text"


# Retrieval
TOP_K = 1   
RETRIEVAL_K = 10
RELEVANCE_THRESHOLD = 0.034