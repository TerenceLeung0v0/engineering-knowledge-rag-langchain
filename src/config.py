from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
CATALOG_DIR = DATA_DIR / "catalog"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
CURATED_QA_DIR =  DATA_DIR / "curated_qa"
LORA_TRAIN_DIR=  DATA_DIR / "lora_train"

DATA_DIRS = [
    DATA_DIR,
    CATALOG_DIR,
    RAW_DOCS_DIR,
    CURATED_QA_DIR,
    LORA_TRAIN_DIR
]

# Artifacts directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
VECTORSTORE_DIR = ARTIFACTS_DIR / "vectorstore"
LORA_ADAPTER_DIR = ARTIFACTS_DIR / "lora_adapter"
TAG_INDEX_DIR = ARTIFACTS_DIR / "tag_index"

ARTIFACTS_DIRS = [
    VECTORSTORE_DIR,
    LORA_ADAPTER_DIR,
    TAG_INDEX_DIR
]

# Artifacts file names
TAG_INDEX_FILENAME = "tag_index.json"
TAG_VECTORS_FILENAME = "tag_vectors.npy"

# Required directories exist before any execution
REQUIRED_DIRS = DATA_DIRS + ARTIFACTS_DIRS

# Retrieval
RETRIEVAL_CONFIG = {
    "fetch_k": 20,          # Fetch from FAISS for gating
    "final_k": 4,           # Feed to the prompt
    "max_l2": 0.8,         # L2 distance gate
    "soft_max_l2": 1.0,     # L2 soft distance gate
    "min_keep": 1,          # Min. docs qty after gating
    "min_gap": 0.05,         # Confidence gap gate (ambiguous)
    "max_options": 3,        # Options when hit ambiguous
    "min_group_gap": 0.1,
    # query-aware signature embedding tie-breaker
    "enable_sig_tiebreak": True,
    "strict_sig": False,
    "min_sig_sim": 0.30,
    "min_sig_sim_gap": 0.015,

    # anchor-content embedding tie-breaker
    "enable_anchor_tiebreak": True,
    "min_anchor_sim": 0.3,
    "min_anchor_sim_gap": 0.01
}

# Text splitting
TEXT_SPLITTING_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 100
}

# Embeddings (local)
EMBEDDING_CONFIG = {
    "backend": "huggingface",
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 64,
    "show_progress": True
}

# LLM (local)
LLM_CONFIG = {
    "backend": "ollama",
    "model_name": "llama3",
    "temperature": 0
}

# Prompt context formatting
PROMPT_CONFIG = {
    "max_chars_per_chunk": 1800,
}

RANDOM_STATE = 42

# Debug configuration
DEBUG_CONFIG = {
    "rag": {
        "ingest": {
            "print_diagnostics": True,
            "print_metadata": True
        },
        "tiebreakers": {
            "print_score": True
        },
        "chain": {
            "print_refusal_reason": True
        },
        "gating": {
            "print_absolute": True,
            "print_gap": True,
            "print_density": True
        },
        "retriever": {
            "print_ambiguous_options": True
        }
    }
}
