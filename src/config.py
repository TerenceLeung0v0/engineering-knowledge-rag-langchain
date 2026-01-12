from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
CURATED_QA_DIR =  DATA_DIR / "curated_qa"
LORA_TRAIN_DIR=  DATA_DIR / "lora_train"

DATA_DIRS = [
    DATA_DIR,
    RAW_DOCS_DIR,
    CURATED_QA_DIR,
    LORA_TRAIN_DIR
]


# Artifacts directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
VECTORSTORE_DIR = ARTIFACTS_DIR / "vectorstore"
LORA_ADAPTER_DIR = ARTIFACTS_DIR / "lora_adapter"

ARTIFACTS_DIRS = [
    VECTORSTORE_DIR,
    LORA_ADAPTER_DIR
]


# Artifacts file names

# Artifacts file paths

# Required directories exist before any execution
REQUIRED_DIRS = DATA_DIRS + ARTIFACTS_DIRS

# Retrieval
RETRIEVAL_CONFIG = {
    "fetch_k": 20,          # Fetch from FAISS for gating
    "final_k": 4,           # Feed to the prompt
    "max_l2": 0.8,         # L2 distance gate
    "min_keep": 1,          # Min. docs qty after gating
    "min_gap": 0.05,         # Confidence gap gate (ambiguous)
    "max_options": 3        # Options when hit ambiguous
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
            "print_source_indexed": True
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
