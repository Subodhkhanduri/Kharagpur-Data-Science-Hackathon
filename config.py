import os
import json
from pathlib import Path

# PATHS

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for directory in [PROCESSED_DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# FILE PATHS

TRAIN_CSV_PATH = RAW_DATA_DIR / "train-1.csv"
NOVELS_DIR = RAW_DATA_DIR

NOVEL_PATHS = {
    "The Count of Monte Cristo": NOVELS_DIR / "The-Count-of-Monte-Cristo.txt",
    "In Search of the Castaways": NOVELS_DIR / "In-search-of-the-castaways.txt",
}

PROCESSED_CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.json"
PROCESSED_CLAIMS_PATH = PROCESSED_DATA_DIR / "claims.json"
RETRIEVAL_CACHE_PATH = PROCESSED_DATA_DIR / "retrieval_cache.pkl"
EMBEDDINGS_CACHE_PATH = PROCESSED_DATA_DIR / "embeddings.pkl"

RESULTS_CSV_PATH = OUTPUT_DIR / "results.csv"

# CHUNKING CONFIGURATION

CHUNK_CONFIG = {
    "chunk_size": 512,           # tokens
    "overlap": 256,              # tokens
    "tokenizer": "gpt2",         # for token counting
}


# RETRIEVAL CONFIGURATION

RETRIEVAL_CONFIG = {
    "top_k": 5,                  # passages per claim
    "max_distance": 500,         # chars between evidence chunks
    "timeline_bias_weight": 0.3, # favor later narrative segments
}

# EMBEDDING CONFIGURATION

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


# LLM CONFIGURATION (for reasoning & claims decomposition)

LLM_CONFIG = {
    "model": "deepseek-chat",           # or "gpt-4-turbo", "claude-3-opus"
    "temperature": 0.3,
    "max_tokens": 500,
    "api_key_env": "DEEPSEEK_API_KEY",  # set in .env file
}

# AGGREGATION & DECISION THRESHOLDS

AGGREGATION_CONFIG = {
    "contradiction_weight": 2.0,  # contradictions count 2x
    "support_weight": 1.0,
    "neutral_weight": 0.0,
    "confidence_threshold": 0.3,  # min confidence to count a score
    "decision_threshold": 0.2,    # aggregate_score > threshold → Consistent (1)
}

# PATHWAY FRAMEWORK CONFIGURATION

PATHWAY_CONFIG = {
    "use_local_mode": True,       # for testing; set False for production
    "vector_store_dir": PROCESSED_DATA_DIR / "pathway_store",
    "enable_caching": True,
}

# LOGGING

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": PROJECT_ROOT / "hackathon.log",
}
