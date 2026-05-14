import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Index paths
FAISS_V2    = BASE_DIR / "data" / "snrt_index_v2.faiss"
META_V2     = BASE_DIR / "data" / "snrt_metadata_v2.json"
# Model paths
BIENC_DIR   = BASE_DIR / "snrt_biencoder_v2"
CROSS_DIR   = BASE_DIR / "data" / "snrt_crossencoder_v2"
WHISPER_DIR = BASE_DIR / "model"

# Storage
UPLOAD_DIR  = BASE_DIR / "uploads"
VIDEOS_DIR  = BASE_DIR / "videos"

# MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27018")
MONGO_DB  = os.getenv("MONGO_DB",  "SNRT_ARCHIVE")

# Fallback model names
BIENC_BASE  = "paraphrase-multilingual-MiniLM-L12-v2"
CROSS_BASE  = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Search defaults
DEFAULT_K       = 2
DEFAULT_SCORE   = 0.33   # matches HARD_SEM_FLOOR in search.py
POOL_MULTIPLIER = 20
WINDOW_SIZE     = 3
MIN_WORDS       = 3

# UI
MAX_HISTORY     = 30
HISTORY_DISPLAY = 10
VIDEO_EXTS      = ["*.mp4", "*.avi", "*.mkv", "*.mov"]

UPLOAD_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)
