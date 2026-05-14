import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import json
import torch
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from config import (
    BIENC_DIR, BIENC_BASE, CROSS_DIR, CROSS_BASE,
    WHISPER_DIR, FAISS_V2, META_V2,
)


@st.cache_resource(show_spinner=False)
def load_biencoder() -> tuple[SentenceTransformer, str]:
    if BIENC_DIR.exists() and (BIENC_DIR / "config.json").exists():
        return SentenceTransformer(str(BIENC_DIR)), "finetuned"
    return SentenceTransformer(BIENC_BASE), "base"


@st.cache_resource(show_spinner=False)
def load_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder
        if CROSS_DIR.exists() and (CROSS_DIR / "config.json").exists():
            return CrossEncoder(str(CROSS_DIR), max_length=512), "finetuned"
        return CrossEncoder(CROSS_BASE, max_length=512), "base"
    except Exception:
        return None, "unavailable"


@st.cache_resource(show_spinner=False)
def load_index() -> tuple:
    if FAISS_V2.exists() and META_V2.exists():
        idx = faiss.read_index(str(FAISS_V2))
        with open(META_V2, encoding="utf-8") as f:
            meta = json.load(f)
        return idx, meta, "v2"
    return None, None, None


@st.cache_resource(show_spinner=False)
def load_whisper():
    # Priority 1: LoRA fine-tuned
    if WHISPER_DIR.exists() and (WHISPER_DIR / "adapter_config.json").exists():
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            from peft import PeftModel
            processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
            base = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-medium", torch_dtype=torch.float32, low_cpu_mem_usage=True,
            )
            base.config.use_cache          = False
            base.config.forced_decoder_ids = None
            base.config.suppress_tokens    = None
            model = PeftModel.from_pretrained(base, str(WHISPER_DIR)).merge_and_unload()
            model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
            return model, processor, "finetuned"
        except Exception:
            pass

    # Priority 2: Faster-Whisper INT8
    try:
        from faster_whisper import WhisperModel
        return WhisperModel("medium", device="cpu", compute_type="int8"), None, "faster"
    except Exception:
        pass

    # Priority 3: baseline openai-whisper
    try:
        import whisper as wlib
        return wlib.load_model("medium"), None, "baseline"
    except Exception:
        return None, None, "unavailable"
