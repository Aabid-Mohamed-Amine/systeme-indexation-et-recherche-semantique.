# =============================================================================
# SNRT — Re-index local videos into the FAISS index
# Usage:  python reindex_local.py
# =============================================================================

import os, json, shutil, subprocess
import numpy as np
import faiss
import torch
import librosa
from pathlib import Path
from collections import Counter
from sentence_transformers import SentenceTransformer

# Shared helpers & config
from utils import clean_whisper, fmt_timecode, find_ffmpeg
from config import (
    BASE_DIR, VIDEOS_DIR, UPLOAD_DIR, FAISS_V2, META_V2,
    WINDOW_SIZE, MIN_WORDS,
)

DATA_DIR = BASE_DIR / "data"

# ── FFmpeg ────────────────────────────────────────────────────────
FFMPEG = find_ffmpeg()
if not FFMPEG:
    raise RuntimeError("FFmpeg not found — check your PATH or installation")

# ── Discover videos ───────────────────────────────────────────────
videos = []
for ext in ["*.mp4", "*.avi", "*.mkv", "*.mov"]:
    videos += list(VIDEOS_DIR.glob(ext))
    videos += list(UPLOAD_DIR.glob(ext))

if not videos:
    raise RuntimeError(f"No videos found in {VIDEOS_DIR} or {UPLOAD_DIR}")

print(f"✅ {len(videos)} video(s) found:")
for v in videos:
    print(f"   {v.name}  ({v.stat().st_size / 1e6:.0f} MB)")


# ── Language detection from filename ─────────────────────────────
def detect_lang(name: str) -> str:
    name = name.lower()
    if any(x in name for x in ("espagnol", "español", "_es_", "telediario")):
        return "es"
    if any(x in name for x in ("arabe", "arab", "_ar_")):
        return "ar"
    return "fr"  # default


# ── Whisper model loading ─────────────────────────────────────────
print("\n🧠 Loading transcription model…")
device      = "cuda" if torch.cuda.is_available() else "cpu"
whisper_mdl = None
model_type  = None

# Priority 1: faster-whisper (recommended on CPU)
try:
    from faster_whisper import WhisperModel
    whisper_mdl = WhisperModel("medium", device="cpu", compute_type="int8")
    model_type  = "faster"
    print("   ✅ faster-whisper medium INT8")
except Exception as e:
    print(f"   ⚠️  faster-whisper unavailable: {e}")

# Priority 2: LoRA fine-tuned
if whisper_mdl is None:
    MODEL_DIR = BASE_DIR / "model"
    if (MODEL_DIR / "adapter_config.json").exists():
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            from peft import PeftModel
            proc        = WhisperProcessor.from_pretrained("openai/whisper-medium")
            base        = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-medium", torch_dtype=torch.float32, low_cpu_mem_usage=True,
            )
            base.config.use_cache          = False
            base.config.forced_decoder_ids = None
            base.config.suppress_tokens    = None
            whisper_mdl = PeftModel.from_pretrained(base, str(MODEL_DIR)).merge_and_unload()
            whisper_mdl.eval().to(device)
            model_type  = "finetuned"
            print("   ✅ LoRA fine-tuned model")
        except Exception as e:
            print(f"   ⚠️  Fine-tuned model unavailable: {e}")

if whisper_mdl is None:
    raise RuntimeError("No Whisper model available")


# ── Transcription ─────────────────────────────────────────────────
def transcribe(audio_path: Path, lang: str) -> list:
    if model_type == "faster":
        segments_gen, _ = whisper_mdl.transcribe(
            str(audio_path), language=lang, task="transcribe",
            beam_size=5, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        return [
            {"id": i, "start": float(s.start), "end": float(s.end),
             "text": clean_whisper(s.text)}
            for i, s in enumerate(segments_gen)
        ]
    else:
        from transformers import WhisperProcessor
        proc      = WhisperProcessor.from_pretrained("openai/whisper-medium")
        audio, _  = librosa.load(str(audio_path), sr=16000, mono=True)
        segs, clip, step = [], 15, 14
        for i, start in enumerate(range(0, int(len(audio) / 16000), step)):
            chunk = audio[int(start * 16000): int((start + clip) * 16000)]
            if len(chunk) < 32000:
                continue
            inp = proc.feature_extractor(
                chunk, sampling_rate=16000, return_tensors="pt",
            ).input_features.to(device)
            with torch.no_grad():
                ids = whisper_mdl.generate(inp, language=lang, task="transcribe")
            text = clean_whisper(proc.batch_decode(ids, skip_special_tokens=True)[0])
            if text:
                segs.append({"id": i, "start": float(start),
                             "end": float(start + clip), "text": text})
        return segs


# ── Main pipeline ─────────────────────────────────────────────────
print("\n🔧 Loading embedding model…")
embed_model  = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
all_segments = []

for video in videos:
    lang = detect_lang(video.name)
    print(f"\n🎬 {video.name}  [{lang.upper()}]")

    # Step 1: extract audio
    audio_path = DATA_DIR / (video.stem + ".wav")
    print("   🔊 Extracting audio…")
    r = subprocess.run([
        FFMPEG, "-y", "-i", str(video),
        "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(audio_path),
    ], capture_output=True)
    if r.returncode != 0:
        print(f"   ❌ FFmpeg error: {r.stderr.decode(errors='ignore')[:200]}")
        continue

    # Step 2: transcribe
    print(f"   📝 Transcribing ({model_type})…")
    raw_segs = transcribe(audio_path, lang)
    print(f"   ✅ {len(raw_segs)} segments")

    # Step 3: context windows
    valid = [s for s in raw_segs if len(s.get("text", "").split()) >= MIN_WORDS]
    for i in range(len(valid)):
        window  = valid[i: i + WINDOW_SIZE]
        text    = " ".join(s["text"] for s in window)
        if len(text.split()) < 5:
            continue
        first, last = window[0], window[-1]
        all_segments.append({
            "text"        : text,
            "text_brut"   : first["text"],
            "timecode"    : fmt_timecode(first["start"]),
            "timecode_end": fmt_timecode(last["end"]),
            "start_sec"   : round(float(first["start"]), 2),
            "end_sec"     : round(float(last["end"]), 2),
            "lang"        : lang,
            "source_audio": video.name,
            "segment_id"  : i,
            "window_size" : len(window),
        })

    audio_path.unlink(missing_ok=True)

print(f"\n✅ Total: {len(all_segments)} segments")
print("Languages:", Counter(s["lang"] for s in all_segments))

# ── Embeddings + FAISS ────────────────────────────────────────────
print("\n🧠 Generating embeddings…")
texts      = [s["text_brut"] for s in all_segments]
embeddings = embed_model.encode(
    texts, normalize_embeddings=True, convert_to_numpy=True,
    batch_size=64, show_progress_bar=True,
).astype(np.float32)

print("🔍 Building FAISS index…")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# ── Save ──────────────────────────────────────────────────────────
if FAISS_V2.exists():
    shutil.copy(FAISS_V2,  DATA_DIR / "snrt_index_old.faiss")
    shutil.copy(META_V2,   DATA_DIR / "snrt_metadata_old.json")

faiss.write_index(index, str(FAISS_V2))
with open(META_V2, "w", encoding="utf-8") as f:
    json.dump(all_segments, f, ensure_ascii=False, indent=2)

print(f"""
✅ Done!
   Videos indexed : {len(videos)}
   Segments       : {len(all_segments)}
   FAISS vectors  : {index.ntotal}
   Backup saved   : data/snrt_index_old.faiss

👉 Run:  streamlit run app.py
""")
