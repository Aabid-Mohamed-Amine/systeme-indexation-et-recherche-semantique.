import subprocess
import numpy as np
import faiss
import json
import librosa
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from utils import clean_whisper, fmt_timecode
from config import WINDOW_SIZE, MIN_WORDS


def extract_audio(ffmpeg: str, video_path: Path, audio_path: Path) -> tuple[bool, str]:
    result = subprocess.run([
        ffmpeg, "-y", "-i", str(video_path),
        "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(audio_path),
    ], capture_output=True)
    if result.returncode != 0:
        return False, result.stderr.decode(errors="ignore")[:300]
    return True, ""


def transcribe(whisper_model, processor, model_type: str, audio_path: Path, lang: str) -> list:
    if model_type == "finetuned":
        return _transcribe_lora(whisper_model, processor, audio_path, lang)
    if model_type == "faster":
        return _transcribe_faster(whisper_model, audio_path, lang)
    return _transcribe_baseline(whisper_model, audio_path, lang)


def _transcribe_lora(model, processor, audio_path: Path, lang: str) -> list:
    audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    duration = len(audio) / 16000
    device   = next(model.parameters()).device
    segs, i  = [], 0
    for start in range(0, int(duration), 14):
        chunk = audio[int(start * 16000): int((start + 15) * 16000)]
        if len(chunk) < 32000:
            continue
        inp = processor.feature_extractor(chunk, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            ids = model.generate(inp, language=lang, task="transcribe")
        text = clean_whisper(processor.batch_decode(ids, skip_special_tokens=True)[0])
        if text:
            segs.append({"id": i, "start": float(start), "end": float(min(start + 15, duration)), "text": text})
            i += 1
    return segs


def _transcribe_faster(model, audio_path: Path, lang: str) -> list:
    gen, _ = model.transcribe(
        str(audio_path), language=lang, task="transcribe",
        beam_size=5, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    return [
        {"id": i, "start": float(s.start), "end": float(s.end), "text": clean_whisper(s.text)}
        for i, s in enumerate(gen) if clean_whisper(s.text).strip()
    ]


def _transcribe_baseline(model, audio_path: Path, lang: str) -> list:
    import whisper as wlib
    audio  = wlib.load_audio(str(audio_path))
    result = model.transcribe(audio, language=lang, task="transcribe", verbose=False)
    return [
        {**s, "text": clean_whisper(s.get("text", ""))}
        for s in result.get("segments", [])
    ]


def build_segments(raw_segs: list, source_name: str, lang: str) -> tuple[list, list]:
    valid   = [s for s in raw_segs if len(s.get("text", "").split()) >= MIN_WORDS]
    entries, texts = [], []
    for i, seg in enumerate(valid):
        window  = valid[i: i + WINDOW_SIZE]
        text_ctx = " ".join(s["text"] for s in window)
        text_seg = window[0]["text"].strip()
        entries.append({
            "text"           : text_ctx,
            "text_clean"     : text_ctx,
            "text_brut"      : text_seg,
            "text_brut_clean": text_seg,
            "timecode"       : fmt_timecode(window[0]["start"]),
            "timecode_end"   : fmt_timecode(window[-1]["end"]),
            "start_sec"      : round(float(window[0]["start"]), 2),
            "end_sec"        : round(float(window[-1]["end"]), 2),
            "lang"           : lang,
            "source_audio"   : source_name,
            "segment_id"     : i,
            "window_size"    : len(window),
        })
        texts.append(text_seg)
    return entries, texts


def add_to_index(
    embed_model: SentenceTransformer,
    index,
    metadata: list,
    segments: list,
    texts: list,
    index_path: Path,
    meta_path: Path,
) -> int:
    embeddings = embed_model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
    ).astype(np.float32)
    index.add(embeddings)
    metadata.extend(segments)
    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return index.ntotal
