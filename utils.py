import io, csv, os, re, shutil
from html import escape as h
from pathlib import Path

# ── Whisper latin-1 byte fix ─────────────────────────────────────
_BYTE_FIX = str.maketrans({
    "º":"°","à":"à","â":"â","é":"é","è":"è","ê":"ê","ë":"ë",
    "î":"î","ï":"ï","ô":"ô","ù":"ù","û":"û","ü":"ü","ç":"ç",
    "ñ":"ñ","á":"á","í":"í","ó":"ó","ú":"ú","¿":"¿","¡":"¡",
})


def clean_whisper(text: str) -> str:
    if not text:
        return text
    try:
        text = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return text.translate(_BYTE_FIX).strip()


def fmt_timecode(sec: float) -> str:
    h_, r = divmod(int(sec), 3600)
    m, s  = divmod(r, 60)
    return f"{h_:02d}:{m:02d}:{s:02d}"


def highlight(text: str, keywords: list[str]) -> str:
    escaped = h(text)
    for kw in sorted(keywords, key=len, reverse=True):
        escaped = re.sub(
            f"({re.escape(h(kw))})", r"<mark>\1</mark>",
            escaped, flags=re.IGNORECASE,
        )
    return escaped


def adjust_timestamp(seg: dict, keywords: list[str]) -> int:
    text  = seg.get("text", "")
    start = seg.get("start_sec", 0)
    end   = seg.get("end_sec", start + 15)
    if not keywords or not text or end <= start:
        return int(start)
    tl = text.lower()
    positions = [tl.find(k.lower()) for k in keywords if tl.find(k.lower()) >= 0]
    if not positions:
        return int(start)
    ratio = min(min(positions) / max(len(text), 1), 0.7)
    return int(start + ratio * (end - start))


def score_pct(score: float, ce_type: str | None) -> float:
    if ce_type == "finetuned":
        return max(0.0, min(100.0, score * 100.0))
    if ce_type == "base":
        return max(0.0, min(100.0, (score + 10.0) / 20.0 * 100.0))
    return max(0.0, min(100.0, score * 100.0))


def display_text(seg: dict) -> str:
    return seg.get("text_brut_clean", seg.get("text_brut", seg.get("text", "")))


def ce_text(seg: dict) -> str:
    return seg.get("text_clean", seg.get("text", ""))


def find_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    candidates = [
        r"C:\Users\hp\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe",
        r"C:\Users\hp\ffmpeg-8.1\bin\ffmpeg.exe",
        r"C:\Users\hp\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]
    return next((c for c in candidates if os.path.exists(c)), None)


def load_videos(videos_dir: Path, upload_dir: Path) -> dict[str, str]:
    videos = {}
    for pattern in ["*.mp4", "*.avi", "*.mkv", "*.mov"]:
        for d in (videos_dir, upload_dir):
            for f in d.glob(pattern):
                if f.name not in videos:
                    videos[f.name] = str(f)
    return videos


# Map lang code → common language tokens found in filenames
_LANG_TOKENS = {
    "fr": ("fr", "french", "francais", "français", "_fr", "FR"),
    "es": ("es", "spanish", "espagnol", "español", "telediario", "_es", "ES"),
    "ar": ("ar", "arabic", "arabe", "_ar", "AR"),
}


def match_video(source: str, videos: dict[str, str], lang: str | None = None) -> str | None:
    """Return the local video file that best matches `source` (the indexed filename).

    Matching priority:
    1. Exact filename match (most reliable — same file was uploaded & indexed).
    2. Language-aware scored match: language prefix in filename scores +4.
    3. YouTube ID match (+3), date match (+2), prefix match (+1).

    The `lang` hint prevents a Spanish segment from being incorrectly paired
    with a French video that happens to share the same broadcast date.
    """
    if not videos:
        return None

    # 1 — exact match (filename stored in source_audio == uploaded video name)
    if source in videos:
        return videos[source]
    # Also try without extension in case source has .wav suffix
    source_stem = re.sub(r"\.(wav|mp4|avi|mkv|mov)$", "", source, flags=re.IGNORECASE)
    for name, path in videos.items():
        name_stem = re.sub(r"\.(mp4|avi|mkv|mov)$", "", name, flags=re.IGNORECASE)
        if name_stem == source_stem:
            return path

    # 2 — fuzzy scored match
    # Strip language prefix to extract date / YouTube ID for comparison
    clean = re.sub(r"^SNRT_(FR|ES|EN|AR)_", "", source, flags=re.IGNORECASE)
    clean = re.sub(r"\.(wav|mp4|avi|mkv|mov)$", "", clean, flags=re.IGNORECASE).strip()

    date  = re.search(r"(\d{8})", clean)
    yt_id = re.search(r"([A-Za-z0-9_-]{11})$", clean)
    d_str = date.group(1)  if date  else ""
    i_str = yt_id.group(1) if yt_id else ""

    lang_tokens = _LANG_TOKENS.get(lang or "", ())

    scored: list[tuple[int, str]] = []
    for name, path in videos.items():
        base  = re.sub(r"\.(mp4|avi|mkv|mov)$", "", name, flags=re.IGNORECASE)
        score = 0

        # Language match — highest weight: prevents cross-language false matches
        if lang_tokens and any(tok in name for tok in lang_tokens):
            score += 4

        if i_str and i_str in base:
            score += 3
        if d_str and d_str in base:
            score += 2
        if clean[:15] and clean[:15] in base:
            score += 1

        if score > 0:
            scored.append((score, path))

    if not scored:
        return None

    # Return the video with the highest match score.
    # If there's a tie, prefer the first one (deterministic ordering).
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def results_to_csv(results: list, query: str, ce_type: str | None) -> str:
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(["Rang","Timecode","Timecode_fin","Langue","Score","Score_%","Texte","Source"])
    for i, r in enumerate(results):
        pct = score_pct(r.get("score", 0), ce_type)
        w.writerow([
            i + 1,
            fmt_timecode(r.get("start_sec", 0)),
            fmt_timecode(r.get("end_sec", 0)),
            r.get("lang", ""),
            f"{r.get('score', 0):.4f}",
            f"{pct:.1f}%",
            display_text(r).replace("\n", " ")[:500],
            r.get("source_audio", ""),
        ])
    return buf.getvalue()
