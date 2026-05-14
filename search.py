import re
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import ce_text

# Stopwords excluded from keyword matching (too generic → false positives)
_STOPWORDS = {
    "les", "des", "une", "pour", "dans", "sur", "avec", "par", "que", "qui",
    "est", "son", "ses", "aux", "pas", "plus", "tout", "cette", "aussi",
    "los", "las", "del", "una", "con", "por", "como",
    "the", "and", "for", "that", "this", "with", "are", "from",
}

# Hard minimum cosine similarity.
# FAISS always returns the nearest neighbours even for completely unrelated
# queries.  Any segment below this floor is treated as "not in the corpus"
# and is never shown to the user.
# 0.33 gives a good balance: filters real noise while keeping multilingual
# results where FR/ES embeddings may align slightly lower than mono-lingual ones.
HARD_SEM_FLOOR = 0.33


def extract_keywords(query: str) -> list[str]:
    return [
        w.lower().strip()
        for w in re.split(r"\s+", query)
        if len(w) >= 3 and w.lower() not in _STOPWORDS
    ]


def _keyword_score(text: str, keywords: list[str]) -> tuple[int, bool]:
    """Return (hit_count, all_keywords_present)."""
    tl = text.lower()
    hits = sum(1 for kw in keywords if kw in tl)
    all_present = len(keywords) > 1 and all(kw in tl for kw in keywords)
    return hits, all_present


def deduplicate(candidates: dict, time_gap: float = 20.0) -> dict:
    """
    Remove overlapping windows from the same source within `time_gap` seconds.
    Keep the highest-scoring segment per time zone.
    """
    by_source: dict[str, list] = {}
    for idx, seg in candidates.items():
        src = seg.get("source_audio", "")
        by_source.setdefault(src, []).append((idx, seg))

    kept = {}
    for segs in by_source.values():
        segs_sorted = sorted(segs, key=lambda x: x[1]["score"], reverse=True)
        accepted: list[float] = []
        for idx, seg in segs_sorted:
            start = seg.get("start_sec", 0)
            if all(abs(start - a) >= time_gap for a in accepted):
                kept[idx] = seg
                accepted.append(start)
    return kept


def run_search(
    query: str,
    embed_model: SentenceTransformer,
    index,
    metadata: list,
    cross_encoder,
    ce_type: str,
    k: int,
    lang_filter: str | None,
    score_min: float,
    use_reranker: bool,
) -> tuple[list, float, str]:
    t0       = time.time()
    keywords = extract_keywords(query)

    # ── Phase 1: Semantic retrieval ───────────────────────────────
    vec = embed_model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)

    # Larger pool gives the cross-encoder more material to re-rank
    pool = min(k * 40, index.ntotal)
    scores_raw, indices = index.search(vec, pool)

    # ── Phase 2: Build candidates ─────────────────────────────────
    candidates: dict = {}
    for sem_score, idx in zip(scores_raw[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue
        seg = metadata[idx]
        if lang_filter and seg.get("lang", "") != lang_filter:
            continue

        sem = float(sem_score)

        # Reject immediately if below the hard semantic floor.
        # This is the primary guard against returning irrelevant results
        # when the search term simply does not exist in the corpus.
        if sem < HARD_SEM_FLOOR:
            continue

        kw_hits, all_present = _keyword_score(seg.get("text", ""), keywords)

        # Small keyword boost — bounded to prevent keyword gaming.
        boost = min(kw_hits * 0.02, 0.06)      # max +0.06 for individual hits
        if all_present:
            boost = min(boost + 0.08, 0.12)    # max +0.12 when all keywords match

        candidates[idx] = {
            **seg,
            "sem_score"   : sem,
            "score"       : sem + boost,
            "keyword_hits": kw_hits,
        }

    # ── Phase 3: Cross-encoder re-ranking ────────────────────────
    if use_reranker and cross_encoder and candidates:
        items     = list(candidates.items())
        pairs     = [(query, ce_text(seg)) for _, seg in items]
        ce_scores = cross_encoder.predict(pairs, show_progress_bar=False)
        for (idx, _), ce_sc in zip(items, ce_scores):
            candidates[idx]["score"] = float(ce_sc)

    # ── Phase 4: Deduplicate overlapping windows ──────────────────
    candidates = deduplicate(candidates)

    # ── Phase 5: Filter & sort ────────────────────────────────────
    if use_reranker:
        threshold = 0.1 if ce_type == "finetuned" else -2.0
    else:
        # Enforce HARD_SEM_FLOOR as an absolute minimum; the user slider can
        # only raise the bar further, never lower it below the floor.
        threshold = max(score_min, HARD_SEM_FLOOR)

    results = [
        s for s in candidates.values()
        if (s["score"] if use_reranker else s.get("sem_score", s["score"])) >= threshold
    ]
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]

    elapsed = time.time() - t0

    if use_reranker:
        mode = f"sémantique + cross-encoder ({ce_type})"
    elif any(r.get("keyword_hits", 0) > 0 for r in results):
        mode = "sémantique + mot-clé"
    else:
        mode = "sémantique"

    return results, elapsed, mode
