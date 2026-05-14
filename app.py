import logging
import os
import re
import sys
import time
import warnings
from html import escape as h
from pathlib import Path

# Ensure local modules are always found regardless of CWD
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from config import (
    DEFAULT_K,
    DEFAULT_SCORE,
    FAISS_V2,
    META_V2,
    UPLOAD_DIR,
    VIDEOS_DIR,
)
from indexer import add_to_index, build_segments, extract_audio, transcribe
from models import load_biencoder, load_cross_encoder, load_index, load_whisper
from search import extract_keywords, run_search
from utils import (
    display_text,
    find_ffmpeg,
    fmt_timecode,
    highlight,
    load_videos,
    match_video,
    results_to_csv,
    score_pct,
)

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="SNRT — Archives Sémantiques",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject Tailwind CDN + minimal overrides ───────────────────────
st.markdown("""
<script src="https://cdn.tailwindcss.com"></script>
<script>
  tailwind.config = {
    theme: {
      extend: {
        colors: {
          navy:   { DEFAULT: '#020817', 50: '#0a1628', 100: '#0f1f38', 200: '#162340' },
          border: { DEFAULT: '#1a2d4a', 2: '#243c5e' },
          gold:   { DEFAULT: '#f59e0b', dark: '#d97706' },
          blue:   { glow: 'rgba(59,130,246,0.25)' },
        },
        fontFamily: {
          sans: ['IBM Plex Sans', 'sans-serif'],
          arab: ['Cairo', 'sans-serif'],
          mono: ['JetBrains Mono', 'monospace'],
        },
      }
    }
  }
</script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=Cairo:wght@400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">

<style>
  /* ── App background ── */
  html, body, .stApp, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background: #020817 !important;
    color: #e2e8f0 !important;
  }
  /* ── Hide sidebar toggle ── */
  [data-testid="collapsedControl"] { display: none !important; }
  section[data-testid="stSidebar"]  { display: none !important; }

  /* ── Inputs ── */
  .stTextInput > div > div > input {
    background: #0f1f38 !important; color: #e2e8f0 !important;
    border: 1px solid #243c5e !important; border-radius: 8px !important;
    padding: 0.75rem 1rem !important; font-size: 1rem !important;
    caret-color: #3b82f6 !important; transition: all 0.2s !important;
  }
  .stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.25) !important;
    outline: none !important;
  }
  .stTextInput > div > div > input::placeholder { color: #64748b !important; }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 12px rgba(59,130,246,0.25) !important;
  }
  .stButton > button:hover { transform: translateY(-1px) !important; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: #0a1628 !important; border-radius: 12px 12px 0 0 !important;
    border: 1px solid #1a2d4a !important; border-bottom: none !important;
    padding: 0.3rem !important;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #94a3b8 !important;
    border-radius: 8px !important; font-weight: 500 !important;
    transition: all 0.2s !important;
  }
  .stTabs [aria-selected="true"] {
    background: #3b82f6 !important; color: white !important;
    box-shadow: 0 2px 10px rgba(59,130,246,0.3) !important;
  }
  .stTabs [data-baseweb="tab-panel"] {
    background: #0a1628 !important; border: 1px solid #1a2d4a !important;
    border-radius: 0 0 12px 12px !important; padding: 1.5rem !important;
  }

  /* ── Select / Slider ── */
  .stSelectbox > div > div {
    background: #0f1f38 !important; border: 1px solid #243c5e !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: #0a1628; }
  ::-webkit-scrollbar-thumb { background: #243c5e; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #3b82f6; }

  /* ── Keyword highlight ── */
  mark {
    background: rgba(245,158,11,0.2) !important; color: #fcd34d !important;
    border-radius: 3px !important; padding: 0 2px !important;
    border-bottom: 1px solid rgba(245,158,11,0.4) !important;
  }

  /* ── Misc ── */
  video, .stVideo > video { border-radius: 10px !important; width: 100% !important; }
  hr { border-color: #1a2d4a !important; margin: 1rem 0 !important; }
  .stStatus  { background: #0f1f38 !important; border: 1px solid #1a2d4a !important; border-radius: 8px !important; }
  .stAlert   { border-radius: 8px !important; }
  [data-testid="stFileUploader"] {
    background: #0f1f38 !important; border: 1px solid #243c5e !important; border-radius: 12px !important;
  }
  .ex-pill .stButton > button {
    background: #0f1f38 !important; color: #94a3b8 !important;
    border: 1px solid #243c5e !important; border-radius: 20px !important;
    font-size: 0.78rem !important; box-shadow: none !important;
  }
  .ex-pill .stButton > button:hover {
    background: #162340 !important; color: #e2e8f0 !important;
    border-color: #3b82f6 !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────
try:
    with st.spinner("Chargement des modèles…"):
        embed_model, embed_type   = load_biencoder()
        cross_encoder, ce_type    = load_cross_encoder()
        index, metadata, idx_ver  = load_index()
        whisper_m, proc, wtype    = load_whisper()
except Exception as _load_err:
    import traceback
    st.error(f"❌ Erreur au chargement : {_load_err}")
    st.code(traceback.format_exc(), language="python")
    st.stop()

FFMPEG = find_ffmpeg()

if index is None:
    st.error("❌ Index FAISS non trouvé — place `snrt_index.faiss` dans `data/`")
    st.stop()

n_segs   = index.ntotal
n_videos = len(set(s.get("source_audio", "") for s in metadata))
videos   = load_videos(VIDEOS_DIR, UPLOAD_DIR)

# ── Session state ─────────────────────────────────────────────────
st.session_state.setdefault("history", [])
st.session_state.setdefault("query_val", "")

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="relative flex items-center justify-between gap-6 px-8 py-5 mb-0 overflow-hidden"
     style="background:linear-gradient(135deg,#020c1f 0%,#051535 50%,#020c1f 100%);
            border-bottom:1px solid #1a2d4a;">
  <div style="position:absolute;inset:0;
    background:radial-gradient(ellipse 60% 80% at 10% 50%,rgba(59,130,246,.07) 0%,transparent 70%),
               radial-gradient(ellipse 40% 60% at 90% 50%,rgba(245,158,11,.05) 0%,transparent 70%)">
  </div>
  <div class="flex items-center gap-4 relative z-10">
    <div class="flex items-center justify-center w-12 h-12 rounded-xl text-2xl"
         style="background:linear-gradient(135deg,#1d4ed8,#3b82f6);box-shadow:0 0 20px rgba(59,130,246,.3)">
      📺
    </div>
    <div>
      <div class="text-2xl font-black text-white" style="font-family:'Cairo',sans-serif;letter-spacing:-.5px">
        SNRT <span style="color:#f59e0b">Archives</span>
      </div>
      <div class="text-xs mt-0.5" style="color:rgba(255,255,255,.5)">
        Système de Recherche Sémantique dans les Archives Audiovisuelles
      </div>
      <div class="text-xs" style="color:rgba(255,255,255,.35);font-family:'Cairo',sans-serif;direction:rtl">
        البحث الدلالي في الأرشيف السمعي البصري · PFE Master SID 2025-2026
      </div>
    </div>
  </div>

  <div class="flex items-center gap-3 relative z-10 flex-wrap">
    <div class="flex gap-1.5">
      <span class="px-2.5 py-1 rounded-full text-xs font-bold"
            style="background:rgba(29,78,216,.2);color:#93c5fd;border:1px solid rgba(59,130,246,.3)">🇫🇷 FR</span>
      <span class="px-2.5 py-1 rounded-full text-xs font-bold"
            style="background:rgba(185,28,28,.2);color:#fca5a5;border:1px solid rgba(239,68,68,.3)">🇪🇸 ES</span>
    </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "🔍  Rechercher",
    "📤  Indexer une vidéo",
])

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 1 — RECHERCHE                                           ║
# ╚══════════════════════════════════════════════════════════════╝
with tab1:
    # ── Search bar ───────────────────────────────────────────────
    default = st.session_state.pop("query_val", "")
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        query = st.text_input(
            "", value=default,
            placeholder="🔍  Rechercher… ex: 'discours royal', 'SIAM agriculture', 'football Botola'",
            label_visibility="collapsed",
        )
    with col_btn:
        do_search = st.button("Rechercher", use_container_width=True)

    # ── Quick examples ───────────────────────────────────────────
    EXAMPLES = [
        "politique économique Maroc",
        "SIAM agriculture Meknès",
        "Mohamed VI discours royal",
        "Bayern Munich football",
        "Nasser Bourita diplomatie",
    ]
    ex_cols = st.columns(5)
    for i, ex in enumerate(EXAMPLES):
        with ex_cols[i]:
            st.markdown('<div class="ex-pill">', unsafe_allow_html=True)
            if st.button(f"💬 {ex[:17]}..", key=f"ex{i}", use_container_width=True):
                query     = ex
                do_search = True
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Inline filters ───────────────────────────────────────────
    with st.expander("⚙️ Filtres & options", expanded=False):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            k_results = st.slider("Résultats", 1, 5, DEFAULT_K)
        with fc2:
            lang_opt  = st.selectbox("Langue", ["Toutes", "Français 🇫🇷", "Espagnol 🇪🇸"])
        with fc3:
            score_min = st.slider("Score minimum", 0.0, 1.0, DEFAULT_SCORE, 0.05)
        with fc4:
            use_rer   = st.toggle(
                "Re-ranking",
                value=(cross_encoder is not None),
                disabled=(cross_encoder is None),
                help="Active le cross-encoder pour un meilleur classement",
            )

    # ── History (collapsible) ────────────────────────────────────
    if st.session_state.history:
        with st.expander(f"🕐 Historique ({len(st.session_state.history)} recherches)", expanded=False):
            h_cols = st.columns(2)
            for j, entry in enumerate(reversed(st.session_state.history[-10:])):
                q, n, dt = entry["query"], entry["n_results"], entry["dt"]
                with h_cols[j % 2]:
                    if st.button(
                        f"🔍 {q[:30]}{'…' if len(q)>30 else ''}  ·  {n} rés. · {dt:.1f}s",
                        key=f"hist_{hash(q+str(entry['ts']))}",
                        use_container_width=True,
                    ):
                        st.session_state.query_val = q
                        st.rerun()
            if st.button("🗑 Effacer l'historique"):
                st.session_state.history = []
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Results ──────────────────────────────────────────────────
    if query:
        lang_map = {"Toutes": None, "Français 🇫🇷": "fr", "Espagnol 🇪🇸": "es"}
        lang_sel = lang_map.get(lang_opt)
        use_ce   = use_rer and cross_encoder is not None

        with st.spinner("🔍 Recherche sémantique…"):
            results, elapsed, mode = run_search(
                query, embed_model, index, metadata,
                cross_encoder, ce_type,
                k_results, lang_sel, score_min, use_ce,
            )

        st.session_state.history.append({
            "query": query, "n_results": len(results),
            "dt": elapsed, "ts": time.time(),
        })
        if len(st.session_state.history) > 30:
            st.session_state.history = st.session_state.history[-30:]

        videos = load_videos(VIDEOS_DIR, UPLOAD_DIR)

        if not results:
            st.markdown(f"""
            <div class="flex flex-col items-center justify-center py-20 gap-4"
                 style="color:#64748b">
              <div style="font-size:4rem;animation:float 3s ease-in-out infinite">🔎</div>
              <div class="text-xl font-bold" style="color:#e2e8f0">
                Aucune vidéo trouvée pour cette recherche
              </div>
              <div class="text-sm text-center" style="color:#94a3b8;max-width:480px;line-height:1.6">
                Le terme <b style="color:#f59e0b">« {h(query)} »</b> n'a pas été trouvé
                dans les vidéos indexées. Le corpus SNRT ne contient probablement
                pas de contenu sur ce sujet.
              </div>
              <div class="flex flex-col gap-1 text-xs text-center" style="color:#64748b">
                <span>💡 Essaie des mots-clés différents ou reformule la requête</span>
                <span>💡 Vérifie que la vidéo concernée a bien été indexée dans <b>Indexer une vidéo</b></span>
              </div>
            </div>
            <style>@keyframes float{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-8px)}}}}</style>
            """, unsafe_allow_html=True)
        else:
            ce_used  = ce_type if use_ce else None
            keywords = extract_keywords(query)

            # ── Low-confidence warning ────────────────────────────
            top_sem = results[0].get("sem_score", results[0].get("score", 0))
            if not use_ce and top_sem < 0.40:
                st.markdown(f"""
                <div class="flex items-start gap-3 px-4 py-3 rounded-xl mb-3"
                     style="background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25)">
                  <span style="font-size:1.1rem">⚠️</span>
                  <div class="text-sm" style="color:#fcd34d">
                    <b>Pertinence faible</b> — le meilleur résultat a un score sémantique de
                    <b>{top_sem*100:.0f}%</b>. Le corpus SNRT contient probablement peu de contenu
                    sur ce sujet. Active le <b>Re-ranking</b> dans les filtres pour un meilleur tri,
                    ou reformule la requête.
                  </div>
                </div>
                """, unsafe_allow_html=True)

            col_info, col_exp = st.columns([4, 1])
            with col_info:
                st.markdown(
                    f"<div class='text-sm' style='color:#94a3b8'>"
                    f"<b style='color:#e2e8f0'>{len(results)} résultat(s)</b> pour "
                    f"<b style='color:#f59e0b'>« {h(query)} »</b> — "
                    f"<span style='color:#3b82f6'>{mode}</span> "
                    f"<span class='px-2 py-0.5 rounded-full text-xs font-mono font-semibold' "
                    f"style='background:rgba(139,92,246,.12);color:#c4b5fd;"
                    f"border:1px solid rgba(139,92,246,.25)'>⏱ {elapsed:.2f}s</span></div>",
                    unsafe_allow_html=True,
                )
            with col_exp:
                csv_data = results_to_csv(results, query, ce_used)
                st.download_button(
                    "⬇️ CSV", csv_data,
                    file_name=f"snrt_{re.sub(r'[^a-z0-9]','_',query.lower()[:20])}.csv",
                    mime="text/csv", use_container_width=True,
                )

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            for i, res in enumerate(results):
                lang      = res.get("lang", "auto")
                source    = res.get("source_audio", "")
                start_sec = res.get("start_sec", 0)
                end_sec   = res.get("end_sec", start_sec + 15)
                # Use start_sec directly — exact segment start from Whisper timestamps.
                # This is more reliable than keyword-offset interpolation for video seek.
                ts_video  = int(start_sec)
                kw_hits   = res.get("keyword_hits", 0)
                texte     = display_text(res)
                score     = res.get("score", 0)
                sem_score = res.get("sem_score", score)
                pct       = score_pct(score, ce_used)
                sem_pct   = int(sem_score * 100)
                html_text = highlight(texte, keywords)

                lang_labels = {"fr": "🇫🇷 FR", "es": "🇪🇸 ES", "ar": "🇲🇦 AR"}
                lang_colors = {
                    "fr": ("rgba(29,78,216,.2)", "#93c5fd", "rgba(59,130,246,.3)"),
                    "es": ("rgba(185,28,28,.2)", "#fca5a5", "rgba(239,68,68,.3)"),
                    "ar": ("rgba(6,78,59,.2)",   "#6ee7b7", "rgba(16,185,129,.3)"),
                }
                lbg, lfg, lbd = lang_colors.get(lang, ("rgba(100,116,139,.2)","#94a3b8","rgba(100,116,139,.2)"))

                kw_html = (
                    f'<span class="px-2 py-0.5 rounded-full text-xs font-semibold" '
                    f'style="background:rgba(245,158,11,.12);color:#fcd34d;border:1px solid rgba(245,158,11,.25)">'
                    f'🔑 {kw_hits} mot{"s" if kw_hits>1 else ""}</span>'
                ) if kw_hits > 0 else ""

                vid_path = match_video(source, videos, lang)
                src_name = os.path.basename(vid_path) if vid_path else source

                card = f"""
<div class="relative rounded-xl p-5 mb-1 transition-all duration-200 hover:-translate-y-px group"
     style="background:linear-gradient(135deg,rgba(10,22,40,.95),rgba(15,31,56,.95));
            border:1px solid #243c5e;overflow:hidden">
  <div class="absolute left-0 top-0 bottom-0 w-1 rounded-l-xl"
       style="background:linear-gradient(180deg,#f59e0b,#3b82f6)"></div>
  <div class="absolute top-4 right-4 flex items-center justify-center w-7 h-7 rounded-full text-xs font-bold font-mono text-white"
       style="background:linear-gradient(135deg,#1d4ed8,#3b82f6);box-shadow:0 2px 8px rgba(59,130,246,.3)">
    {i+1}
  </div>
  <div class="flex flex-wrap items-center gap-2 mb-2">
    <span class="font-mono text-lg font-semibold" style="color:#f59e0b;letter-spacing:1px">
      ⏱ {h(fmt_timecode(ts_video))}
    </span>
    <span class="text-sm" style="color:#64748b">→ {h(fmt_timecode(end_sec))}</span>
    <span class="px-2.5 py-0.5 rounded-full text-xs font-bold"
          style="background:{lbg};color:{lfg};border:1px solid {lbd}">{lang_labels.get(lang,"❓")}</span>
    {kw_html}
  </div>
  <div class="flex items-center gap-2 mb-1">
    <div class="flex-1 h-1 rounded-full overflow-hidden" style="background:#1a2d4a">
      <div class="h-full rounded-full"
           style="width:{pct:.0f}%;background:linear-gradient(90deg,#3b82f6,#f59e0b);transition:width .5s"></div>
    </div>
    <span class="text-xs font-semibold font-mono" style="color:#94a3b8;min-width:38px">{pct:.0f}%</span>
  </div>
  <div class="flex items-center gap-2 mb-2">
    <span class="text-xs" style="color:#475569">similarité sémantique :</span>
    <span class="text-xs font-mono font-semibold" style="color:{'#6ee7b7' if sem_pct>=50 else '#fcd34d' if sem_pct>=35 else '#fca5a5'}">{sem_pct}%</span>
  </div>
  <div class="text-sm leading-relaxed mb-2" style="color:#e2e8f0">"{html_text}"</div>
  <div class="text-xs font-mono" style="color:#64748b">📁 {h(src_name)}</div>
</div>"""

                if vid_path:
                    c_vid, c_info = st.columns([3, 2])
                    with c_vid:
                        st.video(vid_path, start_time=int(start_sec))
                    with c_info:
                        st.markdown(card, unsafe_allow_html=True)
                elif videos:
                    c_vid, c_info = st.columns([3, 2])
                    with c_info:
                        st.markdown(card, unsafe_allow_html=True)
                        sel = st.selectbox("Choisir une vidéo :", list(videos.keys()), key=f"vs{i}")
                    with c_vid:
                        if sel:
                            st.video(videos[sel], start_time=int(start_sec))
                else:
                    st.markdown(card, unsafe_allow_html=True)
                    st.info(f"💡 Place `{source.replace('.wav','')}.mp4` dans `videos/`")
    else:
        st.markdown("""
        <div class="flex flex-col items-center justify-center py-24 gap-3" style="color:#64748b">
          <div style="font-size:4.5rem;animation:float 3s ease-in-out infinite">🎙️</div>
          <div class="text-lg font-semibold" style="color:#94a3b8">
            Tapez une requête pour explorer les archives SNRT
          </div>
          <div class="text-sm">Recherche sémantique multilingue · Français · Español</div>
        </div>
        <style>@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}</style>
        """, unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 2 — INDEXATION                                          ║
# ╚══════════════════════════════════════════════════════════════╝
with tab2:
    st.markdown("""
    <div class="mb-5">
      <div class="text-lg font-bold mb-3" style="color:#e2e8f0">📤 Indexer une nouvelle vidéo</div>
    </div>
    """, unsafe_allow_html=True)

    if whisper_m is None:
        st.error("❌ Le système de transcription est indisponible.")
        st.stop()
    if not FFMPEG:
        st.error("❌ Erreur système : traitement audio indisponible.")
        st.stop()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Choisir une vidéo (MP4, AVI, MKV, MOV)", type=["mp4","avi","mkv","mov"])

    if uploaded:
        save_path = UPLOAD_DIR / uploaded.name
        save_path.write_bytes(uploaded.read())

        cv, ci = st.columns([2, 1])
        with cv:
            st.video(str(save_path))
        with ci:
            st.markdown(f"""
            <div class="rounded-xl p-4" style="background:rgba(10,22,40,.95);border:1px solid #243c5e">
              <div class="font-semibold mb-3" style="color:#e2e8f0">📋 Fichier reçu</div>
              <div class="flex items-center gap-2 p-2 rounded-lg mb-2 text-sm font-mono"
                   style="background:#0f1f38;border:1px solid #1a2d4a;color:#94a3b8">
                📁 {h(uploaded.name)}
              </div>
              <div class="flex items-center gap-2 p-2 rounded-lg text-sm font-mono"
                   style="background:#0f1f38;border:1px solid #1a2d4a;color:#94a3b8">
                📦 {uploaded.size/1e6:.1f} MB
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        cl, cb = st.columns([2, 1])
        with cl:
            lang_sel_ui = st.selectbox(
                "Langue de la vidéo",
                ["Français 🇫🇷", "Espagnol 🇪🇸"],
                help="Spécifier la langue évite les erreurs d'encodage Whisper",
            )
        with cb:
            st.markdown("<br>", unsafe_allow_html=True)
            do_index = st.button("🚀 Transcrire et Indexer", use_container_width=True)

        if do_index:
            lang_map_up = {"Français 🇫🇷": "fr", "Espagnol 🇪🇸": "es"}
            lang_code   = lang_map_up[lang_sel_ui]
            audio_path  = UPLOAD_DIR / (save_path.stem + ".wav")
            idx_path    = FAISS_V2
            meta_path   = META_V2

            with st.status("🎬 Pipeline en cours…", expanded=True) as status:
                st.write("🔊 Étape 1/4 — Extraction audio (FFmpeg 16kHz mono)…")
                ok, err = extract_audio(FFMPEG, save_path, audio_path)
                if not ok:
                    st.error(f"❌ FFmpeg erreur : {err}")
                    st.stop()
                st.write(f"   ✅ Audio extrait : {audio_path.name}")

                st.write(f"📝 Étape 2/4 — Transcription Whisper ({wtype})…")
                bar = st.progress(0, text="Transcription…")
                segs_raw = transcribe(whisper_m, proc, wtype, audio_path, lang_code)
                bar.progress(1.0, text="✅ Transcription terminée")
                st.write(f"   ✅ {len(segs_raw)} segments · langue : {lang_code.upper()}")

                bi_label = "SNRT v2 fine-tuné" if embed_type == "finetuned" else "MiniLM-L12 Base"
                st.write(f"🧠 Étape 3/4 — Embeddings ({bi_label}, d=384)…")
                segments, texts = build_segments(segs_raw, uploaded.name, lang_code)
                if not texts:
                    st.error("❌ Aucun segment valide extrait.")
                    st.stop()

                st.write("🔍 Étape 4/4 — Mise à jour index FAISS…")
                total = add_to_index(embed_model, index, metadata, segments, texts, idx_path, meta_path)
                st.write(f"   ✅ Index mis à jour : {total:,} segments")
                status.update(label="✅ Indexation terminée !", state="complete")

            audio_path.unlink(missing_ok=True)

            st.markdown(f"""
            <div class="rounded-xl p-5 mt-4"
                 style="background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.25)">
              <div class="font-bold" style="color:#6ee7b7;font-size:1rem">
                🎉 {h(uploaded.name)} indexée avec succès !
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.info("🔍 Va dans **Rechercher** pour tester !")
    else:
        st.markdown("""
        <div class="flex flex-col items-center justify-center py-16 rounded-xl cursor-pointer transition-all"
             style="background:#0f1f38;border:2px dashed #243c5e">
          <div style="font-size:3.5rem;margin-bottom:.8rem">🎬</div>
          <div class="font-semibold mb-1" style="color:#94a3b8">
            Glisse une vidéo ici ou clique pour choisir
          </div>
          <div class="text-sm" style="color:#64748b">MP4 · AVI · MKV · MOV · max 2 Go</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        cu, cv = st.columns(2)
        for col, folder, label in [(cu, UPLOAD_DIR, "uploads/"), (cv, VIDEOS_DIR, "videos/")]:
            with col:
                st.markdown(f"<div class='text-sm font-semibold mb-2' style='color:#94a3b8'>{label}</div>", unsafe_allow_html=True)
                vids = list(folder.glob("*.mp4"))
                if vids:
                    for v in vids[:8]:
                        st.markdown(
                            f"<div class='flex items-center gap-2 p-2 mb-1 rounded-lg text-xs font-mono'"
                            f" style='background:#0f1f38;border:1px solid #1a2d4a;color:#94a3b8'>"
                            f"📁 {v.name} <span style='color:#64748b'>— {v.stat().st_size/1e6:.1f} MB</span></div>",
                            unsafe_allow_html=True,
                        )
                    if len(vids) > 8:
                        st.markdown(f"<div class='text-xs' style='color:#f59e0b'>+ {len(vids)-8} autres</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='text-sm' style='color:#64748b'>Aucune vidéo</div>", unsafe_allow_html=True)


