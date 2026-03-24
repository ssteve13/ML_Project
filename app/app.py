"""
Moodify — Emotion-Based Music Recommender  v2.0
================================================
New in v2:
  - Time-aware mood presets  (morning / afternoon / evening / night defaults)
  - Natural-language mood selector → auto-maps to audio features
  - Favourites list  (session-state, persist across rerenders)
  - In-page YouTube iframe embed  (no redirect to YT Music)
  - All v1 bug-fixes retained
"""

from __future__ import annotations

import pickle
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz, process

# ── youtubesearchpython (no API key required) ──────────────────────────────
try:
    from youtubesearchpython import VideosSearch
    YT_SEARCH_AVAILABLE = True
except ImportError:
    YT_SEARCH_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
MODEL_PATH  = BASE / "models" / "emotion_model.pkl"
SCALER_PATH = BASE / "models" / "scaler.pkl"
DATA_PATH   = BASE / "data"   / "songs.csv"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎧 Moodify",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(ellipse at top left, #1a0a2e 0%, #0f0f1a 50%, #0a1628 100%);
    min-height: 100vh;
}

/* ── Hero ── */
.hero { text-align:center; padding:2rem 1rem 1rem; animation: fadeDown .6s ease; }
.hero h1 {
    font-size: clamp(2rem,5vw,3.2rem); font-weight:800;
    background: linear-gradient(90deg,#a78bfa,#7c3aed,#c4b5fd,#818cf8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    margin:0; letter-spacing:-1px;
}
.hero p { color:#94a3b8; font-size:.95rem; margin:.3rem 0 0; }

/* ── Time badge ── */
.time-badge {
    display:inline-flex; align-items:center; gap:.4rem;
    background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.1);
    border-radius:50px; padding:.3rem .9rem; font-size:.78rem;
    color:#94a3b8; margin-top:.6rem;
}

/* ── Glass card ── */
.glass {
    background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
    border-radius:16px; padding:1.4rem 1.6rem; margin-bottom:1rem;
    backdrop-filter:blur(12px);
}

/* ── Section label ── */
.slabel {
    font-size:.68rem; font-weight:700; letter-spacing:.12em;
    text-transform:uppercase; color:#7c3aed; margin-bottom:.45rem;
}

/* ── Mood pill grid ── */
.mood-grid {
    display:flex; flex-wrap:wrap; gap:.5rem; margin-bottom:.8rem;
}
.mood-pill {
    padding:.35rem .85rem; border-radius:50px; font-size:.8rem; font-weight:600;
    border:1px solid; cursor:pointer; transition:all .2s ease;
    user-select:none;
}
.mood-pill:hover { transform:translateY(-2px); }

/* ── Song card ── */
.song-card {
    display:flex; align-items:center; gap:.9rem;
    background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
    border-radius:14px; padding:.9rem 1.1rem; margin-bottom:.6rem;
    transition:all .22s cubic-bezier(.4,0,.2,1);
    animation:fadeUp .35s ease both; cursor:pointer;
}
.song-card:hover {
    background:rgba(167,139,250,.1); border-color:rgba(167,139,250,.45);
    transform:translateX(5px); box-shadow:0 4px 20px rgba(124,58,237,.2);
}
.song-card:nth-child(1){animation-delay:.04s}
.song-card:nth-child(2){animation-delay:.08s}
.song-card:nth-child(3){animation-delay:.12s}
.song-card:nth-child(4){animation-delay:.16s}
.song-card:nth-child(5){animation-delay:.20s}
.song-card:nth-child(6){animation-delay:.24s}
.song-card:nth-child(7){animation-delay:.28s}
.song-card:nth-child(8){animation-delay:.32s}

.song-rank { font-size:1.2rem; font-weight:700; color:rgba(167,139,250,.5); min-width:1.8rem; text-align:center; }
.song-info { flex:1; min-width:0; }
.song-name  { font-weight:600; font-size:.9rem; color:#e2e8f0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.song-artist{ font-size:.76rem; color:#94a3b8; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.song-meta  { font-size:.72rem; color:#64748b; text-align:right; white-space:nowrap; }
.play-icon  { font-size:1.2rem; opacity:.55; transition:opacity .2s,transform .2s; }
.song-card:hover .play-icon { opacity:1; transform:scale(1.15); }

/* ── Genre chip ── */
.gchip {
    display:inline-block; background:rgba(124,58,237,.15);
    border:1px solid rgba(124,58,237,.3); color:#a78bfa;
    border-radius:5px; padding:.1rem .4rem; font-size:.68rem;
    font-weight:600; margin-left:.3rem;
}

/* ── Popularity bar ── */
.pb-bg  { background:rgba(255,255,255,.08); border-radius:4px; height:4px; width:56px; display:inline-block; vertical-align:middle; margin-left:3px; overflow:hidden; }
.pb-fill{ height:100%; border-radius:4px; background:linear-gradient(90deg,#7c3aed,#a78bfa); }

/* ── Emotion badge ── */
.emotion-badge {
    display:inline-flex; align-items:center; gap:.4rem;
    padding:.45rem 1.1rem; border-radius:50px;
    font-weight:700; font-size:1rem; letter-spacing:.4px;
    animation:pulse 2.5s infinite;
}
.emotion-Happy    {background:rgba(251,191,36,.12); border:1px solid rgba(251,191,36,.35); color:#fcd34d;}
.emotion-Sad      {background:rgba(96,165,250,.12);  border:1px solid rgba(96,165,250,.35);  color:#93c5fd;}
.emotion-Calm     {background:rgba(52,211,153,.12);  border:1px solid rgba(52,211,153,.35);  color:#6ee7b7;}
.emotion-Energetic{background:rgba(251,113,133,.12); border:1px solid rgba(251,113,133,.35); color:#fca5a5;}

/* ── YT embed ── */
.yt-wrap {
    position:relative; border-radius:14px; overflow:hidden;
    border:1px solid rgba(255,255,255,.1); background:#000;
    box-shadow:0 8px 32px rgba(0,0,0,.5);
    animation:fadeDown .5s ease;
}
.yt-wrap iframe { display:block; width:100%; border:none; aspect-ratio:16/9; }
.yt-label { font-size:.75rem; color:#94a3b8; padding:.5rem .8rem .6rem; }

/* ── Playing chip ── */
.playing-chip {
    display:inline-flex; align-items:center; gap:.3rem;
    background:rgba(239,68,68,.15); border:1px solid rgba(239,68,68,.35);
    color:#fca5a5; border-radius:50px; padding:.2rem .7rem;
    font-size:.72rem; font-weight:700; animation:pulse 1.5s infinite;
}

/* ── Favs panel ── */
.fav-card {
    display:flex; align-items:center; gap:.7rem;
    background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.06);
    border-radius:10px; padding:.6rem .9rem; margin-bottom:.5rem;
    transition:all .2s;
}
.fav-card:hover { border-color:rgba(239,68,68,.35); background:rgba(239,68,68,.05); }
.fav-name  { flex:1; font-size:.85rem; color:#e2e8f0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.fav-artist{ font-size:.72rem; color:#94a3b8; }

/* ── Divider ── */
.divider { height:1px; background:linear-gradient(90deg,transparent,rgba(167,139,250,.25),transparent); margin:1.2rem 0; }

/* ── Button ── */
.stButton>button {
    width:100%; background:linear-gradient(135deg,#7c3aed,#6d28d9);
    color:#fff; border:none; border-radius:12px; padding:.8rem 1.5rem;
    font-size:.95rem; font-weight:700; font-family:'Inter',sans-serif;
    letter-spacing:.03em; cursor:pointer;
    transition:all .22s ease; box-shadow:0 4px 18px rgba(124,58,237,.38);
}
.stButton>button:hover { transform:translateY(-2px); box-shadow:0 8px 28px rgba(124,58,237,.55); }
.stButton>button:active { transform:translateY(0); }

/* ── Input ── */
div[data-testid="stTextInput"] input {
    background:rgba(255,255,255,.05)!important; border:1px solid rgba(255,255,255,.11)!important;
    border-radius:10px!important; color:#e2e8f0!important; font-family:'Inter',sans-serif!important;
    transition:border-color .2s!important;
}
div[data-testid="stTextInput"] input:focus {
    border-color:rgba(167,139,250,.55)!important;
    box-shadow:0 0 0 3px rgba(124,58,237,.12)!important;
}

/* ── Selectbox ── */
div[data-testid="stSelectbox"] > div { border-radius:10px!important; }

/* ── Animations ── */
@keyframes fadeDown { from{opacity:0;transform:translateY(-18px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeUp   { from{opacity:0;transform:translateY(14px)}  to{opacity:1;transform:translateY(0)} }
@keyframes pulse    { 0%,100%{box-shadow:0 0 0 0 rgba(124,58,237,0)} 50%{box-shadow:0 0 14px 3px rgba(124,58,237,.15)} }

/* ── Responsive ── */
@media(max-width:640px){
    .song-meta,.song-rank{display:none}
    .song-card{padding:.7rem .85rem}
    .song-name{font-size:.84rem}
}

#MainMenu,footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# Constants / mapping tables
# ══════════════════════════════════════════════════════════════════════════

EMOTION_META = {
    "Happy":     ("😄", "Happy"),
    "Sad":       ("😔", "Sad"),
    "Calm":      ("😌", "Calm"),
    "Energetic": ("⚡", "Energetic"),
}

# Mood selector → (energy, valence, danceability, acousticness, tempo)
MOOD_PRESETS: dict[str, dict] = {
    "😄 Happy":       dict(energy=0.78, valence=0.82, danceability=0.72, acousticness=0.15, tempo=128),
    "😔 Sad":         dict(energy=0.25, valence=0.22, danceability=0.35, acousticness=0.65, tempo=78),
    "😌 Chill":       dict(energy=0.38, valence=0.58, danceability=0.50, acousticness=0.55, tempo=96),
    "⚡ Energetic":   dict(energy=0.90, valence=0.65, danceability=0.80, acousticness=0.08, tempo=145),
    "🤍 Romantic":    dict(energy=0.45, valence=0.70, danceability=0.55, acousticness=0.50, tempo=88),
    "🧘 Focus":       dict(energy=0.35, valence=0.50, danceability=0.30, acousticness=0.70, tempo=85),
    "🥳 Party":       dict(energy=0.88, valence=0.80, danceability=0.90, acousticness=0.05, tempo=128),
    "😤 Angry":       dict(energy=0.92, valence=0.20, danceability=0.60, acousticness=0.05, tempo=155),
    "🌙 Late Night":  dict(energy=0.30, valence=0.40, danceability=0.45, acousticness=0.60, tempo=90),
    "☀️ Morning":     dict(energy=0.55, valence=0.68, danceability=0.55, acousticness=0.40, tempo=110),
    "🎸 Nostalgic":   dict(energy=0.48, valence=0.55, danceability=0.48, acousticness=0.55, tempo=100),
}

# Time-of-day → auto mood label
def _time_preset() -> tuple[str, str, dict]:
    """Return (label, icon, feature_dict) based on current hour (IST)."""
    hour = datetime.now().hour
    if 5 <= hour < 10:
        return "☀️ Morning", "Good morning", MOOD_PRESETS["☀️ Morning"]
    if 10 <= hour < 14:
        return "😄 Happy", "Good day", MOOD_PRESETS["😄 Happy"]
    if 14 <= hour < 18:
        return "⚡ Energetic", "Afternoon grind", MOOD_PRESETS["⚡ Energetic"]
    if 18 <= hour < 22:
        return "😌 Chill", "Evening wind-down", MOOD_PRESETS["😌 Chill"]
    return "🌙 Late Night", "Late night vibes", MOOD_PRESETS["🌙 Late Night"]

# ══════════════════════════════════════════════════════════════════════════
# Cached loaders
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], inplace=True)

    def _emotion(r):
        v, e = r["valence"], r["energy"]
        if v > 0.6 and e > 0.6: return "Happy"
        if v < 0.4 and e < 0.4: return "Sad"
        if v > 0.6:              return "Calm"
        return "Energetic"

    df["emotion"]      = df.apply(_emotion, axis=1)
    df["youtube_link"] = df.apply(
        lambda r: "https://music.youtube.com/search?q=" + quote_plus(f"{r['track_name']} {r['artists']}"),
        axis=1,
    )
    df["_track_lower"]  = df["track_name"].str.lower().str.strip()
    df["_artist_lower"] = df["artists"].str.lower().str.strip()
    df["_combined"]     = df["_track_lower"] + " " + df["_artist_lower"]
    return df


# ══════════════════════════════════════════════════════════════════════════
# YouTube search helper (no API key)
# ══════════════════════════════════════════════════════════════════════════
_yt_cache: dict[str, str] = {}
_yt_lock = threading.Lock()

def get_youtube_video_id(track_name: str, artists: str) -> str | None:
    """Search YouTube for the track and return the video ID (cached)."""
    if not YT_SEARCH_AVAILABLE:
        return None
    key = f"{track_name}|||{artists}"
    with _yt_lock:
        if key in _yt_cache:
            return _yt_cache[key]
    try:
        results = VideosSearch(f"{track_name} {artists} official audio", limit=1)
        data    = results.result()
        vid_id  = data["result"][0]["id"] if data["result"] else None
        with _yt_lock:
            _yt_cache[key] = vid_id
        return vid_id
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════
# Fuzzy search
# ══════════════════════════════════════════════════════════════════════════
def fuzzy_find_songs(
    df: pd.DataFrame,
    queries: list[str],
    score_cutoff: int = 60,
    top_n: int = 30,
) -> tuple[pd.DataFrame, list[str]]:
    matched: set[int] = set()
    unmatched: list[str] = []
    choices = df["_combined"].tolist()

    for q in queries:
        q = q.lower().strip()
        if not q:
            continue
        hits = [i for _, _, i in process.extract(q, choices, scorer=fuzz.WRatio, limit=top_n, score_cutoff=score_cutoff)]
        if not hits:
            hits = [i for _, _, i in process.extract(q, df["_track_lower"].tolist(), scorer=fuzz.WRatio, limit=top_n, score_cutoff=score_cutoff)]
        if hits:
            matched.update(hits)
        else:
            unmatched.append(q)

    return (df.iloc[list(matched)].copy(), unmatched) if matched else (pd.DataFrame(), unmatched)


# ══════════════════════════════════════════════════════════════════════════
# Session state helpers
# ══════════════════════════════════════════════════════════════════════════
def init_state() -> None:
    defaults = {
        "favourites": [],        # list[dict] with track_name, artists, youtube_link
        "now_playing": None,     # dict | None
        "now_playing_vid": None, # YouTube video ID | None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def is_fav(track: str, artist: str) -> bool:
    return any(f["track_name"] == track and f["artists"] == artist for f in st.session_state.favourites)


def toggle_fav(row: pd.Series) -> None:
    key = {"track_name": row["track_name"], "artists": row["artists"], "youtube_link": row["youtube_link"]}
    if is_fav(row["track_name"], row["artists"]):
        st.session_state.favourites = [f for f in st.session_state.favourites if not (f["track_name"] == row["track_name"] and f["artists"] == row["artists"])]
    else:
        st.session_state.favourites.append(key)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    init_state()

    try:
        model, scaler = load_model_and_scaler()
        df = load_data()
    except FileNotFoundError as e:
        st.error(f"⚠️ Missing file: `{e.filename}`. Run Streamlit from the project root.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Load error: {e}")
        st.stop()

    # ── Time preset ──────────────────────────────────────────────────────
    auto_mood_label, time_greeting, time_features = _time_preset()
    now_str = datetime.now().strftime("%I:%M %p")

    # ── Hero ─────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="hero">
            <h1>🎵 Moodify</h1>
            <p>AI-powered music tuned to your mood</p>
            <span class="time-badge">🕐 {now_str} &nbsp;·&nbsp; {time_greeting}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════
    # Layout: Left control panel | Right results + player
    # ══════════════════════════════════════════════════════════════════════
    left, right = st.columns([1.05, 1], gap="large")

    # ─────────────────────────── LEFT PANEL ──────────────────────────────
    with left:

        # 1. Mood selector ────────────────────────────────────────────────
        st.markdown('<div class="slabel">✨ What\'s your mood?</div>', unsafe_allow_html=True)
        mood_options = list(MOOD_PRESETS.keys())
        mood_default_idx = mood_options.index(auto_mood_label) if auto_mood_label in mood_options else 0
        selected_mood = st.selectbox(
            "Mood", mood_options, index=mood_default_idx, label_visibility="collapsed"
        )
        mood_features = MOOD_PRESETS[selected_mood]

        # 2. Seed songs ───────────────────────────────────────────────────
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="slabel">🔍 Favourite songs (optional)</div>', unsafe_allow_html=True)
        user_input = st.text_input(
            "Seeds", value="", placeholder="e.g. Vaathi Coming, Levitating, Blinding Lights",
            label_visibility="collapsed",
        )

        raw_queries = [q.strip() for q in user_input.split(",") if q.strip()]
        matched_df, unmatched = fuzzy_find_songs(df, raw_queries) if raw_queries else (pd.DataFrame(), [])

        if unmatched:
            st.caption(f"Couldn't find: {', '.join(f'**{q}**' for q in unmatched)}")

        # Blend mood-preset with matched songs (60 / 40 weight if seeds given)
        if not matched_df.empty:
            w = 0.4  # weight towards song averages
            avg = lambda col: w * float(matched_df[col].mean()) + (1 - w) * mood_features[col]
            e_def = avg("energy"); v_def = avg("valence"); d_def = avg("danceability")
            a_def = avg("acousticness"); t_def = w * float(matched_df["tempo"].mean()) + (1 - w) * mood_features["tempo"]
        else:
            e_def, v_def, d_def = mood_features["energy"], mood_features["valence"], mood_features["danceability"]
            a_def, t_def = mood_features["acousticness"], float(mood_features["tempo"])

        # 3. Fine-tune sliders ────────────────────────────────────────────
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="slabel">🎛️ Fine-tune</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            energy       = st.slider("⚡ Energy",       0.0, 1.0, round(e_def, 2), 0.01)
            danceability = st.slider("💃 Danceability", 0.0, 1.0, round(d_def, 2), 0.01)
            acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, round(a_def, 2), 0.01)
        with c2:
            valence   = st.slider("😊 Positivity",  0.0, 1.0, round(v_def, 2), 0.01)
            tempo     = st.slider("🥁 Tempo (BPM)", 50,  220, int(t_def))
            n_results = st.slider("🎯 Results",     1,   20,  8)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        go = st.button("✨ Discover Music", use_container_width=True)

        # 4. Favourites panel ─────────────────────────────────────────────
        if st.session_state.favourites:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="slabel">❤️ Favourites ({len(st.session_state.favourites)})</div>', unsafe_allow_html=True)
            for fav in st.session_state.favourites:
                col_f, col_btn = st.columns([4, 1])
                with col_f:
                    st.markdown(
                        f'<div class="fav-card">'
                        f'  <div>'
                        f'    <div class="fav-name">{fav["track_name"]}</div>'
                        f'    <div class="fav-artist">{fav["artists"]}</div>'
                        f'  </div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    if st.button("▶", key=f"play_fav_{fav['track_name']}_{fav['artists']}", help="Play"):
                        vid = get_youtube_video_id(fav["track_name"], fav["artists"])
                        st.session_state.now_playing     = fav
                        st.session_state.now_playing_vid = vid
                        st.rerun()

    # ─────────────────────────── RIGHT PANEL ─────────────────────────────
    with right:

        # ── Now Playing Player ────────────────────────────────────────────
        if st.session_state.now_playing:
            np_info = st.session_state.now_playing
            vid_id  = st.session_state.now_playing_vid

            st.markdown(
                f'<div style="margin-bottom:.6rem;">'
                f'  <span class="playing-chip">🔴 NOW PLAYING</span>'
                f'  <span style="font-size:.82rem; color:#94a3b8; margin-left:.5rem;">'
                f'    {np_info["track_name"]} — {np_info["artists"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if vid_id:
                st.markdown(
                    f'<div class="yt-wrap">'
                    f'  <iframe src="https://www.youtube.com/embed/{vid_id}?autoplay=1&rel=0&modestbranding=1"'
                    f'          allow="autoplay; encrypted-media" allowfullscreen></iframe>'
                    f'  <div class="yt-label">▶ YouTube · {np_info["track_name"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Fallback: open YT Music search
                yt_url = "https://music.youtube.com/search?q=" + quote_plus(f"{np_info['track_name']} {np_info['artists']}")
                st.markdown(
                    f'<div class="yt-wrap" style="padding:1.5rem; text-align:center;">'
                    f'  <div style="color:#94a3b8; font-size:.85rem;">🎵 YouTube embed not available for this track.</div>'
                    f'  <a href="{yt_url}" target="_blank" style="color:#a78bfa; font-size:.82rem; margin-top:.5rem; display:block;">Open in YouTube Music ↗</a>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            if st.button("✕ Close player", key="close_player"):
                st.session_state.now_playing     = None
                st.session_state.now_playing_vid = None
                st.rerun()

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Recommendations ───────────────────────────────────────────────
        if go:
            try:
                # Feature order: [danceability, energy, acousticness, tempo]  ← matches scaler fit
                input_scaled = scaler.transform([[danceability, energy, acousticness, tempo]])
                predicted_emotion: str = model.predict(input_scaled)[0]
            except Exception as e:
                st.error(f"⚠️ Prediction error: {e}")
                st.stop()

            emoji, label = EMOTION_META.get(predicted_emotion, ("🎵", predicted_emotion))
            st.markdown(
                f'<div style="margin-bottom:.8rem;">'
                f'  <div class="slabel">🧠 Detected mood</div>'
                f'  <span class="emotion-badge emotion-{predicted_emotion}">{emoji}&nbsp;{label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            pool = df[df["emotion"] == predicted_emotion].copy()
            if pool.empty:
                st.info("No songs found. Try adjusting sliders.")
                return

            pool["score"] = (
                0.30 * abs(pool["energy"]       - energy)       +
                0.25 * abs(pool["danceability"] - danceability) +
                0.20 * abs(pool["valence"]       - valence)      +
                0.15 * abs(pool["acousticness"] - acousticness) +
                0.10 * abs(pool["tempo"].clip(50, 220) / 220 - tempo / 220)
            )
            recs = (
                pool
                .sort_values(["score", "popularity"], ascending=[True, False])
                .drop_duplicates(subset=["track_name", "artists"])
                .head(n_results)
                .reset_index(drop=True)
            )

            st.markdown(f'<div class="slabel">🎵 Top {len(recs)} picks</div>', unsafe_allow_html=True)

            for i, row in recs.iterrows():
                pop   = int(row.get("popularity", 50))
                genre = row.get("track_genre", "")
                gchip = f'<span class="gchip">{genre}</span>' if genre else ""
                fav_icon = "❤️" if is_fav(row["track_name"], row["artists"]) else "🤍"
                pb    = f'<div class="pb-bg"><div class="pb-fill" style="width:{pop}%"></div></div>'

                # Song card HTML (clicking it sets now_playing via button)
                card_id = f"play_{i}_{row['track_name'][:12].replace(' ','_')}"

                col_card, col_actions = st.columns([5, 1])
                with col_card:
                    st.markdown(
                        f'<div class="song-card">'
                        f'  <div class="song-rank">#{i+1}</div>'
                        f'  <div class="song-info">'
                        f'    <div class="song-name">{row["track_name"]}{gchip}</div>'
                        f'    <div class="song-artist">{row["artists"]}</div>'
                        f'  </div>'
                        f'  <div class="song-meta"><div>Pop {pop}{pb}</div><div>{round(row["tempo"])} BPM</div></div>'
                        f'  <div class="play-icon">▶</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with col_actions:
                    # Play button
                    if st.button("▶", key=card_id, help="Play in page"):
                        vid = get_youtube_video_id(row["track_name"], row["artists"])
                        st.session_state.now_playing     = row.to_dict()
                        st.session_state.now_playing_vid = vid
                        st.rerun()
                    # Fav toggle
                    if st.button(fav_icon, key=f"fav_{card_id}", help="Toggle favourite"):
                        toggle_fav(row)
                        st.rerun()

        elif not st.session_state.now_playing:
            # Placeholder
            st.markdown(
                """
                <div style="height:320px; display:flex; flex-direction:column;
                            align-items:center; justify-content:center;
                            text-align:center; gap:1rem; opacity:.45;">
                    <div style="font-size:3.5rem;">🎧</div>
                    <div style="font-size:.92rem; color:#94a3b8; max-width:240px; line-height:1.6;">
                        Choose your mood, add seed songs, then hit
                        <strong>Discover Music</strong>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    with st.expander("ℹ️ How Moodify works"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
**🕐 Time-aware presets**  
Moodify reads the current time and pre-selects a matching mood (e.g. ☀️ Morning → moderate energy & positivity).

**✨ Mood selector**  
Pick from 11 moods. Each maps to a set of Spotify audio features. Seed songs fine-tune the sliders further.

**🔍 Fuzzy seed search**  
Type song names with typos — rapidfuzz's `WRatio` scorer handles it.
""")
        with col_b:
            st.markdown("""
**🧠 AI emotion detection**  
A Random Forest model (trained on Spotify data) maps your audio feature vector to  
😄 Happy · 😔 Sad · 😌 Calm · ⚡ Energetic.

**▶ In-page player**  
Clicking ▶ searches YouTube for the track and embeds it directly — no redirect needed.

**❤️ Favourites**  
Heart any song to save it. Tap ▶ from the favourites panel to replay anytime.
""")


if __name__ == "__main__":
    main()