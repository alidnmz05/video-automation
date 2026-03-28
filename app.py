"""
AI Dark-Story Video Automator — Streamlit UI
=============================================
Launch: streamlit run app.py
"""

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Dark Story Automator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIG_FILE = Path(__file__).parent / "config.json"
OUTPUT_DIR = Path(__file__).parent / "output"
TEMP_DIR = Path(__file__).parent / "temp"

OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Dark cinematic theme */
    .stApp { background-color: #0e0e0e; color: #e0e0e0; }
    section[data-testid="stSidebar"] { background-color: #161616; border-right: 1px solid #2a2a2a; }

    /* Banner */
    .banner {
        background: linear-gradient(135deg, #1a0a00 0%, #2d0000 50%, #0a0a1a 100%);
        border: 1px solid #ff3300;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        text-align: center;
    }
    .banner h1 { color: #ff3300; font-size: 2.4rem; margin: 0; letter-spacing: 2px; }
    .banner p  { color: #888; margin: 6px 0 0; font-size: 0.95rem; }

    /* Step cards */
    .step-card {
        background: #1a1a1a;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 3px solid #333;
        font-size: 0.9rem;
        font-family: monospace;
    }
    .step-done    { border-left-color: #00cc66; color: #00cc66; }
    .step-running { border-left-color: #ffaa00; color: #ffaa00; }
    .step-waiting { border-left-color: #333;    color: #555; }
    .step-error   { border-left-color: #ff3333; color: #ff3333; }

    /* Output video box */
    .video-box {
        background: #111;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }

    /* Metric pill */
    .metric-pill {
        background: #1e1e1e;
        border: 1px solid #2a2a2a;
        border-radius: 20px;
        padding: 6px 16px;
        display: inline-block;
        font-size: 0.85rem;
        color: #aaa;
    }

    /* Hide Streamlit brand */
    #MainMenu, footer { visibility: hidden; }

    /* Buttons */
    .stButton > button {
        background: #cc2200;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1rem;
        padding: 10px 28px;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #ff3300; }

    /* Inputs */
    .stTextInput > div > div > input,
    .stTextArea textarea,
    .stSelectbox > div > div {
        background: #1a1a1a !important;
        color: #e0e0e0 !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"]  { gap: 8px; }
    .stTabs [data-baseweb="tab"]       { background: #1a1a1a; border-radius: 6px 6px 0 0; color: #888; }
    .stTabs [aria-selected="true"]     { background: #2a0000; color: #ff3300; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def save_config(cfg: dict) -> None:
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def get_output_videos() -> list[Path]:
    return sorted(OUTPUT_DIR.glob("*.mp4"), key=os.path.getmtime, reverse=True)


def human_size(path: Path) -> str:
    b = path.stat().st_size
    return f"{b / 1_048_576:.1f} MB" if b >= 1_048_576 else f"{b / 1024:.0f} KB"


# ── QueueHandler for live logs ────────────────────────────────────────────────


class QueueHandler(logging.Handler):
    """Push log records into a queue for Streamlit to consume."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        self.log_queue.put(self.format(record))


# ── Pipeline runner (runs in a background thread) ─────────────────────────────

STEP_NAMES = [
    "Generating script (GPT-4o)",
    "Generating voiceover (ElevenLabs)",
    "Generating subtitles (Whisper)",
    "Downloading stock videos (Pexels)",
    "Assembling final video (MoviePy)",
]

STEP_NAMES_CUSTOM = [
    "Script provided by user ✍️",
    "Generating voiceover (ElevenLabs)",
    "Generating subtitles (Whisper)",
    "Downloading stock videos (Pexels)",
    "Assembling final video (MoviePy)",
]


def _run_pipeline(
    topic: str,
    config: dict,
    log_q: queue.Queue,
    state: dict,
    style: str = "dark_mystery",
    language: str = "en",
    custom_script: str = "",
    preset_keywords: list = None,
) -> None:
    """
    Runs the full 5-step pipeline in a background thread.
    If custom_script is provided, Step 1 (GPT-4o) is skipped.
    Writes progress into `state` dict and logs into `log_q`.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = TEMP_DIR / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)

    # Wire logging into the queue
    handler = QueueHandler(log_q)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    _names = STEP_NAMES_CUSTOM if custom_script else STEP_NAMES

    def step(n: int) -> None:
        state["step"] = n
        log_q.put(f"{'─'*48}")
        log_q.put(f"▶  STEP {n}/5 — {_names[n - 1]}")
        log_q.put(f"{'─'*48}")

    try:
        # Step 1 — Script (skip if custom script provided)
        step(1)
        if custom_script.strip():
            script   = custom_script.strip()
            # Use manually provided keywords, fallback to topic words
            keywords = preset_keywords if preset_keywords else [w for w in topic.split() if len(w) > 3][:5] or [topic]
            log_q.put(f"✍️  Using custom script ({len(script)} chars)")
            log_q.put(f"🔍  Pexels keywords: {keywords}")
        else:
            from modules.script_generator import generate_script
            result   = generate_script(topic, config["api_keys"]["openai"], language=language)
            script   = result["script"]
            keywords = result["keywords"]
        (session_dir / "script.txt").write_text(script, encoding="utf-8")
        state["script"]   = script
        state["keywords"] = keywords

        # Step 2 — Voiceover
        step(2)
        from modules.voiceover import generate_voiceover
        el_cfg = config.get("elevenlabs_tr", config["elevenlabs"]) if language == "tr" else config["elevenlabs"]
        audio_path = generate_voiceover(
            script,
            str(session_dir),
            config["api_keys"]["elevenlabs"],
            el_cfg["voice_id"],
            el_cfg["model_id"],
        )

        # Step 3 — Subtitles
        step(3)
        from modules.subtitle_generator import generate_subtitles
        srt_path = generate_subtitles(audio_path, str(session_dir), config["api_keys"]["openai"])

        # Step 4 — Stock videos
        step(4)
        from modules.asset_sourcer import download_assets
        clips = download_assets(keywords, str(session_dir), config["api_keys"]["pexels"])

        # Step 5 — Assemble
        step(5)
        from modules.video_assembler import assemble_video
        output_path = str(OUTPUT_DIR / f"video_{language}_{timestamp}.mp4")
        assemble_video(audio_path, clips, srt_path, output_path, config)

        state["output"]  = output_path
        state["step"]    = 6          # "done"
        state["running"] = False
        log_q.put("✅  Pipeline complete!")

    except Exception as exc:
        state["error"]   = str(exc)
        state["running"] = False
        log_q.put(f"❌  ERROR: {exc}")
        logging.exception("Pipeline failed")

    finally:
        root_logger.removeHandler(handler)


# ── Session-state initialisation ──────────────────────────────────────────────


def init_state() -> None:
    defaults = {
        "running":  False,
        "log_lines": [],
        "pipeline": {},           # shared dict with thread
        "log_queue": queue.Queue(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ── Sidebar — Configuration ───────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    cfg = load_config()

    with st.expander("🔑  API Keys", expanded=not bool(cfg.get("api_keys", {}).get("openai"))):
        openai_key     = st.text_input("OpenAI Key",     value=cfg.get("api_keys", {}).get("openai", ""),     type="password", key="s_oai")
        elevenlabs_key = st.text_input("ElevenLabs Key", value=cfg.get("api_keys", {}).get("elevenlabs", ""), type="password", key="s_el")
        pexels_key     = st.text_input("Pexels Key",     value=cfg.get("api_keys", {}).get("pexels", ""),     type="password", key="s_px")
        youtube_key    = st.text_input("YouTube Data API Key", value=cfg.get("api_keys", {}).get("youtube", ""), type="password", key="s_yt",
                                       help="Required for Trend Hunt. Get free key at console.cloud.google.com")

    with st.expander("🎙️  Voice Settings"):
        st.markdown("**🇬🇧 English Voice**")
        voice_id = st.text_input(
            "ElevenLabs Voice ID (EN)",
            value=cfg.get("elevenlabs", {}).get("voice_id", "21m00Tcm4TlvDq8ikWAM"),
        )
        model_id = st.selectbox(
            "Model (EN)",
            ["eleven_monolingual_v1", "eleven_multilingual_v2", "eleven_turbo_v2"],
            index=0,
        )
        st.markdown("**🇹🇷 Turkish Voice**")
        tr_voice_id = st.text_input(
            "ElevenLabs Voice ID (TR)",
            value=cfg.get("elevenlabs_tr", {}).get("voice_id", ""),
            placeholder="Multilingual voice ID for Turkish",
        )
        tr_model_id = st.selectbox(
            "Model (TR)",
            ["eleven_multilingual_v2", "eleven_turbo_v2"],
            index=0,
        )

    with st.expander("🖋️  Typography"):
        font_name    = st.selectbox("Font",  ["Impact", "Montserrat"], index=0)
        font_size    = st.slider("Size",     40, 140, cfg.get("typography", {}).get("font_size", 90))
        font_color   = st.selectbox("Color", ["yellow", "white", "red", "cyan"], index=0)
        stroke_width = st.slider("Stroke",   0, 8,   cfg.get("typography", {}).get("stroke_width", 3))

    with st.expander("🎬  Video Settings"):
        overlay_opacity = st.slider(
            "Dark overlay opacity", 0.0, 0.8,
            float(cfg.get("video", {}).get("overlay_opacity", 0.2)), 0.05,
        )
        clips_per_kw = st.slider("Stock clips per keyword", 1, 4, 2)

    if st.button("💾  Save Settings", use_container_width=True):
        cfg.update({
            "api_keys": {
                "openai":     openai_key.strip(),
                "elevenlabs": elevenlabs_key.strip(),
                "pexels":     pexels_key.strip(),
                "youtube":    youtube_key.strip(),
            },
            "elevenlabs":    {"voice_id": voice_id,    "model_id": model_id},
            "elevenlabs_tr": {"voice_id": tr_voice_id, "model_id": tr_model_id},
            "typography": {
                "font":         font_name,
                "font_size":    font_size,
                "font_color":   font_color,
                "stroke_color": "black",
                "stroke_width": stroke_width,
            },
            "video": {
                "width":           cfg.get("video", {}).get("width", 1080),
                "height":          cfg.get("video", {}).get("height", 1920),
                "fps":             cfg.get("video", {}).get("fps", 30),
                "overlay_opacity": overlay_opacity,
            },
            "output_dir": "output",
            "temp_dir":   "temp",
        })
        save_config(cfg)
        st.success("Settings saved!")

    st.markdown("---")
    videos = get_output_videos()
    st.markdown(f"**📦  Generated videos:** {len(videos)}")
    if videos:
        total_mb = sum(v.stat().st_size for v in videos) / 1_048_576
        st.markdown(f'<div class="metric-pill">Total size: {total_mb:.1f} MB</div>', unsafe_allow_html=True)

# ── Banner ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="banner">
        <h1>🎬 DARK STORY AUTOMATOR</h1>
        <p>AI-powered · One click · Ready to upload</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Main tabs ─────────────────────────────────────────────────────────────────

tab_gen, tab_trend, tab_history, tab_preview = st.tabs(["⚡  Generate", "🔥  Trend Hunt", "📂  History", "▶️  Preview"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Generate
# ════════════════════════════════════════════════════════════════════════════
with tab_gen:
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Left: input ──────────────────────────────────────────────────────────
    with col_left:
        # ── Script Mode Toggle ─────────────────────────────────────────────
        script_mode = st.radio(
            "✍️ Script Mode",
            ["🤖 AI Writes the Script", "✍️ I'll Write the Script"],
            horizontal=True,
            key="script_mode",
        )
        use_custom = "I'll Write" in script_mode

        if use_custom:
            st.markdown("### ✍️ Your Script")
            custom_script_text = st.text_area(
                "Paste or type your full script here",
                placeholder="Metni buraya yapıştır. Her cümle kısa ve güçlü olsun.",
                height=220,
                label_visibility="collapsed",
                key="custom_script_text",
            )
            st.markdown("### 🔍 Keywords for Stock Video (Pexels)")
            keywords_input = st.text_input(
                "Comma separated English keywords",
                placeholder="dark alley, hacker, mystery, car crash",
                label_visibility="collapsed",
                key="custom_keywords",
            )
            topic = st.text_input(
                "Short topic label (for filename)",
                placeholder="e.g. nasa secrets",
                key="custom_topic_label",
            )
        else:
            st.markdown("### 📝 Video Topic")
            custom_script_text = ""
            keywords_input = ""
            topic = st.text_area(
                "Describe your story idea",
                placeholder="e.g. The mysterious red room on the dark web…",
                height=120,
                label_visibility="collapsed",
            )

        # Language selector
        lang_choice = st.radio(
            "📺 Channel / Language",
            ["🇬🇧 English", "🇹🇷 Türkçe"],
            horizontal=True,
            key="gen_lang",
        )
        selected_lang = "tr" if "Türkçe" in lang_choice else "en"

        # Content style selector — show EN or TR presets based on language
        from modules.content_adapter import STYLE_PRESETS, TR_STYLE_PRESETS
        _active_presets = TR_STYLE_PRESETS if selected_lang == "tr" else STYLE_PRESETS
        _style_keys   = list(_active_presets.keys())
        _style_labels = [_active_presets[k]["label"] for k in _style_keys]
        _style_idx    = st.selectbox("Content Style", _style_labels, index=0, key="gen_style")
        selected_style = _style_keys[_style_labels.index(_style_idx)]

        # Quick-idea chips
        st.markdown("**Quick ideas:**")
        ideas = [
            "A serial killer's lost diary",
            "The deepest hole ever drilled",
            "NASA's classified moon footage",
            "The Dyatlov Pass incident",
            "A hacker who predicted 9/11",
        ]
        chips = st.columns(len(ideas))
        for col, idea in zip(chips, ideas):
            with col:
                if st.button(idea[:22], key=f"idea_{idea}", use_container_width=True):
                    st.session_state["_injected_topic"] = idea
                    st.rerun()

        # Apply injected topic from chip click (only in AI mode)
        if not use_custom and "_injected_topic" in st.session_state:
            topic = st.session_state.pop("_injected_topic")

        st.markdown("---")

        # Validate before allowing generation
        cfg_now = load_config()
        _required_keys = ("elevenlabs", "pexels") if use_custom else ("openai", "elevenlabs", "pexels")
        keys_ok = all(cfg_now.get("api_keys", {}).get(k) for k in _required_keys)

        if not keys_ok:
            st.warning("⚠️  Fill in all API keys in the sidebar and click **Save Settings**.")

        _ready = topic.strip() and (custom_script_text.strip() if use_custom else True)
        generate_btn = st.button(
            "🚀  Generate Video",
            use_container_width=True,
            disabled=st.session_state.running or not keys_ok or not _ready,
        )

        if generate_btn:
            # Build Pexels keywords from manual input or fallback to topic words
            _custom_kw = [k.strip() for k in keywords_input.split(",") if k.strip()] if use_custom and keywords_input else []
            _final_custom_script = custom_script_text if use_custom else ""

            st.session_state.running   = True
            st.session_state.log_lines = []
            st.session_state.log_queue = queue.Queue()
            pipeline_state = {
                "running": True,
                "step":    0,
                "script":  None,
                "keywords": _custom_kw,
                "output":  None,
                "error":   None,
            }
            st.session_state.pipeline = pipeline_state

            t = threading.Thread(
                target=_run_pipeline,
                args=(topic.strip() or "custom", cfg_now, st.session_state.log_queue, pipeline_state, selected_style, selected_lang, _final_custom_script, _custom_kw or None),
                daemon=True,
            )
            t.start()
            st.session_state.thread = t

    # ── Right: progress ──────────────────────────────────────────────────────
    with col_right:
        st.markdown("### 🔄 Pipeline Progress")

        step_placeholder   = st.empty()
        log_placeholder    = st.empty()
        result_placeholder = st.empty()

        def render_steps(current_step: int, error: bool = False) -> str:
            icons = {0: "⬜", 1: "✅", -1: "❌"}
            html = ""
            for i, name in enumerate(STEP_NAMES, start=1):
                if error and i == current_step:
                    cls, icon = "step-error",   "❌"
                elif i < current_step:
                    cls, icon = "step-done",    "✅"
                elif i == current_step and not error:
                    cls, icon = "step-running", "⏳"
                else:
                    cls, icon = "step-waiting", "⬜"
                html += f'<div class="step-card {cls}">{icon}  Step {i}: {name}</div>'
            return html

        # Live poll loop while running
        if st.session_state.running:
            ps = st.session_state.pipeline

            while st.session_state.running:
                # Drain the log queue
                while not st.session_state.log_queue.empty():
                    st.session_state.log_lines.append(st.session_state.log_queue.get_nowait())

                current = ps.get("step", 0)
                has_err = bool(ps.get("error"))

                step_placeholder.markdown(
                    render_steps(current, error=has_err),
                    unsafe_allow_html=True,
                )

                log_text = "\n".join(st.session_state.log_lines[-60:])
                log_placeholder.code(log_text or "Starting…", language="")

                if not ps.get("running", True):
                    break

                time.sleep(1.0)
                st.rerun()

            # Final drain
            while not st.session_state.log_queue.empty():
                st.session_state.log_lines.append(st.session_state.log_queue.get_nowait())

            st.session_state.running = False
            ps = st.session_state.pipeline

            if ps.get("output"):
                step_placeholder.markdown(render_steps(6), unsafe_allow_html=True)
                out_path = Path(ps["output"])
                with result_placeholder.container():
                    st.success(f"✅  Video ready: `{out_path.name}` ({human_size(out_path)})")
                    with open(out_path, "rb") as vf:
                        st.download_button(
                            "⬇️  Download MP4",
                            data=vf,
                            file_name=out_path.name,
                            mime="video/mp4",
                            use_container_width=True,
                        )
                    if ps.get("script"):
                        with st.expander("📄 Generated Script"):
                            st.write(ps["script"])
                    if ps.get("keywords"):
                        st.markdown("**🔍 Keywords used:** " + " · ".join(f"`{k}`" for k in ps["keywords"]))
            elif ps.get("error"):
                step_placeholder.markdown(
                    render_steps(ps.get("step", 0), error=True), unsafe_allow_html=True
                )
                result_placeholder.error(f"❌ {ps['error']}")

        else:
            # Idle state
            if st.session_state.pipeline.get("output"):
                # Show last result
                out_path = Path(st.session_state.pipeline["output"])
                step_placeholder.markdown(render_steps(6), unsafe_allow_html=True)
                with result_placeholder.container():
                    st.success(f"✅  {out_path.name}  ({human_size(out_path)})")
                    with open(out_path, "rb") as vf:
                        st.download_button("⬇️  Download MP4", data=vf, file_name=out_path.name, mime="video/mp4", use_container_width=True)
            elif st.session_state.pipeline.get("error"):
                step_placeholder.markdown(render_steps(st.session_state.pipeline.get("step", 0), error=True), unsafe_allow_html=True)
                result_placeholder.error(f"❌ {st.session_state.pipeline['error']}")
            else:
                step_placeholder.markdown(render_steps(0), unsafe_allow_html=True)
                log_placeholder.code("Waiting for topic…", language="")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — History
# ════════════════════════════════════════════════════════════════════════════
with tab_history:
    videos = get_output_videos()

    if not videos:
        st.info("No videos generated yet. Go to the **Generate** tab to create your first one!")
    else:
        st.markdown(f"### 📂 All Generated Videos ({len(videos)})")

        for video in videos:
            mtime = datetime.fromtimestamp(video.stat().st_mtime).strftime("%Y-%m-%d  %H:%M")
            size  = human_size(video)

            with st.container():
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    st.markdown(f"**🎥 {video.name}**  \n`{mtime}  ·  {size}`")
                with c2:
                    with open(video, "rb") as vf:
                        st.download_button(
                            "⬇️ Download",
                            data=vf,
                            file_name=video.name,
                            mime="video/mp4",
                            key=f"dl_{video.name}",
                            use_container_width=True,
                        )
                with c3:
                    if st.button("▶️ Preview", key=f"prev_{video.name}", use_container_width=True):
                        st.session_state["preview_video"] = str(video)
                        st.rerun()

                st.markdown('<hr style="border-color:#222; margin:6px 0">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Preview
# ════════════════════════════════════════════════════════════════════════════
with tab_preview:
    preview_target = st.session_state.get("preview_video")

    if not preview_target or not Path(preview_target).exists():
        videos = get_output_videos()
        if videos:
            preview_target = str(videos[0])
            st.session_state["preview_video"] = preview_target
        else:
            st.info("Generate a video first to preview it here.")
            preview_target = None

    if preview_target:
        p = Path(preview_target)
        st.markdown(f"### ▶️  {p.name}")

        # Streamlit can't yet stream large local MP4s natively — we serve bytes
        with open(p, "rb") as vf:
            video_bytes = vf.read()

        col_vid, col_meta = st.columns([1, 1], gap="large")
        with col_vid:
            st.video(video_bytes)

        with col_meta:
            st.markdown("**File details**")
            st.markdown(f"- **Name:** `{p.name}`")
            st.markdown(f"- **Size:** {human_size(p)}")
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"- **Created:** {mtime}")

            st.markdown("---")
            all_videos = get_output_videos()
            st.markdown("**📋 Switch video**")
            names = [v.name for v in all_videos]
            sel = st.selectbox("Select video", names, index=names.index(p.name) if p.name in names else 0, label_visibility="collapsed")
            if sel != p.name:
                st.session_state["preview_video"] = str(OUTPUT_DIR / sel)
                st.rerun()

            with open(p, "rb") as vf:
                st.download_button("⬇️  Download this video", data=vf, file_name=p.name, mime="video/mp4", use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Trend Hunt
# ════════════════════════════════════════════════════════════════════════════
CHANNELS_FILE = Path(__file__).parent / "channels.json"


def _load_channels() -> dict:
    if CHANNELS_FILE.exists():
        with open(CHANNELS_FILE) as f:
            return json.load(f)
    return {"channels": [], "settings": {}}


def _save_channels(data: dict) -> None:
    with open(CHANNELS_FILE, "w") as f:
        json.dump(data, f, indent=2)


with tab_trend:
    from modules.content_adapter import STYLE_PRESETS as _SP, TR_STYLE_PRESETS as _TR_SP, ALL_PRESETS as _ALL_SP

    st.markdown("### \U0001f525 Trend Hunt — Competitor Intelligence")
    cfg_trend = load_config()
    yt_key = cfg_trend.get("api_keys", {}).get("youtube", "").strip()

    ch_data = _load_channels()

    col_ch, col_scan = st.columns([1, 2], gap="large")

    # ── Left: channel management ─────────────────────────────────────────────
    with col_ch:
        st.markdown("#### \U0001f4e1 Monitored Channels")

        for i, ch in enumerate(ch_data["channels"]):
            c1, c2 = st.columns([3, 1])
            with c1:
                niche_label = _SP.get(ch.get("niche", "dark_mystery"), {}).get("label", ch.get("niche", ""))
                st.markdown(f"**{ch['name']}**  \n`{niche_label}`")
            with c2:
                if st.button("\u2715", key=f"rm_ch_{i}", help="Remove channel"):
                    ch_data["channels"].pop(i)
                    _save_channels(ch_data)
                    st.rerun()

        st.markdown("---")
        st.markdown("**Add a channel:**")
        new_ch_id    = st.text_input("Channel ID",   key="new_ch_id",    placeholder="UCxxxxxxxxxxxxxxxx")
        new_ch_name  = st.text_input("Display name", key="new_ch_name",  placeholder="e.g. MrBallen")
        _style_keys_t  = list(_SP.keys())
        _style_labels_t = [_SP[k]["label"] for k in _style_keys_t]
        new_ch_niche_label = st.selectbox("Content style", _style_labels_t, key="new_ch_niche")
        new_ch_niche = _style_keys_t[_style_labels_t.index(new_ch_niche_label)]

        if st.button("\u2795  Add Channel", use_container_width=True):
            if new_ch_id.strip() and new_ch_name.strip():
                ch_data["channels"].append({
                    "id":    new_ch_id.strip(),
                    "name":  new_ch_name.strip(),
                    "niche": new_ch_niche,
                })
                _save_channels(ch_data)
                st.success(f"Added: {new_ch_name}")
                st.rerun()

        st.markdown("---")
        st.markdown("**Scan settings:**")
        new_threshold = st.slider(
            "Viral score threshold", 0.50, 1.0,
            float(ch_data["settings"].get("viral_threshold", 0.80)), 0.05,
        )
        new_vph_ratio = st.slider(
            "VPH alert ratio", 1.0, 5.0,
            float(ch_data["settings"].get("vph_alert_ratio", 1.5)), 0.1,
        )
        if st.button("\U0001f4be  Save scan settings", use_container_width=True):
            ch_data["settings"]["viral_threshold"] = new_threshold
            ch_data["settings"]["vph_alert_ratio"] = new_vph_ratio
            _save_channels(ch_data)
            st.success("Saved!")

    # ── Right: scan results ───────────────────────────────────────────────────
    with col_scan:
        st.markdown("#### \U0001f4ca Scan Results")

        if not yt_key or yt_key.startswith("YOUR_"):
            st.warning(
                "\u26a0\ufe0f  Add your **YouTube Data API v3 key** in the sidebar and save settings.  \n"
                "Get a free key at [console.cloud.google.com](https://console.cloud.google.com/) "
                "(enable YouTube Data API v3, create an API key)."
            )
        else:
            scan_btn = st.button("\U0001f50d  Scan Channels Now", use_container_width=True,
                                  disabled=not ch_data["channels"])

            if scan_btn:
                with st.spinner("Scanning competitor channels..."):
                    try:
                        from modules.trend_detector import scan_channels, get_top_comments
                        from modules.viral_scorer import calculate_viral_score

                        raw_results = scan_channels(
                            ch_data["channels"], yt_key,
                            vph_threshold_ratio=ch_data["settings"].get("vph_alert_ratio", 1.5),
                        )

                        # Score trending videos
                        for vid in raw_results:
                            if vid["is_trending"]:
                                comments = get_top_comments(vid["video_id"], yt_key, 20)
                                vid["scores"] = calculate_viral_score(
                                    views=vid["views"],
                                    likes=vid["likes"],
                                    comments_count=vid["comments"],
                                    vph_ratio=vid["vph_ratio"],
                                    comment_texts=comments,
                                )

                        st.session_state["trend_results"] = raw_results
                    except Exception as exc:
                        st.error(f"Scan failed: {exc}")

            results = st.session_state.get("trend_results", [])

            if results:
                trending_count = sum(1 for r in results if r["is_trending"])
                st.markdown(f"**{len(results)} videos scanned · {trending_count} trending \U0001f525**")

                for vid in results:
                    scores  = vid.get("scores")
                    v_score = scores["viral_score"] if scores else None
                    decision = scores["decision"]  if scores else ""

                    if vid["is_trending"]:
                        border = "#ff3300"
                        badge  = f"\U0001f525 TREND · Viral {v_score:.2f}" if v_score else "\U0001f525 TREND"
                    else:
                        border = "#333"
                        badge  = f"VPH ratio {vid['vph_ratio']:.2f}x"

                    st.markdown(
                        f"""
                        <div style="border:1px solid {border};border-radius:10px;
                                    padding:14px 18px;margin:8px 0;background:#1a1a1a">
                          <div style="display:flex;justify-content:space-between;align-items:center">
                            <div>
                              <b style="color:#e0e0e0">{vid['title'][:70]}</b><br>
                              <span style="color:#888;font-size:0.85rem">
                                {vid['channel_name']} &nbsp;·&nbsp;
                                {vid['views']:,} views &nbsp;·&nbsp;
                                {vid['age_hours']:.1f}h old &nbsp;·&nbsp;
                                VPH {vid['vph']:.0f}
                              </span>
                            </div>
                            <span style="color:{border};font-weight:bold;font-size:0.85rem">{badge}</span>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if vid["is_trending"]:
                        btn_cols = st.columns([1.5, 2, 2, 1])
                        with btn_cols[0]:
                            _lang_choice = st.radio(
                                "Lang", ["🇬🇧 EN", "🇹🇷 TR"],
                                horizontal=True,
                                key=f"lang_{vid['video_id']}",
                                label_visibility="collapsed",
                            )
                            _vid_lang = "tr" if "TR" in _lang_choice else "en"

                        with btn_cols[1]:
                            _active_sp     = _TR_SP if _vid_lang == "tr" else _SP
                            _style_keys_r  = list(_active_sp.keys())
                            _style_labels_r = [_active_sp[k]["label"] for k in _style_keys_r]
                            default_niche = vid.get("niche", "dark_mystery")
                            # map to TR variant if TR selected
                            _tr_niche = f"{default_niche}_tr" if _vid_lang == "tr" else default_niche
                            if _tr_niche not in _style_keys_r:
                                _tr_niche = _style_keys_r[0]
                            _default_idx  = _style_keys_r.index(_tr_niche) if _tr_niche in _style_keys_r else 0
                            chosen_label  = st.selectbox(
                                "Style", _style_labels_r, index=_default_idx,
                                key=f"style_{vid['video_id']}", label_visibility="collapsed"
                            )
                            chosen_style = _style_keys_r[_style_labels_r.index(chosen_label)]

                        with btn_cols[2]:
                            if st.button("🚀  Produce Video", key=f"prod_{vid['video_id']}",
                                          use_container_width=True):
                                st.session_state["trend_produce"] = {
                                    "video_id": vid["video_id"],
                                    "title":    vid["title"],
                                    "style":    chosen_style,
                                    "language": _vid_lang,
                                }
                                st.rerun()

                        with btn_cols[3]:
                            st.link_button("🔗 Open", vid["url"], use_container_width=True)

            # Trigger production from trend ──────────────────────────────────
            if "trend_produce" in st.session_state and not st.session_state.running:
                tp = st.session_state.pop("trend_produce")
                st.session_state.running   = True
                st.session_state.log_lines = []
                st.session_state.log_queue = queue.Queue()
                pipeline_state = {
                    "running": True, "step": 0,
                    "script": None, "keywords": [],
                    "output": None, "error": None,
                }
                st.session_state.pipeline = pipeline_state

                def _run_trend_produce(video_id, title, style, language, config, log_q, state):
                    handler = QueueHandler(log_q)
                    handler.setFormatter(logging.Formatter(
                        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"
                    ))
                    root_logger = logging.getLogger()
                    root_logger.addHandler(handler)
                    root_logger.setLevel(logging.INFO)
                    try:
                        from trend_pipeline import produce_from_video
                        out = produce_from_video(video_id, title, style, language, config)
                        state["output"]  = out
                        state["step"]    = 6
                        state["running"] = False
                        log_q.put("\u2705  Done!")
                    except Exception as exc:
                        state["error"]   = str(exc)
                        state["running"] = False
                        log_q.put(f"\u274c  {exc}")
                    finally:
                        root_logger.removeHandler(handler)

                t2 = threading.Thread(
                    target=_run_trend_produce,
                    args=(tp["video_id"], tp["title"], tp["style"], tp.get("language", "en"),
                          cfg_trend, st.session_state.log_queue, pipeline_state),
                    daemon=True,
                )
                t2.start()
                st.session_state.thread = t2
                st.toast(f"🚀 Producing trend video: {tp['title'][:30]}...", icon="🎬")
                st.info(f"🚀 Producing video from: **{tp['title'][:60]}**  \nWatch progress in the **⚡ Generate** tab.")
                time.sleep(2) # Give it a moment to show the info
                st.rerun()