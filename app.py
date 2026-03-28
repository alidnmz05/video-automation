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


def _run_pipeline(topic: str, config: dict, log_q: queue.Queue, state: dict) -> None:
    """
    Runs the full 5-step pipeline in a background thread.
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

    def step(n: int) -> None:
        state["step"] = n
        log_q.put(f"{'─'*48}")
        log_q.put(f"▶  STEP {n}/5 — {STEP_NAMES[n - 1]}")
        log_q.put(f"{'─'*48}")

    try:
        # Step 1 — Script
        step(1)
        from modules.script_generator import generate_script
        result   = generate_script(topic, config["api_keys"]["openai"])
        script   = result["script"]
        keywords = result["keywords"]
        (session_dir / "script.txt").write_text(script, encoding="utf-8")
        state["script"]   = script
        state["keywords"] = keywords

        # Step 2 — Voiceover
        step(2)
        from modules.voiceover import generate_voiceover
        audio_path = generate_voiceover(
            script,
            str(session_dir),
            config["api_keys"]["elevenlabs"],
            config["elevenlabs"]["voice_id"],
            config["elevenlabs"]["model_id"],
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
        output_path = str(OUTPUT_DIR / f"video_{timestamp}.mp4")
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

    with st.expander("🎙️  Voice Settings"):
        voice_id = st.text_input(
            "ElevenLabs Voice ID",
            value=cfg.get("elevenlabs", {}).get("voice_id", "21m00Tcm4TlvDq8ikWAM"),
        )
        model_id = st.selectbox(
            "Model",
            ["eleven_monolingual_v1", "eleven_multilingual_v2", "eleven_turbo_v2"],
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
            },
            "elevenlabs": {"voice_id": voice_id, "model_id": model_id},
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

tab_gen, tab_history, tab_preview = st.tabs(["⚡  Generate", "📂  History", "▶️  Preview"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Generate
# ════════════════════════════════════════════════════════════════════════════
with tab_gen:
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Left: input ──────────────────────────────────────────────────────────
    with col_left:
        st.markdown("### 📝 Video Topic")
        topic = st.text_area(
            "Describe your story idea",
            placeholder="e.g. The mysterious red room on the dark web…",
            height=120,
            label_visibility="collapsed",
        )

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

        # Apply injected topic from chip click
        if "_injected_topic" in st.session_state:
            topic = st.session_state.pop("_injected_topic")

        st.markdown("---")

        # Validate before allowing generation
        cfg_now = load_config()
        keys_ok = all(
            cfg_now.get("api_keys", {}).get(k)
            for k in ("openai", "elevenlabs", "pexels")
        )

        if not keys_ok:
            st.warning("⚠️  Fill in all API keys in the sidebar and click **Save Settings**.")

        generate_btn = st.button(
            "🚀  Generate Video",
            use_container_width=True,
            disabled=st.session_state.running or not keys_ok or not topic.strip(),
        )

        if generate_btn:
            st.session_state.running   = True
            st.session_state.log_lines = []
            st.session_state.log_queue = queue.Queue()
            pipeline_state = {
                "running": True,
                "step":    0,
                "script":  None,
                "keywords": [],
                "output":  None,
                "error":   None,
            }
            st.session_state.pipeline = pipeline_state

            t = threading.Thread(
                target=_run_pipeline,
                args=(topic.strip(), cfg_now, st.session_state.log_queue, pipeline_state),
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
