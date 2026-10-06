"""
app.py — Steamroller End-to-End Fast Pipeline
Run: streamlit run app.py

Flow:
  [Upload file  OR  Live mic recording]
  → Whisper ASR → NLP cleaning → local repair agents → repaired audio output.
"""

import html
import io
import time
import wave
import os
import uuid

import numpy as np
import streamlit as st

st.set_page_config(page_title="Steamroller", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root {
  --bg:#f4fbff; --surface:#ffffff; --border:#cfe6ee;
  --accent:#08a6c8; --accent2:#13c6b8; --text:#0f172a;
  --green:#00896f; --red:#e53e3e; --radius:16px;
}
html, body, [data-testid="stAppViewContainer"] { background:var(--bg)!important; color:var(--text)!important; }
[data-testid="stSidebar"] { background:#ffffff!important; border-right:1px solid var(--border); }
#MainMenu, footer, header { visibility:hidden; }
.hero { text-align:center; padding:1.2rem 1rem 1.6rem; }
.hero h1 { margin:0; font-size:2.4rem; color:#102a43; font-weight:800; }
.hero p { color:#486581; max-width:700px; margin:.5rem auto 0; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:1.35rem; box-shadow:0 8px 22px rgba(15,23,42,.05); margin-bottom:1rem; }
.card-title { font-weight:800; color:#0f172a; letter-spacing:.04em; margin-bottom:.8rem; text-transform:uppercase; }
.text-block { background:#f8fdff; color:#0f172a; border:1px solid #d9eef5; border-left:5px solid var(--accent); border-radius:12px; padding:1rem; min-height:90px; line-height:1.7; font-size:1rem; word-break:break-word; }
.text-block.final { border-left-color:var(--green); background:#f5fffc; color:#052e2b; font-weight:600; }
.text-block.empty { color:#64748b; font-style:italic; font-weight:400; }
.stButton > button { background:linear-gradient(135deg,var(--accent),var(--accent2))!important; color:white!important; border:0!important; border-radius:12px!important; font-weight:800!important; padding:.75rem 1rem!important; transition:opacity .2s; }
.stButton > button:hover { opacity:.88; }
.stDownloadButton > button { border-radius:12px!important; font-weight:700!important; }
[data-testid="stFileUploader"] section { background:#f8fdff!important; border:2px dashed #9bd8e8!important; border-radius:16px!important; }
.small-note { color:#64748b; font-size:.85rem; line-height:1.5; }
.status-pill { display:inline-block; background:#e7fbf8; color:#006b5f; border:1px solid #baf0e7; border-radius:999px; padding:.25rem .7rem; font-weight:700; font-size:.82rem; margin:.15rem .25rem .15rem 0; }
.rec-pill { display:inline-block; background:#fff0f0; color:#c53030; border:1px solid #feb2b2; border-radius:999px; padding:.28rem .8rem; font-weight:800; font-size:.88rem; animation:pulse 1s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.45} }
.tab-hint { color:#486581; font-size:.9rem; margin-bottom:.75rem; }
/* Style Streamlit tab headers */
[data-testid="stTabs"] [role="tab"] { font-weight:700; font-size:1rem; }
/* Force sidebar to always be visible — can't be collapsed or hidden */
[data-testid="stSidebar"] {
  min-width: 320px !important;
  max-width: 320px !important;
  transform: none !important;
  visibility: visible !important;
  display: block !important;
}
[data-testid="collapsedControl"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def html_text(text: str) -> str:
    return html.escape(text or "")


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert a float32 numpy array to WAV bytes."""
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()

def save_uploaded_voice(audio_bytes, filename="reference.wav"):
    """
    Save uploaded/live audio as WAV for F5-TTS.
    """

    os.makedirs("uploads", exist_ok=True)

    input_path = os.path.join(
        "uploads",
        f"temp_{uuid.uuid4().hex}_{filename}"
    )

    with open(input_path, "wb") as f:
        f.write(audio_bytes)

    return input_path


def trim_reference_audio(audio_bytes, filename, max_seconds=12):
    """
    Trim raw audio down to the first `max_seconds` and normalize to a clean
    mono 24kHz WAV. F5-TTS misjudges speaking pace badly on long or messy
    reference clips (this is why uploaded files, which have no length limit,
    sound rushed/off compared to short live recordings) — so every reference
    clip sent to F5-TTS goes through here first, regardless of source.
    """
    import subprocess

    os.makedirs("uploads", exist_ok=True)
    tmp_id = uuid.uuid4().hex
    ext = os.path.splitext(filename)[1] or ".wav"
    raw_path = os.path.join("uploads", f"raw_{tmp_id}{ext}")
    trimmed_path = os.path.join("uploads", f"ref_{tmp_id}.wav")

    with open(raw_path, "wb") as f:
        f.write(audio_bytes)

    cmd = [
        "ffmpeg", "-y",
        "-i", raw_path,
        "-t", str(max_seconds),
        "-ac", "1",
        "-ar", "24000",
        trimmed_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 or not os.path.exists(trimmed_path):
        # ffmpeg failed for some reason — fall back to the untrimmed original
        # rather than blocking the whole pipeline
        return raw_path

    return trimmed_path
# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎙️ STEAMROLLER")
    st.caption("Fast stuttered audio → fluent audio")
    st.markdown("---")
    st.markdown("### Speed Settings")

    whisper_model = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small"],
        index=0,
        help="tiny is fastest — usually under 1 minute for short clips.",
    )

    language = st.selectbox(
        "Language",
        [
            "Auto-detect",
            "English",
            "Hindi",
            "Spanish",
            "French",
            "German"
        ],
        index=0
    )

    lang_code_map = {
        "Auto-detect": None,
        "English": "en",
        "Hindi": "hi",
        "Spanish": "es",
        "French": "fr",
        "German": "de"
    }

    lang_code = lang_code_map[language]


    tts_backend = st.radio(
        "Output voice",
        [
            "System voice (faster)",
            "Google TTS",
            "F5-TTS Voice Clone"
        ],
        index=0
    )

    f5tts_speed = st.slider(
        "F5-TTS speaking speed",
        min_value=0.5,
        max_value=1.3,
        value=0.9,
        step=0.05,
        help="Lower = slower speech. F5-TTS often sounds rushed by default — try 0.8–0.9 first.",
    )


    st.markdown("---")

    st.markdown(
        "<div class='small-note'>"
        "For speed, use clips around 5–30 seconds. "
        "Longer audio takes more time."
        "</div>",
        unsafe_allow_html=True
    )


    if st.button(
        "Clear Session",
        use_container_width=True
    ):
        reset_session()
        st.rerun()

# ── session state defaults ────────────────────────────────────────────────────

for key, default in {
    "raw_audio": None,
    "raw_text": "",
    "final_text": "",
    "tts_bytes": None,
    "tts_format": "audio/wav",
    "timings": {},
    "error": None,
    "filename": "audio.wav",
    "detected_gender": None,
}.items():
    st.session_state.setdefault(key, default)

# ── hero ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class='hero'>
  <h1>FLUENT FORCE — Speech Repair Assistant</h1>
  <p>Record your voice live <em>or</em> upload an existing audio file. The system transcribes, repairs stuttered speech, and returns fluent audio.</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.error:
    st.error(st.session_state.error)

# ── layout ────────────────────────────────────────────────────────────────────

left, right = st.columns([1, 1], gap="large")

# ── LEFT: input card with two tabs ───────────────────────────────────────────

with left:
    st.markdown("<div class='card'><div class='card-title'>Voice Input</div>", unsafe_allow_html=True)

    tab_upload, tab_live = st.tabs(["📂  Upload Audio File", "🎤  Live Recording"])

    # ── Tab 1 : Upload ────────────────────────────────────────────────────────
    with tab_upload:
        st.markdown("<p class='tab-hint'>Drag & drop or browse an existing audio file.</p>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Supported formats: WAV, MP3, M4A, OGG, FLAC",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            label_visibility="collapsed",
        )
        if uploaded:
            audio_bytes = uploaded.getvalue()
            # Reset gender when a new file is uploaded
            if st.session_state.filename != uploaded.name:
                st.session_state.detected_gender = None
            st.session_state.raw_audio = audio_bytes
            st.session_state.filename = uploaded.name
            st.audio(audio_bytes)

            if st.session_state.detected_gender is None:
                with st.spinner("Detecting speaker gender…"):
                    from audio_pipeline import detect_speaker_gender
                    st.session_state.detected_gender = detect_speaker_gender(
                        audio_bytes, filename=uploaded.name
                    )

            gender_icon  = "♀️" if st.session_state.detected_gender == "female" else "♂️"
            gender_label = st.session_state.detected_gender.capitalize()
            st.markdown(
                f"<span class='status-pill'>{gender_icon} Detected: {gender_label} voice</span>",
                unsafe_allow_html=True,
            )

    # ── Tab 2 : Live Recording ────────────────────────────────────────────────
    with tab_live:
        st.markdown("<p class='tab-hint'>Choose a duration, then click <strong>Start Recording</strong> and speak naturally.</p>", unsafe_allow_html=True)

        rec_duration = st.slider("Recording duration (seconds)", min_value=5, max_value=60, value=10, step=5)

        if st.button("🎤  Start Recording", use_container_width=True, key="btn_record"):
            SAMPLE_RATE = 16000
            placeholder = st.empty()
            placeholder.markdown(
                f"<div class='rec-pill'>🔴 Recording for {rec_duration}s — speak now…</div>",
                unsafe_allow_html=True,
            )
            try:
                import sounddevice as sd
                audio_np = sd.rec(
                    int(rec_duration * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()                         # blocks until recording done
                audio_np = audio_np.flatten()

                # Normalize amplitude — mic recordings are often very quiet and
                # cause Whisper to hallucinate on low-energy input
                from audio_pipeline import normalize_audio
                audio_np = normalize_audio(audio_np)

                wav_bytes = numpy_to_wav_bytes(audio_np, SAMPLE_RATE)
                st.session_state.raw_audio = wav_bytes
                st.session_state.filename  = "live_recording.wav"
                st.session_state.detected_gender = None   # re-detect for new clip

                placeholder.empty()
                st.success(f"✅ Recording captured ({rec_duration}s). Click **Repair Speech** to process.")
                st.audio(wav_bytes, format="audio/wav")

                with st.spinner("Detecting speaker gender…"):
                    from audio_pipeline import detect_speaker_gender
                    st.session_state.detected_gender = detect_speaker_gender(
                        wav_bytes, filename="live_recording.wav"
                    )

                gender_icon  = "♀️" if st.session_state.detected_gender == "female" else "♂️"
                gender_label = st.session_state.detected_gender.capitalize()
                st.markdown(
                    f"<span class='status-pill'>{gender_icon} Detected: {gender_label} voice</span>",
                    unsafe_allow_html=True,
                )

            except Exception as exc:
                placeholder.empty()
                st.error(f"Microphone error: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)   # close .card

    process = st.button(
        "🔧  Repair Speech",
        disabled=not bool(st.session_state.raw_audio),
        use_container_width=True,
        key="btn_process",
    )

# ── RIGHT: output cards ───────────────────────────────────────────────────────

with right:
    st.markdown("<div class='card'><div class='card-title'>Repaired Output</div>", unsafe_allow_html=True)
    if st.session_state.final_text:
        st.markdown(f"<div class='text-block final'>{html_text(st.session_state.final_text)}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='text-block empty'>The repaired sentence will appear here after processing.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='card-title'>Repaired Audio</div>", unsafe_allow_html=True)
    if st.session_state.tts_bytes:
        st.audio(st.session_state.tts_bytes, format=st.session_state.tts_format)
        ext = "mp3" if st.session_state.tts_format == "audio/mp3" else "wav"
        st.download_button(
            "⬇️  Download Repaired Audio",
            data=st.session_state.tts_bytes,
            file_name=f"steamroller_repaired.{ext}",
            mime=st.session_state.tts_format,
            use_container_width=True,
        )
    else:
        st.markdown("<div class='text-block empty'>Repaired audio will appear here.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── pipeline ──────────────────────────────────────────────────────────────────

if process:
    st.session_state.error      = None
    st.session_state.raw_text   = ""
    st.session_state.final_text = ""
    st.session_state.tts_bytes  = None
    st.session_state.timings    = {}

    overall_start = time.perf_counter()
    progress = st.progress(0)
    status   = st.empty()

    try:
        status.info("1/4 Transcribing audio with Whisper…")
        t0 = time.perf_counter()
        from audio_pipeline import transcribe_uploaded_file
        is_live = st.session_state.filename == "live_recording.wav"
        raw_text, _ = transcribe_uploaded_file(
            st.session_state.raw_audio,
            filename=st.session_state.filename,
            model_size=whisper_model,
            language=lang_code,
            is_live=is_live,
        )
        st.session_state.raw_text = raw_text
        st.session_state.timings["ASR"] = round(time.perf_counter() - t0, 2)
        progress.progress(35)

        status.info("2/4 Cleaning stutter patterns…")
        t0 = time.perf_counter()
        from nlp_cleaner import clean_asr_output
        report = clean_asr_output(raw_text)
        st.session_state.timings["NLP"] = round(time.perf_counter() - t0, 2)
        progress.progress(55)

        status.info("3/4 Repairing fluency with local agents…")
        t0 = time.perf_counter()
        from repair_agents import run_multi_agent_repair
        verdict = run_multi_agent_repair(report.cleaned, original_text=raw_text)
        st.session_state.final_text = verdict.final_text
        st.session_state.timings["Repair"] = round(time.perf_counter() - t0, 2)
        progress.progress(75)

        status.info("4/4 Generating repaired audio…")
        t0 = time.perf_counter()

        if tts_backend == "F5-TTS Voice Clone":

            from voice_clone import VoiceClone

            if "f5tts_model" not in st.session_state:
                st.session_state.f5tts_model = VoiceClone()

            reference_audio = trim_reference_audio(
                st.session_state.raw_audio,
                st.session_state.filename
            )

            output_file = st.session_state.f5tts_model.generate(
                text=st.session_state.final_text,
                speaker_wav=reference_audio,
                ref_text="",  # let F5-TTS auto-transcribe the trimmed clip for an exact match
                language=lang_code or "en",
                speed=f5tts_speed
            )

            with open(output_file, "rb") as f:
                st.session_state.tts_bytes = f.read()

            st.session_state.tts_format = "audio/wav"


        else:

            from tts_engine import synthesise_speech

            backend = (
                "pyttsx3"
                if tts_backend.startswith("System")
                else "gtts"
            )

            gender = (
                st.session_state.detected_gender
                or "female"
            )

            st.session_state.tts_bytes = synthesise_speech(
                st.session_state.final_text,
                backend=backend,
                lang=lang_code or "en",
                slow=False,
                gender=gender,
            )

            st.session_state.tts_format = (
                "audio/wav"
                if backend == "pyttsx3"
                else "audio/mp3"
            )
        st.session_state.timings["TTS"]   = round(time.perf_counter() - t0, 2)
        st.session_state.timings["Total"] = round(time.perf_counter() - overall_start, 2)
        progress.progress(100)
        status.success(f"✅ Repair complete in {st.session_state.timings['Total']}s.")
        time.sleep(0.5)
        st.rerun()

    except Exception as exc:
        st.session_state.error = f"Processing failed: {exc}"
        st.rerun()

# ── timing summary ────────────────────────────────────────────────────────────

if st.session_state.timings:
    st.markdown("<div class='card'><div class='card-title'>Processing Time</div>", unsafe_allow_html=True)
    st.markdown(
        "".join(f"<span class='status-pill'>{k}: {v}s</span>" for k, v in st.session_state.timings.items()),
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='small-note'>Voice gender is auto-detected from pitch analysis. "
        "True voice cloning uses F5-TTS running on a connected Kaggle GPU notebook. "
        "System TTS matches detected gender as closely as available voices allow.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
#commands to run
# 1. Check the server is alive:

# cmd
# Check the Kaggle TTS server:
# curl https://YOUR-NGROK-URL/health

# Should return {"status":"ok"}.

# 2. Set the environment variable (same Command Prompt window):

# cmd
# Windows:
# set KAGGLE_TTS_URL=https://YOUR-NGROK-URL

# 3. Go into your project folder:

# cmd
# cd Steamroller

# 4. Run the app:

# cmd
# streamlit run app.py

# That's the full sequence, all in the same terminal window, one after another.