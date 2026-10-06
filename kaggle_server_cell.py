# Paste these as new cells at the END of final-f5tts.ipynb, AFTER the
# existing F5-TTS install cells (the git clone + pip install -e . ones).
# This turns your working f5-tts_infer-cli setup into an HTTP server that
# voice_clone.py can call from your local Streamlit app.

# --- Cell A: install server deps ---
# !pip install -q flask pyngrok

# --- Cell B: the Flask server ---
import os
import uuid
import subprocess
from flask import Flask, request, send_file

app = Flask(__name__)

WORKDIR = "/kaggle/working"
UPLOAD_DIR = os.path.join(WORKDIR, "uploads")
OUTPUT_DIR = os.path.join(WORKDIR, "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}


@app.route("/clone", methods=["POST"])
def clone():
    text = request.form.get("text")
    ref_text = request.form.get("ref_text", "")
    speaker_file = request.files.get("speaker_wav")

    if not text or not speaker_file:
        return {"error": "text and speaker_wav are required"}, 400

    uid = uuid.uuid4().hex
    ref_path = os.path.join(UPLOAD_DIR, f"ref_{uid}.wav")
    speaker_file.save(ref_path)

    output_filename = f"clone_{uid}.wav"

    cmd = [
        "f5-tts_infer-cli",
        "--model", "F5TTS_Base",
        "--ref_audio", ref_path,
        "--ref_text", ref_text,  # empty string -> F5-TTS auto-transcribes with Whisper
        "--gen_text", text,
        "--output_dir", OUTPUT_DIR,
        "--output_file", output_filename,
        "--remove_silence",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return {"error": result.stderr[-2000:]}, 500

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    if not os.path.exists(output_path):
        return {"error": "F5-TTS ran but no output file was produced", "log": result.stdout[-2000:]}, 500

    return send_file(output_path, mimetype="audio/wav")


# --- Cell C: expose it publicly and start serving ---
# from pyngrok import ngrok
#
# ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")   # free at https://ngrok.com
# public_url = ngrok.connect(5000)
# print("Public URL:", public_url)
# print("Copy this into KAGGLE_TTS_URL on your local machine.")
#
# app.run(port=5000)
