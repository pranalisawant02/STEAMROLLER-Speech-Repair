"""
tts_engine.py — Fast speech synthesis.

"""

import io
import os
import tempfile
from typing import Literal

TTSBackend = Literal["pyttsx3", "gtts"]


def text_to_speech_pyttsx3(text: str, gender: str = "male", rate: int = 165) -> bytes:
    """Synthesise text using the system TTS engine.

    Args:
        text:   The text to synthesise.
        gender: "female" or "male" — used to pick the best matching system voice.
        rate:   Speech rate in words per minute.
    """
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", rate)

    voices = engine.getProperty("voices") or []
    chosen_voice_id = None

    if gender == "female":
        female_keywords = ["zira", "hazel", "susan", "female", "woman", "heera", "sabina"]
        for voice in voices:
            desc = f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')}".lower()
            if any(k in desc for k in female_keywords):
                chosen_voice_id = voice.id
                break
    else:
        male_keywords = ["david", "mark", "ravi", "george", "male", "man"]
        for voice in voices:
            desc = f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')}".lower()
            if any(k in desc for k in male_keywords) and "zira" not in desc:
                chosen_voice_id = voice.id
                break

    # Fallback: first available voice
    if not chosen_voice_id and voices:
        chosen_voice_id = voices[0].id
    if chosen_voice_id:
        engine.setProperty("voice", chosen_voice_id)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        engine.save_to_file(text, tmp.name)
        engine.runAndWait()
        with open(tmp.name, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def text_to_speech_gtts(text: str, lang: str = "en", slow: bool = False) -> bytes:
    from gtts import gTTS
    tts = gTTS(text=text, lang=lang, slow=slow)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()


def synthesise_speech(
    text: str,
    backend: TTSBackend = "pyttsx3",
    lang: str = "en",
    slow: bool = False,
    gender: str = "male",
) -> bytes:
    """Synthesise speech, matching the voice gender of the original speaker.

    Args:
        text:    Repaired text to synthesise.
        backend: "pyttsx3" (local, fast) or "gtts" (Google, network).
        lang:    BCP-47 language code used by gTTS.
        slow:    Use slower gTTS speech rate.
        gender:  "female" or "male" — selects the system voice gender for pyttsx3.
                 gTTS does not support voice switching, so this only affects pyttsx3.
    """
    if not text.strip():
        raise ValueError("Cannot synthesise empty text.")
    if backend == "gtts":
        try:
            return text_to_speech_gtts(text, lang=lang, slow=slow)
        except Exception:
            return text_to_speech_pyttsx3(text, gender=gender)
    return text_to_speech_pyttsx3(text, gender=gender)
