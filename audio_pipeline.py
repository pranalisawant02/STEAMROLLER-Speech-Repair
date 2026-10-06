"""
audio_pipeline.py — Upload audio + optimized Whisper ASR transcription.
"""

import io
import os
import wave
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

SAMPLE_RATE = 16000
CHANNELS = 1

# Pitch threshold for approximate voice classification.
# Typical adult male voices: approximately 80–165 Hz
# Typical adult female voices: approximately 165–300 Hz
_FEMALE_F0_THRESHOLD = 165.0


def detect_speaker_gender(
    file_bytes: bytes,
    filename: str = "audio.wav"
) -> str:
    """
    Estimate speaker gender from audio using fundamental frequency (F0).

    Returns:
        "female" or "male"

    Note:
        This is an approximate pitch-based classification. It is not
        guaranteed to correctly classify every speaker.
    """

    suffix = Path(filename).suffix or ".wav"

    tmp = tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False
    )

    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()

        import scipy.io.wavfile as wavfile
        from scipy.signal import resample_poly
        import math

        sample_rate, data = wavfile.read(tmp.name)

        # Convert stereo audio to mono.
        if data.ndim > 1:
            data = data.mean(axis=1)

        data = data.astype(np.float32)

        # Normalize integer PCM audio.
        if np.max(np.abs(data)) > 1.0:
            data /= 32768.0

        # Resample audio to 16 kHz.
        if sample_rate != SAMPLE_RATE:
            gcd = math.gcd(sample_rate, SAMPLE_RATE)

            data = resample_poly(
                data,
                SAMPLE_RATE // gcd,
                sample_rate // gcd
            )

            sample_rate = SAMPLE_RATE

        if data.size == 0:
            return "male"

        # Remove DC offset.
        data = data - np.mean(data)

        # 40 ms analysis frames with 20 ms hop.
        frame_length = int(sample_rate * 0.04)
        hop_length = int(sample_rate * 0.02)

        # Search range for adult speech fundamental frequency.
        min_f0 = 75.0
        max_f0 = 300.0

        min_period = int(sample_rate / max_f0)
        max_period = int(sample_rate / min_f0)

        detected_pitches = []

        for start in range(
            0,
            len(data) - frame_length,
            hop_length
        ):
            frame = data[start:start + frame_length]

            # Apply a Hann window.
            frame = frame * np.hanning(len(frame))

            # Ignore silent or extremely quiet frames.
            rms = float(np.sqrt(np.mean(frame ** 2)))

            if rms < 0.015:
                continue

            # Autocorrelation.
            autocorrelation = np.correlate(
                frame,
                frame,
                mode="full"
            )

            autocorrelation = autocorrelation[
                len(autocorrelation) // 2:
            ]

            if autocorrelation.size == 0:
                continue

            if autocorrelation[0] <= 0:
                continue

            # Normalize autocorrelation.
            autocorrelation = (
                autocorrelation / autocorrelation[0]
            )

            search_region = autocorrelation[
                min_period:max_period + 1
            ]

            if search_region.size == 0:
                continue

            # Locate the strongest periodic peak.
            peak_index = (
                int(np.argmax(search_region)) + min_period
            )

            peak_strength = float(
                autocorrelation[peak_index]
            )

            # Accept only reasonably reliable pitch estimates.
            if peak_strength < 0.35:
                continue

            fundamental_frequency = (
                sample_rate / peak_index
            )

            if min_f0 <= fundamental_frequency <= max_f0:
                detected_pitches.append(
                    fundamental_frequency
                )

        # If pitch detection fails, do not force "female".
        # Male is used as a neutral fallback for the existing UI.
        if len(detected_pitches) < 3:
            return "male"

        median_f0 = float(
            np.median(detected_pitches)
        )

        if median_f0 >= _FEMALE_F0_THRESHOLD:
            return "female"

        return "male"

    except Exception:
        # Prevent the previous issue where every error returned female.
        return "male"

    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def record_audio(duration_seconds: int = 10) -> np.ndarray:
    """Record mono audio from the microphone."""

    audio = sd.rec(
        int(duration_seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32"
    )

    sd.wait()

    return audio.flatten()


def normalize_audio(
    audio: np.ndarray,
    target_peak: float = 0.9
) -> np.ndarray:
    """
    Peak-normalize a float32 audio array so Whisper receives
    a sufficiently strong signal without clipping.
    """

    peak = np.abs(audio).max()

    if peak < 1e-6:
        return audio

    return audio * (target_peak / peak)


def check_audio_has_speech(
    audio: np.ndarray,
    rms_threshold: float = 0.01
) -> bool:
    """
    Return True if the audio contains enough energy to likely
    contain speech.

    Uses RMS energy on the loudest 50% of frames as a simple
    voice-activity-detection approximation.
    """

    if audio.size == 0:
        return False

    frame_length = int(SAMPLE_RATE * 0.02)

    frames = [
        audio[i:i + frame_length]
        for i in range(
            0,
            len(audio) - frame_length,
            frame_length
        )
    ]

    if not frames:
        rms = float(
            np.sqrt(np.mean(audio ** 2))
        )

        return rms >= rms_threshold

    energies = [
        float(np.sqrt(np.mean(frame ** 2)))
        for frame in frames
    ]

    energies.sort(reverse=True)

    top_half = energies[
        :max(1, len(energies) // 2)
    ]

    return float(np.mean(top_half)) >= rms_threshold


def numpy_to_wav_bytes(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE
) -> bytes:
    """Convert a NumPy audio array into WAV bytes."""

    audio_int16 = (
        np.clip(audio, -1.0, 1.0) * 32767
    ).astype(np.int16)

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    return buffer.getvalue()


def save_audio_to_temp(audio: np.ndarray) -> str:
    """Save NumPy audio as a temporary WAV file."""

    tmp = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    wav_write(
        tmp.name,
        SAMPLE_RATE,
        (audio * 32767).astype(np.int16)
    )

    tmp.close()

    return tmp.name


_whisper_models = {}


def _load_whisper(model_size: str = "small"):
    """Load and cache a Whisper model."""

    if model_size not in _whisper_models:
        import whisper

        _whisper_models[model_size] = (
            whisper.load_model(model_size)
        )

    return _whisper_models[model_size]


# Whisper anti-hallucination options.
_WHISPER_OPTS = dict(
    verbose=False,
    fp16=False,
    beam_size=1,
    best_of=1,
    temperature=0,
    condition_on_previous_text=False,
    no_speech_threshold=0.6,
    logprob_threshold=-1.0,
    compression_ratio_threshold=2.4,
)


def transcribe_uploaded_file(
    file_bytes: bytes,
    filename: str = "audio.wav",
    model_size: str = "small",
    language: Optional[str] = None,
    is_live: bool = False,
    **kwargs,
) -> Tuple[str, dict]:
    """
    Transcribe an uploaded audio file.

    Returns:
        (transcribed_text, whisper_result)
    """

    suffix = Path(filename).suffix or ".wav"

    tmp = tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False
    )

    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()

        import whisper

        # Whisper handles WAV, MP3, M4A, and other supported formats.
        audio = whisper.load_audio(tmp.name)

        return transcribe_audio(
            audio,
            model_size=model_size,
            language=language,
            is_live=is_live,
        )

    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def transcribe_audio(
    audio: np.ndarray,
    model_size: str = "small",
    language: Optional[str] = None,
    is_live: bool = False,
    **kwargs,
) -> Tuple[str, dict]:
    """
    Transcribe a NumPy audio array using Whisper.

    Returns:
        (transcribed_text, whisper_result)
    """

    # Normalize microphone/upload audio.
    audio = normalize_audio(audio)

    if not check_audio_has_speech(audio):
        raise ValueError(
            "No speech detected. Please speak clearly and try again."
        )

    model = _load_whisper(model_size)

    temporary_audio_path = save_audio_to_temp(audio)

    try:
        result = model.transcribe(
            temporary_audio_path,
            language=language,
            initial_prompt=(
                "Transcribe exactly what is spoken. "
                "Do not guess missing words. "
                "Do not complete unfinished sentences."
            ),
            **_WHISPER_OPTS,
        )

    finally:
        try:
            os.unlink(temporary_audio_path)
        except OSError:
            pass

    text = result.get("text", "").strip()

    if not text:
        raise ValueError(
            "Whisper could not detect speech."
        )

    # Optional confidence protection.
    segments = result.get("segments", [])

    if segments:
        average_log_probability = np.mean(
            [
                segment.get("avg_logprob", -10)
                for segment in segments
            ]
        )

        if average_log_probability < -1.5:
            raise ValueError(
                "Low confidence transcription. "
                "Please record again."
            )

    return text, result