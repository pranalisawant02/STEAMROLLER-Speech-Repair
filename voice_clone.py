import os
import uuid
import requests


class VoiceClone:
    def __init__(self, api_url=None):
        self.api_url = (
            api_url or os.getenv("KAGGLE_TTS_URL", "")
        ).rstrip("/")

        if not self.api_url:
            raise RuntimeError(
                "KAGGLE_TTS_URL is not configured."
            )

    def health_check(self):
        url = f"{self.api_url}/health"

        try:
            response = requests.get(
                url,
                timeout=20
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"F5-TTS health check failed: {exc}"
            ) from exc

    def generate(
        self,
        text,
        speaker_wav,
        ref_text="",
        language="en",
        speed=0.9
    ):
        if not text or not text.strip():
            raise ValueError("Text for voice generation is empty.")

        if not speaker_wav or not os.path.exists(speaker_wav):
            raise FileNotFoundError(
                f"Reference audio not found: {speaker_wav}"
            )

        self.health_check()

        url = f"{self.api_url}/clone"

        data = {
            "text": text.strip(),
            "ref_text": ref_text.strip(),
        }

        with open(speaker_wav, "rb") as audio_file:
            files = {
                "speaker_wav": (
                    os.path.basename(speaker_wav),
                    audio_file,
                    "audio/wav"
                )
            }

            try:
                response = requests.post(
                    url,
                    data=data,
                    files=files,
                    timeout=(20, 300)
                )
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"Could not connect to Kaggle F5-TTS server: {exc}"
                ) from exc

        if response.status_code != 200:
            error_text = response.text[:3000]
            raise RuntimeError(
                f"F5-TTS server returned "
                f"{response.status_code}: {error_text}"
            )

        if not response.content:
            raise RuntimeError(
                "F5-TTS server returned an empty audio response."
            )

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir,
            f"clone_{uuid.uuid4().hex}.wav"
        )

        with open(output_path, "wb") as output_file:
            output_file.write(response.content)

        return output_path