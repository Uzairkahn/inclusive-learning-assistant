from datetime import datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from gtts import gTTS


SUPPORTED_TTS_LANGUAGES = {
    "en": "English",
    "ur": "Urdu",
}


def normalize_tts_text(text):
    """
    Normalize incoming text before sending it to the speech engine.
    """
    return " ".join(text.split()).strip()


def validate_tts_language(lang):
    """
    Ensure the requested language is supported by this application.
    """
    normalized_lang = (lang or "").strip().lower()
    if normalized_lang not in SUPPORTED_TTS_LANGUAGES:
        raise ValueError("Language must be 'en' or 'ur'")
    return normalized_lang


def ensure_audio_directory(base_dir):
    """
    Create the static/audio directory if it does not already exist.
    """
    audio_dir = Path(base_dir) / "static" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir


def build_audio_filename(lang, text=None):
    """
    Generate a unique filename for each TTS request.
    """
    if text:
        digest = sha256(f"{lang}\n{text}".encode("utf-8")).hexdigest()[:16]
        return f"tts_{lang}_{digest}.mp3"

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"tts_{lang}_{timestamp}_{uuid4().hex[:8]}.mp3"


def generate_tts_audio(text, lang, output_dir):
    """
    Generate a speech file from text and return the saved file path and filename.
    """
    normalized_text = normalize_tts_text(text)
    if not normalized_text:
        raise ValueError("Text input cannot be empty")

    normalized_lang = validate_tts_language(lang)
    audio_dir = Path(output_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    filename = build_audio_filename(normalized_lang, normalized_text)
    output_path = audio_dir / filename
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path, filename

    speech = gTTS(text=normalized_text, lang=normalized_lang, slow=False)
    speech.save(str(output_path))

    return output_path, filename
