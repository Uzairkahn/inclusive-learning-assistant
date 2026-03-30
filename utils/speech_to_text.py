import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

try:
    import whisper
except ImportError:  # pragma: no cover - depends on local environment
    whisper = None


ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav"}
WHISPER_MODEL_NAME = "base"
WHISPER_MODEL = None
WHISPER_LOAD_ERROR = None


def validate_audio_extension(filename):
    """
    Ensure the uploaded file has a supported audio extension.
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise ValueError("Only MP3 and WAV files are supported")
    return suffix


def save_uploaded_audio_temporarily(file_storage, temp_dir):
    """
    Persist the uploaded file to a temporary location for Whisper to read.
    """
    if file_storage is None:
        raise ValueError("No audio file was provided")

    filename = (file_storage.filename or "").strip()
    if not filename:
        raise ValueError("Please choose an audio file")

    suffix = validate_audio_extension(filename)
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_file = NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir)
    temp_path = Path(temp_file.name)
    temp_file.close()
    file_storage.save(temp_path)

    if temp_path.stat().st_size == 0:
        temp_path.unlink(missing_ok=True)
        raise ValueError("Uploaded file is empty")

    return temp_path


def initialize_whisper_model():
    """
    Load the Whisper base model once and store any load error for later reporting.
    """
    global WHISPER_MODEL, WHISPER_LOAD_ERROR

    try:
        if whisper is None:
            raise RuntimeError("openai-whisper is not installed")

        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not installed or not available on PATH")

        WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_NAME)
        WHISPER_LOAD_ERROR = None
    except Exception as exc:  # pragma: no cover - depends on local environment
        WHISPER_MODEL = None
        WHISPER_LOAD_ERROR = exc

    return WHISPER_MODEL


def get_whisper_status():
    """
    Return the cached Whisper model and any initialization error.
    """
    return WHISPER_MODEL, WHISPER_LOAD_ERROR


def transcribe_audio_file(file_path):
    """
    Transcribe the provided audio file with the globally loaded Whisper model.
    """
    if WHISPER_MODEL is None:
        error_message = str(WHISPER_LOAD_ERROR) if WHISPER_LOAD_ERROR else "Whisper model is unavailable"
        raise RuntimeError(error_message)

    result = WHISPER_MODEL.transcribe(str(file_path), fp16=torch.cuda.is_available())
    return (result.get("text") or "").strip()
