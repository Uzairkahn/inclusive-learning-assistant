# Copilot instructions for Inclusive Learning Assistant

Purpose: Give AI coding agents the minimal, actionable knowledge to be productive in this repository.

- **Run / dev flow**: Activate the provided virtualenv then run the Flask app.
  - Windows PowerShell: `& "./fyp_env/Scripts/Activate.ps1"`
  - Install dependencies (if needed): `pip install -r requirements.txt`
  - Start app: `python app.py` (runs Flask on `http://127.0.0.1:5000` by default)

- **Big picture architecture**:
  - This is a small Flask monolith implemented in `app.py` that renders UI from `templates/` and serves JS/CSS from `static/`.
  - `utils/` contains helper modules; `app.py` adds `utils/` to `sys.path` and tries to dynamically load `utils/db.py` (or a top-level `db.py`) at runtime.
  - Persistent data: SQLite DB at `data/users.db` (created by `utils/db.py`). Logs stored at `logs/transcripts.json`. Uploaded files saved to `uploads/`.

- **Key files to read first**:
  - `app.py` — central routing, auth, API endpoints (STT, TTS, upload, summarize, translate, logs).
  - `utils/db.py` — DB schema and `create_user` / `verify_user` helpers.
  - `utils/speech_to_text.py` — microphone-based STT helper (used as a reference implementation).
  - `requirements.txt` — external deps (notably `transformers`, `torch`, `SpeechRecognition`, `PyAudio`, `gTTS`).

- **Data flows & integration points**:
  - Upload -> `app.extract_text_from_file()` (PDF/Docx/PPTX handled inline in `app.py` using `PyPDF2`, `python-docx`, `python-pptx`).
  - Summarization uses `transformers.pipeline("summarization")` in `app.py`. If the model fails to load, `summarizer` is set to `None` and the API returns a fallback message.
  - TTS uses `gTTS` and plays audio on Windows via `os.system(f"start {path}")` (keep Windows-specific behaviour in mind when changing playback).
  - STT uses `speech_recognition` and the microphone; `utils/speech_to_text.py` contains a reusable `recognize_speech()` function.

- **Project-specific conventions and patterns**:
  - Dynamic DB import: `app.py` will load `utils/db.py` if present, otherwise it falls back to inline stubs. Prefer updating `utils/db.py` for DB changes.
  - Session gating: many routes use `session['user_id']` — add new protected APIs with the same pattern.
  - Minimal front-end: use `templates/*.html` and `static/js/main.js` for client behavior. Keep template names consistent with existing routes (`stt.html`, `tts.html`, `upload.html`, `summarize.html`).
  - Logs: use `save_log(text)` in `app.py` to append transcript entries to `logs/transcripts.json`.

- **Developer notes / gotchas**:
  - Large dependencies: `transformers` / `torch` can be heavy and may try to download models at first run. The app already handles missing summarizer gracefully.
  - PyAudio installation on Windows often requires a prebuilt wheel. If STT fails, check that `PyAudio` is installed in the virtualenv.
  - Secret: `app.secret_key` is hard-coded in `app.py` for development — do not commit secrets when moving to production.
  - Database reset: to recreate schema delete `data/users.db` and restart app.

- **When changing code**:
  - Prefer adding helpers in `utils/` and calling them from `app.py` rather than large inline edits.
  - If you add database functions, register them in `utils/db.py` and keep their names `create_user`, `verify_user`, `init_db` so `app.py` can load them dynamically.
  - For TTS/STT updates, mirror the patterns in `utils/speech_to_text.py` and avoid breaking the synchronous flows that call system audio on Windows.

- **Quick examples**:
  - Guard an API route: `if 'user_id' not in session: return jsonify({...}), 401`
  - Append a log entry: call `save_log(text)` from `app.py` or replicate its pattern (open `logs/transcripts.json`, append, write).

If anything here is unclear or you'd like me to include extra examples (e.g., step-by-step for installing PyAudio on Windows or instructions for adding tests), tell me which area and I'll iterate. 
