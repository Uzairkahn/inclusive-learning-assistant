# =============== IMPORTS ===============
# Flask modules for web application framework, templating, and session management
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from datetime import datetime  # For timestamp logging
import os  # Operating system operations (file handling, paths)
import sqlite3  # SQLite database for storing user credentials
from pathlib import Path  # Modern file path handling
import hashlib  # Password hashing for security
from threading import Lock

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from gtts.tts import gTTSError
from werkzeug.exceptions import BadRequest

from db import (
    get_stats,
    increment_audio,
    increment_summaries,
    increment_translations,
    increment_uploads,
    init_stats_table,
)
from utils.speech_to_text import (
    get_whisper_status,
    initialize_whisper_model,
    save_uploaded_audio_temporarily,
    transcribe_audio_file,
)
from utils.file_handler import extract_text_from_file, save_uploaded_document_temporarily
from utils.text_to_speech import ensure_audio_directory, generate_tts_audio

# =============== FLASK APP INITIALIZATION ===============
# Create Flask application instance
app = Flask(__name__)
# Set secret key for session encryption (hardcoded for development - NOT for production!)
app.secret_key = 'inclusive_learning_assistant_secret_key_2025'
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# =============== DATABASE CONFIGURATION ===============
# Database file location: stores user credentials (email, password, name, registration date)
# Located in 'data/users.db' relative to app.py directory
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "users.db"
TTS_AUDIO_DIR = ensure_audio_directory(BASE_DIR)
UPLOAD_TEMP_DIR = BASE_DIR / "uploads"
STT_TEMP_DIR = UPLOAD_TEMP_DIR
SUMMARIZER_MODEL_NAME = "facebook/bart-large-cnn"
SUMMARIZER_DEVICE = 0 if torch.cuda.is_available() else -1
TRANSLATOR_MODEL_NAME = "Helsinki-NLP/opus-mt-en-ur"
TRANSLATOR_DEVICE = 0 if torch.cuda.is_available() else -1
MAX_INPUT_CHARACTERS = 50000
CHUNK_TOKEN_SIZE = 900
CHUNK_TOKEN_OVERLAP = 100
MAX_SUMMARY_PASSES = 3
TRANSLATION_CHUNK_TOKEN_SIZE = 384
TRANSLATION_MAX_LENGTH = 256
SUMMARIZER = None
SUMMARIZER_TOKENIZER = None
SUMMARIZER_LOAD_ERROR = None
SUMMARIZER_LOCK = Lock()
TRANSLATOR = None
TRANSLATOR_TOKENIZER = None
TRANSLATOR_LOAD_ERROR = None
TRANSLATOR_LOCK = Lock()


def get_db_connection():
    """
    Create a SQLite connection with row access by column name.
    """
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection

def init_db():
    """
    Initialize SQLite database and create users table if it doesn't exist.
    Called once on app startup to ensure database schema exists.
    """
    # Create 'data' directory if it doesn't exist
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect to SQLite database (creates file if it doesn't exist)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Create users table with columns: id (auto-increment), email (unique), password, name, creation timestamp
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TEXT
        )
    """)

    # Save changes and close connection
    conn.commit()
    conn.close()

def hash_password(password):
    """
    Hash a password using SHA-256 algorithm for secure storage.
    Never stores plain-text passwords in the database.
    Args: password (str) - plain text password
    Returns: str - hashed password (64-character hex string)
    """
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(email, password, name):
    """
    Register a new user by storing their credentials in the database.
    Prevents duplicate emails (unique constraint).
    Args: email, password, name (str)
    Returns: True if successful, False if email already exists
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Hash password before storing (security best practice)
        hashed_pw = hash_password(password)
        # Record registration timestamp
        created_at = datetime.now().isoformat()
        # Insert new user into database
        cur.execute(
            "INSERT INTO users (email, password, name, created_at) VALUES (?, ?, ?, ?)",
            (email, hashed_pw, name, created_at)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Email already exists (UNIQUE constraint violation)
        return False
    finally:
        conn.close()

def verify_user(email, password):
    """
    Authenticate user during login by checking email and password.
    Args: email, password (str)
    Returns: (user_id, user_name) tuple if credentials valid, None if invalid
    """
    conn = get_db_connection()
    cur = conn.cursor()
    # Hash the provided password and compare with database record
    hashed_pw = hash_password(password)
    # Query: find user with matching email AND hashed password
    cur.execute("SELECT id, name FROM users WHERE email = ? AND password = ?", (email, hashed_pw))
    user = cur.fetchone()  # Returns tuple of (id, name) or None
    conn.close()
    return user

# =============== DATABASE INITIALIZATION ===============
# Call init_db() on app startup to ensure database exists and is properly configured
init_db()
init_stats_table()


# =============== SUMMARIZATION SERVICE ===============
def initialize_summarizer():
    """
    Load the summarization model once during app startup.
    If loading fails, the API returns a clear 503 error instead of crashing the app.
    """
    global SUMMARIZER, SUMMARIZER_TOKENIZER, SUMMARIZER_LOAD_ERROR

    try:
        SUMMARIZER_TOKENIZER = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL_NAME)
        summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL_NAME)
        summarizer_model.eval()

        if SUMMARIZER_DEVICE >= 0:
            summarizer_model = summarizer_model.to("cuda")

        SUMMARIZER = pipeline(
            "summarization",
            model=summarizer_model,
            tokenizer=SUMMARIZER_TOKENIZER,
            device=SUMMARIZER_DEVICE,
        )
        SUMMARIZER_LOAD_ERROR = None
        app.logger.info("Loaded summarization model: %s", SUMMARIZER_MODEL_NAME)
    except Exception as exc:
        SUMMARIZER = None
        SUMMARIZER_TOKENIZER = None
        SUMMARIZER_LOAD_ERROR = exc
        app.logger.exception("Failed to load summarization model")


def initialize_translator():
    """
    Load the translation model once during app startup.
    If loading fails, the API returns a clear 503 error instead of crashing the app.
    """
    global TRANSLATOR, TRANSLATOR_TOKENIZER, TRANSLATOR_LOAD_ERROR

    try:
        TRANSLATOR_TOKENIZER = AutoTokenizer.from_pretrained(TRANSLATOR_MODEL_NAME)
        translator_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR_MODEL_NAME)
        translator_model.eval()

        if TRANSLATOR_DEVICE >= 0:
            translator_model = translator_model.to("cuda")

        TRANSLATOR = pipeline(
            "translation_en_to_ur",
            model=translator_model,
            tokenizer=TRANSLATOR_TOKENIZER,
            device=TRANSLATOR_DEVICE,
        )
        TRANSLATOR_LOAD_ERROR = None
        app.logger.info("Loaded translation model: %s", TRANSLATOR_MODEL_NAME)
    except Exception as exc:
        TRANSLATOR = None
        TRANSLATOR_TOKENIZER = None
        TRANSLATOR_LOAD_ERROR = exc
        app.logger.exception("Failed to load translation model")


def json_error(message, status_code):
    """
    Return a consistent JSON error response.
    """
    return jsonify({"error": message}), status_code


def normalize_text(text):
    """
    Collapse repeated whitespace and trim the input.
    Very large payloads are capped to protect the server from pathological requests.
    """
    normalized_text = " ".join(text.split()).strip()
    if len(normalized_text) <= MAX_INPUT_CHARACTERS:
        return normalized_text

    truncated_text = normalized_text[:MAX_INPUT_CHARACTERS].rsplit(" ", 1)[0].strip()
    return truncated_text or normalized_text[:MAX_INPUT_CHARACTERS].strip()


def split_text_into_chunks(text, chunk_token_size=CHUNK_TOKEN_SIZE, overlap_tokens=CHUNK_TOKEN_OVERLAP):
    """
    Break long input into overlapping tokenizer-aware chunks that fit BART's context window.
    """
    token_ids = SUMMARIZER_TOKENIZER.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    chunks = []
    start_index = 0
    step = max(chunk_token_size - overlap_tokens, 1)

    while start_index < len(token_ids):
        end_index = min(start_index + chunk_token_size, len(token_ids))
        chunk_text = SUMMARIZER_TOKENIZER.decode(
            token_ids[start_index:end_index],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end_index >= len(token_ids):
            break
        start_index += step

    return chunks


def get_summary_lengths(text, aggressive=False):
    """
    Pick sensible generation lengths based on the amount of source text.
    """
    word_count = len(text.split())

    if aggressive:
        max_length = 90
        min_length = 25
    elif word_count < 80:
        max_length = 60
        min_length = 15
    elif word_count < 180:
        max_length = 90
        min_length = 25
    else:
        max_length = 130
        min_length = 40

    if min_length >= max_length:
        min_length = max(10, max_length - 5)

    return min_length, max_length


def summarize_chunk(text, aggressive=False):
    """
    Summarize a single chunk of text with deterministic generation settings.
    """
    if len(text.split()) < 25:
        return text

    min_length, max_length = get_summary_lengths(text, aggressive=aggressive)

    with SUMMARIZER_LOCK:
        with torch.inference_mode():
            result = SUMMARIZER(
                text,
                do_sample=False,
                truncation=True,
                min_length=min_length,
                max_length=max_length,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

    return result[0]["summary_text"].strip()


def generate_summary(text):
    """
    Generate a final summary, chunking and re-summarizing when the input is too large.
    """
    current_text = text

    for pass_number in range(MAX_SUMMARY_PASSES):
        chunks = split_text_into_chunks(current_text)
        if not chunks:
            return ""

        if len(chunks) == 1:
            return summarize_chunk(chunks[0], aggressive=pass_number > 0)

        chunk_summaries = [
            summarize_chunk(chunk, aggressive=pass_number > 0)
            for chunk in chunks
        ]
        current_text = " ".join(summary for summary in chunk_summaries if summary).strip()

    final_chunks = split_text_into_chunks(current_text, overlap_tokens=0)
    final_summary_parts = [
        summarize_chunk(chunk, aggressive=True)
        for chunk in final_chunks
    ]
    return " ".join(part for part in final_summary_parts if part).strip()


def split_translation_text_into_chunks(text, chunk_token_size=TRANSLATION_CHUNK_TOKEN_SIZE):
    """
    Break translation input into tokenizer-aware chunks that fit the model context window.
    """
    token_ids = TRANSLATOR_TOKENIZER.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    chunks = []
    for start_index in range(0, len(token_ids), chunk_token_size):
        end_index = min(start_index + chunk_token_size, len(token_ids))
        chunk_text = TRANSLATOR_TOKENIZER.decode(
            token_ids[start_index:end_index],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def generate_translation(text):
    """
    Translate English text to Urdu, chunking large input to avoid model truncation.
    """
    translated_parts = []

    for chunk in split_translation_text_into_chunks(text):
        with TRANSLATOR_LOCK:
            with torch.inference_mode():
                result = TRANSLATOR(
                    chunk,
                    do_sample=False,
                    truncation=True,
                    max_length=TRANSLATION_MAX_LENGTH,
                    clean_up_tokenization_spaces=True,
                )

        translated_text = result[0]["translation_text"].strip()
        if translated_text:
            translated_parts.append(translated_text)

    return " ".join(translated_parts).strip()


initialize_summarizer()
initialize_translator()
initialize_whisper_model()

# =============== AUTHENTICATION ROUTES ===============
# These routes handle user login, registration, and logout
# Protected routes check for 'user_id' in session to ensure user is authenticated

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Login route: Displays login form (GET) and processes login (POST).
    On successful login, stores user_id and user_name in session and redirects to dashboard.
    """
    if request.method == 'POST':
        # Get credentials from login form
        email = request.form.get('email')
        password = request.form.get('password')
        # Verify credentials against database
        user = verify_user(email, password)
        if user:
            # Store user info in session (browser cookie, encrypted with secret_key)
            session['user_id'] = user[0]  # Store user ID
            session['user_name'] = user[1]  # Store user name
            # Redirect to dashboard
            return redirect(url_for('index'))
        else:
            # Invalid credentials - show error and re-render login form
            return render_template('login.html', error='Invalid email or password')
    # GET request: display login form
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    Registration route: Displays signup form (GET) and creates new user account (POST).
    """
    if request.method == 'POST':
        # Get registration form data
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Attempt to create new user
        if create_user(email, password, name):
            # Success: redirect to login page
            return redirect(url_for('login'))
        else:
            # Failure: email already exists - show error
            return render_template('register.html', error='Email already exists')
    # GET request: display registration form
    return render_template('register.html')

@app.route('/logout')
def logout():
    """
    Logout route: Clears all session data and redirects to login page.
    """
    # Remove all session data (user_id, user_name, etc.)
    session.clear()
    return redirect(url_for('login'))

# =============== PAGE ROUTES ===============
# Each route checks if user is logged in (has user_id in session)
# If not logged in, redirects to login page

@app.route('/')
def index():
    """
    Dashboard/Home page: Main landing page after login.
    Displays usage statistics and recent activity.
    """
    # Check if user is authenticated
    if 'user_id' not in session:
        return redirect(url_for('login'))
    # Render dashboard with datetime module available in template
    return render_template("index.html", datetime=datetime)

@app.route("/stt")
def stt():
    """
    Speech-to-Text page: Allows users to record audio and convert it to text.
    """
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template("stt.html")

@app.route("/tts-page")
def tts_page():
    """
    Text-to-Speech page: Allows users to convert text to speech (audio).
    """
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template("tts.html")

@app.route("/readability")
def readability():
    """
    Readability/Accessibility page: Allows users to adjust text size, font, contrast.
    """
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template("readability.html")

@app.route("/summarize")
def summarize():
    """
    Summarization page: Allows users to generate AI summaries of text content.
    """
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template("summarize.html")

@app.route("/about")
def about():
    """
    About page: Displays project information, team details, and technology stack.
    """
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template("about.html")


# =============== DASHBOARD & ACTIVITY APIs ===============
# Endpoints that provide data for the dashboard and activity feeds.

@app.route("/api/dashboard-stats")
def dashboard_stats():
    """
    Dashboard Statistics API: Returns persisted application usage counters.
    """
    if 'user_id' not in session:
        return json_error("Please login first", 401)

    try:
        stats = get_stats()
        app.logger.info("Dashboard stats loaded: %s", stats)
    except Exception:
        app.logger.exception("Failed to load dashboard stats")
        return json_error("Failed to load dashboard stats", 500)

    return jsonify(stats)

@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    """
    Summarization API: Accepts JSON input and returns an AI-generated summary.
    Handles invalid payloads, empty text, model availability, and inference failures.
    """
    if 'user_id' not in session:
        return json_error("Please login first", 401)

    if not request.is_json:
        return json_error("Request body must be valid JSON", 400)

    try:
        payload = request.get_json()
    except BadRequest:
        return json_error("Malformed JSON payload", 400)

    if not isinstance(payload, dict):
        return json_error("JSON body must be an object", 400)

    text = payload.get("text")
    if not isinstance(text, str):
        return json_error("The 'text' field must be a string", 400)

    normalized_text = normalize_text(text)
    if not normalized_text:
        return json_error("Text input cannot be empty", 400)

    if SUMMARIZER is None:
        app.logger.error("Summarizer unavailable: %s", SUMMARIZER_LOAD_ERROR)
        return json_error("Summarization model is unavailable. Please try again later.", 503)

    try:
        summary = generate_summary(normalized_text)
    except Exception:
        app.logger.exception("Summarization request failed")
        return json_error("Failed to summarize text", 500)

    if not summary:
        return json_error("Unable to generate a summary for the provided text", 500)

    try:
        updated_stats = increment_summaries()
        app.logger.info("Summary stats incremented: %s", updated_stats)
    except Exception:
        app.logger.exception("Failed to record summary usage")

    return jsonify({"summary": summary})

@app.route("/api/translate", methods=["POST"])
def api_translate():
    """
    Translation API: Accepts JSON input and returns Urdu translation.
    Handles invalid payloads, empty text, model availability, and inference failures.
    """
    if 'user_id' not in session:
        return json_error("Please login first", 401)

    if not request.is_json:
        return json_error("Request body must be valid JSON", 400)

    try:
        payload = request.get_json()
    except BadRequest:
        return json_error("Malformed JSON payload", 400)

    if not isinstance(payload, dict):
        return json_error("JSON body must be an object", 400)

    text = payload.get("text")
    if not isinstance(text, str):
        return json_error("The 'text' field must be a string", 400)

    normalized_text = normalize_text(text)
    if not normalized_text:
        return json_error("Text input cannot be empty", 400)

    if TRANSLATOR is None:
        app.logger.error("Translator unavailable: %s", TRANSLATOR_LOAD_ERROR)
        return json_error("Translation model is unavailable. Please try again later.", 503)

    try:
        translated_text = generate_translation(normalized_text)
    except Exception:
        app.logger.exception("Translation request failed")
        return json_error("Failed to translate text", 500)

    if not translated_text:
        return json_error("Unable to translate the provided text", 500)

    try:
        updated_stats = increment_translations()
        app.logger.info("Translation stats incremented: %s", updated_stats)
    except Exception:
        app.logger.exception("Failed to record translation usage")

    return jsonify({"translated_text": translated_text})

@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    File upload API: Accepts PDF, DOCX, or TXT files and returns extracted text.
    Saves the upload temporarily and always removes it after processing.
    """
    if 'user_id' not in session:
        return json_error("Please login first", 401)

    uploaded_file = request.files.get("file")
    if uploaded_file is None:
        return json_error("File is required", 400)

    temp_file_path = None

    try:
        temp_file_path = save_uploaded_document_temporarily(uploaded_file, UPLOAD_TEMP_DIR)
        extracted_text = extract_text_from_file(temp_file_path)
    except ValueError as exc:
        return json_error(str(exc), 400)
    except Exception:
        app.logger.exception("Document upload processing failed")
        return json_error("Failed to extract text from the uploaded file", 500)
    finally:
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)

    try:
        updated_stats = increment_uploads()
        app.logger.info("Upload stats incremented: %s", updated_stats)
    except Exception:
        app.logger.exception("Failed to record upload usage")

    return jsonify({"text": extracted_text})

@app.route("/api/tts", methods=["POST"])
def api_tts():
    """
    Text-to-Speech API: Accepts JSON input and returns the generated audio file URL.
    Handles invalid payloads, empty text, unsupported languages, and TTS failures.
    """
    if 'user_id' not in session:
        return json_error("Please login first", 401)

    if not request.is_json:
        return json_error("Request body must be valid JSON", 400)

    try:
        payload = request.get_json()
    except BadRequest:
        return json_error("Malformed JSON payload", 400)

    if not isinstance(payload, dict):
        return json_error("JSON body must be an object", 400)

    text = payload.get("text")
    lang = payload.get("lang")

    if not isinstance(text, str):
        return json_error("The 'text' field must be a string", 400)

    if not isinstance(lang, str):
        return json_error("The 'lang' field must be a string", 400)

    try:
        _, filename = generate_tts_audio(text=text, lang=lang, output_dir=TTS_AUDIO_DIR)
    except ValueError as exc:
        return json_error(str(exc), 400)
    except gTTSError:
        app.logger.exception("TTS generation failed")
        return json_error("Text-to-speech service is currently unavailable. Please try again later.", 502)
    except Exception:
        app.logger.exception("Unexpected TTS generation failure")
        return json_error("Failed to generate audio", 500)

    audio_url = url_for("static", filename=f"audio/{filename}")

    try:
        updated_stats = increment_audio()
        app.logger.info("Audio stats incremented: %s", updated_stats)
    except Exception:
        app.logger.exception("Failed to record audio usage")

    return jsonify({"audio_url": audio_url})

@app.route("/api/stt", methods=["POST"])
def api_stt():
    """
    Speech-to-Text API: Accepts an uploaded audio file and returns its transcript.
    Uses a globally loaded Whisper base model and removes temp files after processing.
    """
    if 'user_id' not in session:
        return json_error("Please login first", 401)

    whisper_model, whisper_error = get_whisper_status()
    if whisper_model is None:
        app.logger.error("Whisper unavailable: %s", whisper_error)
        return json_error("Speech-to-text model is unavailable. Please check Whisper and ffmpeg setup.", 503)

    audio_file = request.files.get("audio")
    if audio_file is None:
        return json_error("Audio file is required", 400)

    temp_file_path = None

    try:
        temp_file_path = save_uploaded_audio_temporarily(audio_file, STT_TEMP_DIR)
        transcript = transcribe_audio_file(temp_file_path)
    except ValueError as exc:
        return json_error(str(exc), 400)
    except RuntimeError as exc:
        app.logger.exception("STT processing failed")
        return json_error(str(exc), 500)
    except Exception:
        app.logger.exception("Unexpected STT failure")
        return json_error("Failed to transcribe audio", 500)
    finally:
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)

    return jsonify({"text": transcript})

@app.route("/api/recent-activity") 
def recent_activity():
    """
    Recent Activity API: Returns list of user's recent activities.
    Currently returns demo activities.
    In production: would read from logs/transcripts.json or database.
    Each activity has: icon, color, title, timestamp, description.
    """
    if 'user_id' not in session:
        return jsonify({"error": "Please login first"}), 401
    
    # Demo activities shown on dashboard
    activities = [
        {
            "icon": "🔊",
            "color": "sky",
            "title": "Text converted to speech", 
            "time": "2 minutes ago",
            "description": "Chapter 3 of Computer Science book"
        },
        {
            "icon": "📄",
            "color": "emerald",
            "title": "PDF file uploaded",
            "time": "1 hour ago",
            "description": "Mathematics_Notes.pdf"
        },
        {
            "icon": "🧠",
            "color": "purple", 
            "title": "Chapter summarized",
            "time": "3 hours ago",
            "description": "AI and Machine Learning basics"
        },
        {
            "icon": "👁️",
            "color": "blue",
            "title": "Readability settings adjusted", 
            "time": "5 hours ago",
            "description": "Font size and contrast customized"
        }
    ]
    
    return jsonify(activities)

# =============== APPLICATION ENTRY POINT ===============
if __name__ == "__main__":
    """
    Start Flask development server.
    debug=True: Auto-reloads on code changes, enables debugger, shows detailed error pages.
    Note: Only for development! Do NOT use in production.
    Access app at: http://127.0.0.1:5000
    """
    app.run(debug=True)
