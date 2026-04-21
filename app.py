# =============== IMPORTS ===============
# Flask modules for web application framework, templating, and session management
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from datetime import datetime  # For timestamp logging
import os  # Operating system operations (file handling, paths)
import re  # Sentence-aware chunking for translation
import sqlite3  # SQLite database for storing user credentials
from pathlib import Path  # Modern file path handling
import bcrypt  # Password hashing for security
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
SUMMARIZER_MODEL_NAME = os.environ.get("SUMMARIZER_MODEL_NAME", "facebook/bart-large-cnn")
SUMMARIZER_DEVICE = 0 if torch.cuda.is_available() else -1
TRANSLATOR_MODEL_NAME = os.environ.get("TRANSLATOR_MODEL_NAME", "Helsinki-NLP/opus-mt-en-ur")
TRANSLATOR_DEVICE = 0 if torch.cuda.is_available() else -1
MAX_INPUT_CHARACTERS = 50000
SUMMARY_MAX_INPUT_CHARACTERS = 30000
SUMMARY_MODEL_CHARACTER_LIMIT = 16000
CHUNK_TOKEN_SIZE = 900
CHUNK_TOKEN_OVERLAP = 40
MAX_SUMMARY_PASSES = 2
SUMMARY_NUM_BEAMS = 2
TRANSLATION_CHUNK_TOKEN_SIZE = 170
TRANSLATION_MAX_INPUT_CHARACTERS = 12000
TRANSLATION_MAX_LENGTH = 512
TRANSLATION_NUM_BEAMS = 5
TRANSLATION_REPETITION_PENALTY = 1.12
TRANSLATION_NO_REPEAT_NGRAM_SIZE = 3
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
    Hash a password using bcrypt for secure storage.
    Never stores plain-text passwords in the database.
    Args: password (str) - plain text password
    Returns: bytes - hashed password
    """
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

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
    # Query: find user with matching email
    cur.execute("SELECT id, name, password FROM users WHERE email = ?", (email,))
    user = cur.fetchone()  # Returns tuple of (id, name, password_hash) or None
    conn.close()

    # If user found, verify password using bcrypt
    if user:
        user_id, name, stored_hash = user
        try:
            # Convert stored_hash to bytes if it's a string (for bcrypt compatibility)
            if isinstance(stored_hash, str):
                stored_hash = stored_hash.encode()
            if bcrypt.checkpw(password.encode(), stored_hash):
                return (user_id, name)
        except (ValueError, TypeError):
            pass  # Invalid hash format

    return None

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


def truncate_text_to_limit(text, max_characters):
    """
    Trim text to a word boundary so huge requests do not overload the models.
    """
    if len(text) <= max_characters:
        return text

    truncated_text = text[:max_characters].rsplit(" ", 1)[0].strip()
    return truncated_text or text[:max_characters].strip()


def normalize_text(text, max_characters=MAX_INPUT_CHARACTERS, preserve_paragraphs=False):
    """
    Collapse repeated whitespace and trim the input.
    Very large payloads are capped to protect the server from pathological requests.
    """
    if preserve_paragraphs:
        compact_lines = []
        last_line_blank = False

        for raw_line in text.replace("\x00", "").splitlines():
            line = " ".join(raw_line.split()).strip()
            if line:
                compact_lines.append(line)
                last_line_blank = False
            elif not last_line_blank:
                compact_lines.append("")
                last_line_blank = True

        normalized_text = "\n".join(compact_lines).strip()
    else:
        normalized_text = " ".join(text.split()).strip()

    return truncate_text_to_limit(normalized_text, max_characters)


def split_sentences(text):
    """
    Split text into English/Urdu sentence-like units for fast pre-processing.
    """
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?;\u061f\u06d4])\s+", text)
        if sentence.strip()
    ]


def contains_urdu_script(text):
    """
    Detect whether text already contains Urdu/Arabic-script characters.
    """
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def prepare_text_for_fast_summary(text):
    """
    Reduce very long documents before neural summarization.

    Transformer summarization is the slowest part of the app. For uploaded notes,
    keeping the first few sentences from each paragraph preserves section coverage
    while avoiding dozens of expensive model calls.
    """
    if len(text) <= SUMMARY_MODEL_CHARACTER_LIMIT:
        return text

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", text)
        if paragraph.strip()
    ]

    selected_parts = []
    selected_length = 0

    if len(paragraphs) > 1:
        for paragraph in paragraphs:
            paragraph_sentences = split_sentences(paragraph)
            candidates = paragraph_sentences[:2] if paragraph_sentences else [paragraph]

            for sentence in candidates:
                if selected_length + len(sentence) + 2 > SUMMARY_MODEL_CHARACTER_LIMIT:
                    if not selected_parts:
                        return truncate_text_to_limit(sentence, SUMMARY_MODEL_CHARACTER_LIMIT)
                    return "\n\n".join(selected_parts).strip()
                selected_parts.append(sentence)
                selected_length += len(sentence) + 2

        prepared_text = "\n\n".join(selected_parts).strip()
        if prepared_text:
            return prepared_text

    selected_sentences = []
    selected_length = 0
    for sentence in split_sentences(text):
        if selected_length + len(sentence) + 1 > SUMMARY_MODEL_CHARACTER_LIMIT:
            break
        selected_sentences.append(sentence)
        selected_length += len(sentence) + 1

    prepared_text = " ".join(selected_sentences).strip()
    return prepared_text or truncate_text_to_limit(text, SUMMARY_MODEL_CHARACTER_LIMIT)


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
        max_length = 75
        min_length = 20
    elif word_count < 80:
        max_length = 60
        min_length = 15
    elif word_count < 180:
        max_length = 80
        min_length = 22
    else:
        max_length = 110
        min_length = 32

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
                num_beams=SUMMARY_NUM_BEAMS,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

    return result[0]["summary_text"].strip()


def generate_summary(text):
    """
    Generate a final summary, chunking and re-summarizing when the input is too large.
    """
    current_text = prepare_text_for_fast_summary(text)

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


def split_long_translation_unit(text, chunk_token_size=TRANSLATION_CHUNK_TOKEN_SIZE):
    """
    Split one oversized sentence/paragraph into tokenizer-sized chunks.
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


def normalize_translation_source(text):
    """
    Clean common extracted-document noise before English-to-Urdu translation.
    """
    normalized = text.replace("\x00", " ")
    normalized = re.sub(r"(\w)-\s+(\w)", r"\1\2", normalized)
    normalized = re.sub(r"\s*([,.;:!?])\s*", r"\1 ", normalized)
    normalized = re.sub(r"[\u2022\u25cf\u25aa]\s*", "- ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def split_translation_text_into_chunks(text, chunk_token_size=TRANSLATION_CHUNK_TOKEN_SIZE):
    """
    Break translation input into sentence-aware chunks.

    Arbitrary token cuts can split sentences in half, which hurts Urdu output quality,
    especially with text extracted from uploaded documents. This keeps complete
    sentences together when possible and only token-splits unusually long units.
    """
    if not TRANSLATOR_TOKENIZER:
        raise RuntimeError("Translation tokenizer is not initialized")

    normalized = normalize_translation_source(text)
    if not normalized:
        return []

    sentence_units = [
        unit.strip()
        for unit in re.split(r"(?<=[.!?;:\u061f\u06d4])\s+", normalized)
        if unit.strip()
    ]

    chunks = []
    current_parts = []
    current_token_count = 0

    for unit in sentence_units:
        unit_token_count = len(TRANSLATOR_TOKENIZER.encode(unit, add_special_tokens=False))

        if unit_token_count > chunk_token_size:
            if current_parts:
                chunks.append(" ".join(current_parts).strip())
                current_parts = []
                current_token_count = 0
            chunks.extend(split_long_translation_unit(unit, chunk_token_size))
            continue

        if current_parts and current_token_count + unit_token_count > chunk_token_size:
            chunks.append(" ".join(current_parts).strip())
            current_parts = [unit]
            current_token_count = unit_token_count
        else:
            current_parts.append(unit)
            current_token_count += unit_token_count

    if current_parts:
        chunks.append(" ".join(current_parts).strip())

    return [chunk for chunk in chunks if chunk]


def clean_urdu_translation(text):
    """
    Light cleanup for generated Urdu text without changing its meaning.
    """
    cleaned = " ".join(text.split()).strip()
    cleaned = re.sub("\\s+([\\u06d4\\u060c\\u061f!])", r"\1", cleaned)
    cleaned = re.sub("([\\u06d4\\u060c\\u061f!])(?=\\S)", r"\1 ", cleaned)
    return cleaned


def generate_translation(text):
    """
    Translate English text to Urdu, chunking large input to avoid model truncation.
    """
    if not TRANSLATOR:
        raise RuntimeError("Translation model is not initialized")

    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    translated_parts = []

    chunks = split_translation_text_into_chunks(text)
    if not chunks:
        raise ValueError("Unable to tokenize the input text")

    for chunk in chunks:
        try:
            with TRANSLATOR_LOCK:
                with torch.inference_mode():
                    result = TRANSLATOR(
                        chunk,
                        do_sample=False,
                        truncation=True,
                        max_length=TRANSLATION_MAX_LENGTH,
                        num_beams=TRANSLATION_NUM_BEAMS,
                        repetition_penalty=TRANSLATION_REPETITION_PENALTY,
                        no_repeat_ngram_size=TRANSLATION_NO_REPEAT_NGRAM_SIZE,
                        early_stopping=True,
                        clean_up_tokenization_spaces=True,
                    )

            if result and len(result) > 0:
                translated_text = clean_urdu_translation(result[0].get("translation_text", ""))
                if translated_text:
                    translated_parts.append(translated_text)
        except Exception as exc:
            app.logger.error("Error translating chunk: %s", exc)
            raise

    final_translation = clean_urdu_translation(" ".join(translated_parts))
    if not final_translation:
        raise RuntimeError("Translation resulted in empty output")

    return final_translation


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

    normalized_text = normalize_text(
        text,
        max_characters=SUMMARY_MAX_INPUT_CHARACTERS,
        preserve_paragraphs=True,
    )
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

    normalized_text = normalize_text(
        text,
        max_characters=TRANSLATION_MAX_INPUT_CHARACTERS,
    )
    if not normalized_text:
        return json_error("Text input cannot be empty", 400)

    if TRANSLATOR is None:
        error_msg = str(TRANSLATOR_LOAD_ERROR) if TRANSLATOR_LOAD_ERROR else "Unknown error"
        app.logger.error("Translator unavailable: %s", error_msg)
        return json_error("Translation model is unavailable. Please try again later.", 503)

    try:
        translated_text = generate_translation(normalized_text)
    except ValueError as exc:
        app.logger.warning("Translation validation error: %s", exc)
        return json_error(f"Translation error: {str(exc)}", 400)
    except Exception as exc:
        app.logger.exception("Translation request failed: %s", exc)
        return json_error("Failed to translate text. Please try shorter text.", 500)

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
    File upload API: Accepts PDF, DOCX, TXT, or PPTX files and returns extracted text.
    Saves the upload temporarily and always removes it after processing.
    """
    if 'user_id' not in session:
        return json_error("Please login first", 401)

    uploaded_file = request.files.get("file")
    if uploaded_file is None:
        return json_error("File is required", 400)

    requested_max_chars = request.form.get("max_chars")
    upload_max_characters = None
    if requested_max_chars:
        try:
            upload_max_characters = max(1000, min(int(requested_max_chars), MAX_INPUT_CHARACTERS))
        except (TypeError, ValueError):
            return json_error("The 'max_chars' field must be a number", 400)

    temp_file_path = None

    try:
        temp_file_path = save_uploaded_document_temporarily(uploaded_file, UPLOAD_TEMP_DIR)
        extracted_text = extract_text_from_file(
            temp_file_path,
            max_characters=upload_max_characters,
        )
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

    returned_length = len(extracted_text)
    is_likely_truncated = bool(
        upload_max_characters and returned_length >= max(upload_max_characters - 100, 0)
    )

    return jsonify({
        "text": extracted_text,
        "truncated": is_likely_truncated,
        "returned_length": returned_length,
    })

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

    normalized_lang = lang.strip().lower()
    text_for_audio = " ".join(text.split()).strip()
    was_auto_translated = False

    if not text_for_audio:
        return json_error("Text input cannot be empty", 400)

    if normalized_lang == "ur" and not contains_urdu_script(text_for_audio):
        if TRANSLATOR is None:
            app.logger.error("Translator unavailable for Urdu TTS: %s", TRANSLATOR_LOAD_ERROR)
            return json_error(
                "Urdu auto-translation is unavailable. Please paste Urdu text or try again later.",
                503,
            )

        try:
            text_for_audio = generate_translation(text_for_audio)
            was_auto_translated = True
        except Exception as exc:
            app.logger.exception("Urdu TTS auto-translation failed: %s", exc)
            return json_error("Failed to translate text before Urdu audio generation.", 500)

    try:
        _, filename = generate_tts_audio(
            text=text_for_audio,
            lang=normalized_lang,
            output_dir=TTS_AUDIO_DIR,
        )
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

    return jsonify({
        "audio_url": audio_url,
        "spoken_text": text_for_audio,
        "translated": was_auto_translated,
    })

@app.route("/api/stt", methods=["POST"])
def api_stt():
    """
    Speech-to-Text API: Accepts an uploaded audio file and returns its transcript and SRT subtitles.
    Uses a globally loaded Whisper base model and removes temp files after processing.
    Returns: JSON with "text" (plain text) and "srt" (SRT subtitle format)
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
        transcript, srt_content = transcribe_audio_file(temp_file_path)
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

    return jsonify({"text": transcript, "srt": srt_content})

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
