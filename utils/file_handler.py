from pathlib import Path
from tempfile import NamedTemporaryFile

from PyPDF2 import PdfReader
from docx import Document
from werkzeug.utils import secure_filename


ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt"}


def validate_document_extension(filename):
    """
    Ensure the uploaded document uses a supported extension.
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_DOCUMENT_EXTENSIONS:
        raise ValueError("Only PDF, DOCX, and TXT files are supported")
    return suffix


def save_uploaded_document_temporarily(file_storage, temp_dir):
    """
    Save an uploaded document into the temporary uploads directory.
    """
    if file_storage is None:
        raise ValueError("No file was provided")

    filename = (file_storage.filename or "").strip()
    if not filename:
        raise ValueError("Please choose a file")

    suffix = validate_document_extension(filename)
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = secure_filename(Path(filename).stem) or "upload"
    temp_file = NamedTemporaryFile(
        delete=False,
        dir=temp_dir,
        prefix=f"{safe_stem}_",
        suffix=suffix,
    )
    temp_path = Path(temp_file.name)
    temp_file.close()

    file_storage.save(temp_path)

    if temp_path.stat().st_size == 0:
        temp_path.unlink(missing_ok=True)
        raise ValueError("Uploaded file is empty")

    return temp_path


def clean_extracted_text(text):
    """
    Trim noisy whitespace while preserving paragraph breaks.
    """
    compact_lines = []
    last_line_blank = False

    for raw_line in text.replace("\x00", "").splitlines():
        line = raw_line.strip()
        if line:
            compact_lines.append(line)
            last_line_blank = False
        elif not last_line_blank:
            compact_lines.append("")
            last_line_blank = True

    cleaned_text = "\n".join(compact_lines).strip()
    if not cleaned_text:
        raise ValueError("No readable text found in the uploaded file")

    return cleaned_text


def extract_text_from_pdf(file_path):
    """
    Extract readable text from a PDF document.
    """
    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:
        raise ValueError("Unable to read the uploaded PDF file") from exc

    page_text = []
    for page in reader.pages:
        extracted_text = page.extract_text() or ""
        if extracted_text.strip():
            page_text.append(extracted_text.strip())

    return clean_extracted_text("\n\n".join(page_text))


def extract_text_from_docx(file_path):
    """
    Extract text from DOCX paragraphs and tables.
    """
    try:
        document = Document(str(file_path))
    except Exception as exc:
        raise ValueError("Unable to read the uploaded DOCX file") from exc

    sections = []

    sections.extend(
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text and paragraph.text.strip()
    )

    for table in document.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
            if row_text:
                sections.append(" | ".join(row_text))

    return clean_extracted_text("\n\n".join(sections))


def extract_text_from_txt(file_path):
    """
    Read plain text from a TXT file.
    """
    path = Path(file_path)

    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="latin-1")
        except Exception as exc:
            raise ValueError("Unable to read the uploaded TXT file") from exc
    except Exception as exc:
        raise ValueError("Unable to read the uploaded TXT file") from exc

    return clean_extracted_text(text)


def extract_text_from_file(file_path):
    """
    Dispatch text extraction based on the uploaded document extension.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix == ".docx":
        return extract_text_from_docx(path)
    if suffix == ".txt":
        return extract_text_from_txt(path)

    raise ValueError("Only PDF, DOCX, and TXT files are supported")
