"""
app/services/ocr.py
Handles PDF / image ingestion, direct digital text extraction,
and Tesseract OCR with automatic Windows path discovery.
"""

from __future__ import annotations

import io
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageEnhance, ImageOps
import pytesseract

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    PdfReader = None
    PdfWriter = None

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".json"}
SUPPORTED_EXTENSIONS = SUPPORTED_DOCUMENT_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS

# ---------------------------------------------------------------------------
# Auto-discover Tesseract on Windows if not already in system PATH
# ---------------------------------------------------------------------------
def _configure_tesseract() -> bool:
    if shutil.which("tesseract"):
        return True

    custom_path = os.getenv("TESSERACT_PATH")
    if custom_path and Path(custom_path).exists():
        pytesseract.pytesseract.tesseract_cmd = custom_path
        return True

    user_home = Path.home()
    common_windows_paths = [
        user_home / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe",
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    for p in common_windows_paths:
        if p.exists():
            pytesseract.pytesseract.tesseract_cmd = str(p)
            return True
    return False

_configure_tesseract()


def preprocess_image(image: Image.Image) -> Image.Image:
    """Convert to greyscale, normalise contrast, and sharpen slightly."""
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.5)
    return image


def _extract_pdf_digital_text(file_path: Path) -> Tuple[str, int]:
    """Extract embedded text directly from digital PDFs without OCR dependencies."""
    if PdfReader is None:
        return "", 0

    try:
        reader = PdfReader(str(file_path))
        page_texts: List[str] = []
        for page in reader.pages:
            t = page.extract_text() or ""
            page_texts.append(t)
        full_text = "\n".join(page_texts).strip()
        return full_text, len(reader.pages)
    except Exception as exc:
        print(f"[Warning] pypdf extraction failed: {exc}")
        return "", 0


def get_pdf_page_count(file_path: str | Path) -> int:
    """Return the number of pages in a PDF file."""
    path = Path(file_path)
    if path.suffix.lower() != ".pdf" or PdfReader is None:
        return 1
    try:
        reader = PdfReader(str(path))
        return len(reader.pages)
    except Exception:
        return 1


def split_pdf_to_pages(file_path: str | Path, output_dir: Optional[Path] = None) -> List[Tuple[Path, int]]:
    """
    Split a multi-page PDF into single-page PDF files.
    Returns a list of (page_file_path, page_number_1_indexed).
    If the file is not a PDF, or has only 1 page, returns [(Path(file_path), 1)].
    """
    path = Path(file_path)
    if path.suffix.lower() != ".pdf" or PdfReader is None:
        return [(path, 1)]

    try:
        reader = PdfReader(str(path))
        num_pages = len(reader.pages)
        if num_pages <= 1:
            return [(path, 1)]

        if PdfWriter is None:
            return [(path, 1)]

        out_dir = Path(output_dir) if output_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        base_stem = path.stem
        page_results: List[Tuple[Path, int]] = []

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            writer = PdfWriter()
            writer.add_page(page)
            page_file = out_dir / f"{base_stem}_page_{page_num}.pdf"
            with open(page_file, "wb") as f_out:
                writer.write(f_out)
            page_results.append((page_file, page_num))

        return page_results
    except Exception as exc:
        print(f"[Warning] PDF splitting failed: {exc}, using whole file")
        return [(path, 1)]


def render_pages(file_path: str | Path, dpi: int = 300) -> List[Image.Image]:
    """
    Return a list of PIL Images — one per page.
    Supports PDF (via pdf2image with fallback to pypdf embedded image extraction)
    and common image formats.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".pdf":
        try:
            from pdf2image import convert_from_path
            return convert_from_path(str(path), dpi=dpi)
        except Exception as exc:
            # Fallback: attempt to extract raw embedded images from the PDF using pypdf
            if PdfReader is not None:
                try:
                    reader = PdfReader(str(path))
                    extracted_imgs: List[Image.Image] = []
                    for page in reader.pages:
                        for img in page.images:
                            extracted_imgs.append(Image.open(io.BytesIO(img.data)).convert("RGB"))
                    if extracted_imgs:
                        return extracted_imgs
                except Exception:
                    pass
            raise RuntimeError(
                f"PDF to image conversion failed. Ensure Poppler is installed on your system: {exc}"
            )

    return [Image.open(path).convert("RGB")]


def extract_text(file_path: str | Path, dpi: int = 300) -> Tuple[str, int]:
    """
    Extract text from a PDF or image file:
    1. For PDFs: Attempts direct digital extraction first (fast, 100% accurate, zero external binaries needed).
    2. If not a PDF or if PDF has no digital text (scanned copy): Falls back to Tesseract OCR.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    # Step 0: Direct text / JSON extraction (pre-extracted OCR text)
    if ext == ".txt":
        raw_text = path.read_text(encoding="utf-8", errors="replace").strip()
        return raw_text, 1

    if ext == ".json":
        raw_text = path.read_text(encoding="utf-8", errors="replace").strip()
        try:
            import json
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict) and parsed.get("ocr_text"):
                return parsed["ocr_text"], parsed.get("page_count", 1)
        except Exception:
            pass
        return raw_text, 1

    # Step 1: Direct PDF text extraction
    if ext == ".pdf":
        digital_text, page_count = _extract_pdf_digital_text(path)
        if digital_text and len(digital_text.strip()) > 30:
            return digital_text, page_count

    # Step 2: OCR for images or scanned PDFs
    try:
        pages = render_pages(path, dpi=dpi)
    except Exception as exc:
        raise ValueError(
            f"Could not read invoice file: {exc}. "
            "For PDFs without embedded text, Poppler and Tesseract are required."
        )

    _configure_tesseract()
    page_texts: List[str] = []
    try:
        for page in pages:
            preprocessed = preprocess_image(page)
            page_text = pytesseract.image_to_string(preprocessed, config="--psm 6")
            page_texts.append(page_text)
    except pytesseract.TesseractNotFoundError:
        raise ValueError(
            "Tesseract OCR is not installed or not in your system PATH.\n"
            "Options to fix:\n"
            "1. Upload a digital/text-searchable PDF invoice (which works directly without OCR).\n"
            "2. Install Tesseract on Windows: Download installer from "
            "https://github.com/UB-Mannheim/tesseract/wiki and install to default location.\n"
            "3. Or run using Docker where Tesseract is pre-configured: 'docker compose up'."
        )
    except Exception as exc:
        raise ValueError(f"OCR processing failed: {exc}")

    full_ocr = "\n".join(page_texts).strip()
    return full_ocr, len(pages)
