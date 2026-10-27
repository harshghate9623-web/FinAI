"""
app/services/ingest.py
Orchestrates the full invoice processing pipeline:
  upload → validate file → OCR → field extraction → rule validation → anomaly scoring → duplicate check
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from app.services.ocr import extract_text, SUPPORTED_EXTENSIONS
from app.services.parse_invoice import (
    parse_invoice_to_json,
    save_structured_json,
    load_structured_json,
    structured_to_flat_fields,
    flat_to_structured_json,
)
from app.services.extract_fields import get_missing_fields
from app.services.validate_invoice import validate_single
from app.services.anomaly import get_detector
from app.services.duplicates import find_near_duplicates, find_exact_duplicates, DuplicateMatch

# Upload directory (override via env var)
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "30"))


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------

def validate_upload(filename: str, size_bytes: int) -> None:
    """Raise ValueError for unsupported file type or file too large."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Accepted: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        raise ValueError(
            f"File size {size_bytes / 1024 / 1024:.1f} MB exceeds "
            f"limit of {MAX_FILE_SIZE_MB} MB."
        )


def save_upload(file_bytes: bytes, original_filename: str) -> Path:
    """Save uploaded file to the upload directory with a unique name."""
    ext = Path(original_filename).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / unique_name
    dest.write_bytes(file_bytes)
    return dest


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def process_invoice(
    file_path: Path | str,
    original_filename: str,
    existing_invoices: pd.DataFrame | None = None,
    output_json_path: Path | str | None = None,
) -> Dict[str, Any]:
    """
    Run the complete ingestion pipeline for a single invoice file:
    1. Extract plain text (via OCR for PDF/image, or direct read for .txt/.json).
    2. Parse the extracted plain text into an organized, well-structured JSON file.
    3. Save the organized JSON to disk.
    4. Feed the organized JSON data into the validation, ML anomaly detection,
       and duplicate detection pipeline.

    Args:
        file_path:          Path to the saved upload file (.pdf, .png, .jpg, .txt, .json).
        original_filename:  Original name of the uploaded file.
        existing_invoices:  DataFrame of previously processed invoices (used for duplicate detection).
        output_json_path:   Optional custom path to save the organized JSON file.

    Returns:
        Dict containing structured_json, json_file_path, line_items, extracted,
        validated, duplicates, missing_fields, ocr_text, etc.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    # Case A: Input is an existing JSON file
    if ext == ".json":
        raw_content = path.read_text(encoding="utf-8", errors="replace").strip()
        data = json.loads(raw_content)
        if "invoice_metadata" in data or "structured_json" in data:
            structured_json = data.get("structured_json", data)
            fields = structured_to_flat_fields(
                structured_json,
                file_name=original_filename,
                file_path=str(path),
            )
        else:
            fields = data
            structured_json = flat_to_structured_json(
                fields,
                file_name=original_filename,
                file_path=str(path),
            )

        ocr_text = fields.get("ocr_text", raw_content)
        page_count = fields.get("page_count", 1)
        out_json_path = Path(output_json_path) if output_json_path else path
        save_structured_json(structured_json, out_json_path)
        json_str = json.dumps(structured_json, indent=2, ensure_ascii=False)

    # Case B: Input is PDF, image, or .txt (OCR plain text)
    else:
        # 1. OCR / text extraction
        ocr_text, page_count = extract_text(path)

        if not ocr_text.strip():
            raise ValueError("OCR returned no readable text. Try a clearer image or PDF.")

        if page_count > MAX_PAGES:
            raise ValueError(
                f"Document has {page_count} pages, which exceeds the limit of {MAX_PAGES}."
            )

        # 2. Parse extracted plain text into organized JSON
        parse_res = parse_invoice_to_json(
            ocr_text,
            file_name=original_filename,
            file_path=str(path),
            page_count=page_count,
        )
        structured_json = parse_res["structured_json"]
        fields = parse_res["flat_fields"]
        json_str = parse_res["json_str"]

        # 3. Save organized JSON file to disk
        if output_json_path:
            out_json_path = Path(output_json_path)
        else:
            out_json_path = path.parent / f"{path.stem}_extracted.json"

        save_structured_json(structured_json, out_json_path)

    fields["file_name"] = original_filename
    fields["file_path"] = str(path)
    fields["page_count"] = page_count
    fields["ocr_text"] = ocr_text
    fields["json_file_path"] = str(out_json_path)
    fields["structured_json"] = structured_json

    # 4. Check missing fields
    missing_fields = get_missing_fields(fields)

    # 5. Rule-based validation
    validated = validate_single(fields)

    # 6. Anomaly detection (Isolation Forest)
    try:
        detector = get_detector()
        is_anomaly, score, input_invalid = detector.predict(validated)
        validated["isolation_anomaly"] = is_anomaly
        validated["isolation_score"] = score
        validated["model_input_invalid"] = input_invalid
    except FileNotFoundError:
        validated["isolation_anomaly"] = False
        validated["isolation_score"] = None
        validated["model_input_invalid"] = True

    # 7. Combined final anomaly flag
    validated["final_anomaly"] = (
        validated.get("rule_anomaly", False)
        or validated.get("isolation_anomaly", False)
        or validated.get("model_input_invalid", False)
    )

    # 8. Duplicate detection
    duplicates: List[DuplicateMatch] = []
    if existing_invoices is not None and not existing_invoices.empty:
        duplicates = find_near_duplicates(validated, existing_invoices)

    # 9. Review status
    if validated.get("model_input_invalid"):
        validated["review_status"] = "Manual Verification Required"
    elif validated.get("rule_anomaly"):
        validated["review_status"] = "Pending Manual Review"
    elif validated.get("isolation_anomaly"):
        validated["review_status"] = "Flagged for Review"
    elif duplicates:
        validated["review_status"] = "Possible Duplicate — Review"
    else:
        validated["review_status"] = "Passed Initial Checks"

    return sanitize_for_json({
        "extracted": fields,
        "validated": validated,
        "structured_json": structured_json,
        "json_file_path": str(out_json_path),
        "json_str": json_str,
        "line_items": structured_json.get("line_items", []),
        "duplicates": [d.__dict__ for d in duplicates],
        "missing_fields": missing_fields,
        "ocr_text": ocr_text,
        "page_count": page_count,
        "file_name": original_filename,
        "file_path": str(path),
    })


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars, dates, and paths to native JSON types."""
    import numpy as np
    import datetime

    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(x) for x in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    elif isinstance(obj, Path):
        return str(obj)
    return obj


def process_ocr_text(
    ocr_text: str,
    original_filename: str = "ocr_extracted.txt",
    existing_invoices: pd.DataFrame | None = None,
    output_json_path: Path | str | None = None,
) -> Dict[str, Any]:
    """
    Directly parse raw OCR text into an organized JSON file, save it,
    and feed it through the FinDoc AI validation/anomaly pipeline.
    """
    if not ocr_text or not ocr_text.strip():
        raise ValueError("OCR text is empty.")

    # Save OCR text to temporary/upload store for provenance
    txt_filename = f"{uuid.uuid4().hex}_{original_filename}"
    txt_path = UPLOAD_DIR / txt_filename
    txt_path.write_text(ocr_text, encoding="utf-8")

    out_json = output_json_path or UPLOAD_DIR / f"{txt_path.stem}_extracted.json"
    return process_invoice(
        file_path=txt_path,
        original_filename=original_filename,
        existing_invoices=existing_invoices,
        output_json_path=out_json,
    )


def process_structured_json(
    json_data_or_path: Union[str, Path, Dict[str, Any]],
    original_filename: str = "invoice.json",
    existing_invoices: pd.DataFrame | None = None,
    output_json_path: Path | str | None = None,
) -> Dict[str, Any]:
    """
    Accept an organized JSON document (as a dictionary, file path, or JSON string)
    and feed it through the FinDoc AI validation/anomaly pipeline.
    """
    if isinstance(json_data_or_path, (str, Path)) and Path(str(json_data_or_path)).is_file():
        return process_invoice(
            file_path=Path(str(json_data_or_path)),
            original_filename=original_filename,
            existing_invoices=existing_invoices,
            output_json_path=output_json_path,
        )

    # If it is a dictionary or raw json string
    if isinstance(json_data_or_path, str):
        data = json.loads(json_data_or_path)
    else:
        data = json_data_or_path

    # Save to file
    json_filename = f"{uuid.uuid4().hex}_{original_filename}"
    json_path = UPLOAD_DIR / json_filename
    save_structured_json(data, json_path)

    return process_invoice(
        file_path=json_path,
        original_filename=original_filename,
        existing_invoices=existing_invoices,
        output_json_path=output_json_path or json_path,
    )
