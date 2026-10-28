"""
app/services/gemini_extractor.py
─────────────────────────────────────────────────────────────────────────────
AI-Powered Multimodal Invoice Extractor using Google Gemini Flash.
Extracts structured B2B invoice fields (GSTIN, vendor, buyer, line items,
financial breakdown) with zero regex dependencies and high visual accuracy.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from app.services.parse_invoice import structured_to_flat_fields

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()

CANDIDATE_MODELS = [
    DEFAULT_MODEL,
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
]

_deprecated_models: set = set()


def is_gemini_available() -> bool:
    """Return True if a Gemini API key is configured in the environment."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return bool(key)


def _build_extraction_prompt() -> str:
    return """You are a senior auditor and expert in Indian B2B GST tax compliance.
Examine this invoice document thoroughly and extract the exact structured data into JSON.

Adhere strictly to these extraction instructions:
1. GSTIN:
   - Extract the 15-character Goods and Services Tax Identification Number for both Vendor and Buyer.
   - If there are OCR character confusions (e.g., 'O' instead of '0', 'I' instead of '1', 'Z' instead of '2'), restore the correct character based on the standard GSTIN schema: 2 digits (state) + 5 letters (PAN) + 4 digits + 1 letter + 1 char + 'Z' + 1 check char.
2. Invoice Number & Dates:
   - Extract the invoice number accurately. Do not confuse 'TAX INVOICE' with the invoice number.
   - Convert all dates to standard ISO format (YYYY-MM-DD).
3. Financial Breakdown:
   - Extract subtotal (taxable value), CGST amount, SGST amount, IGST amount, total tax, and total amount (grand total).
   - Ensure all numeric values are numbers (not strings, strip currency symbols and commas).
4. Line Items:
   - Extract each item with description, HSN/SAC code, quantity, unit price, tax rate (%), and line total.
5. Generate a faithful, complete text transcription of all text on the document in "ocr_text".

Output JSON format must match this structure exactly:
{
  "invoice_metadata": {
    "invoice_number": "string",
    "invoice_date": "YYYY-MM-DD",
    "due_date": "YYYY-MM-DD or null",
    "po_number": "string or null",
    "place_of_supply": "string or null",
    "currency": "INR"
  },
  "vendor": {
    "name": "string",
    "gstin": "15-char string",
    "address": "string or null",
    "state": "string or null",
    "state_code": "string or null",
    "phone": "string or null",
    "email": "string or null"
  },
  "buyer": {
    "name": "string or null",
    "gstin": "15-char string or null",
    "address": "string or null",
    "state": "string or null",
    "state_code": "string or null"
  },
  "line_items": [
    {
      "item_order": 1,
      "description": "string",
      "hsn_sac_code": "string or null",
      "quantity": 1.0,
      "unit": "PCS",
      "unit_price": 0.0,
      "tax_rate": 18.0,
      "line_total": 0.0
    }
  ],
  "financial_summary": {
    "subtotal": 0.0,
    "discount_amount": 0.0,
    "cgst_amount": 0.0,
    "sgst_amount": 0.0,
    "igst_amount": 0.0,
    "tax_amount": 0.0,
    "total_amount": 0.0,
    "item_count": 1,
    "total_quantity": 1.0
  },
  "ocr_text": "complete readable plain text of the invoice"
}
"""


def extract_invoice_with_gemini(
    file_path: str | Path,
    original_filename: str = "",
    page_count: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    Extract structured invoice data using Google Gemini Vision.
    Returns a dictionary with 'structured_json', 'flat_fields', 'ocr_text', and 'json_str'.
    Returns None if extraction fails or API is unavailable.
    """
    path = Path(file_path)
    if not path.exists():
        return None

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    ext = path.suffix.lower()

    # Determine MIME type
    if ext == ".pdf":
        mime_type = "application/pdf"
    elif ext in (".jpg", ".jpeg"):
        mime_type = "image/jpeg"
    elif ext == ".png":
        mime_type = "image/png"
    elif ext == ".webp":
        mime_type = "image/webp"
    elif ext == ".txt":
        mime_type = "text/plain"
    else:
        mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"

    file_bytes = path.read_bytes()
    b64_data = base64.b64encode(file_bytes).decode("utf-8")

    prompt = _build_extraction_prompt()

    # Prepare parts for Gemini generateContent
    if mime_type == "text/plain":
        text_content = file_bytes.decode("utf-8", errors="replace")
        parts = [
            {"text": prompt},
            {"text": f"--- INVOICE TEXT CONTENT ---\n{text_content}"},
        ]
    else:
        parts = [
            {"text": prompt},
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": b64_data,
                }
            },
        ]

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.1,
        },
    }

    # Deduplicate models preserving order, skipping any previously known deprecated/failed models
    seen = set(_deprecated_models)
    models_to_try = []
    for m in CANDIDATE_MODELS:
        if m and m not in seen:
            seen.add(m)
            models_to_try.append(m)

    extracted_dict: Optional[Dict[str, Any]] = None

    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            resp = requests.post(url, json=payload, timeout=25)
            if resp.status_code == 200:
                result_json = resp.json()
                raw_text = result_json["candidates"][0]["content"]["parts"][0]["text"]
                extracted_dict = json.loads(raw_text)
                print(f"[Gemini] Successfully extracted invoice using model: {model_name}")
                break
            elif resp.status_code in (404, 400):
                # Model is deprecated, not found, or unsupported; remember to not retry
                _deprecated_models.add(model_name)
                print(f"[Gemini] Model '{model_name}' unavailable ({resp.status_code}): {resp.text[:120]}")
            else:
                print(f"[Gemini] Model '{model_name}' returned status {resp.status_code}: {resp.text[:120]}")
        except Exception as exc:
            print(f"[Gemini] Error calling model '{model_name}': {exc}")
            continue

    if not extracted_dict or not isinstance(extracted_dict, dict):
        return None

    # Harmonize with FinDoc AI internal structure
    structured_json = extracted_dict
    raw_ocr_text = extracted_dict.get("ocr_text") or ""

    # Ensure metadata contains file info
    structured_json.setdefault("invoice_metadata", {})
    structured_json.setdefault("vendor", {})
    structured_json.setdefault("buyer", {})
    structured_json.setdefault("line_items", [])
    structured_json.setdefault("financial_summary", {})

    # Compute item_count and total_quantity if not present
    items = structured_json.get("line_items", [])
    summary = structured_json.get("financial_summary", {})
    if "item_count" not in summary:
        summary["item_count"] = len(items) if items else 1
    if "total_quantity" not in summary:
        summary["total_quantity"] = sum(float(i.get("quantity", 1.0)) for i in items) if items else 1.0

    # If raw_ocr_text was not generated by Gemini, create a readable plain-text summary
    if not raw_ocr_text.strip():
        meta = structured_json["invoice_metadata"]
        vend = structured_json["vendor"]
        raw_ocr_text = (
            f"TAX INVOICE\n"
            f"Vendor: {vend.get('name', '')}\n"
            f"GSTIN: {vend.get('gstin', '')}\n"
            f"Address: {vend.get('address', '')}\n"
            f"Invoice No: {meta.get('invoice_number', '')}\n"
            f"Invoice Date: {meta.get('invoice_date', '')}\n"
            f"Subtotal: {summary.get('subtotal', 0.0)}\n"
            f"CGST: {summary.get('cgst_amount', 0.0)}\n"
            f"SGST: {summary.get('sgst_amount', 0.0)}\n"
            f"Total: {summary.get('total_amount', 0.0)}\n"
        )

    # Flatten fields for model inference & DB insertion
    fname = original_filename or path.name
    flat = structured_to_flat_fields(
        structured_json,
        file_name=fname,
        file_path=str(path),
    )
    flat["ocr_text"] = raw_ocr_text
    flat["page_count"] = page_count
    flat["extraction_engine"] = "gemini"

    return {
        "structured_json": structured_json,
        "flat_fields": flat,
        "ocr_text": raw_ocr_text,
        "json_str": json.dumps(structured_json, indent=2, ensure_ascii=False),
        "extraction_engine": "gemini",
    }
