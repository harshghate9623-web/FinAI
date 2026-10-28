"""
app/services/extract_fields.py
Regex-based field extraction from raw OCR text.
Refactored from untitled2.py (Colab) to work as a standalone module.
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_text(
    patterns: str | List[str],
    text: str,
    default: Optional[str] = None,
) -> Optional[str]:
    """Return the first capture group from the first matching pattern."""
    if isinstance(patterns, str):
        patterns = [patterns]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

    return default


def _find_amount(
    patterns: str | List[str],
    text: str,
) -> Optional[float]:
    """Extract a numeric amount, stripping currency symbols and commas."""
    raw = _find_text(patterns, text)
    if raw is None:
        return None

    raw = raw.replace(",", "")
    raw = re.sub(r"₹|Rs\.?", "", raw, flags=re.IGNORECASE).strip()

    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_fields(ocr_text: str) -> Dict[str, Any]:
    """
    Parse structured invoice fields from raw OCR text using the robust parser.

    Returns a dict whose keys match the `invoices` DB table and the
    Isolation Forest model feature list.
    """
    from app.services.parse_invoice import parse_invoice_to_json
    return parse_invoice_to_json(ocr_text)["flat_fields"]


def _legacy_extract_fields(ocr_text: str) -> Dict[str, Any]:
    from app.services.parse_invoice import normalize_gstin
    raw_gstin = _find_text(
        [
            r"(?:GSTIN(?:\s*/\s*UIN)?|GST\s*NO\.?|GST\s*NUMBER|GSTIN\s*NO\.?|G\.?S\.?T\.?I\.?N\.?|GST)\s*[:.\-]?\s*([0-9A-Za-z\s\-_]{14,24})",
            r"\b([0-9OIZBS]{2}[\s\-_]*[A-Z0-9]{5}[\s\-_]*[0-9OIZBS]{4}[\s\-_]*[A-Z0-9]{1}[\s\-_]*[0-9A-Z]{1}[\s\-_]*[Z27S][\s\-_]*[0-9A-Z]{1})\b",
        ],
        ocr_text,
        default="",
    )
    cleaned_gstin = normalize_gstin(raw_gstin) or ""

    fields: Dict[str, Any] = {
        "gstin": cleaned_gstin,

        "invoice_number": _find_text(
            [
                r"(?:Invoice\s*(?:No\.?|Num|Number|Na|N0|#|ID))\s*[:.\-]?\s*([A-Za-z0-9/\-_\.]+)",
                r"(?:Bill\s*(?:No\.?|Num|Number|Na|N0|#|ID))\s*[:.\-]?\s*([A-Za-z0-9/\-_\.]+)",
                r"(?:Tax\s*Invoice\s*(?:No\.?|Num|Number|#))\s*[:.\-]?\s*([A-Za-z0-9/\-_\.]+)",
                r"(?:Inv\s*(?:No\.?|Num|#)?)\s*[:.\-]?\s*([A-Za-z0-9/\-_\.]+)",
                r"\b(INV[\.\-/_]?[0-9A-Za-z\-_\.]+)",
            ],
            ocr_text,
            default="",
        ),

        "invoice_date": _find_text(
            [
                r"(?:Invoice\s*Date|Bill\s*Date|Date)\s*:?\s*"
                r"([0-9]{1,2}[/\-][0-9]{1,2}[/\-][0-9]{2,4})",
                r"(?:Invoice\s*Date|Bill\s*Date|Date)\s*:?\s*"
                r"([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})",
            ],
            ocr_text,
        ),

        "vendor_name": _find_text(
            [
                r"(?:Vendor\s*Name|Supplier\s*Name|Bill\s*From|From)\s*:?\s*(.+)",
                r"(?:M/s\.?|M/S\.?)\s*(.+)",
            ],
            ocr_text,
        ),

        "vendor_address": _find_text(
            r"(?:Vendor\s*Address|Address)\s*:?\s*(.+)",
            ocr_text,
            default="",
        ),

        "subtotal": _find_amount(
            [
                r"(?:Sub\s*Total|Subtotal|Taxable\s*Value|Total\s*Taxable\s*Value)"
                r"\s*:?\s*(?:₹|Rs\.?\s*)?([\d,]+\.?\d*)"
            ],
            ocr_text,
        ),

        "cgst_amount": _find_amount(
            r"\bCGST(?:\s*@\s*[\d.]+\s*%)?\s*:?\s*(?:₹|Rs\.?\s*)?([\d,]+\.?\d*)",
            ocr_text,
        ),

        "sgst_amount": _find_amount(
            r"\bSGST(?:\s*@\s*[\d.]+\s*%)?\s*:?\s*(?:₹|Rs\.?\s*)?([\d,]+\.?\d*)",
            ocr_text,
        ),

        "igst_amount": _find_amount(
            r"\bIGST(?:\s*@\s*[\d.]+\s*%)?\s*:?\s*(?:₹|Rs\.?\s*)?([\d,]+\.?\d*)",
            ocr_text,
        ),

        "tax_amount": _find_amount(
            [
                r"(?:Total\s*Tax|Tax\s*Amount|Total\s*GST)\s*:?\s*"
                r"(?:₹|Rs\.?\s*)?([\d,]+\.?\d*)"
            ],
            ocr_text,
        ),

        "discount_amount": _find_amount(
            r"\bDiscount(?:\s*\([^)]*\))?\s*:?\s*(?:₹|Rs\.?\s*)?([\d,]+\.?\d*)",
            ocr_text,
        ),

        "total_amount": _find_amount(
            [
                r"(?:Grand\s*Total|Invoice\s*Total|Total\s*Amount|"
                r"Net\s*Payable|Amount\s*Due)\s*:?\s*(?:₹|Rs\.?\s*)?([\d,]+\.?\d*)"
            ],
            ocr_text,
        ),

        # Line-item counts require structured parsing; set to None for now.
        "item_count": None,
        "quantity": None,
    }

    # Derive total tax from components when the header value is missing.
    if fields["tax_amount"] is None:
        components = [fields["cgst_amount"], fields["sgst_amount"], fields["igst_amount"]]
        if all(v is not None for v in components):
            fields["tax_amount"] = sum(components)

    return fields


def get_missing_fields(fields: Dict[str, Any]) -> List[str]:
    """Return field names that are None or empty string."""
    required = [
        "gstin", "invoice_number", "invoice_date",
        "subtotal", "cgst_amount", "sgst_amount",
        "igst_amount", "tax_amount", "discount_amount",
        "total_amount", "item_count", "quantity",
    ]
    return [
        f for f in required
        if fields.get(f) is None or fields.get(f) == ""
    ]
