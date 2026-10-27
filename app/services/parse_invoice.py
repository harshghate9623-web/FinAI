"""
app/services/parse_invoice.py
─────────────────────────────────────────────────────────────────────────────
Robust structured invoice parser.
Converts unstructured OCR text into a well-organized, validated JSON structure
and feeds both hierarchical JSON and normalized flat fields into the FinDoc AI
pipeline.

Output structure:
{
    "invoice_metadata": {
        "invoice_number": str,
        "invoice_date": str (YYYY-MM-DD),
        "due_date": Optional[str],
        "po_number": Optional[str],
        "place_of_supply": Optional[str],
        "currency": str
    },
    "vendor": {
        "name": Optional[str],
        "gstin": Optional[str],
        "address": Optional[str],
        "state": Optional[str],
        "state_code": Optional[str],
        "phone": Optional[str],
        "email": Optional[str]
    },
    "buyer": {
        "name": Optional[str],
        "gstin": Optional[str],
        "address": Optional[str],
        "state": Optional[str],
        "state_code": Optional[str]
    },
    "line_items": [
        {
            "item_order": int,
            "description": str,
            "hsn_sac_code": Optional[str],
            "quantity": float,
            "unit": Optional[str],
            "unit_price": float,
            "tax_rate": Optional[float],
            "line_total": float
        }
    ],
    "financial_summary": {
        "subtotal": float,
        "discount_amount": float,
        "cgst_amount": float,
        "sgst_amount": float,
        "igst_amount": float,
        "tax_amount": float,
        "total_amount": float,
        "item_count": int,
        "total_quantity": float
    },
    "ocr_metadata": {
        "raw_text_length": int,
        "line_count": int
    }
}
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from app.services.validate_invoice import STATE_ALIASES


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def clean_amount(raw: Any) -> Optional[float]:
    """Strip currency symbols, commas, and whitespace, returning float."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().replace(",", "")
    s = re.sub(r"[₹$€£\s]|Rs\.?|INR", "", s, flags=re.IGNORECASE).strip()
    # match leading number including decimals
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def parse_date_str(val: Any) -> Optional[str]:
    """Normalize date strings into ISO format YYYY-MM-DD."""
    if not val:
        return None
    if isinstance(val, (datetime.date, datetime.datetime)):
        return val.strftime("%Y-%m-%d")

    val_str = str(val).strip()
    # Common separators: / - . or whitespace
    patterns = [
        ("%Y-%m-%d", r"^\d{4}-\d{1,2}-\d{1,2}"),
        ("%d/%m/%Y", r"^\d{1,2}/\d{1,2}/\d{4}"),
        ("%d-%m-%Y", r"^\d{1,2}-\d{1,2}-\d{4}"),
        ("%d.%m.%Y", r"^\d{1,2}\.\d{1,2}\.\d{4}"),
        ("%m/%d/%Y", r"^\d{1,2}/\d{1,2}/\d{4}"),
        ("%d %b %Y", r"^\d{1,2}\s+[A-Za-z]{3}\s+\d{4}"),
        ("%d %B %Y", r"^\d{1,2}\s+[A-Za-z]{4,9}\s+\d{4}"),
    ]
    for fmt, regex in patterns:
        if re.search(regex, val_str, re.IGNORECASE):
            try:
                dt = datetime.datetime.strptime(val_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

    # Fallback search inside string
    m = re.search(r"(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})", val_str)
    if m:
        sub = m.group(1).replace(".", "/")
        for fmt in ("%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%Y/%m/%d"):
            try:
                return datetime.datetime.strptime(sub, fmt).strftime("%Y-%m-%d")
            except ValueError:
                pass
    return val_str


def find_regex(patterns: List[str] | str, text: str, default: Optional[str] = None) -> Optional[str]:
    """Find first matching regex capture group."""
    if isinstance(patterns, str):
        patterns = [patterns]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if m and m.lastindex:
            res = m.group(1).strip()
            if res:
                return res
    return default


def find_amount_regex(patterns: List[str] | str, text: str) -> Optional[float]:
    raw = find_regex(patterns, text)
    return clean_amount(raw)


def extract_state_info(text: str, gstin: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Determine state name and state code from GSTIN or address text."""
    state_code = None
    state_name = None

    if gstin and len(gstin) >= 2 and gstin[:2].isdigit():
        code = gstin[:2]
        # Reverse lookup code to name
        for name, c in STATE_ALIASES.items():
            if c == code:
                state_code = code
                state_name = name.title()
                break

    if not state_name and text:
        upper = text.upper()
        for name, code in STATE_ALIASES.items():
            if re.search(r"\b" + re.escape(name) + r"\b", upper):
                state_name = name.title()
                state_code = code
                break

    return state_name, state_code


# ---------------------------------------------------------------------------
# Section extractors
# ---------------------------------------------------------------------------

def extract_metadata(text: str) -> Dict[str, Any]:
    invoice_number = find_regex(
        [
            r"(?:Invoice\s*(?:No\.?|Number|#))\s*[:\-]?\s*([A-Za-z0-9/\-_]+)",
            r"(?:Bill\s*(?:No\.?|Number|#))\s*[:\-]?\s*([A-Za-z0-9/\-_]+)",
            r"(?:Tax\s*Invoice\s*(?:No\.?|#)?)\s*[:\-]?\s*([A-Za-z0-9/\-_]+)",
            r"\bINV[-/_]?[0-9A-Za-z\-_]+",
        ],
        text,
        default="",
    )

    raw_date = find_regex(
        [
            r"(?:Invoice\s*Date|Bill\s*Date|Date\s*of\s*Invoice|Date)\s*[:\-]?\s*([0-9]{1,2}[/\-\.][0-9]{1,2}[/\-\.][0-9]{2,4})",
            r"(?:Invoice\s*Date|Bill\s*Date|Date)\s*[:\-]?\s*([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})",
            r"Date\s*[:\-]?\s*([0-9]{4}[/\-][0-9]{1,2}[/\-][0-9]{1,2})",
        ],
        text,
    )
    invoice_date = parse_date_str(raw_date)

    due_date_raw = find_regex(
        [
            r"(?:Due\s*Date|Payment\s*Due)\s*[:\-]?\s*([0-9]{1,2}[/\-\.][0-9]{1,2}[/\-\.][0-9]{2,4})",
            r"(?:Due\s*Date|Payment\s*Due)\s*[:\-]?\s*([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})",
        ],
        text,
    )
    due_date = parse_date_str(due_date_raw)

    po_number = find_regex(
        [
            r"(?:PO\s*(?:No\.?|Number|#)|Purchase\s*Order\s*(?:No\.?|#)?)\s*[:\-]?\s*([A-Za-z0-9/\-_]+)",
        ],
        text,
    )

    place_of_supply = find_regex(
        [
            r"(?:Place\s*of\s*Supply|POS|State\s*of\s*Supply)\s*[:\-]?\s*([A-Za-z0-9\s\-]+)",
        ],
        text,
    )
    if place_of_supply:
        place_of_supply = place_of_supply.splitlines()[0].strip()

    currency = "INR"
    if "$" in text or "USD" in text:
        currency = "USD"
    elif "€" in text or "EUR" in text:
        currency = "EUR"

    return {
        "invoice_number": invoice_number or "",
        "invoice_date": invoice_date or "",
        "due_date": due_date,
        "po_number": po_number,
        "place_of_supply": place_of_supply,
        "currency": currency,
    }


def extract_vendor_and_buyer(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract vendor (seller) and buyer (client / bill to) blocks."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Find GSTINs in document (up to 2: vendor and buyer)
    all_gstins = re.findall(r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})\b", text)

    vendor_gstin = all_gstins[0] if len(all_gstins) >= 1 else None
    buyer_gstin = all_gstins[1] if len(all_gstins) >= 2 else None

    # Check for labeled GSTINs
    v_gst_match = find_regex(
        [
            r"(?:Supplier|Vendor|Seller|From)[\s\S]{1,60}?GSTIN(?:\s*/\s*UIN)?\s*[:\-]?\s*([0-9A-Z]{15})",
            r"(?:GSTIN|UIN)\s*[:\-]?\s*([0-9A-Z]{15})",
        ],
        text,
    )
    if v_gst_match:
        vendor_gstin = v_gst_match

    b_gst_match = find_regex(
        [
            r"(?:Buyer|Bill\s*To|Customer|Client)[\s\S]{1,60}?GSTIN(?:\s*/\s*UIN)?\s*[:\-]?\s*([0-9A-Z]{15})",
            r"Buyer(?:\'s)?\s*GSTIN\s*[:\-]?\s*([0-9A-Z]{15})",
        ],
        text,
    )
    if b_gst_match:
        buyer_gstin = b_gst_match

    # Vendor Name
    vendor_name = find_regex(
        [
            r"(?:Vendor\s*Name|Supplier\s*Name|Seller|From|Sold\s*By)\s*[:\-]?\s*([A-Za-z0-9\s\.\,&'-]+)",
            r"(?:M/s\.?|M/S\.?)\s*([A-Za-z0-9\s\.\,&'-]+)",
        ],
        text,
    )

    # If vendor name not found by explicit label, take the first non-generic top line
    if not vendor_name and lines:
        for line in lines[:5]:
            if not re.search(r"tax|invoice|bill|original|duplicate|gstin|date|phone", line, re.IGNORECASE):
                if len(line) > 3 and not re.match(r"^[-=_*#]+$", line):
                    vendor_name = line
                    break

    # Buyer Name
    buyer_name = find_regex(
        [
            r"(?:Bill\s*To|Billed\s*To|Buyer|Customer\s*Name|Client\s*Name|Consignee)\s*[:\-]?\s*([A-Za-z0-9\s\.\,&'-]+)",
        ],
        text,
    )

    # Vendor Address
    vendor_address = find_regex(
        [
            r"(?:Vendor\s*Address|Supplier\s*Address|Address)\s*[:\-]?\s*([A-Za-z0-9\s\.,\-\/]+)",
        ],
        text,
    )

    # If vendor address not found by explicit label, search lines under vendor name
    if not vendor_address and lines:
        for line in lines[:8]:
            if line == vendor_name:
                continue
            lower_l = line.lower()
            if any(k in lower_l for k in ("road", "street", "marg", "lane", "nagar", "park", "floor", "plot", "bldg", "building", "cross", "east", "west", "north", "south", "mumbai", "delhi", "bengaluru", "chennai", "kolkata", "hyderabad", "pune", "sector", "estate", "industrial")) or re.search(r"\b\d{6}\b", line):
                if not any(k in lower_l for k in ("invoice", "tax", "bill", "gstin", "phone", "email", "date", "due date")):
                    vendor_address = line
                    break

    # Buyer Address
    buyer_address = find_regex(
        [
            r"(?:Bill\s*To|Buyer\s*Address|Shipping\s*Address)[\s\S]{1,40}?Address\s*[:\-]?\s*([A-Za-z0-9\s\.,\-\/]+)",
        ],
        text,
    )

    # Heuristic fallback for buyer address
    if not buyer_address and lines:
        for idx, line in enumerate(lines):
            if any(k in line.lower() for k in ("billed to", "bill to", "buyer")):
                for next_l in lines[idx + 1:idx + 4]:
                    lower_nl = next_l.lower()
                    if any(k in lower_nl for k in ("road", "street", "marg", "lane", "nagar", "park", "floor", "plot", "mumbai", "pune", "delhi", "sector", "estate")) or re.search(r"\b\d{6}\b", next_l):
                        buyer_address = next_l
                        break
                break

    # Contact Info
    phone = find_regex(r"(?:Phone|Mob(?:ile)?|Tel)\s*[:\-]?\s*([+0-9\s\-]{10,15})", text)
    email = find_regex(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b", text)

    # Clean multiline captures
    if vendor_name:
        vendor_name = vendor_name.splitlines()[0].strip()
    if buyer_name:
        buyer_name = buyer_name.splitlines()[0].strip()
    if vendor_address:
        vendor_address = vendor_address.splitlines()[0].strip()
    if buyer_address:
        buyer_address = buyer_address.splitlines()[0].strip()

    v_state, v_state_code = extract_state_info(vendor_address or text[:500], vendor_gstin)
    b_state, b_state_code = extract_state_info(buyer_address or text[500:], buyer_gstin)

    vendor = {
        "name": vendor_name or "",
        "gstin": vendor_gstin or "",
        "address": vendor_address or "",
        "state": v_state or "",
        "state_code": v_state_code or "",
        "phone": phone,
        "email": email,
    }

    buyer = {
        "name": buyer_name or "",
        "gstin": buyer_gstin or "",
        "address": buyer_address or "",
        "state": b_state or "",
        "state_code": b_state_code or "",
    }

    return vendor, buyer


# ---------------------------------------------------------------------------
# Line item table parser
# ---------------------------------------------------------------------------

def extract_line_items(text: str) -> List[Dict[str, Any]]:
    """
    Parses table line items from OCR text using multi-column patterns.
    Recognizes typical invoice row patterns like:
    [order] [description] [HSN/SAC] [Qty] [Unit] [Rate/Unit Price] [Tax %] [Amount]
    """
    items: List[Dict[str, Any]] = []
    lines = text.splitlines()

    item_regex = re.compile(
        r"^(?P<order>\d+[\.\)])?\s*"
        r"(?P<desc>[A-Za-z0-9\s\-\(\)\/\.,#]{3,40}?)\s+"
        r"(?:(?P<hsn>\d{4,8})\s+)?"
        r"(?P<qty>\d+(?:\.\d+)?)\s*"
        r"(?:(?P<unit>PCS|NOS|KG|MTR|BOX|SET|UNITS?|PACKS?)\s+)?"
        r"(?:[@₹Rs\.]*\s*(?P<price>[\d,]+\.?\d*))\s+"
        r"(?:(?P<tax>\d+(?:\.\d+)?%?)\s+)?"
        r"(?:[@₹Rs\.]*\s*(?P<total>[\d,]+\.?\d*))$",
        re.IGNORECASE,
    )

    # Simpler fallback row: Description ... Qty ... Price ... Total
    simple_row_regex = re.compile(
        r"^(?P<desc>[A-Za-z][A-Za-z0-9\s\-\.\/]{2,35})\s+"
        r"(?P<qty>\d+(?:\.\d+)?)\s+"
        r"(?P<price>[\d,]+\.?\d*)\s+"
        r"(?P<total>[\d,]+\.?\d*)$",
        re.IGNORECASE,
    )

    order_idx = 1
    in_table_section = False

    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue

        lower = cleaned.lower()
        # Look for table header
        if any(h in lower for h in ("description", "particulars", "item name", "hsn/sac", "rate", "qty", "quantity")):
            in_table_section = True
            continue

        # Look for summary/footer boundary to stop table parsing
        if any(f in lower for f in ("sub total", "subtotal", "total amount", "taxable value", "grand total", "cgst", "sgst", "igst")):
            if in_table_section and items:
                break

        # Pipe-delimited table rows (common OCR output for invoice tables)
        if "|" in cleaned:
            parts = [p.strip() for p in cleaned.split("|") if p.strip()]
            if len(parts) >= 3:
                first_lower = parts[0].lower()
                # Skip header row
                if any(h in first_lower for h in ("item", "desc", "particular", "sl", "s.no", "#")):
                    in_table_section = True
                    continue
                # Extract numeric values from the trailing columns
                extracted_nums = []
                desc_idx = 1 if parts[0].isdigit() and len(parts) > 1 else 0
                desc = parts[desc_idx]
                hsn_code = None
                trailing_parts = parts[desc_idx + 1:]

                # Check if first trailing part is HSN/SAC code (4 to 8 digit integer)
                if len(trailing_parts) >= 3 and re.fullmatch(r"\d{4,8}", trailing_parts[0]):
                    hsn_code = trailing_parts[0]
                    trailing_parts = trailing_parts[1:]

                for p in trailing_parts:
                    val = clean_amount(p)
                    if val is not None:
                        extracted_nums.append(val)

                if len(extracted_nums) >= 2:
                    if len(extracted_nums) == 2:
                        qty, price = extracted_nums[0], extracted_nums[1]
                        line_tot = round(qty * price, 2)
                    elif len(extracted_nums) >= 3:
                        qty, price, line_tot = extracted_nums[0], extracted_nums[1], extracted_nums[-1]
                    else:
                        qty, price, line_tot = 1.0, extracted_nums[0], extracted_nums[0]

                    if not any(k in desc.lower() for k in ("subtotal", "tax", "total", "cgst", "sgst", "igst", "discount")):
                        items.append({
                            "item_order": order_idx,
                            "description": desc,
                            "hsn_sac_code": hsn_code,
                            "quantity": qty,
                            "unit": "PCS",
                            "unit_price": price,
                            "tax_rate": 18.0,
                            "line_total": line_tot,
                        })
                        order_idx += 1
                        continue

        # Attempt detailed regex
        m = item_regex.match(cleaned)
        if m:
            gd = m.groupdict()
            qty = clean_amount(gd.get("qty")) or 1.0
            price = clean_amount(gd.get("price")) or 0.0
            tot = clean_amount(gd.get("total")) or (qty * price)
            tax_rate = clean_amount(gd.get("tax"))

            items.append({
                "item_order": order_idx,
                "description": gd.get("desc", "").strip(),
                "hsn_sac_code": gd.get("hsn") or None,
                "quantity": qty,
                "unit": gd.get("unit") or "PCS",
                "unit_price": price,
                "tax_rate": tax_rate or 18.0,
                "line_total": tot,
            })
            order_idx += 1
            continue

        # Attempt simpler regex if in table section or looks like an item line
        m_simple = simple_row_regex.match(cleaned)
        if m_simple:
            gd = m_simple.groupdict()
            qty = clean_amount(gd.get("qty")) or 1.0
            price = clean_amount(gd.get("price")) or 0.0
            tot = clean_amount(gd.get("total")) or (qty * price)

            # Avoid false positives like "Total 1 500 500"
            if not any(k in gd.get("desc", "").lower() for k in ("total", "tax", "gst", "subtotal", "discount", "phone")):
                items.append({
                    "item_order": order_idx,
                    "description": gd.get("desc", "").strip(),
                    "hsn_sac_code": None,
                    "quantity": qty,
                    "unit": "PCS",
                    "unit_price": price,
                    "tax_rate": 18.0,
                    "line_total": tot,
                })
                order_idx += 1

    return items


# ---------------------------------------------------------------------------
# Financial summary and cross-derivation
# ---------------------------------------------------------------------------

def extract_financials(text: str, line_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract financial summary and infer missing values using arithmetic."""
    subtotal = find_amount_regex(
        [
            r"(?:Sub\s*Total|Subtotal|Taxable\s*Value|Total\s*Taxable\s*Value)\s*[:\-]?\s*(?:₹|Rs\.?\s*)?([\d,]+\.?\d*)",
        ],
        text,
    )

    cgst = find_amount_regex(
        [
            r"\bCGST(?:\s*@\s*[\d.]+\s*%)?\s*[:\-]?\s*(?:₹|Rs\.?\s*)?([\d,]+\.?\d*)",
            r"(?:Central\s*GST)\s*[:\-]?\s*(?:₹|Rs\.?\s*)?([\d,]+\.?\d*)",
        ],
        text,
    )

    sgst = find_amount_regex(
        [
            r"\bSGST(?:\s*@\s*[\d.]+\s*%)?\s*[:\-]?\s*(?:₹|Rs\.?\s*)?([\d,]+\.?\d*)",
            r"(?:State\s*GST)\s*[:\-]?\s*(?:₹|Rs\.?\s*)?([\d,]+\.?\d*)",
        ],
        text,
    )

    igst = find_amount_regex(
        [
            r"\bIGST(?:\s*@\s*[\d.]+\s*%)?\s*[:\-]?\s*(?:₹|Rs\.?\s*)?([\d,]+\.?\d*)",
            r"(?:Integrated\s*GST)\s*[:\-]?\s*(?:₹|Rs\.?\s*)?([\d,]+\.?\d*)",
        ],
        text,
    )

    tax_amount = find_amount_regex(
        [
            r"(?:Total\s*Tax|Tax\s*Amount|Total\s*GST)\s*[:\-]?\s*(?:₹|Rs\.?\s*)?([\d,]+\.?\d*)",
        ],
        text,
    )

    discount_amount = find_amount_regex(
        [
            r"\bDiscount(?:\s*\([^)]*\))?\s*[:\-]?\s*(?:₹|Rs\.?\s*)?([\d,]+\.?\d*)",
        ],
        text,
    ) or 0.0

    total_amount = find_amount_regex(
        [
            r"(?:Grand\s*Total|Invoice\s*Total|Total\s*Amount|Net\s*Payable|Amount\s*Due)\s*[:\-]?\s*(?:₹|Rs\.?\s*)?([\d,]+\.?\d*)",
            r"\bTotal\s*[:\-]?\s*(?:₹|Rs\.?\s*)?([\d,]+\.?\d*)",
        ],
        text,
    )

    # Line item derived totals
    items_total_sum = sum(item["line_total"] for item in line_items) if line_items else 0.0
    items_total_qty = sum(item["quantity"] for item in line_items) if line_items else 0.0
    items_count = len(line_items)

    # Total quantity fallback
    if items_total_qty == 0.0:
        extracted_qty = find_amount_regex(
            [
                r"(?:Total\s*Qty|Total\s*Quantity|Quantity)\s*[:\-]?\s*([\d,]+\.?\d*)",
            ],
            text,
        )
        items_total_qty = extracted_qty if extracted_qty is not None else 1.0

    if items_count == 0:
        items_count = 1  # Standard single-invoice baseline for model inference

    # Derive Subtotal
    if subtotal is None:
        if items_total_sum > 0.0:
            subtotal = items_total_sum
        elif total_amount is not None and tax_amount is not None:
            subtotal = round(total_amount - tax_amount + discount_amount, 2)

    # Derive Tax Components
    cgst_val = cgst or 0.0
    sgst_val = sgst or 0.0
    igst_val = igst or 0.0

    if tax_amount is None:
        if (cgst is not None) or (sgst is not None) or (igst is not None):
            tax_amount = round(cgst_val + sgst_val + igst_val, 2)
        elif subtotal is not None and total_amount is not None and total_amount >= subtotal:
            tax_amount = round(total_amount - subtotal + discount_amount, 2)
        else:
            tax_amount = 0.0

    # Derive Total Amount
    if total_amount is None:
        if subtotal is not None:
            total_amount = round(subtotal + (tax_amount or 0.0) - discount_amount, 2)
        elif items_total_sum > 0.0:
            total_amount = round(items_total_sum + (tax_amount or 0.0) - discount_amount, 2)

    return {
        "subtotal": subtotal or 0.0,
        "discount_amount": discount_amount or 0.0,
        "cgst_amount": cgst or 0.0,
        "sgst_amount": sgst or 0.0,
        "igst_amount": igst or 0.0,
        "tax_amount": tax_amount or 0.0,
        "total_amount": total_amount or 0.0,
        "item_count": items_count,
        "total_quantity": items_total_qty,
    }


# ---------------------------------------------------------------------------
# Main structured parser
# ---------------------------------------------------------------------------

def save_structured_json(structured_doc: Dict[str, Any], output_path: str | Path) -> Path:
    """Save structured invoice dictionary as a formatted JSON file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(structured_doc, f, indent=2, ensure_ascii=False)
    return out


def load_structured_json(json_path: str | Path) -> Dict[str, Any]:
    """Load and parse structured invoice JSON file."""
    p = Path(json_path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def structured_to_flat_fields(
    structured_doc: Dict[str, Any],
    file_name: str = "",
    file_path: str = "",
    page_count: int = 1,
    ocr_text: str = "",
) -> Dict[str, Any]:
    """Convert hierarchical structured invoice JSON into flat fields for pipeline & DB models."""
    meta = structured_doc.get("invoice_metadata", {})
    vendor = structured_doc.get("vendor", {})
    buyer = structured_doc.get("buyer", {})
    fin = structured_doc.get("financial_summary", {})
    line_items = structured_doc.get("line_items", [])
    prov = structured_doc.get("provenance", {})

    subtotal = float(fin.get("subtotal") or 0.0)
    cgst = float(fin.get("cgst_amount") or 0.0)
    sgst = float(fin.get("sgst_amount") or 0.0)
    igst = float(fin.get("igst_amount") or 0.0)
    tax = float(fin.get("tax_amount") or (cgst + sgst + igst))
    discount = float(fin.get("discount_amount") or 0.0)
    total = float(fin.get("total_amount") or (subtotal - discount + tax))

    item_count = int(fin.get("item_count") or (len(line_items) if line_items else 1))
    qty = float(fin.get("total_quantity") or (sum(float(i.get("quantity") or 1.0) for i in line_items) if line_items else 1.0))

    return {
        "invoice_number": str(meta.get("invoice_number") or ""),
        "invoice_date": str(meta.get("invoice_date") or ""),
        "vendor_name": str(vendor.get("name") or ""),
        "gstin": str(vendor.get("gstin") or ""),
        "vendor_address": str(vendor.get("address") or ""),
        "extracted_vendor_state": str(vendor.get("state") or ""),
        "extracted_vendor_state_code": str(vendor.get("state_code") or ""),
        "buyer_name": str(buyer.get("name") or ""),
        "buyer_gstin": str(buyer.get("gstin") or ""),
        "buyer_address": str(buyer.get("address") or ""),
        "subtotal": subtotal,
        "cgst_amount": cgst,
        "sgst_amount": sgst,
        "igst_amount": igst,
        "tax_amount": tax,
        "discount_amount": discount,
        "total_amount": total,
        "item_count": item_count,
        "quantity": qty,
        "items": line_items,
        "structured_json": structured_doc,
        "file_name": file_name or prov.get("file_name", ""),
        "file_path": file_path or prov.get("file_path", ""),
        "page_count": page_count or prov.get("page_count", 1),
        "ocr_text": ocr_text or prov.get("ocr_text", ""),
    }


def flat_to_structured_json(
    flat_fields: Dict[str, Any],
    file_name: str = "",
    file_path: str = "",
) -> Dict[str, Any]:
    """Convert flat fields dictionary into the organized structured JSON document."""
    items = flat_fields.get("items") or []
    subtotal = float(flat_fields.get("subtotal") or 0.0)
    cgst = float(flat_fields.get("cgst_amount") or 0.0)
    sgst = float(flat_fields.get("sgst_amount") or 0.0)
    igst = float(flat_fields.get("igst_amount") or 0.0)
    tax = float(flat_fields.get("tax_amount") or (cgst + sgst + igst))
    discount = float(flat_fields.get("discount_amount") or 0.0)
    total = float(flat_fields.get("total_amount") or (subtotal - discount + tax))
    item_count = int(flat_fields.get("item_count") or (len(items) or 1))
    qty = float(flat_fields.get("quantity") or (sum(float(i.get("quantity") or 1.0) for i in items) or 1.0))

    return {
        "invoice_metadata": {
            "invoice_number": str(flat_fields.get("invoice_number") or ""),
            "invoice_date": str(flat_fields.get("invoice_date") or ""),
            "due_date": flat_fields.get("due_date"),
            "po_number": flat_fields.get("po_number"),
            "place_of_supply": flat_fields.get("place_of_supply"),
            "currency": flat_fields.get("currency", "INR"),
        },
        "vendor": {
            "name": flat_fields.get("vendor_name", ""),
            "gstin": flat_fields.get("gstin", ""),
            "address": flat_fields.get("vendor_address", ""),
            "state": flat_fields.get("extracted_vendor_state", ""),
            "state_code": flat_fields.get("extracted_vendor_state_code", ""),
            "phone": flat_fields.get("phone"),
            "email": flat_fields.get("email"),
        },
        "buyer": {
            "name": flat_fields.get("buyer_name", ""),
            "gstin": flat_fields.get("buyer_gstin", ""),
            "address": flat_fields.get("buyer_address", ""),
            "state": flat_fields.get("buyer_state", ""),
            "state_code": flat_fields.get("buyer_state_code", ""),
        },
        "line_items": items,
        "financial_summary": {
            "subtotal": subtotal,
            "discount_amount": discount,
            "cgst_amount": cgst,
            "sgst_amount": sgst,
            "igst_amount": igst,
            "tax_amount": tax,
            "total_amount": total,
            "item_count": item_count,
            "total_quantity": qty,
        },
        "provenance": {
            "file_name": file_name or flat_fields.get("file_name", ""),
            "file_path": file_path or flat_fields.get("file_path", ""),
            "page_count": flat_fields.get("page_count", 1),
            "parsed_at": datetime.datetime.utcnow().isoformat(),
        },
    }


def parse_invoice_to_json(
    ocr_text: str,
    file_name: str = "",
    file_path: str = "",
    page_count: int = 1,
    output_json_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Takes raw OCR text and structures it into a comprehensive JSON schema.
    Optionally saves the formatted JSON to output_json_path.

    Returns:
    {
        "structured_json": <hierarchical JSON dict>,
        "flat_fields": <flattened fields matching DB/ML models>,
        "json_str": <formatted JSON string>,
        "json_file_path": <Path to saved JSON file, if output_json_path is given>
    }
    """
    metadata = extract_metadata(ocr_text)
    vendor, buyer = extract_vendor_and_buyer(ocr_text)
    line_items = extract_line_items(ocr_text)
    financials = extract_financials(ocr_text, line_items)

    structured_doc = {
        "invoice_metadata": metadata,
        "vendor": vendor,
        "buyer": buyer,
        "line_items": line_items,
        "financial_summary": financials,
        "provenance": {
            "file_name": file_name,
            "file_path": file_path,
            "page_count": page_count,
            "parsed_at": datetime.datetime.utcnow().isoformat(),
            "raw_text_length": len(ocr_text),
            "line_count": len(ocr_text.splitlines()),
        },
    }

    flat_fields = structured_to_flat_fields(
        structured_doc,
        file_name=file_name,
        file_path=file_path,
        page_count=page_count,
        ocr_text=ocr_text,
    )

    json_str = json.dumps(structured_doc, indent=2, ensure_ascii=False)
    saved_path_str: Optional[str] = None

    if output_json_path is not None:
        saved_file = save_structured_json(structured_doc, output_json_path)
        saved_path_str = str(saved_file)
        flat_fields["json_file_path"] = saved_path_str

    result: Dict[str, Any] = {
        "structured_json": structured_doc,
        "flat_fields": flat_fields,
        "json_str": json_str,
    }
    if saved_path_str:
        result["json_file_path"] = saved_path_str

    return result


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Parse OCR plain text into organized JSON and optionally feed into FinDoc AI pipeline.")
    parser.add_argument("input_file", help="Path to plain text OCR output file (.txt) or invoice document")
    parser.add_argument("-o", "--output", help="Path to save organized output JSON file (default: <input>_extracted.json)", default=None)
    parser.add_argument("--pipeline", action="store_true", help="Feed the organized JSON into the complete validation & anomaly pipeline")

    args = parser.parse_args()
    in_path = Path(args.input_file)

    if not in_path.exists():
        print(f"Error: File '{in_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_extracted.json")

    if args.pipeline:
        from app.services.ingest import process_invoice
        print(f"Feeding '{in_path}' into the FinDoc AI pipeline...")
        pipeline_res = process_invoice(in_path, in_path.name, output_json_path=out_path)
        print(f"[OK] Pipeline complete! Organized JSON saved to: {out_path}")
        print(f"Review Status: {pipeline_res['validated'].get('review_status')}")
        print(f"Final Anomaly: {pipeline_res['validated'].get('final_anomaly')}")
        if pipeline_res.get("missing_fields"):
            print(f"Missing fields: {pipeline_res['missing_fields']}")
    else:
        from app.services.ocr import extract_text
        text, p_count = extract_text(in_path)
        res = parse_invoice_to_json(text, file_name=in_path.name, file_path=str(in_path), page_count=p_count, output_json_path=out_path)
        print(f"[OK] Parsed {len(res['structured_json']['line_items'])} line items.")
        print(f"[OK] Saved organized JSON to: {out_path}")
        print("\nStructured JSON summary:")
        print(json.dumps(res["structured_json"]["financial_summary"], indent=2))
