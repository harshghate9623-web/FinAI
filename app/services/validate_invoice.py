"""
app/services/validate_invoice.py
Deterministic rule-based validation engine for Indian GST invoices.
Evaluates statutory rules, arithmetic balances, and anomaly criteria.
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOLERANCE = 2.00            # ₹ rounding tolerance for arithmetic checks
MAX_DISCOUNT_RATIO = 0.30   # 30% of subtotal is the excessive-discount threshold
SPLITTING_THRESHOLD = 100_000  # ₹1 lakh threshold-splitting check
STALE_DAYS = 730            # Invoices older than 2 years are flagged

# ---------------------------------------------------------------------------
# GST state-code mapping (complete as of 2024)
# ---------------------------------------------------------------------------

STATE_ALIASES: Dict[str, str] = {
    "JAMMU AND KASHMIR": "01", "JAMMU & KASHMIR": "01",
    "JAMMU": "01", "SRINAGAR": "01",
    "HIMACHAL PRADESH": "02", "HP": "02",
    "PUNJAB": "03", "CHANDIGARH": "04",
    "UTTARAKHAND": "05", "UTTARANCHAL": "05",
    "HARYANA": "06",
    "DELHI": "07", "NEW DELHI": "07",
    "RAJASTHAN": "08",
    "UTTAR PRADESH": "09", "UP": "09",
    "BIHAR": "10", "SIKKIM": "11",
    "ARUNACHAL PRADESH": "12", "NAGALAND": "13",
    "MANIPUR": "14", "MIZORAM": "15",
    "TRIPURA": "16", "MEGHALAYA": "17",
    "ASSAM": "18", "WEST BENGAL": "19",
    "JHARKHAND": "20",
    "ODISHA": "21", "ORISSA": "21",
    "CHHATTISGARH": "22",
    "MADHYA PRADESH": "23", "MP": "23",
    "GUJARAT": "24",
    "DAMAN AND DIU": "25", "DAMAN & DIU": "25",
    "DADRA AND NAGAR HAVELI": "26",
    "DADRA & NAGAR HAVELI": "26",
    "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "26",
    "MAHARASHTRA": "27", "MH": "27",
    "ANDHRA PRADESH": "28",
    "KARNATAKA": "29", "KA": "29",
    "GOA": "30", "LAKSHADWEEP": "31",
    "KERALA": "32", "TAMIL NADU": "33",
    "PUDUCHERRY": "34", "PONDICHERRY": "34",
    "ANDAMAN AND NICOBAR ISLANDS": "35",
    "TELANGANA": "36", "TS": "36",
    "ANDHRA PRADESH (NEW)": "37",
    "LADAKH": "38",
}

STATE_CODES = set(STATE_ALIASES.values())
# Sort longer names first to avoid partial matches.
STATE_NAMES = sorted(STATE_ALIASES, key=len, reverse=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_state_from_address(address: str) -> Tuple[str | None, str | None]:
    """Return (state_name, GST_state_code) inferred from a vendor address string."""
    if pd.isna(address) or not str(address).strip():
        return None, None

    text = str(address).upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    for name in STATE_NAMES:
        pattern = r"(?<![A-Z])" + re.escape(name) + r"(?![A-Z])"
        if re.search(pattern, text):
            return name, STATE_ALIASES[name]

    return None, None


def gstin_format_ok(gstin: str) -> bool:
    """Return True when GSTIN matches the 15-character Indian format."""
    if not re.fullmatch(r"\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]", gstin):
        return False
    return gstin[:2] in STATE_CODES


# ---------------------------------------------------------------------------
# Core validation — operates on a DataFrame
# ---------------------------------------------------------------------------

def validate_invoices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all deterministic rule checks to a DataFrame of invoices.

    Adds columns:
        rule_flags          list[str]   — individual flag messages
        rule_flags_text     str         — semicolon-joined flags or 'No issue detected'
        rule_anomaly        bool        — True when any flag was raised
        extracted_vendor_state          — state name parsed from vendor_address
        extracted_vendor_state_code     — GST state code parsed from vendor_address
    """
    df = df.copy()
    numeric_cols = [
        "subtotal", "cgst_amount", "sgst_amount", "igst_amount",
        "tax_amount", "discount_amount", "total_amount",
    ]

    # -- Normalise types -------------------------------------------------
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["gstin", "invoice_number"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip().str.upper()

    if "invoice_date" in df.columns:
        def _parse_safe_date(val: Any) -> Any:
            if not val or pd.isna(val):
                return pd.NaT
            s = str(val).strip()
            if re.match(r"^\d{4}-\d{2}-\d{2}", s):
                return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
            return pd.to_datetime(s, dayfirst=True, errors="coerce")

        df["invoice_date_parsed"] = df["invoice_date"].apply(_parse_safe_date)

    df["rule_flags"] = [[] for _ in range(len(df))]

    def flag(mask: Any, message: str) -> None:
        mask_s = pd.Series(mask, index=df.index).fillna(False).astype(bool)
        for idx in df.index[mask_s]:
            df.at[idx, "rule_flags"].append(message)

    # -- Fill calc copy (for arithmetic; never mutates display values) ---
    calc = df[numeric_cols].copy()
    for col in numeric_cols:
        if col in calc.columns:
            calc[col] = calc[col].fillna(0)

    # 1. Required text fields
    for col in ["gstin", "invoice_number"]:
        if col in df.columns:
            flag(df[col].eq(""), f"Missing required field: {col}")

    if "invoice_date" in df.columns:
        flag(df["invoice_date"].fillna("").eq(""), "Missing required field: invoice_date")

    if "vendor_address" in df.columns:
        df["vendor_address"] = df["vendor_address"].fillna("").astype(str).str.strip()
        flag(df["vendor_address"].eq(""), "Missing vendor address")

    # 2. Missing / invalid numeric values
    for col in numeric_cols:
        if col in df.columns:
            flag(df[col].isna(), f"Missing or invalid numeric value: {col}")

    # 3. Date checks
    if "invoice_date_parsed" in df.columns:
        today = pd.Timestamp.today().normalize()
        flag(df["invoice_date_parsed"].isna(), "Invalid invoice date")
        flag(df["invoice_date_parsed"] > today, "Invoice date is in the future")
        flag(
            df["invoice_date_parsed"] < today - pd.Timedelta(days=STALE_DAYS),
            "Invoice date is stale: older than 730 days",
        )

    # 4. GSTIN format
    if "gstin" in df.columns:
        flag(
            df["gstin"].ne("") & ~df["gstin"].apply(gstin_format_ok),
            "Invalid GSTIN format or state code",
        )

    # 5. Vendor address → state extraction + GSTIN state match
    if "vendor_address" in df.columns:
        extracted = df["vendor_address"].apply(extract_state_from_address)
        df["extracted_vendor_state"] = extracted.apply(lambda x: x[0])
        df["extracted_vendor_state_code"] = extracted.apply(lambda x: x[1])

        flag(
            df["vendor_address"].ne("") & df["extracted_vendor_state_code"].isna(),
            "Could not extract state from vendor address: review",
        )

        if "gstin" in df.columns:
            state_mismatch = (
                df["gstin"].str.match(r"^\d{2}", na=False)
                & df["extracted_vendor_state_code"].notna()
                & df["gstin"].str[:2].ne(df["extracted_vendor_state_code"].fillna(""))
            )
            flag(state_mismatch, "GSTIN state differs from address-derived state: review")

    # 6. Tax arithmetic: CGST + SGST + IGST = tax_amount
    if all(c in df.columns for c in ["cgst_amount", "sgst_amount", "igst_amount", "tax_amount"]):
        tax_components = calc["cgst_amount"] + calc["sgst_amount"] + calc["igst_amount"]
        flag(
            df["tax_amount"].notna()
            & df[["cgst_amount", "sgst_amount", "igst_amount"]].notna().all(axis=1)
            & ((calc["tax_amount"] - tax_components).abs() > TOLERANCE),
            "Tax amount does not equal CGST + SGST + IGST",
        )

    # 7. Invoice total arithmetic: subtotal - discount + tax = total
    if all(c in df.columns for c in ["subtotal", "discount_amount", "tax_amount", "total_amount"]):
        expected_total = calc["subtotal"] - calc["discount_amount"] + calc["tax_amount"]
        flag(
            df[["subtotal", "discount_amount", "tax_amount", "total_amount"]].notna().all(axis=1)
            & ((calc["total_amount"] - expected_total).abs() > TOLERANCE),
            "Invoice total mismatch",
        )

    # 8. Negative / zero amounts
    if "subtotal" in df.columns and "total_amount" in df.columns:
        flag(
            (calc["subtotal"] < 0) | (calc["total_amount"] <= 0),
            "Negative subtotal or non-positive invoice total",
        )
    if "discount_amount" in df.columns:
        flag(calc["discount_amount"] < 0, "Negative discount amount")

    # 9. Excessive discount (> 30% of subtotal)
    if "subtotal" in df.columns and "discount_amount" in df.columns:
        flag(
            (calc["subtotal"] > 0)
            & (calc["discount_amount"] > calc["subtotal"] * MAX_DISCOUNT_RATIO),
            "High discount requires review",
        )

    # 10. CGST ≠ SGST (intra-state invoices should have equal CGST & SGST)
    if "cgst_amount" in df.columns and "sgst_amount" in df.columns:
        flag(
            df[["cgst_amount", "sgst_amount"]].notna().all(axis=1)
            & ((calc["cgst_amount"] - calc["sgst_amount"]).abs() > TOLERANCE),
            "CGST and SGST amounts differ: review",
        )

    # 11. Threshold splitting (batch-level, within this DataFrame only)
    if "total_amount" in df.columns and "gstin" in df.columns:
        df["_invoice_day"] = (
            df.get("invoice_date_parsed", pd.Series(dtype="datetime64[ns]")).dt.date
        )
        df["_near_threshold"] = (
            (calc["total_amount"] >= 0.80 * SPLITTING_THRESHOLD)
            & (calc["total_amount"] < SPLITTING_THRESHOLD)
            & df["gstin"].ne("")
        )
        candidate = df[df["_near_threshold"]].copy()
        if len(candidate) > 0 and "invoice_number" in df.columns:
            candidate_counts = candidate.groupby(
                ["gstin", "_invoice_day"]
            )["invoice_number"].transform("count")
            splitting_idx = candidate.index[candidate_counts >= 2]
            flag(df.index.isin(splitting_idx), "Possible invoice splitting: manual review")
        df.drop(columns=["_invoice_day", "_near_threshold"], errors="ignore", inplace=True)

    # -- Finalise --------------------------------------------------------
    df["rule_anomaly"] = df["rule_flags"].apply(bool)
    df["rule_flags_text"] = df["rule_flags"].apply(
        lambda items: "; ".join(items) if items else "No rule issue detected"
    )

    return df


def validate_single(fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience wrapper: validate a single invoice dict and return
    a dict with 'rule_anomaly', 'rule_flags_text', and extracted state info.
    """
    df = pd.DataFrame([fields])
    result = validate_invoices(df)
    return result.iloc[0].to_dict()
