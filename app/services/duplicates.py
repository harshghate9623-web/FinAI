"""
app/services/duplicates.py
Detects exact and near-duplicate invoices.
Stores findings in the `duplicates` DB table (via the DB layer, TBD).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import pandas as pd


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

EXACT_KEYS = ["gstin", "invoice_number"]                     # exact match fields
NEAR_AMOUNT_TOLERANCE_PCT = 0.05                             # 5% amount difference
NEAR_DATE_TOLERANCE_DAYS = 7                                 # within 7 days
NEAR_SIMILARITY_THRESHOLD = 0.80                             # minimum score to report


@dataclass
class DuplicateMatch:
    invoice_id_1: int
    invoice_id_2: int
    match_type: str                 # 'exact' | 'near_duplicate'
    similarity_score: float         # 0.0 – 1.0
    matched_fields: str             # human-readable list of matched fields


# ---------------------------------------------------------------------------
# Exact duplicate detection
# ---------------------------------------------------------------------------

def find_exact_duplicates(df: pd.DataFrame) -> List[DuplicateMatch]:
    """
    Return pairs of invoices that share the same GSTIN + invoice_number.
    Both fields must be non-empty.
    """
    if not {"gstin", "invoice_number", "invoice_id"}.issubset(df.columns):
        return []

    matches: List[DuplicateMatch] = []
    groups = df[df["gstin"].ne("") & df["invoice_number"].ne("")]

    for _, group in groups.groupby(["gstin", "invoice_number"]):
        ids = group["invoice_id"].tolist()
        if len(ids) >= 2:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    matches.append(
                        DuplicateMatch(
                            invoice_id_1=ids[i],
                            invoice_id_2=ids[j],
                            match_type="exact",
                            similarity_score=1.0,
                            matched_fields="gstin, invoice_number",
                        )
                    )

    return matches


# ---------------------------------------------------------------------------
# Near-duplicate detection (fuzzy, field-based)
# ---------------------------------------------------------------------------

def _amount_similarity(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Return a 0–1 similarity score for two amounts, or None if either is missing."""
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    if a == 0 and b == 0:
        return 1.0
    diff_pct = abs(a - b) / max(abs(a), abs(b), 1e-9)
    return max(0.0, 1.0 - diff_pct / NEAR_AMOUNT_TOLERANCE_PCT)


def _date_similarity(d1: Any, d2: Any) -> Optional[float]:
    """Return 1.0 for same date, 0.0 for dates > NEAR_DATE_TOLERANCE_DAYS apart."""
    try:
        t1 = pd.Timestamp(d1)
        t2 = pd.Timestamp(d2)
        days_apart = abs((t1 - t2).days)
        if days_apart > NEAR_DATE_TOLERANCE_DAYS:
            return 0.0
        return 1.0 - (days_apart / NEAR_DATE_TOLERANCE_DAYS)
    except Exception:
        return None


def _gstin_match(g1: str, g2: str) -> float:
    """1.0 if same GSTIN (non-empty), else 0.0."""
    if g1 and g2 and g1.upper() == g2.upper():
        return 1.0
    return 0.0


def compute_similarity(row1: Dict[str, Any], row2: Dict[str, Any]) -> tuple[float, List[str]]:
    """
    Compute an overall similarity score between two invoice records.
    Returns (score 0-1, list of matched field names).
    """
    scored_fields: List[tuple[str, Optional[float], float]] = []  # (name, score, weight)

    scored_fields.append(("gstin", _gstin_match(
        str(row1.get("gstin", "")), str(row2.get("gstin", ""))
    ), 2.0))

    scored_fields.append(("total_amount", _amount_similarity(
        row1.get("total_amount"), row2.get("total_amount")
    ), 2.0))

    scored_fields.append(("invoice_date", _date_similarity(
        row1.get("invoice_date"), row2.get("invoice_date")
    ), 1.5))

    scored_fields.append(("subtotal", _amount_similarity(
        row1.get("subtotal"), row2.get("subtotal")
    ), 1.0))

    scored_fields.append(("tax_amount", _amount_similarity(
        row1.get("tax_amount"), row2.get("tax_amount")
    ), 1.0))

    total_weight = 0.0
    weighted_sum = 0.0
    matched: List[str] = []

    for field, score, weight in scored_fields:
        if score is None:
            continue
        total_weight += weight
        weighted_sum += score * weight
        if score >= 0.95:
            matched.append(field)

    if total_weight == 0:
        return 0.0, []

    return round(weighted_sum / total_weight, 4), matched


def find_near_duplicates(
    new_invoice: Dict[str, Any],
    existing_invoices: pd.DataFrame,
) -> List[DuplicateMatch]:
    """
    Compare a single new invoice against all existing invoices in the DB.
    Returns near-duplicate matches above the similarity threshold.
    Used in the upload pipeline for incremental processing.
    """
    if existing_invoices.empty:
        return []

    new_id = new_invoice.get("invoice_id")
    matches: List[DuplicateMatch] = []

    # Pre-filter: only compare invoices from the same GSTIN vendor.
    gstin = str(new_invoice.get("gstin", "")).upper()
    if gstin:
        candidates = existing_invoices[
            existing_invoices["gstin"].astype(str).str.upper() == gstin
        ]
    else:
        candidates = existing_invoices

    for _, row in candidates.iterrows():
        other_id = row.get("invoice_id")
        if other_id == new_id:
            continue

        score, matched_fields = compute_similarity(new_invoice, row.to_dict())

        if score >= NEAR_SIMILARITY_THRESHOLD and matched_fields:
            matches.append(
                DuplicateMatch(
                    invoice_id_1=int(new_id) if new_id is not None else 0,
                    invoice_id_2=int(other_id),
                    match_type="near_duplicate",
                    similarity_score=score,
                    matched_fields=", ".join(matched_fields),
                )
            )

    return matches
