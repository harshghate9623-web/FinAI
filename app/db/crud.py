"""
app/db/crud.py
Database access layer — functions to insert, update, and query invoice records.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db.models import AnomalyResult, AuditLog, Duplicate, Invoice, InvoiceItem


def parse_date(val: Any) -> Optional[datetime.date]:
    """Helper to parse a date string or timestamp into a datetime.date object."""
    if not val:
        return None
    if isinstance(val, (datetime.date, datetime.datetime)):
        return val if isinstance(val, datetime.date) else val.date()
    val_str = str(val).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.datetime.strptime(val_str, fmt).date()
        except ValueError:
            continue
    return None


def save_processed_invoice(
    db: Session,
    pipeline_result: Dict[str, Any],
    file_path: Optional[str] = None,
    performed_by: str = "system",
) -> Invoice:
    """
    Persist the entire outcome of the ingest pipeline to PostgreSQL:
    1. Insert header invoice record
    2. Insert anomaly detection findings
    3. Insert duplicate detections (if any)
    4. Write audit log
    """
    extracted = pipeline_result.get("extracted", {})
    validated = pipeline_result.get("validated", {})
    duplicates_list = pipeline_result.get("duplicates", [])

    inv_date = parse_date(extracted.get("invoice_date"))

    invoice = Invoice(
        invoice_number=str(extracted.get("invoice_number") or ""),
        vendor_name=str(extracted.get("vendor_name") or ""),
        gstin=str(extracted.get("gstin") or ""),
        invoice_date=inv_date,
        vendor_address=str(extracted.get("vendor_address") or ""),
        extracted_vendor_state=str(validated.get("extracted_vendor_state") or ""),
        extracted_vendor_state_code=str(validated.get("extracted_vendor_state_code") or ""),
        subtotal=extracted.get("subtotal") or 0.0,
        cgst_amount=extracted.get("cgst_amount") or 0.0,
        sgst_amount=extracted.get("sgst_amount") or 0.0,
        igst_amount=extracted.get("igst_amount") or 0.0,
        tax_amount=extracted.get("tax_amount") or 0.0,
        discount_amount=extracted.get("discount_amount") or 0.0,
        total_amount=extracted.get("total_amount") or 0.0,
        item_count=int(extracted.get("item_count") or 0),
        quantity=float(extracted.get("quantity") or 0.0),
        file_name=pipeline_result.get("file_name", ""),
        file_path=file_path or "",
        page_count=int(pipeline_result.get("page_count", 1)),
        ocr_text=pipeline_result.get("ocr_text", ""),
        extracted_json=pipeline_result.get("structured_json") or extracted,
        review_status=validated.get("review_status", "Pending Manual Review"),
    )
    db.add(invoice)
    db.flush()  # Populates invoice.invoice_id

    # 1b. Line items
    line_items = pipeline_result.get("line_items") or extracted.get("items") or []
    for item in line_items:
        inv_item = InvoiceItem(
            invoice_id=invoice.invoice_id,
            item_order=int(item.get("item_order", 1)),
            description=str(item.get("description", "")),
            hsn_sac_code=item.get("hsn_sac_code"),
            quantity=float(item.get("quantity") or 1.0),
            unit_price=float(item.get("unit_price") or 0.0),
            line_total=float(item.get("line_total") or 0.0),
            tax_rate=float(item.get("tax_rate") or 18.0),
        )
        db.add(inv_item)

    # 2. Anomaly results
    anomaly = AnomalyResult(
        invoice_id=invoice.invoice_id,
        rule_anomaly=bool(validated.get("rule_anomaly", False)),
        isolation_anomaly=bool(validated.get("isolation_anomaly", False)),
        final_anomaly=bool(validated.get("final_anomaly", False)),
        model_input_invalid=bool(validated.get("model_input_invalid", False)),
        isolation_score=validated.get("isolation_score"),
        rule_flags_text=str(validated.get("rule_flags_text", "")),
        model_version=str(validated.get("model_version", "isolation_forest_v1")),
    )
    db.add(anomaly)

    # 3. Duplicates
    for dup in duplicates_list:
        other_id = dup.get("matched_with_invoice_id")
        if other_id and str(other_id).isdigit():
            dup_record = Duplicate(
                invoice_id_1=invoice.invoice_id,
                invoice_id_2=int(other_id),
                match_type=dup.get("match_type", "near_duplicate"),
                similarity_score=dup.get("similarity_score"),
                matched_fields=", ".join(dup.get("matched_fields", [])),
                review_status="Pending Review",
            )
            db.add(dup_record)

    # 4. Audit Log
    log = AuditLog(
        invoice_id=invoice.invoice_id,
        action="uploaded_and_processed",
        performed_by=performed_by,
        details={
            "final_anomaly": validated.get("final_anomaly", False),
            "flags": validated.get("rule_flags_text", ""),
            "review_status": invoice.review_status,
        },
    )
    db.add(log)

    db.commit()
    db.refresh(invoice)
    return invoice


def get_all_invoices_df(db: Session, limit: int = 1000) -> pd.DataFrame:
    """Fetch recent invoices as a pandas DataFrame for duplicate comparison."""
    query = select(Invoice).order_by(desc(Invoice.invoice_id)).limit(limit)
    records = db.execute(query).scalars().all()

    if not records:
        return pd.DataFrame()

    data = [
        {
            "invoice_id": r.invoice_id,
            "invoice_number": r.invoice_number,
            "gstin": r.gstin,
            "invoice_date": str(r.invoice_date) if r.invoice_date else "",
            "total_amount": float(r.total_amount) if r.total_amount else 0.0,
            "vendor_name": r.vendor_name,
            "subtotal": float(r.subtotal) if r.subtotal else 0.0,
            "cgst_amount": float(r.cgst_amount) if r.cgst_amount else 0.0,
            "sgst_amount": float(r.sgst_amount) if r.sgst_amount else 0.0,
            "igst_amount": float(r.igst_amount) if r.igst_amount else 0.0,
            "tax_amount": float(r.tax_amount) if r.tax_amount else 0.0,
            "discount_amount": float(r.discount_amount) if r.discount_amount else 0.0,
            "quantity": float(r.quantity) if r.quantity else 0.0,
            "item_count": r.item_count,
            "review_status": r.review_status,
            "file_name": r.file_name,
        }
        for r in records
    ]
    return pd.DataFrame(data)


def list_invoices_with_anomalies(db: Session, limit: int = 100) -> List[Dict[str, Any]]:
    """List recent invoices joined with their anomaly findings."""
    query = (
        select(Invoice, AnomalyResult)
        .outerjoin(AnomalyResult, Invoice.invoice_id == AnomalyResult.invoice_id)
        .order_by(desc(Invoice.invoice_id))
        .limit(limit)
    )
    rows = db.execute(query).all()

    results = []
    for inv, anom in rows:
        results.append({
            "invoice_id": inv.invoice_id,
            "invoice_number": inv.invoice_number,
            "vendor_name": inv.vendor_name,
            "gstin": inv.gstin,
            "invoice_date": str(inv.invoice_date) if inv.invoice_date else None,
            "total_amount": float(inv.total_amount) if inv.total_amount else 0.0,
            "review_status": inv.review_status,
            "file_name": inv.file_name,
            "rule_anomaly": anom.rule_anomaly if anom else False,
            "isolation_anomaly": anom.isolation_anomaly if anom else False,
            "final_anomaly": anom.final_anomaly if anom else False,
            "isolation_score": float(anom.isolation_score) if anom and anom.isolation_score is not None else None,
            "rule_flags_text": anom.rule_flags_text if anom else "",
            "uploaded_at": inv.uploaded_at.isoformat() if inv.uploaded_at else None,
        })
    return results


def get_exception_invoices(db: Session) -> List[Dict[str, Any]]:
    """Fetch all invoices marked as final_anomaly = True."""
    query = (
        select(Invoice, AnomalyResult)
        .join(AnomalyResult, Invoice.invoice_id == AnomalyResult.invoice_id)
        .where(AnomalyResult.final_anomaly.is_(True))
        .order_by(desc(Invoice.invoice_id))
    )
    rows = db.execute(query).all()

    return [
        {
            "invoice_id": inv.invoice_id,
            "invoice_number": inv.invoice_number,
            "vendor_name": inv.vendor_name,
            "gstin": inv.gstin,
            "invoice_date": str(inv.invoice_date) if inv.invoice_date else None,
            "total_amount": float(inv.total_amount) if inv.total_amount else 0.0,
            "tax_amount": float(inv.tax_amount) if inv.tax_amount else 0.0,
            "review_status": inv.review_status,
            "rule_flags": anom.rule_flags_text,
            "isolation_score": float(anom.isolation_score) if anom.isolation_score is not None else None,
            "uploaded_at": inv.uploaded_at.isoformat() if inv.uploaded_at else None,
        }
        for inv, anom in rows
    ]


def update_invoice_review_status(
    db: Session,
    invoice_id: int,
    new_status: str,
    performed_by: str = "reviewer",
) -> Optional[Invoice]:
    """Update review_status of an invoice and record an audit log."""
    invoice = db.get(Invoice, invoice_id)
    if not invoice:
        return None

    old_status = invoice.review_status
    invoice.review_status = new_status
    invoice.updated_at = datetime.datetime.utcnow()

    log = AuditLog(
        invoice_id=invoice.invoice_id,
        action="review_status_updated",
        performed_by=performed_by,
        details={
            "old_status": old_status,
            "new_status": new_status,
        },
    )
    db.add(log)
    db.commit()
    db.refresh(invoice)
    return invoice
