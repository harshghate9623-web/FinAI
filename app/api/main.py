"""
app/api/main.py
FastAPI application — REST endpoints for FinDoc AI.
Integrates OCR, rule engine, ML anomaly detector, duplicate detection,
and PostgreSQL persistence (Aiven / Local) with fallback in-memory mode.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.db.connection import get_db, init_db, is_db_connected
from app.db import crud
from app.services.ingest import (
    MAX_FILE_SIZE_MB,
    process_invoice,
    process_upload_file,
    save_upload,
    validate_upload,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FinDoc AI",
    description=(
        "Invoice OCR, field extraction, rule-based validation, "
        "anomaly detection, and duplicate detection for Indian B2B invoices."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory fallback store if database is not configured
_in_memory_invoices: List[Dict[str, Any]] = []


@app.on_event("startup")
def on_startup() -> None:
    """Initialize DB tables on app start if database is configured."""
    try:
        init_db()
    except Exception as exc:
        print(f"[Warning] Could not initialize database tables: {exc}")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
def health() -> Dict[str, Any]:
    db_ok = is_db_connected()
    return {
        "status": "ok",
        "version": "0.1.0",
        "database": "connected" if db_ok else "in_memory_fallback",
    }


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------

from pydantic import BaseModel


class ProcessTextRequest(BaseModel):
    ocr_text: str
    filename: Optional[str] = "ocr_extracted.txt"


class ProcessJsonRequest(BaseModel):
    invoice_data: Dict[str, Any]
    filename: Optional[str] = "invoice.json"


@app.post("/upload", tags=["invoices"])
async def upload_invoice(
    file: UploadFile = File(...),
    db: Optional[Session] = Depends(get_db),
) -> JSONResponse:
    """
    Accept a PDF, image, plain OCR text (.txt), or structured JSON file,
    parse into an organized JSON file, run the validation/anomaly pipeline,
    and persist results to PostgreSQL or in-memory store.
    """
    content = await file.read()

    # -- Validate file ------------------------------------------------------
    try:
        validate_upload(file.filename or "unknown", len(content))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    # -- Save to disk -------------------------------------------------------
    saved_path = save_upload(content, file.filename or "invoice")

    # -- Retrieve existing invoices for duplicate detection -----------------
    if db is not None:
        try:
            existing_df = crud.get_all_invoices_df(db)
        except Exception:
            existing_df = (
                pd.DataFrame(_in_memory_invoices)
                if _in_memory_invoices
                else pd.DataFrame()
            )
    else:
        existing_df = (
            pd.DataFrame(_in_memory_invoices)
            if _in_memory_invoices
            else pd.DataFrame()
        )

    # -- Run pipeline (automatically separates multi-page PDFs into distinct invoices) --
    try:
        results = process_upload_file(saved_path, file.filename or "invoice", existing_df)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {exc}",
        )

    # -- Persist each detected invoice to DB or in-memory store -------------
    invoices_list = []
    for res in results:
        inv_id = None
        inv_file_path = str(res.get("file_path", saved_path))
        if db is not None:
            try:
                saved_inv = crud.save_processed_invoice(db, res, inv_file_path)
                inv_id = saved_inv.invoice_id
            except Exception as exc:
                print(f"[Warning] Failed to persist invoice to DB, storing in-memory: {exc}")
                inv_id = len(_in_memory_invoices) + 1
                res["validated"]["invoice_id"] = inv_id
                res["validated"]["file_path"] = inv_file_path
                res["validated"]["file_name"] = res.get("file_name")
                _in_memory_invoices.append(res["validated"])
        else:
            inv_id = len(_in_memory_invoices) + 1
            res["validated"]["invoice_id"] = inv_id
            res["validated"]["file_path"] = inv_file_path
            res["validated"]["file_name"] = res.get("file_name")
            _in_memory_invoices.append(res["validated"])

        res["invoice_id"] = inv_id

        inv_dict = {
            "invoice_id": inv_id,
            "file_name": res["file_name"],
            "page_number": res.get("page_number", 1),
            "total_pages": res.get("total_pages", 1),
            "page_count": res.get("page_count", 1),
            "file_path": inv_file_path,
            "file_bytes_base64": res.get("file_bytes_base64"),
            "json_file_path": res.get("json_file_path"),
            "structured_json": res.get("structured_json"),
            "line_items": res.get("line_items", []),
            "extracted_fields": res["extracted"],
            "missing_fields": res["missing_fields"],
            "rule_anomaly": res["validated"].get("rule_anomaly", False),
            "rule_flags": res["validated"].get("rule_flags_text", ""),
            "isolation_anomaly": res["validated"].get("isolation_anomaly", False),
            "isolation_score": res["validated"].get("isolation_score"),
            "model_input_invalid": res["validated"].get("model_input_invalid", False),
            "final_anomaly": res["validated"].get("final_anomaly", False),
            "review_status": res["validated"].get("review_status", ""),
            "duplicates": res["duplicates"],
            "ocr_text": res.get("ocr_text", ""),
        }
        invoices_list.append(inv_dict)

    first_inv = invoices_list[0]
    response_payload = {
        "status": "success",
        "total_invoices": len(invoices_list),
        "is_multi_invoice": len(invoices_list) > 1,
        "invoices": invoices_list,
        **first_inv,
    }

    return JSONResponse(content=jsonable_encoder(response_payload))


@app.post("/process-text", tags=["invoices"])
def process_text_endpoint(
    payload: ProcessTextRequest,
    db: Optional[Session] = Depends(get_db),
) -> JSONResponse:
    """
    Accept OCR plain text directly, parse it into an organized JSON file,
    and feed it through the FinDoc AI validation/anomaly pipeline.
    """
    from app.services.ingest import process_ocr_text

    # Retrieve existing invoices for duplicate detection
    if db is not None:
        try:
            existing_df = crud.get_all_invoices_df(db)
        except Exception:
            existing_df = pd.DataFrame(_in_memory_invoices) if _in_memory_invoices else pd.DataFrame()
    else:
        existing_df = pd.DataFrame(_in_memory_invoices) if _in_memory_invoices else pd.DataFrame()

    try:
        result = process_ocr_text(payload.ocr_text, payload.filename or "ocr_extracted.txt", existing_df)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Processing error: {exc}")

    invoice_id = None
    if db is not None:
        try:
            saved_inv = crud.save_processed_invoice(db, result, result.get("file_path"))
            invoice_id = saved_inv.invoice_id
        except Exception as exc:
            print(f"[Warning] Failed to persist invoice to DB, storing in-memory: {exc}")
            _in_memory_invoices.append(result["validated"])
    else:
        _in_memory_invoices.append(result["validated"])

    inv_dict = {
        "invoice_id": invoice_id,
        "file_name": result["file_name"],
        "page_number": 1,
        "total_pages": 1,
        "page_count": result.get("page_count", 1),
        "file_path": result.get("file_path"),
        "json_file_path": result.get("json_file_path"),
        "structured_json": result.get("structured_json"),
        "line_items": result.get("line_items", []),
        "extracted_fields": result["extracted"],
        "missing_fields": result["missing_fields"],
        "rule_anomaly": result["validated"].get("rule_anomaly", False),
        "rule_flags": result["validated"].get("rule_flags_text", ""),
        "isolation_anomaly": result["validated"].get("isolation_anomaly", False),
        "isolation_score": result["validated"].get("isolation_score"),
        "model_input_invalid": result["validated"].get("model_input_invalid", False),
        "final_anomaly": result["validated"].get("final_anomaly", False),
        "review_status": result["validated"].get("review_status", ""),
        "duplicates": result["duplicates"],
        "ocr_text": result.get("ocr_text", ""),
    }

    return JSONResponse(content=jsonable_encoder({
        "status": "success",
        "total_invoices": 1,
        "is_multi_invoice": False,
        "invoices": [inv_dict],
        **inv_dict,
    }))


@app.post("/process-json", tags=["invoices"])
def process_json_endpoint(
    payload: ProcessJsonRequest,
    db: Optional[Session] = Depends(get_db),
) -> JSONResponse:
    """
    Accept an already-structured or raw invoice JSON document and feed it
    through the FinDoc AI validation/anomaly pipeline.
    """
    from app.services.ingest import process_structured_json

    if db is not None:
        try:
            existing_df = crud.get_all_invoices_df(db)
        except Exception:
            existing_df = pd.DataFrame(_in_memory_invoices) if _in_memory_invoices else pd.DataFrame()
    else:
        existing_df = pd.DataFrame(_in_memory_invoices) if _in_memory_invoices else pd.DataFrame()

    try:
        result = process_structured_json(payload.invoice_data, payload.filename or "invoice.json", existing_df)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Processing error: {exc}")

    invoice_id = None
    if db is not None:
        try:
            saved_inv = crud.save_processed_invoice(db, result, result.get("file_path"))
            invoice_id = saved_inv.invoice_id
        except Exception as exc:
            print(f"[Warning] Failed to persist invoice to DB, storing in-memory: {exc}")
            _in_memory_invoices.append(result["validated"])
    else:
        _in_memory_invoices.append(result["validated"])

    inv_dict = {
        "invoice_id": invoice_id,
        "file_name": result["file_name"],
        "page_number": 1,
        "total_pages": 1,
        "page_count": result.get("page_count", 1),
        "file_path": result.get("file_path"),
        "json_file_path": result.get("json_file_path"),
        "structured_json": result.get("structured_json"),
        "line_items": result.get("line_items", []),
        "extracted_fields": result["extracted"],
        "missing_fields": result["missing_fields"],
        "rule_anomaly": result["validated"].get("rule_anomaly", False),
        "rule_flags": result["validated"].get("rule_flags_text", ""),
        "isolation_anomaly": result["validated"].get("isolation_anomaly", False),
        "isolation_score": result["validated"].get("isolation_score"),
        "model_input_invalid": result["validated"].get("model_input_invalid", False),
        "final_anomaly": result["validated"].get("final_anomaly", False),
        "review_status": result["validated"].get("review_status", ""),
        "duplicates": result["duplicates"],
        "ocr_text": result.get("ocr_text", ""),
    }

    return JSONResponse(content=jsonable_encoder({
        "status": "success",
        "total_invoices": 1,
        "is_multi_invoice": False,
        "invoices": [inv_dict],
        **inv_dict,
    }))


# ---------------------------------------------------------------------------
# List all processed invoices
# ---------------------------------------------------------------------------

@app.get("/invoices", tags=["invoices"])
def list_invoices(
    limit: int = 100,
    db: Optional[Session] = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Return all processed invoices (from PostgreSQL or in-memory)."""
    if db is not None:
        try:
            return crud.list_invoices_with_anomalies(db, limit=limit)
        except Exception as exc:
            print(f"[Warning] DB query failed: {exc}")
    return _in_memory_invoices


# ---------------------------------------------------------------------------
# Export exception report
# ---------------------------------------------------------------------------

@app.get("/report", tags=["reports"])
def exception_report(db: Optional[Session] = Depends(get_db)) -> StreamingResponse:
    """
    Download a CSV of all flagged invoices (rule or isolation anomaly).
    """
    flagged: List[Dict[str, Any]] = []

    if db is not None:
        try:
            flagged = crud.get_exception_invoices(db)
        except Exception as exc:
            print(f"[Warning] Failed to fetch exceptions from DB: {exc}")

    if not flagged:
        flagged = [
            inv for inv in _in_memory_invoices
            if inv.get("final_anomaly", False)
        ]

    if not flagged:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No flagged invoices to export.",
        )

    df = pd.DataFrame(flagged)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return StreamingResponse(
        iter([csv_buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=exception_report.csv"},
    )


# ---------------------------------------------------------------------------
# Status update & invoice file access
# ---------------------------------------------------------------------------

class UpdateInvoiceStatusRequest(BaseModel):
    review_status: str
    performed_by: Optional[str] = "reviewer"


@app.patch("/invoices/{invoice_id}/status", tags=["invoices"])
@app.post("/invoices/{invoice_id}/status", tags=["invoices"])
def update_invoice_status(
    invoice_id: int,
    payload: UpdateInvoiceStatusRequest,
    db: Optional[Session] = Depends(get_db),
) -> JSONResponse:
    """Update review status (e.g. 'Checked', 'Marked for Review') and record audit log."""
    if db is not None:
        updated = crud.update_invoice_review_status(
            db=db,
            invoice_id=invoice_id,
            new_status=payload.review_status,
            performed_by=payload.performed_by or "reviewer",
        )
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Invoice ID {invoice_id} not found in database.",
            )
        return JSONResponse(content={
            "status": "success",
            "invoice_id": updated.invoice_id,
            "review_status": updated.review_status,
            "updated_at": updated.updated_at.isoformat() if updated.updated_at else None,
        })
    else:
        # Fallback in-memory
        for inv in _in_memory_invoices:
            if inv.get("invoice_id") == invoice_id:
                inv["review_status"] = payload.review_status
                return JSONResponse(content={
                    "status": "success",
                    "invoice_id": invoice_id,
                    "review_status": payload.review_status,
                })
        return JSONResponse(content={
            "status": "success",
            "invoice_id": invoice_id,
            "review_status": payload.review_status,
            "note": "Updated in-memory store",
        })


@app.get("/invoices/{invoice_id}/file", tags=["invoices"])
def get_invoice_file(
    invoice_id: int,
    db: Optional[Session] = Depends(get_db),
):
    """Retrieve the original uploaded invoice file."""
    file_p: Optional[Path] = None
    filename: Optional[str] = None

    if db is not None:
        try:
            from app.db.models import Invoice
            invoice = db.get(Invoice, invoice_id)
            if invoice and invoice.file_path:
                p = Path(invoice.file_path)
                if p.exists():
                    file_p = p
                    filename = invoice.file_name or p.name
        except Exception as exc:
            print(f"[Warning] DB error fetching invoice file: {exc}")

    if file_p is None:
        # Check in-memory store
        for inv in _in_memory_invoices:
            if inv.get("invoice_id") == invoice_id and inv.get("file_path"):
                p = Path(inv["file_path"])
                if p.exists():
                    file_p = p
                    filename = inv.get("file_name") or p.name
                    break

    if not file_p or not file_p.exists():
        raise HTTPException(status_code=404, detail="Invoice file not found on server disk.")

    return FileResponse(
        path=str(file_p),
        filename=filename or file_p.name,
    )

