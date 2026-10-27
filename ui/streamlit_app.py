"""
ui/streamlit_app.py
FinDoc AI — Streamlit review dashboard.
Calls the FastAPI backend to process invoices, displays the bill preview,
and allows human reviewers to update database status ("Checked" or "Marked for Review").
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Optional
import pandas as pd
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="FinDoc AI — Invoice Review",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for centered cards, badges, and organized anomaly list items
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .sub-caption {
        text-align: center;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.45rem 1.4rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.15rem;
        text-align: center;
    }
    .status-passed {
        background-color: #d1e7dd;
        color: #0f5132;
        border: 1px solid #badbcc;
    }
    .status-flagged {
        background-color: #fff3cd;
        color: #664d03;
        border: 1px solid #ffecb5;
    }
    .status-error {
        background-color: #f8d7da;
        color: #842029;
        border: 1px solid #f5c2c7;
    }
    .anomaly-card {
        border-left: 4px solid #dc3545;
        background-color: rgba(220, 53, 69, 0.06);
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
        border-radius: 4px 8px 8px 4px;
    }
    .anomaly-title {
        font-weight: 600;
        color: #dc3545;
        margin-bottom: 0.2rem;
    }
    .anomaly-desc {
        color: inherit;
        font-size: 0.95rem;
    }
    .anomaly-clean {
        border-left: 4px solid #198754;
        background-color: rgba(25, 135, 84, 0.06);
        padding: 1rem;
        border-radius: 4px 8px 8px 4px;
    }
    .doc-viewer-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 8px;
        padding: 0.75rem;
        background-color: var(--secondary-background-color);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='main-title'>📄 FinDoc AI — Invoice Review Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-caption'>Upload Indian B2B invoice PDF/image or OCR text to parse, validate, detect anomalies, and review.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — health check & actions
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚡ System Health")
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        if resp.ok:
            info = resp.json()
            st.success("API Online ✅")
            st.caption(f"Database: **{info.get('database', 'connected')}**")
        else:
            st.error("API returned an error")
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach API server (http://localhost:8000)")

    st.markdown("---")
    st.markdown("**API Documentation:** [Swagger UI](http://localhost:8000/docs)")

    st.markdown("---")
    st.subheader("📊 Export Reports")
    if st.button("📥 Download Flagged Exceptions (CSV)", use_container_width=True):
        try:
            r_report = requests.get(f"{API_BASE}/report", timeout=10)
            if r_report.ok:
                st.download_button(
                    "💾 Save CSV",
                    data=r_report.content,
                    file_name="invoice_exceptions_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("No flagged exception invoices found to export.")
        except Exception as exc:
            st.error(f"Download error: {exc}")


# ---------------------------------------------------------------------------
# Session State initialization for persistent display
# ---------------------------------------------------------------------------
if "invoice_result" not in st.session_state:
    st.session_state["invoice_result"] = None
if "uploaded_file_bytes" not in st.session_state:
    st.session_state["uploaded_file_bytes"] = None
if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = None


# ---------------------------------------------------------------------------
# Input section: Upload or Paste
# ---------------------------------------------------------------------------
input_mode = st.radio(
    "Choose Input Method:",
    ["📁 Upload Invoice / OCR File", "✍️ Paste Plain OCR Text"],
    horizontal=True,
)

if input_mode == "📁 Upload Invoice / OCR File":
    uploaded_file = st.file_uploader(
        "Upload Invoice (PDF, PNG, JPG, JPEG, TIFF, BMP, TXT, JSON)",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "txt", "json"],
    )

    if uploaded_file is not None:
        st.info(f"Selected: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

        if st.button("🔍 Process Invoice", type="primary"):
            file_bytes = uploaded_file.getvalue()
            with st.spinner("Processing OCR, parsing structures, and running validation models..."):
                try:
                    response = requests.post(
                        f"{API_BASE}/upload",
                        files={"file": (uploaded_file.name, file_bytes)},
                        timeout=90,
                    )
                    if response.ok:
                        st.session_state["invoice_result"] = response.json()
                        st.session_state["uploaded_file_bytes"] = file_bytes
                        st.session_state["uploaded_file_name"] = uploaded_file.name
                        st.success("Invoice analyzed and persisted to database!")
                    else:
                        st.error(f"API error {response.status_code}: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach backend API. Make sure FastAPI server is running.")

else:
    pasted_text = st.text_area(
        "Paste OCR plain text output here:",
        height=180,
        placeholder="e.g.\nTAX INVOICE\nVendor: Acme Corp\nGSTIN: 27AAPFU0939F1ZV\nInvoice No: INV-2024-001\nDate: 15/04/2024\nSubtotal: 2500.00\nTotal: 2950.00",
    )
    custom_filename = st.text_input("Invoice reference filename (optional):", value="pasted_invoice_ocr.txt")

    if st.button("🔍 Parse Plain Text & Run Pipeline", type="primary"):
        if not pasted_text.strip():
            st.warning("Please paste OCR text before proceeding.")
        else:
            with st.spinner("Structuring text and executing validation pipeline..."):
                try:
                    response = requests.post(
                        f"{API_BASE}/process-text",
                        json={"ocr_text": pasted_text, "filename": custom_filename},
                        timeout=90,
                    )
                    if response.ok:
                        st.session_state["invoice_result"] = response.json()
                        st.session_state["uploaded_file_bytes"] = pasted_text.encode("utf-8")
                        st.session_state["uploaded_file_name"] = custom_filename or "pasted_invoice_ocr.txt"
                        st.success("Plain text parsed and saved to database!")
                    else:
                        st.error(f"API error {response.status_code}: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach backend API. Make sure FastAPI server is running.")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def extract_anomalies_list(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract and structure all detected anomalies into clean, organized items."""
    anomalies: List[Dict[str, str]] = []

    # 1. Rule Engine Violations
    rule_flags_raw = data.get("rule_flags", "") or ""
    if data.get("rule_anomaly") and rule_flags_raw:
        flags = [f.strip() for f in rule_flags_raw.split(";") if f.strip() and f.strip().lower() != "no rule issue detected"]
        for f in flags:
            anomalies.append({
                "category": "Deterministic Rule Violation",
                "icon": "⚠️",
                "title": f,
                "detail": "Failed GST validation rule check."
            })

    # 2. Isolation Forest Outlier
    if data.get("isolation_anomaly"):
        score = data.get("isolation_score")
        score_fmt = f"{float(score):.4f}" if score is not None else "N/A"
        anomalies.append({
            "category": "Statistical ML Outlier",
            "icon": "🤖",
            "title": f"Unusual Invoice Pattern Detected by Isolation Forest (Score: {score_fmt})",
            "detail": "Financial and quantity distribution deviates significantly from normal baseline profiles."
        })
    elif data.get("model_input_invalid"):
        anomalies.append({
            "category": "ML Model Warning",
            "icon": "ℹ️",
            "title": "Model Input Incomplete",
            "detail": "Some financial fields were missing; ML anomaly scoring was bypassed for this record."
        })

    # 3. Duplicate Invoices
    duplicates = data.get("duplicates", [])
    if duplicates:
        for d in duplicates:
            match_id = d.get("matched_with_invoice_id", "Existing")
            match_type = str(d.get("match_type", "Duplicate")).replace("_", " ").title()
            sim = d.get("similarity_score")
            sim_str = f" ({float(sim):.1%} similarity)" if sim is not None else ""
            fields_str = ", ".join(d.get("matched_fields", []))
            anomalies.append({
                "category": "Duplicate Detection",
                "icon": "👥",
                "title": f"{match_type} with Invoice #{match_id}{sim_str}",
                "detail": f"Matched attributes: {fields_str}" if fields_str else "Potential duplicate detected in database."
            })

    # 4. Critical Missing Fields
    missing_fields = data.get("missing_fields", [])
    if missing_fields:
        anomalies.append({
            "category": "Incomplete Extraction",
            "icon": "📋",
            "title": f"Missing Essential Fields: {', '.join(missing_fields)}",
            "detail": "Invoice document is missing required GST/Tax identifiers."
        })

    return anomalies


def render_bill_preview(file_bytes: bytes, file_name: str):
    """Render the bill preview based on file extension."""
    ext = file_name.split(".")[-1].lower() if "." in file_name else ""

    if ext in ["png", "jpg", "jpeg", "webp", "tiff", "bmp"]:
        st.image(file_bytes, caption=f"📄 {file_name}", use_container_width=True)
    elif ext == "pdf":
        base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
        pdf_display = (
            f'<iframe src="data:application/pdf;base64,{base64_pdf}#toolbar=0" '
            f'width="100%" height="480" type="application/pdf" '
            f'style="border: 1px solid rgba(128,128,128,0.25); border-radius: 8px;"></iframe>'
        )
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        try:
            text_preview = file_bytes.decode("utf-8", errors="replace")
        except Exception:
            text_preview = str(file_bytes)
        st.markdown(
            f"""
            <div style="
                background-color: var(--secondary-background-color);
                border: 1px solid rgba(128, 128, 128, 0.25);
                padding: 1rem;
                border-radius: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 0.86rem;
                white-space: pre-wrap;
                max-height: 480px;
                overflow-y: auto;
            ">{text_preview}</div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Results Display
# ---------------------------------------------------------------------------
data = st.session_state["invoice_result"]

if data:
    fields = data.get("extracted_fields", {})
    status_val = data.get("review_status", "Pending Review")
    inv_id = data.get("invoice_id")
    file_name = st.session_state.get("uploaded_file_name") or data.get("file_name") or "invoice"
    file_bytes = st.session_state.get("uploaded_file_bytes")

    # If bytes not in session state but we have an invoice_id, attempt to fetch from backend
    if file_bytes is None and inv_id:
        try:
            f_resp = requests.get(f"{API_BASE}/invoices/{inv_id}/file", timeout=5)
            if f_resp.ok:
                file_bytes = f_resp.content
                st.session_state["uploaded_file_bytes"] = file_bytes
        except Exception:
            pass

    total_val = fields.get("total_amount")
    total_str = f"₹{float(total_val):,.2f}" if total_val is not None else "N/A"

    st.markdown("---")

    # =========================================================================
    # 🎯 MAIN RESULT — CENTERED
    # =========================================================================
    left_spacer, center_col, right_spacer = st.columns([1, 8, 1])

    with center_col:
        st.markdown("<h2 style='text-align: center; margin-bottom: 0.5rem;'>🎯 Primary Review Outcome</h2>", unsafe_allow_html=True)

        # Centered Status Badge
        if status_val in ["Checked", "Passed Initial Checks"]:
            badge_class = "status-passed"
            badge_icon = "✅"
        elif "Review" in status_val or "Flagged" in status_val or "Pending" in status_val or "Duplicate" in status_val:
            badge_class = "status-flagged"
            badge_icon = "⚠️"
        else:
            badge_class = "status-error"
            badge_icon = "🔴"

        id_str = f"Invoice #{inv_id} &bull; " if inv_id else ""
        st.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 1.25rem;">
                <span class="status-badge {badge_class}">{badge_icon} {id_str}{status_val}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # High-level Metrics Row in Center
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Amount", total_str, help="Invoice Grand Total")
        with m2:
            st.metric("Rule Validation", "Passed ✅" if not data.get("rule_anomaly") else "Flagged ❌")
        with m3:
            st.metric("ML Anomaly Check", "Normal ✅" if not data.get("isolation_anomaly") else "Outlier ❌")
        with m4:
            dups = data.get("duplicates", [])
            st.metric("Duplicate Check", "Clear ✅" if not dups else f"{len(dups)} Match(es) ⚠️")

        # ---------------------------------------------------------------------
        # Organised List of Detected Anomalies
        # ---------------------------------------------------------------------
        st.markdown("### 🔍 Anomaly & Findings Breakdown")
        anomalies_list = extract_anomalies_list(data)

        if anomalies_list:
            st.markdown(f"**Total Flags Identified:** `{len(anomalies_list)}`")
            for idx, item in enumerate(anomalies_list, start=1):
                st.markdown(
                    f"""
                    <div class="anomaly-card">
                        <div class="anomaly-title">{item['icon']} #{idx} [{item['category']}] — {item['title']}</div>
                        <div class="anomaly-desc">{item['detail']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <div class="anomaly-clean">
                    <strong>🎉 Zero Anomalies Detected!</strong><br>
                    <span>This invoice satisfies all statutory GST arithmetic checks, matches normal statistical distributions, and has no registered duplicate records in the database.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # 📄 DOCUMENT REVIEW & ACTIONS (Left) + 📦 DATA & MINIMISED FIELDS (Right)
    # =========================================================================
    st.markdown("---")
    col_bill_review, col_data_view = st.columns([1, 1], gap="large")

    # -------------------------------------------------------------------------
    # LEFT COLUMN: Bill Preview & Review Action Buttons
    # -------------------------------------------------------------------------
    with col_bill_review:
        st.subheader("📄 Invoice Bill Document")
        st.caption(f"Review document: **{file_name}**")

        # Render the bill
        if file_bytes:
            render_bill_preview(file_bytes, file_name)
        else:
            st.info("Original bill document preview unavailable. Raw OCR text is shown in the tabs.")

        # Action Buttons Directly Below the Bill
        st.markdown("#### ✍️ Human Review Decision")
        st.write("Confirm invoice verification or flag for further auditing:")

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            if st.button("⚠️ Mark as review", type="secondary", use_container_width=True):
                new_status = "Marked for Review"
                if inv_id:
                    try:
                        res = requests.patch(
                            f"{API_BASE}/invoices/{inv_id}/status",
                            json={"review_status": new_status, "performed_by": "reviewer"},
                            timeout=8,
                        )
                        if res.ok:
                            st.session_state["invoice_result"]["review_status"] = new_status
                            st.toast("⚠️ Status updated to 'Marked for Review' in PostgreSQL database!", icon="⚠️")
                            st.rerun()
                        else:
                            st.error(f"Failed to update database: {res.text}")
                    except Exception as err:
                        st.error(f"API request failed: {err}")
                else:
                    st.session_state["invoice_result"]["review_status"] = new_status
                    st.toast("Updated review status in-memory", icon="⚠️")
                    st.rerun()

        with btn_col2:
            if st.button("✅ Checked", type="primary", use_container_width=True):
                new_status = "Checked"
                if inv_id:
                    try:
                        res = requests.patch(
                            f"{API_BASE}/invoices/{inv_id}/status",
                            json={"review_status": new_status, "performed_by": "reviewer"},
                            timeout=8,
                        )
                        if res.ok:
                            st.session_state["invoice_result"]["review_status"] = new_status
                            st.toast("✅ Status updated to 'Checked' in PostgreSQL database!", icon="✅")
                            st.rerun()
                        else:
                            st.error(f"Failed to update database: {res.text}")
                    except Exception as err:
                        st.error(f"API request failed: {err}")
                else:
                    st.session_state["invoice_result"]["review_status"] = new_status
                    st.toast("Updated review status in-memory", icon="✅")
                    st.rerun()

        # Download bill button
        if file_bytes:
            st.download_button(
                "📥 Download Original Bill File",
                data=file_bytes,
                file_name=file_name,
                use_container_width=True,
            )

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: Structured Data & Minimized Extracted Fields (Bottom-Right)
    # -------------------------------------------------------------------------
    with col_data_view:
        st.subheader("📦 Parsed Data & Export")

        struct_json = data.get("structured_json")
        json_bytes = json.dumps(struct_json, indent=2, ensure_ascii=False) if struct_json else ""

        if json_bytes:
            st.download_button(
                "⬇️ Download Structured JSON",
                data=json_bytes,
                file_name=f"{str(file_name).replace('.txt', '').replace('.pdf', '')}_extracted.json",
                mime="application/json",
                use_container_width=True,
            )

        tab_items, tab_json, tab_ocr = st.tabs(["📑 Parsed Line Items", "📄 Structured JSON", "📝 Raw OCR Text"])

        with tab_items:
            line_items = data.get("line_items", [])
            if line_items:
                df_items = pd.DataFrame(line_items)
                st.dataframe(df_items, use_container_width=True, height=280)
            else:
                st.info("No individual line items parsed from this invoice.")

        with tab_json:
            if struct_json:
                st.json(struct_json)
            else:
                st.info("No structured JSON available.")

        with tab_ocr:
            ocr_text = fields.get("ocr_text", "") or data.get("ocr_text", "")
            st.text_area("Raw Extracted OCR Text", value=ocr_text, height=260, disabled=True)

        # Minimized Extracted Fields in Bottom Right Corner
        st.markdown("---")
        st.subheader("📋 Extracted Fields")
        with st.expander("🔍 Extracted Fields Overview (Minimised)", expanded=False):
            st.caption("Expand to review raw extracted attributes:")

            display_keys = [
                ("Invoice Number", "invoice_number"),
                ("Invoice Date", "invoice_date"),
                ("Vendor Name", "vendor_name"),
                ("Vendor GSTIN", "gstin"),
                ("Subtotal", "subtotal"),
                ("CGST Amount", "cgst_amount"),
                ("SGST Amount", "sgst_amount"),
                ("IGST Amount", "igst_amount"),
                ("Tax Amount", "tax_amount"),
                ("Discount", "discount_amount"),
                ("Grand Total", "total_amount"),
                ("Item Count", "item_count"),
                ("Total Quantity", "quantity"),
            ]

            field_records = []
            for label, key in display_keys:
                val = fields.get(key)
                val_str = str(val) if val is not None and str(val).strip() != "" else "-"
                field_records.append({"Field": label, "Extracted Value": val_str})

            st.dataframe(pd.DataFrame(field_records), hide_index=True, use_container_width=True)

            if data.get("missing_fields"):
                st.warning("⚠️ Incomplete fields: " + ", ".join(data["missing_fields"]))
