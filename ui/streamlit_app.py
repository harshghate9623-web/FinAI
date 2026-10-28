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
    .invoice-selector-box {
        background: linear-gradient(135deg, rgba(13, 110, 253, 0.08) 0%, rgba(13, 110, 253, 0.02) 100%);
        border: 1px solid rgba(13, 110, 253, 0.35);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .invoice-badge-count {
        display: inline-block;
        background-color: #0d6efd;
        color: white;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.15rem 0.65rem;
        border-radius: 12px;
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
if "selected_invoice_idx" not in st.session_state:
    st.session_state["selected_invoice_idx"] = 0
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
                        st.session_state["selected_invoice_idx"] = 0
                        st.session_state["uploaded_file_bytes"] = file_bytes
                        st.session_state["uploaded_file_name"] = uploaded_file.name
                        total_invs = st.session_state["invoice_result"].get("total_invoices", 1)
                        if total_invs > 1:
                            st.success(f"Multi-invoice PDF detected! Separated {total_invs} pages into distinct invoices.")
                        else:
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
                        st.session_state["selected_invoice_idx"] = 0
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
    # Handle both multi-invoice array and legacy single-invoice payload
    invoices = data.get("invoices")
    if not invoices or not isinstance(invoices, list):
        invoices = [data]

    total_detected = len(invoices)

    # Ensure selected index is valid
    if "selected_invoice_idx" not in st.session_state or st.session_state["selected_invoice_idx"] >= total_detected:
        st.session_state["selected_invoice_idx"] = 0

    selected_idx = st.session_state["selected_invoice_idx"]
    current_invoice = invoices[selected_idx]

    fields = current_invoice.get("extracted_fields") or current_invoice
    status_val = current_invoice.get("review_status", "Pending Review")
    inv_id = current_invoice.get("invoice_id")
    file_name = current_invoice.get("file_name") or st.session_state.get("uploaded_file_name") or "invoice"
    page_num = current_invoice.get("page_number", selected_idx + 1)

    # Resolve document bytes for currently selected invoice
    file_bytes = None
    if current_invoice.get("file_bytes_base64"):
        try:
            file_bytes = base64.b64decode(current_invoice["file_bytes_base64"])
        except Exception:
            pass

    if file_bytes is None and current_invoice.get("file_path"):
        fp = Path(current_invoice["file_path"])
        if fp.exists():
            try:
                file_bytes = fp.read_bytes()
            except Exception:
                pass

    if file_bytes is None and inv_id:
        try:
            f_resp = requests.get(f"{API_BASE}/invoices/{inv_id}/file", timeout=5)
            if f_resp.ok:
                file_bytes = f_resp.content
        except Exception:
            pass

    if file_bytes is None:
        file_bytes = st.session_state.get("uploaded_file_bytes")

    total_val = fields.get("total_amount")
    total_str = f"₹{float(total_val):,.2f}" if total_val is not None else "N/A"

    st.markdown("---")

    # =========================================================================
    # 🎯 MIDDLE SCREEN: Primary Review Outcome (Left) & Detected Invoice Selector (Right)
    # =========================================================================
    mid_header_left, mid_header_right = st.columns([5, 4], gap="large")

    with mid_header_left:
        st.markdown("<h2 style='margin-bottom: 0.2rem;'>🎯 Primary Review Outcome</h2>", unsafe_allow_html=True)

        if total_detected > 1:
            st.markdown(
                f"<div style='margin-bottom: 0.75rem; font-size: 0.95rem; color: #0d6efd; font-weight: 500;'>"
                f"📑 <strong>Multi-Invoice Document:</strong> Viewing <strong>Invoice {selected_idx + 1} of {total_detected}</strong> "
                f"(Page {page_num})"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Standard single invoice verification result.")

        # Status badge for currently selected invoice
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
            <div style="margin-bottom: 0.85rem;">
                <span class="status-badge {badge_class}">{badge_icon} {id_str}{status_val}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with mid_header_right:
        # RIGHT SIDE OF THE MIDDLE SCREEN: The Detected Invoice Option Selector
        st.markdown(
            f"""
            <div class="invoice-selector-box">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span style="font-weight: 700; font-size: 1.05rem; color: var(--text-color, inherit);">📑 Select Detected Invoice</span>
                    <span class="invoice-badge-count">{total_detected} Detected</span>
                </div>
                <div style="font-size: 0.84rem; color: #6c757d; margin-bottom: 0.4rem;">
                    Select an invoice to inspect its individual validation, preview, and findings:
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        def format_invoice_option(idx: int) -> str:
            inv_item = invoices[idx]
            p_num = inv_item.get("page_number", idx + 1)
            f_item = inv_item.get("extracted_fields") or inv_item
            inv_no = f_item.get("invoice_number") or "N/A"
            v_name = f_item.get("vendor_name") or "Unknown Vendor"
            v_clean = (v_name[:18] + "...") if len(v_name) > 20 else v_name
            tot = f_item.get("total_amount")
            tot_str = f"₹{float(tot):,.2f}" if tot is not None else "₹0.00"
            stat = inv_item.get("review_status", "")
            icon = "✅" if stat in ["Checked", "Passed Initial Checks"] and not inv_item.get("final_anomaly") else ("⚠️" if inv_item.get("final_anomaly") or "Review" in stat or "Flagged" in stat else "📄")
            return f"{icon} Page {p_num}: {v_clean} (Inv #{inv_no}) — {tot_str}"

        selected_dropdown = st.selectbox(
            "Select Detected Invoice:",
            options=list(range(total_detected)),
            index=selected_idx,
            format_func=format_invoice_option,
            key="detected_invoice_selector_dropdown",
            label_visibility="collapsed",
        )

        if selected_dropdown != selected_idx:
            st.session_state["selected_invoice_idx"] = selected_dropdown
            st.rerun()

        if total_detected > 1:
            nav_c1, nav_c2, nav_c3 = st.columns([1, 2, 1])
            with nav_c1:
                if st.button("◀ Prev", disabled=(selected_idx == 0), key="prev_inv_btn", use_container_width=True):
                    st.session_state["selected_invoice_idx"] = max(0, selected_idx - 1)
                    st.rerun()
            with nav_c2:
                st.markdown(
                    f"<div style='text-align: center; font-size: 0.85rem; padding-top: 0.4rem; color: #6c757d;'>"
                    f"Invoice {selected_idx + 1} of {total_detected}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with nav_c3:
                if st.button("Next ▶", disabled=(selected_idx == total_detected - 1), key="next_inv_btn", use_container_width=True):
                    st.session_state["selected_invoice_idx"] = min(total_detected - 1, selected_idx + 1)
                    st.rerun()

    # -------------------------------------------------------------------------
    # Middle Screen Metrics & Anomaly Breakdown for Selected Invoice
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Amount", total_str, help="Invoice Grand Total for this detected invoice")
    with m2:
        st.metric("Rule Validation", "Passed ✅" if not current_invoice.get("rule_anomaly") else "Flagged ❌")
    with m3:
        st.metric("ML Anomaly Check", "Normal ✅" if not current_invoice.get("isolation_anomaly") else "Outlier ❌")
    with m4:
        dups = current_invoice.get("duplicates", [])
        st.metric("Duplicate Check", "Clear ✅" if not dups else f"{len(dups)} Match(es) ⚠️")

    # -------------------------------------------------------------------------
    # Organised List of Detected Anomalies for Selected Invoice
    # -------------------------------------------------------------------------
    st.markdown("### 🔍 Anomaly & Findings Breakdown")
    anomalies_list = extract_anomalies_list(current_invoice)

    if anomalies_list:
        st.markdown(f"**Total Flags Identified for this Invoice:** `{len(anomalies_list)}`")
        for idx_item, item in enumerate(anomalies_list, start=1):
            st.markdown(
                f"""
                <div class="anomaly-card">
                    <div class="anomaly-title">{item['icon']} #{idx_item} [{item['category']}] — {item['title']}</div>
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
    # LEFT COLUMN: Bill Preview & Review Action Buttons for Selected Invoice
    # -------------------------------------------------------------------------
    with col_bill_review:
        st.subheader("📄 Invoice Bill Document")
        if total_detected > 1:
            st.caption(f"Review document: **{file_name}** &bull; **Invoice {selected_idx + 1} of {total_detected} (Page {page_num})**")
        else:
            st.caption(f"Review document: **{file_name}**")

        # Render the bill
        if file_bytes:
            render_bill_preview(file_bytes, file_name)
        else:
            st.info("Original bill document preview unavailable. Raw OCR text is shown in the tabs.")

        # Action Buttons Directly Below the Bill
        st.markdown("#### ✍️ Human Review Decision")
        st.write(f"Confirm verification or flag **Invoice #{inv_id or (selected_idx + 1)}** for further auditing:")

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            if st.button("⚠️ Mark as review", type="secondary", use_container_width=True, key=f"btn_review_{selected_idx}"):
                new_status = "Marked for Review"
                if inv_id:
                    try:
                        res = requests.patch(
                            f"{API_BASE}/invoices/{inv_id}/status",
                            json={"review_status": new_status, "performed_by": "reviewer"},
                            timeout=8,
                        )
                        if res.ok:
                            st.toast(f"⚠️ Invoice #{inv_id} status updated to 'Marked for Review' in database!", icon="⚠️")
                        else:
                            st.error(f"Failed to update database: {res.text}")
                    except Exception as err:
                        st.error(f"API request failed: {err}")
                else:
                    st.toast(f"Updated Invoice #{selected_idx + 1} review status in-memory", icon="⚠️")

                current_invoice["review_status"] = new_status
                if "invoices" in st.session_state["invoice_result"]:
                    st.session_state["invoice_result"]["invoices"][selected_idx]["review_status"] = new_status
                if selected_idx == 0:
                    st.session_state["invoice_result"]["review_status"] = new_status
                st.rerun()

        with btn_col2:
            if st.button("✅ Checked", type="primary", use_container_width=True, key=f"btn_checked_{selected_idx}"):
                new_status = "Checked"
                if inv_id:
                    try:
                        res = requests.patch(
                            f"{API_BASE}/invoices/{inv_id}/status",
                            json={"review_status": new_status, "performed_by": "reviewer"},
                            timeout=8,
                        )
                        if res.ok:
                            st.toast(f"✅ Invoice #{inv_id} status updated to 'Checked' in database!", icon="✅")
                        else:
                            st.error(f"Failed to update database: {res.text}")
                    except Exception as err:
                        st.error(f"API request failed: {err}")
                else:
                    st.toast(f"Updated Invoice #{selected_idx + 1} review status in-memory", icon="✅")

                current_invoice["review_status"] = new_status
                if "invoices" in st.session_state["invoice_result"]:
                    st.session_state["invoice_result"]["invoices"][selected_idx]["review_status"] = new_status
                if selected_idx == 0:
                    st.session_state["invoice_result"]["review_status"] = new_status
                st.rerun()

        # Download bill button
        if file_bytes:
            st.download_button(
                f"📥 Download Invoice Bill File ({file_name})",
                data=file_bytes,
                file_name=file_name,
                use_container_width=True,
                key=f"dl_bill_btn_{selected_idx}",
            )

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: Structured Data & Minimized Extracted Fields for Selected Invoice
    # -------------------------------------------------------------------------
    with col_data_view:
        st.subheader("📦 Parsed Data & Export")

        struct_json = current_invoice.get("structured_json")
        json_bytes = json.dumps(struct_json, indent=2, ensure_ascii=False) if struct_json else ""

        if json_bytes:
            st.download_button(
                f"⬇️ Download Structured JSON (Invoice #{inv_id or (selected_idx + 1)})",
                data=json_bytes,
                file_name=f"{str(file_name).replace('.txt', '').replace('.pdf', '')}_extracted.json",
                mime="application/json",
                use_container_width=True,
                key=f"dl_json_btn_{selected_idx}",
            )

        tab_items, tab_json, tab_ocr = st.tabs(["📑 Parsed Line Items", "📄 Structured JSON", "📝 Raw OCR Text"])

        with tab_items:
            line_items = current_invoice.get("line_items", [])
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
            ocr_text = fields.get("ocr_text", "") or current_invoice.get("ocr_text", "")
            st.text_area("Raw Extracted OCR Text", value=ocr_text, height=260, disabled=True, key=f"ocr_text_area_{selected_idx}")

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

            if current_invoice.get("missing_fields"):
                st.warning("⚠️ Incomplete fields: " + ", ".join(current_invoice["missing_fields"]))
