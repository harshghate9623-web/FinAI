# FinDoc AI: Intelligent B2B Invoice Anomaly Detection & Review System

FinDoc AI is an end-to-end document processing and compliance verification system designed for Indian B2B tax invoices. The system combines OCR, deterministic statutory GST rule validation, Isolation Forest anomaly scoring, and duplicate detection with a reviewer dashboard.

---

## Key Features

- **Document Ingestion**: Supports PDF documents, scanned images (PNG, JPG), and raw OCR text with validation on file size and format.
- **OCR & Field Extraction**: Optical Character Recognition using Tesseract OCR with image thresholding; extracts GSTIN, dates, invoice numbers, amounts, and item breakdown.
- **Statutory GST Validation**: Deterministic rule engine verifying GSTIN structure, state code matching, CGST/SGST balance, arithmetic reconciliations, and stale dates.
- **Machine Learning Outlier Detection**: Isolation Forest model trained on multi-dimensional invoice financial distributions to flag irregular patterns.
- **Duplicate Detection**: Levenshtein and exact matching algorithms to detect duplicate or near-duplicate bill submissions.
- **Human-in-the-Loop Review Dashboard**: Streamlit web interface displaying the original bill document, centered decision outcomes, categorized anomaly lists, and one-click database review actions (`Checked` / `Marked for Review`).
- **Audit Trail & Persistence**: Fully relational PostgreSQL database storing invoices, line items, anomaly results, and complete audit history.

---

## Project Structure

```text
findoc-ai/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py              # FastAPI REST endpoints
│   ├── db/
│   │   ├── __init__.py
│   │   ├── connection.py        # SQLAlchemy session & engine management
│   │   ├── crud.py              # Database queries, updates & audit logging
│   │   ├── models.py            # ORM tables (invoices, items, anomalies, audit)
│   │   └── init_db.sql          # SQL schema definition reference
│   └── services/
│       ├── __init__.py
│       ├── anomaly.py           # Isolation Forest model inference wrapper
│       ├── duplicates.py        # Exact and fuzzy duplicate detector
│       ├── extract_fields.py    # Regex & heuristic invoice parser
│       ├── ingest.py            # Ingestion pipeline coordinator
│       ├── ocr.py               # Tesseract OCR & image preprocessing
│       ├── parse_invoice.py     # Structured JSON parsing & line-item extractor
│       └── validate_invoice.py  # GST rule verification engine
├── ui/
│   └── streamlit_app.py         # Streamlit reviewer dashboard & document viewer
├── scripts/
│   ├── train_model.py           # ML training script for Isolation Forest
│   └── migrate_schema.py        # Database schema verification & migration utility
├── artifacts/
│   └── isolation_forest_tuned.pkl # Serialized Isolation Forest model
├── sample_data/
│   ├── sample_ocr_invoice.txt
│   └── sample_ocr_invoice_extracted.json
├── uploads/                     # Upload folder (cleaned, with .gitkeep)
├── Dockerfile                   # Multi-stage Docker container specification
├── docker-compose.yml           # Multi-service setup (FastAPI + Streamlit)
├── requirements.txt             # Python project dependencies
├── .env.example                 # Example configuration template
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Prerequisites
- Python 3.8 to 3.11
- Tesseract OCR (`tesseract-ocr` and `tesseract-ocr-eng`)
- Poppler (`poppler-utils`) for PDF rendering

### 2. Environment Setup
```bash
# Clone the repository and enter directory
cd findoc-ai

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # On Linux/macOS
# .\venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

### 3. Running Locally
You can run both services together with a single command, or run them in separate terminals:

**Option A: Single command (Recommended)**
```bash
# From findoc-ai/ directory with venv activated:
python run_app.py

# Or on Windows, double-click:
start_local.bat
```

**Option B: Separate terminals**
```bash
# Terminal 1: Start FastAPI backend (Port 8000)
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start Streamlit dashboard (Port 8501)
streamlit run ui/streamlit_app.py
```

- **Dashboard UI**: [http://localhost:8501](http://localhost:8501)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 4. Running with Docker
```bash
docker-compose up --build
```

---

## Core REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health and database connectivity check |
| `POST` | `/upload` | Upload invoice file (PDF/image) and run full pipeline |
| `POST` | `/process-text` | Parse raw OCR text directly into structured records |
| `GET` | `/invoices` | List recently processed invoices with anomaly findings |
| `PATCH` | `/invoices/{id}/status` | Update reviewer status (`Checked` / `Marked for Review`) |
| `GET` | `/invoices/{id}/file` | Fetch original uploaded invoice bill file |
| `GET` | `/report` | Export flagged anomaly records as CSV |
