-- =============================================================================
-- FinDoc AI — Complete PostgreSQL Database Schema (Aiven & DBeaver Ready)
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS invoices (
    invoice_id                  BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    invoice_number              VARCHAR(100),
    vendor_name                 VARCHAR(255),
    gstin                       VARCHAR(15),
    invoice_date                DATE,
    vendor_address              TEXT,
    extracted_vendor_state      VARCHAR(100),
    extracted_vendor_state_code VARCHAR(2),
    
    subtotal                    NUMERIC(14,2) DEFAULT 0.00,
    cgst_amount                 NUMERIC(14,2) DEFAULT 0.00,
    sgst_amount                 NUMERIC(14,2) DEFAULT 0.00,
    igst_amount                 NUMERIC(14,2) DEFAULT 0.00,
    tax_amount                  NUMERIC(14,2) DEFAULT 0.00,
    discount_amount             NUMERIC(14,2) DEFAULT 0.00,
    total_amount                NUMERIC(14,2) DEFAULT 0.00,
    
    item_count                  INTEGER DEFAULT 0,
    quantity                    NUMERIC(14,3) DEFAULT 0.000,
    
    file_name                   VARCHAR(255),
    file_path                   TEXT,
    page_count                  INTEGER DEFAULT 1,
    source_currency             VARCHAR(10) DEFAULT 'INR',
    ocr_text                    TEXT,
    extracted_json              JSONB,
    
    review_status               VARCHAR(50) DEFAULT 'Pending Manual Review',
    uploaded_at                 TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at                 TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS invoice_items (
    item_id                     BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    invoice_id                  BIGINT NOT NULL REFERENCES invoices(invoice_id) ON DELETE CASCADE,
    item_order                  INTEGER DEFAULT 1,
    description                 TEXT,
    hsn_sac_code                VARCHAR(20),
    quantity                    NUMERIC(14,3) DEFAULT 1.000,
    unit_price                  NUMERIC(14,2) DEFAULT 0.00,
    line_total                  NUMERIC(14,2) DEFAULT 0.00,
    tax_rate                    NUMERIC(6,3) DEFAULT 18.000,
    created_at                  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS anomaly_results (
    result_id                   BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    invoice_id                  BIGINT NOT NULL REFERENCES invoices(invoice_id) ON DELETE CASCADE,
    rule_anomaly                BOOLEAN DEFAULT FALSE,
    isolation_anomaly           BOOLEAN DEFAULT FALSE,
    final_anomaly               BOOLEAN DEFAULT FALSE,
    model_input_invalid         BOOLEAN DEFAULT FALSE,
    isolation_score             NUMERIC(10,6),
    rule_flags_text             TEXT,
    model_version               VARCHAR(100) DEFAULT 'isolation_forest_v1',
    checked_at                  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS duplicates (
    duplicate_id                BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    invoice_id_1                BIGINT NOT NULL REFERENCES invoices(invoice_id) ON DELETE CASCADE,
    invoice_id_2                BIGINT NOT NULL REFERENCES invoices(invoice_id) ON DELETE CASCADE,
    match_type                  VARCHAR(50) NOT NULL,
    similarity_score            NUMERIC(5,4),
    matched_fields              TEXT,
    review_status               VARCHAR(50) DEFAULT 'Pending Review',
    detected_at                 TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT no_self_match CHECK (invoice_id_1 <> invoice_id_2)
);

CREATE TABLE IF NOT EXISTS audit_logs (
    log_id                      BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    invoice_id                  BIGINT REFERENCES invoices(invoice_id) ON DELETE SET NULL,
    action                      VARCHAR(100) NOT NULL,
    performed_by                VARCHAR(255) DEFAULT 'system',
    details                     JSONB,
    created_at                  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_invoices_gstin ON invoices(gstin);
CREATE INDEX IF NOT EXISTS idx_invoices_invoice_number ON invoices(invoice_number);
CREATE INDEX IF NOT EXISTS idx_invoices_date ON invoices(invoice_date);
CREATE INDEX IF NOT EXISTS idx_invoices_status ON invoices(review_status);
CREATE INDEX IF NOT EXISTS idx_invoices_uploaded_at ON invoices(uploaded_at DESC);
CREATE INDEX IF NOT EXISTS idx_invoice_items_invoice_id ON invoice_items(invoice_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_final ON anomaly_results(final_anomaly);
CREATE INDEX IF NOT EXISTS idx_anomaly_invoice_id ON anomaly_results(invoice_id);
CREATE INDEX IF NOT EXISTS idx_duplicates_pair ON duplicates(invoice_id_1, invoice_id_2);
CREATE INDEX IF NOT EXISTS idx_duplicates_match_type ON duplicates(match_type);
CREATE INDEX IF NOT EXISTS idx_audit_invoice_id ON audit_logs(invoice_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);
