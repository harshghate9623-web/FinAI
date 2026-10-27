"""
app/db/models.py
SQLAlchemy ORM models corresponding to the PostgreSQL schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.connection import Base


class Invoice(Base):
    __tablename__ = "invoices"

    invoice_id = Column(BigInteger, primary_key=True, autoincrement=True)
    invoice_number = Column(String(100), index=True)
    vendor_name = Column(String(255))
    gstin = Column(String(15), index=True)
    invoice_date = Column(Date, index=True)
    vendor_address = Column(Text)
    extracted_vendor_state = Column(String(100))
    extracted_vendor_state_code = Column(String(2))

    # Financials
    subtotal = Column(Numeric(14, 2), default=0.0)
    cgst_amount = Column(Numeric(14, 2), default=0.0)
    sgst_amount = Column(Numeric(14, 2), default=0.0)
    igst_amount = Column(Numeric(14, 2), default=0.0)
    tax_amount = Column(Numeric(14, 2), default=0.0)
    discount_amount = Column(Numeric(14, 2), default=0.0)
    total_amount = Column(Numeric(14, 2), default=0.0)

    # Counts
    item_count = Column(Integer, default=0)
    quantity = Column(Numeric(14, 3), default=0.0)

    # Provenance
    file_name = Column(String(255))
    file_path = Column(Text)
    page_count = Column(Integer, default=1)
    source_currency = Column(String(10), default="INR")
    ocr_text = Column(Text)
    extracted_json = Column(JSONB)

    # Status & Timestamps
    review_status = Column(String(50), default="Pending Manual Review", index=True)
    uploaded_at = Column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    items = relationship("InvoiceItem", back_populates="invoice", cascade="all, delete-orphan")
    anomaly_result = relationship("AnomalyResult", back_populates="invoice", uselist=False, cascade="all, delete-orphan")


class InvoiceItem(Base):
    __tablename__ = "invoice_items"

    item_id = Column(BigInteger, primary_key=True, autoincrement=True)
    invoice_id = Column(BigInteger, ForeignKey("invoices.invoice_id", ondelete="CASCADE"), nullable=False, index=True)
    item_order = Column(Integer, default=1)
    description = Column(Text)
    hsn_sac_code = Column(String(20))
    quantity = Column(Numeric(14, 3), default=1.0)
    unit_price = Column(Numeric(14, 2), default=0.0)
    line_total = Column(Numeric(14, 2), default=0.0)
    tax_rate = Column(Numeric(6, 3), default=18.0)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    invoice = relationship("Invoice", back_populates="items")


class AnomalyResult(Base):
    __tablename__ = "anomaly_results"

    result_id = Column(BigInteger, primary_key=True, autoincrement=True)
    invoice_id = Column(BigInteger, ForeignKey("invoices.invoice_id", ondelete="CASCADE"), nullable=False, index=True)
    rule_anomaly = Column(Boolean, default=False)
    isolation_anomaly = Column(Boolean, default=False)
    final_anomaly = Column(Boolean, default=False, index=True)
    model_input_invalid = Column(Boolean, default=False)
    isolation_score = Column(Numeric(10, 6), nullable=True)
    rule_flags_text = Column(Text, default="")
    model_version = Column(String(100), default="isolation_forest_v1")
    checked_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    invoice = relationship("Invoice", back_populates="anomaly_result")


class Duplicate(Base):
    __tablename__ = "duplicates"

    duplicate_id = Column(BigInteger, primary_key=True, autoincrement=True)
    invoice_id_1 = Column(BigInteger, ForeignKey("invoices.invoice_id", ondelete="CASCADE"), nullable=False)
    invoice_id_2 = Column(BigInteger, ForeignKey("invoices.invoice_id", ondelete="CASCADE"), nullable=False)
    match_type = Column(String(50), nullable=False, index=True)
    similarity_score = Column(Numeric(5, 4), nullable=True)
    matched_fields = Column(Text, default="")
    review_status = Column(String(50), default="Pending Review")
    detected_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("invoice_id_1 <> invoice_id_2", name="no_self_match"),
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    log_id = Column(BigInteger, primary_key=True, autoincrement=True)
    invoice_id = Column(BigInteger, ForeignKey("invoices.invoice_id", ondelete="SET NULL"), nullable=True, index=True)
    action = Column(String(100), nullable=False, index=True)
    performed_by = Column(String(255), default="system")
    details = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
