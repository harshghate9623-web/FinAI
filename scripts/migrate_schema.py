"""
scripts/migrate_schema.py
Adds missing columns to the live PostgreSQL database so the SQLAlchemy model
and remote table are in sync.
Run once: python scripts/migrate_schema.py
"""
from app.db.connection import engine
from sqlalchemy import text, inspect

MIGRATIONS = [
    # invoices table
    "ALTER TABLE invoices ADD COLUMN IF NOT EXISTS source_currency VARCHAR(10) DEFAULT 'INR'",
    "ALTER TABLE invoices ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW()",
    # invoice_items table
    "ALTER TABLE invoice_items ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW()",
]

with engine.begin() as conn:
    for sql in MIGRATIONS:
        print(f"Running: {sql}")
        conn.execute(text(sql))
        print("  OK")

print("\nVerifying columns now in DB...")
inspector = inspect(engine)

checks = {
    "invoices":      ["source_currency", "updated_at"],
    "invoice_items": ["created_at"],
}
all_ok = True
for table, expected_cols in checks.items():
    db_cols = {c["name"] for c in inspector.get_columns(table)}
    for col in expected_cols:
        status = "OK" if col in db_cols else "STILL MISSING"
        if status != "OK":
            all_ok = False
        print(f"  {table}.{col}: {status}")

print("\nAll columns present!" if all_ok else "\nWARNING: Some columns are still missing.")
