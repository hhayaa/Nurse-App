
import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = str(Path(__file__).parent / "triage_hil.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS cases (
        case_id           TEXT PRIMARY KEY,
        ticket_number     TEXT NOT NULL,
        patient_symptoms  TEXT NOT NULL,
        status            TEXT DEFAULT 'processing',
        llm_urgency       TEXT,
        llm_reasoning     TEXT,
        llm_recommendation TEXT,
        llm_next_steps    TEXT,
        llm_sources       TEXT,
        rag_mode          TEXT,
        nurse_tier        TEXT,
        nurse_action      TEXT,
        nurse_notes       TEXT,
        nurse_timestamp   TEXT,
        final_tier        TEXT,
        booking_status    TEXT,
        booking_details   TEXT,
        created_at        TEXT,
        updated_at        TEXT
    )""")
    conn.commit()
    conn.close()

def insert_case(case: dict):
    conn = sqlite3.connect(DB_PATH)
    cols = ', '.join(case.keys())
    placeholders = ', '.join(['?'] * len(case))
    conn.execute(f'INSERT OR REPLACE INTO cases ({cols}) VALUES ({placeholders})',
                 list(case.values()))
    conn.commit()
    conn.close()

def update_case(case_id: str, updates: dict):
    conn = sqlite3.connect(DB_PATH)
    updates['updated_at'] = datetime.utcnow().isoformat()
    set_clause = ', '.join(f'{k} = ?' for k in updates)
    conn.execute(f'UPDATE cases SET {set_clause} WHERE case_id = ?',
                 list(updates.values()) + [case_id])
    conn.commit()
    conn.close()

def get_cases(status=None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if status:
        rows = conn.execute('SELECT * FROM cases WHERE status = ?', (status,)).fetchall()
    else:
        rows = conn.execute('SELECT * FROM cases ORDER BY created_at DESC').fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_case(case_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute('SELECT * FROM cases WHERE case_id = ?', (case_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_patient_cases(ticket_number):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT * FROM cases WHERE ticket_number = ? ORDER BY created_at DESC',
                        (ticket_number,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# Initialize on import
init_db()
