import streamlit as st
import sqlite3
import json
import uuid
import random
import re
import os
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================================
# DATABASE
# ============================================================================
DB_PATH = "triage_hil.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS cases (
        case_id TEXT PRIMARY KEY, ticket_number TEXT, patient_symptoms TEXT,
        status TEXT DEFAULT 'processing',
        llm_urgency TEXT, llm_reasoning TEXT, llm_recommendation TEXT,
        llm_next_steps TEXT, llm_sources TEXT, rag_mode TEXT,
        nurse_tier TEXT, nurse_action TEXT, nurse_notes TEXT, nurse_timestamp TEXT,
        final_tier TEXT, booking_status TEXT, booking_details TEXT,
        created_at TEXT, updated_at TEXT
    )""")
    conn.commit()
    conn.close()

def db_insert(case):
    conn = sqlite3.connect(DB_PATH)
    cols = ', '.join(case.keys())
    phs = ', '.join(['?'] * len(case))
    conn.execute(f'INSERT OR REPLACE INTO cases ({cols}) VALUES ({phs})', list(case.values()))
    conn.commit()
    conn.close()

def db_update(case_id, updates):
    conn = sqlite3.connect(DB_PATH)
    updates['updated_at'] = datetime.utcnow().isoformat()
    s = ', '.join(f'{k} = ?' for k in updates)
    conn.execute(f'UPDATE cases SET {s} WHERE case_id = ?', list(updates.values()) + [case_id])
    conn.commit()
    conn.close()

def db_get_all(status=None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if status:
        rows = conn.execute('SELECT * FROM cases WHERE status=? ORDER BY created_at DESC', (status,)).fetchall()
    else:
        rows = conn.execute('SELECT * FROM cases ORDER BY created_at DESC').fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_get_one(case_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute('SELECT * FROM cases WHERE case_id=?', (case_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def db_get_by_ticket(ticket):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT * FROM cases WHERE ticket_number=? ORDER BY created_at DESC', (ticket,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

init_db()

# ============================================================================
# TRIAGE (Gemini or demo fallback)
# ============================================================================
def run_triage(symptoms):
    try:
        from google import genai
        from google.genai import types
        client = genai.Client()
        SYSTEM = (
            'You are an ER triage assistant. Classify urgency.\n'
            'Categories: Urgent / Routine / Self-care\n'
            'Output:\nUrgency: <tier>\n\nReasoning:\n<explanation>\n\n'
            'Recommendation:\n<action>\n\nNext steps:\n- <bullets>\n\nSources:\nNone'
        )
        resp = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f'Patient symptoms:\n"""{symptoms}"""',
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM, temperature=0.0,
                max_output_tokens=900,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return parse_triage(resp.text or '')
    except Exception:
        pass

    # Fallback demo mode
    s = symptoms.lower()
    urgent = ['chest pain', 'stroke', 'breathing', 'unconscious', 'bleeding', 'swelling throat', 'suicidal']
    routine = ['cough', 'burning urin', 'back pain', 'rash', 'headache', 'depressed', 'knee pain']
    if any(k in s for k in urgent):
        return {'urgency': 'Urgent', 'reasoning': 'Red-flag symptoms detected.',
                'recommendation': 'Immediate ER evaluation', 'next_steps': '- Emergency protocol',
                'sources': 'None', 'rag_mode': 'demo'}
    if any(k in s for k in routine):
        return {'urgency': 'Routine', 'reasoning': 'Non-emergency symptoms needing follow-up.',
                'recommendation': 'Schedule appointment in 1-2 weeks', 'next_steps': '- Monitor symptoms',
                'sources': 'None', 'rag_mode': 'demo'}
    return {'urgency': 'Self-care', 'reasoning': 'Self-limiting symptoms, no red flags.',
            'recommendation': 'Home management', 'next_steps': '- Rest and monitor',
            'sources': 'None', 'rag_mode': 'demo'}

def parse_triage(text):
    result = {'urgency': 'Unknown', 'reasoning': '', 'recommendation': '',
              'next_steps': '', 'sources': '', 'rag_mode': 'gemini'}
    m = re.search(r'Urgency:\s*(Urgent|Routine|Self-care|Self care)', text, re.IGNORECASE)
    if m:
        u = m.group(1).strip()
        result['urgency'] = 'Self-care' if 'self' in u.lower() else u.capitalize()
    for section in re.split(r'\n(?=(?:Reasoning|Recommendation|Next [Ss]teps|Sources):)', text):
        s = section.strip()
        if s.lower().startswith('reasoning:'):
            result['reasoning'] = s.split(':', 1)[1].strip()
        elif s.lower().startswith('recommendation:'):
            result['recommendation'] = s.split(':', 1)[1].strip()
        elif 'next steps' in s.lower()[:12]:
            result['next_steps'] = s.split(':', 1)[1].strip()
        elif s.lower().startswith('sources:'):
            result['sources'] = s.split(':', 1)[1].strip()
    return result

# ============================================================================
# BOOKING
# ============================================================================
def book_action(case_id, tier):
    if tier == 'Urgent':
        return {'status': 'urgent_referral', 'doctor': 'Dr. Khalil (Emergency)',
                'time': (datetime.now() + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M'),
                'dept': 'Emergency', 'booking_id': f'UR-{uuid.uuid4().hex[:6]}'}
    elif tier == 'Routine':
        d = random.randint(2, 10)
        return {'status': 'booked', 'doctor': 'Dr. Ahmed (Internal Medicine)',
                'time': (datetime.now() + timedelta(days=d)).strftime('%Y-%m-%d %H:%M'),
                'dept': 'Internal Medicine', 'booking_id': f'BK-{uuid.uuid4().hex[:6]}'}
    return {'status': 'self_care_issued', 'doctor': None, 'time': None, 'dept': None,
            'booking_id': f'SC-{uuid.uuid4().hex[:6]}',
            'guidance': 'Rest, monitor symptoms. Return if worsening.'}

# ============================================================================
# STREAMLIT APP
# ============================================================================
st.set_page_config(page_title="Triage Decision Support", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Dashboard:", ["Patient Dashboard", "Nurse Dashboard"])

# ---------- PATIENT DASHBOARD ------------------------------------------------
if page == "Patient Dashboard":
    st.title("Patient Triage Dashboard")
    st.markdown("Enter your ticket number and describe your symptoms.")

    with st.form("patient_form"):
        ticket = st.text_input("Ticket Number", placeholder="e.g. 12")
        symptoms = st.text_area("Describe your symptoms",
                                placeholder="I have been feeling...", height=150)
        submitted = st.form_submit_button("Submit", type="primary")

    if submitted and ticket and symptoms:
        case_id = f"CASE-{uuid.uuid4().hex[:8]}"
        with st.spinner("Analyzing your symptoms..."):
            triage = run_triage(symptoms)
            db_insert({
                'case_id': case_id, 'ticket_number': ticket,
                'patient_symptoms': symptoms, 'status': 'pending',
                'llm_urgency': triage['urgency'], 'llm_reasoning': triage['reasoning'],
                'llm_recommendation': triage['recommendation'],
                'llm_next_steps': triage.get('next_steps', ''),
                'llm_sources': triage.get('sources', ''),
                'rag_mode': triage.get('rag_mode', ''),
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
            })
        st.success(f"Submitted! Case ID: **{case_id}**. A nurse will review shortly.")

    st.markdown("---")
    st.subheader("Check Your Results")
    check = st.text_input("Enter ticket number to check:", key="chk")
    if check:
        cases = db_get_by_ticket(check)
        if not cases:
            st.info("No cases found.")
        for c in cases:
            if c['status'] == 'reviewed':
                tier = c.get('final_tier', 'Unknown')
                colors = {'Urgent': 'red', 'Routine': 'orange', 'Self-care': 'green'}
                st.markdown(f"### Case {c['case_id']}")
                st.markdown(f"**Final Decision:** :{colors.get(tier, 'gray')}[{tier}]")
                bd = {}
                try:
                    bd = json.loads(c.get('booking_details', '{}') or '{}')
                except:
                    pass
                if bd.get('doctor'):
                    st.markdown(f"**Doctor:** {bd['doctor']}")
                if bd.get('time'):
                    st.markdown(f"**Appointment:** {bd['time']}")
                if bd.get('dept'):
                    st.markdown(f"**Department:** {bd['dept']}")
                if c.get('llm_recommendation'):
                    st.markdown(f"**Recommendation:** {c['llm_recommendation']}")
            elif c['status'] == 'pending':
                st.warning(f"Case {c['case_id']} — Waiting for nurse review...")
            else:
                st.info(f"Case {c['case_id']} — Processing...")

# ---------- NURSE DASHBOARD --------------------------------------------------
elif page == "Nurse Dashboard":
    st.title("Nurse Triage Review Dashboard")

    pending = db_get_all(status='pending')
    reviewed = db_get_all(status='reviewed')
    st.markdown(f"**Pending:** {len(pending)}  |  **Reviewed:** {len(reviewed)}")

    if not pending:
        st.info("No pending cases. Waiting for patient submissions...")
        if st.button("Refresh"):
            st.rerun()
    else:
        opts = {f"{c['case_id']} — Ticket #{c['ticket_number']}": c['case_id'] for c in pending}
        sel = st.selectbox("Select a case:", list(opts.keys()))
        case = db_get_one(opts[sel])

        if case:
            col1, col2 = st.columns([3, 2])

            with col1:
                st.subheader("Patient Symptoms")
                st.text_area("", value=case['patient_symptoms'], height=120, disabled=True)
                st.subheader("LLM Assessment")
                tier = case.get('llm_urgency', 'Unknown')
                tc = {'Urgent': 'red', 'Routine': 'orange', 'Self-care': 'green'}
                st.markdown(f"**Urgency:** :{tc.get(tier, 'gray')}[{tier}]")
                st.markdown(f"**Mode:** `{case.get('rag_mode', '')}`")
                if case.get('llm_reasoning'):
                    st.markdown(f"**Reasoning:** {case['llm_reasoning']}")
                if case.get('llm_recommendation'):
                    st.markdown(f"**Recommendation:** {case['llm_recommendation']}")
                if case.get('llm_next_steps'):
                    st.markdown(f"**Next Steps:** {case['llm_next_steps']}")

            with col2:
                st.subheader("Your Decision")
                tiers = ['Urgent', 'Routine', 'Self-care']
                idx = tiers.index(tier) if tier in tiers else 0
                nurse_tier = st.selectbox("Final Tier:", tiers, index=idx)
                notes = st.text_area("Notes (optional):", height=80)

                st.markdown("---")
                st.markdown("**Rate this assessment:**")
                clarity = st.slider("Clarity", 1, 5, 3)
                usefulness = st.slider("Usefulness", 1, 5, 3)
                trust = st.slider("Trustworthiness", 1, 5, 3)
                safety = st.slider("Perceived Safety", 1, 5, 3)

                if st.button("Confirm Decision", type="primary"):
                    if nurse_tier == tier:
                        action = 'approve'
                    elif tiers.index(nurse_tier) < tiers.index(tier):
                        action = 'override_upgrade'
                    else:
                        action = 'override_downgrade'
                    booking = book_action(case['case_id'], nurse_tier)
                    db_update(case['case_id'], {
                        'nurse_tier': nurse_tier, 'nurse_action': action,
                        'nurse_notes': notes,
                        'nurse_timestamp': datetime.utcnow().isoformat(),
                        'final_tier': nurse_tier,
                        'booking_status': booking['status'],
                        'booking_details': json.dumps(booking),
                        'status': 'reviewed',
                    })
                    st.success(f"Recorded: **{nurse_tier}** ({action})")
                    st.balloons()
                    st.rerun()

    st.markdown("---")
    st.subheader("Recently Reviewed")
    for c in reviewed[:5]:
        emoji = {'approve': '✅', 'override_upgrade': '⬆️', 'override_downgrade': '⬇️'}.get(c.get('nurse_action', ''), '❓')
        st.markdown(f"{emoji} **{c['case_id']}** — LLM: {c.get('llm_urgency', '?')} → Final: {c.get('final_tier', '?')} ({c.get('nurse_action', '?')})")
