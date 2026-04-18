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
        llm_next_steps TEXT, llm_sources TEXT, llm_evidence TEXT,
        llm_patient_explanation TEXT, llm_confidence TEXT,
        rag_mode TEXT,
        nurse_tier TEXT, nurse_action TEXT, nurse_notes TEXT,
        nurse_override_reason TEXT, nurse_timestamp TEXT,
        nurse_clarity INTEGER, nurse_usefulness INTEGER,
        nurse_trust INTEGER, nurse_safety INTEGER,
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

def db_stats():
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute('SELECT COUNT(*) FROM cases').fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM cases WHERE status='pending'").fetchone()[0]
    reviewed = conn.execute("SELECT COUNT(*) FROM cases WHERE status='reviewed'").fetchone()[0]
    overrides = conn.execute("SELECT COUNT(*) FROM cases WHERE nurse_action LIKE 'override%'").fetchone()[0]
    conn.close()
    return {'total': total, 'pending': pending, 'reviewed': reviewed, 'overrides': overrides}

init_db()

# ============================================================================
# TRIAGE (Enhanced Gemini prompt with evidence + patient explanation)
# ============================================================================
TRIAGE_SYSTEM = """You are an Emergency Department triage assistant supporting an ER nurse.
You have access to clinical knowledge about emergency medicine, common conditions, and triage protocols.

## Your Task
Given patient symptoms, perform a structured triage assessment.

## Urgency Categories (choose EXACTLY ONE)
- Urgent: Immediate or same-day ER evaluation needed. Red-flag symptoms present.
- Routine: Non-urgent clinical evaluation needed within days to weeks. Yellow-flag symptoms.
- Self-care: Can be safely managed at home with monitoring. No red or yellow flags.

## Output Format (STRICT — follow exactly)

Urgency: <Urgent / Routine / Self-care>

Confidence: <High / Medium / Low>

Evidence:
[1] <First relevant clinical finding or guideline supporting your assessment>
[2] <Second relevant clinical finding or guideline>
[3] <Third relevant clinical finding (if applicable)>

Reasoning:
<2-4 sentences explaining your clinical reasoning, referencing the evidence above with [1], [2], etc.>

Recommendation:
<Specific clinical recommendation for the nurse>

Next steps:
- <Action item 1>
- <Action item 2>
- <Action item 3>

Patient explanation:
<2-3 sentences in simple, reassuring language explaining to the patient what is happening and what to expect. Use no medical jargon. Write as if speaking directly to the patient.>

Sources:
<List any clinical guidelines or references used, or "Based on clinical triage protocols" if general knowledge>"""

def run_triage(symptoms):
    try:
        from google import genai
        from google.genai import types
        client = genai.Client()
        resp = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f'Patient symptoms:\n"""{symptoms}"""',
            config=types.GenerateContentConfig(
                system_instruction=TRIAGE_SYSTEM, temperature=0.0,
                max_output_tokens=1200,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return parse_triage(resp.text or '', mode='gemini')
    except Exception as e:
        pass

    # Fallback demo mode
    s = symptoms.lower()
    urgent_kw = ['chest pain', 'stroke', 'breathing', 'unconscious', 'bleeding heavily',
                 'swelling throat', 'suicidal', 'seizure', 'facial droop', 'slurred']
    routine_kw = ['cough', 'burning urin', 'back pain', 'rash', 'headache weeks',
                  'depressed', 'knee pain', 'heartburn', 'anxiety']

    if any(k in s for k in urgent_kw):
        return {
            'urgency': 'Urgent', 'confidence': 'High',
            'evidence': '[1] Patient presents with symptoms consistent with a medical emergency.\n[2] Red-flag findings identified in symptom description.',
            'reasoning': 'Red-flag symptoms detected requiring immediate evaluation [1]. Clinical guidelines recommend emergency assessment for these presentations [2].',
            'recommendation': 'Immediate ER evaluation and stabilization.',
            'next_steps': '- Activate emergency protocol\n- Continuous monitoring\n- Prepare for urgent workup',
            'patient_explanation': 'Your symptoms need immediate medical attention. A doctor will see you right away to make sure you are safe. Please stay calm and let the medical team take care of you.',
            'sources': 'Based on emergency triage protocols',
            'rag_mode': 'demo'
        }
    elif any(k in s for k in routine_kw):
        return {
            'urgency': 'Routine', 'confidence': 'Medium',
            'evidence': '[1] Symptoms suggest a non-emergency condition.\n[2] No red-flag findings identified.\n[3] Clinical evaluation recommended within standard timeframe.',
            'reasoning': 'The symptoms described do not indicate an immediate emergency [2] but warrant clinical follow-up to rule out underlying conditions [3].',
            'recommendation': 'Schedule appointment with primary care within 1-2 weeks.',
            'next_steps': '- Monitor symptoms\n- Schedule follow-up appointment\n- Return to ER if symptoms worsen',
            'patient_explanation': 'Your symptoms do not appear to be an emergency, but you should see a doctor within the next week or two. If anything gets worse before then, come back right away.',
            'sources': 'Based on clinical triage protocols',
            'rag_mode': 'demo'
        }
    else:
        return {
            'urgency': 'Self-care', 'confidence': 'Medium',
            'evidence': '[1] Symptoms appear self-limiting based on description.\n[2] No red or yellow flag findings identified.',
            'reasoning': 'Based on the symptoms described, this appears to be a self-limiting condition [1] with no concerning features [2]. Home management is appropriate.',
            'recommendation': 'Home management with symptom monitoring.',
            'next_steps': '- Rest and adequate hydration\n- Over-the-counter symptom relief as appropriate\n- Return if symptoms worsen or persist beyond 1 week',
            'patient_explanation': 'Based on what you have described, you can manage this at home with rest. Keep an eye on your symptoms, and if anything changes or gets worse, please come back or see your doctor.',
            'sources': 'Based on clinical triage protocols',
            'rag_mode': 'demo'
        }


def parse_triage(text, mode='gemini'):
    result = {
        'urgency': 'Unknown', 'confidence': 'Medium', 'evidence': '',
        'reasoning': '', 'recommendation': '', 'next_steps': '',
        'patient_explanation': '', 'sources': '', 'rag_mode': mode
    }
    if not text:
        return result

    # Extract urgency
    m = re.search(r'Urgency:\s*(Urgent|Routine|Self-care|Self care)', text, re.IGNORECASE)
    if m:
        u = m.group(1).strip()
        result['urgency'] = 'Self-care' if 'self' in u.lower() else u.capitalize()

    # Extract confidence
    m = re.search(r'Confidence:\s*(High|Medium|Low)', text, re.IGNORECASE)
    if m:
        result['confidence'] = m.group(1).capitalize()

    # Extract sections
    section_map = {
        'evidence': r'Evidence:', 'reasoning': r'Reasoning:',
        'recommendation': r'Recommendation:', 'next steps': r'Next [Ss]teps:',
        'patient explanation': r'Patient [Ee]xplanation:', 'sources': r'Sources:',
    }
    for key, pattern in section_map.items():
        m = re.search(pattern + r'\s*\n?(.*?)(?=\n(?:Urgency|Confidence|Evidence|Reasoning|Recommendation|Next [Ss]teps|Patient [Ee]xplanation|Sources):|\Z)',
                       text, re.DOTALL | re.IGNORECASE)
        if m:
            field = key.replace(' ', '_')
            if field == 'next_steps':
                result['next_steps'] = m.group(1).strip()
            elif field == 'patient_explanation':
                result['patient_explanation'] = m.group(1).strip()
            elif field in result:
                result[field] = m.group(1).strip()

    return result


# ============================================================================
# BOOKING
# ============================================================================
DOCTORS = {
    'Emergency': 'Dr. Khalil',
    'Internal Medicine': 'Dr. Ahmed',
    'Family Medicine': 'Dr. Noor',
    'Cardiology': 'Dr. Hasan',
    'Neurology': 'Dr. Lina',
}

def infer_dept(symptoms):
    s = symptoms.lower()
    if any(k in s for k in ['chest', 'heart', 'cardiac']):
        return 'Cardiology'
    if any(k in s for k in ['head', 'brain', 'stroke', 'vision', 'speech']):
        return 'Neurology'
    return 'Internal Medicine'

def book_action(case_id, tier, symptoms=''):
    dept = infer_dept(symptoms)
    doctor = DOCTORS.get(dept, 'Dr. Ahmed')

    if tier == 'Urgent':
        return {
            'status': 'urgent_referral', 'type': 'Urgent Referral',
            'doctor': f'{doctor} ({dept})',
            'time': (datetime.now() + timedelta(hours=1)).strftime('%B %d, %Y at %I:%M %p'),
            'dept': dept, 'room': f'ER-{random.randint(1,15)}',
            'booking_id': f'UR-{uuid.uuid4().hex[:6]}',
            'instructions': 'Please proceed to the Emergency Room immediately. A nurse will meet you at triage.',
        }
    elif tier == 'Routine':
        d = random.randint(2, 10)
        return {
            'status': 'booked', 'type': 'Scheduled Appointment',
            'doctor': f'{doctor} ({dept})',
            'time': (datetime.now() + timedelta(days=d)).strftime('%B %d, %Y at %I:%M %p'),
            'dept': dept, 'room': f'Clinic-{random.randint(100,350)}',
            'booking_id': f'BK-{uuid.uuid4().hex[:6]}',
            'instructions': 'Please arrive 15 minutes before your appointment. Bring your ID and insurance card.',
        }
    return {
        'status': 'self_care_issued', 'type': 'Self-Care Guidance',
        'doctor': None, 'time': None, 'dept': None, 'room': None,
        'booking_id': f'SC-{uuid.uuid4().hex[:6]}',
        'instructions': 'You can manage your symptoms at home. Please return to the ER or see your doctor if symptoms worsen.',
        'guidance': '- Get plenty of rest\n- Stay hydrated\n- Monitor your symptoms\n- Take over-the-counter medication as needed\n- Return if symptoms worsen or new symptoms develop',
    }


# ============================================================================
# STREAMLIT APP
# ============================================================================
st.set_page_config(page_title="Triage Decision Support System", layout="wide",
                   initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; }
    div[data-testid="stMetric"] { background-color: #f8f9fa; padding: 12px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Dashboard:", ["Patient Dashboard", "Nurse Dashboard"])

# Show stats in sidebar
stats = db_stats()
st.sidebar.markdown("---")
st.sidebar.markdown("**System Status**")
st.sidebar.markdown(f"Total cases: **{stats['total']}**")
st.sidebar.markdown(f"Pending review: **{stats['pending']}**")
st.sidebar.markdown(f"Completed: **{stats['reviewed']}**")
if stats['reviewed'] > 0:
    st.sidebar.markdown(f"Override rate: **{stats['overrides']}/{stats['reviewed']}** "
                        f"({stats['overrides']/stats['reviewed']*100:.0f}%)")

# ==================== PATIENT DASHBOARD ======================================
if page == "Patient Dashboard":
    st.title("Patient Triage Dashboard")
    st.markdown("Welcome. Please enter your ticket number and describe your symptoms below.")

    tab1, tab2 = st.tabs(["Submit New Case", "Check Results"])

    # ---- TAB 1: Submit -------------------------------------------------------
    with tab1:
        with st.form("patient_form"):
            col_a, col_b = st.columns([1, 3])
            with col_a:
                ticket = st.text_input("Ticket Number", placeholder="e.g. 12")
            with col_b:
                symptoms = st.text_area(
                    "Describe your symptoms",
                    placeholder="Please describe what you are feeling, when it started, and how severe it is...",
                    height=150
                )
            submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)

        if submitted:
            if not ticket or not symptoms:
                st.error("Please fill in both your ticket number and symptoms.")
            elif len(symptoms.strip()) < 10:
                st.warning("Please describe your symptoms in more detail so we can help you better.")
            else:
                case_id = f"CASE-{uuid.uuid4().hex[:8]}"
                with st.spinner("Analyzing your symptoms... This may take a moment."):
                    triage = run_triage(symptoms)
                    db_insert({
                        'case_id': case_id, 'ticket_number': ticket,
                        'patient_symptoms': symptoms, 'status': 'pending',
                        'llm_urgency': triage['urgency'],
                        'llm_reasoning': triage['reasoning'],
                        'llm_recommendation': triage['recommendation'],
                        'llm_next_steps': triage.get('next_steps', ''),
                        'llm_sources': triage.get('sources', ''),
                        'llm_evidence': triage.get('evidence', ''),
                        'llm_patient_explanation': triage.get('patient_explanation', ''),
                        'llm_confidence': triage.get('confidence', ''),
                        'rag_mode': triage.get('rag_mode', ''),
                        'created_at': datetime.utcnow().isoformat(),
                        'updated_at': datetime.utcnow().isoformat(),
                    })
                st.success(f"Your case has been submitted successfully!")
                st.info(f"**Your Case ID:** `{case_id}`\n\n"
                        f"A nurse will review your case shortly. "
                        f"Use the **Check Results** tab with your ticket number to see your results.")

    # ---- TAB 2: Results ------------------------------------------------------
    with tab2:
        check = st.text_input("Enter your ticket number:", key="chk",
                              placeholder="e.g. 12")
        if check:
            cases = db_get_by_ticket(check)
            if not cases:
                st.info("No cases found for this ticket number. Please check and try again.")

            for c in cases:
                if c['status'] == 'reviewed':
                    tier = c.get('final_tier', 'Unknown')
                    colors = {'Urgent': 'red', 'Routine': 'orange', 'Self-care': 'green'}
                    icons = {'Urgent': '🔴', 'Routine': '🟡', 'Self-care': '🟢'}
                    color = colors.get(tier, 'gray')
                    icon = icons.get(tier, '⚪')

                    st.markdown(f"### {icon} Case {c['case_id']}")
                    st.markdown(f"**Final Decision:** :{color}[{tier}]")

                    # Booking details
                    bd = {}
                    try:
                        bd = json.loads(c.get('booking_details', '{}') or '{}')
                    except:
                        pass

                    if bd:
                        st.markdown("---")
                        st.markdown("#### Appointment Details")

                        if bd.get('type'):
                            st.markdown(f"**Type:** {bd['type']}")
                        if bd.get('doctor'):
                            st.markdown(f"**Doctor:** {bd['doctor']}")
                        if bd.get('time'):
                            st.markdown(f"**When:** {bd['time']}")
                        if bd.get('dept'):
                            st.markdown(f"**Department:** {bd['dept']}")
                        if bd.get('room'):
                            st.markdown(f"**Room:** {bd['room']}")
                        if bd.get('instructions'):
                            st.info(f"{bd['instructions']}")
                        if bd.get('guidance'):
                            st.markdown("**Self-Care Guidance:**")
                            st.markdown(bd['guidance'])

                    # Patient explanation
                    if c.get('llm_patient_explanation'):
                        st.markdown("---")
                        st.markdown("#### What This Means For You")
                        st.markdown(c['llm_patient_explanation'])

                    # Nurse notes (if any)
                    if c.get('nurse_notes') and c['nurse_notes'].strip():
                        st.markdown("---")
                        st.markdown("#### Nurse Notes")
                        st.markdown(f"> {c['nurse_notes']}")

                    # Recommendation
                    if c.get('llm_recommendation'):
                        st.markdown("---")
                        st.markdown("#### Clinical Recommendation")
                        st.markdown(c['llm_recommendation'])

                    # Next steps
                    if c.get('llm_next_steps'):
                        st.markdown("#### Next Steps")
                        st.markdown(c['llm_next_steps'])

                    # Case timeline
                    st.markdown("---")
                    st.markdown("#### Case Timeline")
                    if c.get('created_at'):
                        st.markdown(f"- **Submitted:** {c['created_at'][:19].replace('T', ' ')}")
                    nurse_action = c.get('nurse_action', '')
                    action_label = {'approve': 'Approved', 'override_upgrade': 'Upgraded (more urgent)',
                                    'override_downgrade': 'Adjusted (less urgent)'}.get(nurse_action, nurse_action)
                    if c.get('nurse_timestamp'):
                        st.markdown(f"- **Nurse reviewed:** {c['nurse_timestamp'][:19].replace('T', ' ')} — {action_label}")
                    if c.get('updated_at'):
                        st.markdown(f"- **Completed:** {c['updated_at'][:19].replace('T', ' ')}")

                elif c['status'] == 'pending':
                    st.warning(f"**Case {c['case_id']}** — Your case is in the queue and will be reviewed by a nurse shortly.")
                else:
                    st.info(f"**Case {c['case_id']}** — Processing your symptoms...")


# ==================== NURSE DASHBOARD ========================================
elif page == "Nurse Dashboard":
    st.title("Nurse Triage Review Dashboard")

    pending = db_get_all(status='pending')
    reviewed = db_get_all(status='reviewed')

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pending", len(pending))
    col2.metric("Reviewed", len(reviewed))
    overrides = sum(1 for c in reviewed if c.get('nurse_action', '').startswith('override'))
    col3.metric("Overrides", overrides)
    avg_trust = 0
    trust_vals = [c.get('nurse_trust') for c in reviewed if c.get('nurse_trust')]
    if trust_vals:
        avg_trust = sum(int(v) for v in trust_vals) / len(trust_vals)
    col4.metric("Avg Trust", f"{avg_trust:.1f}/5" if trust_vals else "N/A")

    tab_review, tab_history = st.tabs(["Review Cases", "Review History"])

    # ---- TAB 1: Review -------------------------------------------------------
    with tab_review:
        if not pending:
            st.info("No pending cases. Waiting for patient submissions...")
            if st.button("Refresh", use_container_width=True):
                st.rerun()
        else:
            opts = {f"{c['case_id']} — Ticket #{c['ticket_number']}": c['case_id'] for c in pending}
            sel = st.selectbox("Select a case to review:", list(opts.keys()))
            case = db_get_one(opts[sel])

            if case:
                st.markdown("---")
                col_left, col_right = st.columns([3, 2])

                with col_left:
                    # Patient symptoms
                    st.subheader("Patient Symptoms")
                    st.text_area("", value=case['patient_symptoms'], height=100, disabled=True,
                                 label_visibility="collapsed")

                    # LLM Assessment
                    st.subheader("AI Triage Assessment")
                    tier = case.get('llm_urgency', 'Unknown')
                    conf = case.get('llm_confidence', 'Unknown')
                    tier_colors = {'Urgent': 'red', 'Routine': 'orange', 'Self-care': 'green'}
                    conf_colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.markdown(f"**Urgency:** :{tier_colors.get(tier, 'gray')}[{tier}]")
                    mc2.markdown(f"**Confidence:** :{conf_colors.get(conf, 'gray')}[{conf}]")
                    mc3.markdown(f"**Mode:** `{case.get('rag_mode', 'unknown')}`")

                    # Retrieved evidence
                    if case.get('llm_evidence'):
                        st.markdown("**Retrieved Evidence:**")
                        with st.expander("View evidence passages", expanded=True):
                            st.markdown(case['llm_evidence'])

                    # Reasoning
                    if case.get('llm_reasoning'):
                        st.markdown(f"**Clinical Reasoning:** {case['llm_reasoning']}")

                    # Recommendation
                    if case.get('llm_recommendation'):
                        st.markdown(f"**Recommendation:** {case['llm_recommendation']}")

                    # Next steps
                    if case.get('llm_next_steps'):
                        st.markdown(f"**Next Steps:** {case['llm_next_steps']}")

                    # Patient explanation
                    if case.get('llm_patient_explanation'):
                        st.markdown(f"**Patient Explanation:** _{case['llm_patient_explanation']}_")

                    # Sources
                    if case.get('llm_sources') and case['llm_sources'] != 'None':
                        st.markdown(f"**Sources:** {case['llm_sources']}")

                with col_right:
                    st.subheader("Your Clinical Decision")

                    # Tier selection
                    tiers = ['Urgent', 'Routine', 'Self-care']
                    idx = tiers.index(tier) if tier in tiers else 0
                    nurse_tier = st.selectbox("Final Urgency Tier:", tiers, index=idx)

                    # Override reason (required if changing)
                    is_override = (nurse_tier != tier)
                    if is_override:
                        direction = "upgrading" if tiers.index(nurse_tier) < tiers.index(tier) else "downgrading"
                        st.warning(f"You are **{direction}** from {tier} to {nurse_tier}.")
                        override_reason = st.text_area(
                            "Override reason (required):",
                            height=80,
                            placeholder=f"Why are you {direction} this case?"
                        )
                    else:
                        override_reason = ''

                    # General notes
                    notes = st.text_area("Clinical notes (optional):", height=80,
                                        placeholder="Any additional observations...")

                    # Likert ratings
                    st.markdown("---")
                    st.markdown("**Rate this AI assessment:**")
                    clarity = st.slider("Clarity of reasoning", 1, 5, 3,
                                       help="How clear and understandable is the AI's reasoning?")
                    usefulness = st.slider("Usefulness for decision-making", 1, 5, 3,
                                          help="How useful was this assessment in making your triage decision?")
                    trust = st.slider("Trustworthiness", 1, 5, 3,
                                     help="How much do you trust this assessment?")
                    safety = st.slider("Perceived safety", 1, 5, 3,
                                      help="How safe would it be to follow this recommendation?")

                    # Confirm button
                    can_submit = True
                    if is_override and not override_reason.strip():
                        can_submit = False
                        st.error("Please provide a reason for the override.")

                    if st.button("Confirm Decision", type="primary",
                                 use_container_width=True, disabled=not can_submit):
                        if nurse_tier == tier:
                            action = 'approve'
                        elif tiers.index(nurse_tier) < tiers.index(tier):
                            action = 'override_upgrade'
                        else:
                            action = 'override_downgrade'

                        booking = book_action(case['case_id'], nurse_tier,
                                             case.get('patient_symptoms', ''))

                        db_update(case['case_id'], {
                            'nurse_tier': nurse_tier,
                            'nurse_action': action,
                            'nurse_notes': notes,
                            'nurse_override_reason': override_reason,
                            'nurse_timestamp': datetime.utcnow().isoformat(),
                            'nurse_clarity': clarity,
                            'nurse_usefulness': usefulness,
                            'nurse_trust': trust,
                            'nurse_safety': safety,
                            'final_tier': nurse_tier,
                            'booking_status': booking['status'],
                            'booking_details': json.dumps(booking),
                            'status': 'reviewed',
                        })
                        st.success(f"Decision recorded: **{nurse_tier}** ({action})")
                        st.balloons()
                        st.rerun()

    # ---- TAB 2: History -------------------------------------------------------
    with tab_history:
        if not reviewed:
            st.info("No reviewed cases yet.")
        else:
            # Filter options
            filter_action = st.selectbox("Filter by action:",
                                         ['All', 'Approved', 'Overridden'])

            for c in reviewed:
                action = c.get('nurse_action', '')
                if filter_action == 'Approved' and action != 'approve':
                    continue
                if filter_action == 'Overridden' and not action.startswith('override'):
                    continue

                emoji = {'approve': '✅', 'override_upgrade': '⬆️',
                         'override_downgrade': '⬇️'}.get(action, '❓')
                label = {'approve': 'Approved', 'override_upgrade': 'Upgraded',
                         'override_downgrade': 'Downgraded'}.get(action, action)

                with st.expander(
                    f"{emoji} {c['case_id']} — "
                    f"AI: {c.get('llm_urgency', '?')} → Final: {c.get('final_tier', '?')} "
                    f"({label})"
                ):
                    st.markdown(f"**Symptoms:** {c.get('patient_symptoms', '')[:200]}...")
                    st.markdown(f"**AI said:** {c.get('llm_urgency', '?')} → **Nurse decided:** {c.get('final_tier', '?')}")
                    if c.get('nurse_override_reason'):
                        st.markdown(f"**Override reason:** {c['nurse_override_reason']}")
                    if c.get('nurse_notes'):
                        st.markdown(f"**Notes:** {c['nurse_notes']}")
                    ratings = []
                    for r in ['nurse_clarity', 'nurse_usefulness', 'nurse_trust', 'nurse_safety']:
                        v = c.get(r)
                        if v:
                            ratings.append(f"{r.replace('nurse_', '').capitalize()}: {v}/5")
                    if ratings:
                        st.markdown(f"**Ratings:** {' | '.join(ratings)}")
