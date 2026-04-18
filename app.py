import streamlit as st
import sqlite3, json, uuid, random, re, os
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = "triage_hil.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS cases (
        case_id TEXT PRIMARY KEY, ticket_number TEXT, patient_symptoms TEXT,
        status TEXT DEFAULT 'processing',
        llm_urgency TEXT, llm_reasoning TEXT, llm_recommendation TEXT,
        llm_next_steps TEXT, llm_sources TEXT, llm_evidence TEXT,
        llm_patient_explanation TEXT, llm_confidence TEXT, rag_mode TEXT,
        nurse_tier TEXT, nurse_action TEXT, nurse_notes TEXT,
        nurse_override_reason TEXT, nurse_timestamp TEXT,
        nurse_clarity INTEGER, nurse_usefulness INTEGER,
        nurse_trust INTEGER, nurse_safety INTEGER,
        final_tier TEXT, booking_status TEXT, booking_details TEXT,
        created_at TEXT, updated_at TEXT)""")
    conn.commit(); conn.close()

def db_insert(case):
    conn = sqlite3.connect(DB_PATH)
    cols = ', '.join(case.keys()); phs = ', '.join(['?']*len(case))
    conn.execute(f'INSERT OR REPLACE INTO cases ({cols}) VALUES ({phs})', list(case.values()))
    conn.commit(); conn.close()

def db_update(cid, upd):
    conn = sqlite3.connect(DB_PATH)
    upd['updated_at'] = datetime.utcnow().isoformat()
    s = ', '.join(f'{k}=?' for k in upd)
    conn.execute(f'UPDATE cases SET {s} WHERE case_id=?', list(upd.values())+[cid])
    conn.commit(); conn.close()

def db_get_all(status=None):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    q = 'SELECT * FROM cases WHERE status=? ORDER BY created_at DESC' if status else 'SELECT * FROM cases ORDER BY created_at DESC'
    rows = conn.execute(q, (status,) if status else ()).fetchall()
    conn.close(); return [dict(r) for r in rows]

def db_get_one(cid):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    row = conn.execute('SELECT * FROM cases WHERE case_id=?', (cid,)).fetchone()
    conn.close(); return dict(row) if row else None

def db_get_by_ticket(t):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT * FROM cases WHERE ticket_number=? ORDER BY created_at DESC', (t,)).fetchall()
    conn.close(); return [dict(r) for r in rows]

def db_stats():
    conn = sqlite3.connect(DB_PATH)
    t = conn.execute('SELECT COUNT(*) FROM cases').fetchone()[0]
    p = conn.execute("SELECT COUNT(*) FROM cases WHERE status='pending'").fetchone()[0]
    r = conn.execute("SELECT COUNT(*) FROM cases WHERE status='reviewed'").fetchone()[0]
    o = conn.execute("SELECT COUNT(*) FROM cases WHERE nurse_action LIKE 'override%'").fetchone()[0]
    conn.close(); return {'total':t,'pending':p,'reviewed':r,'overrides':o}

init_db()

# ============================================================================
# RAG KNOWLEDGE BASE — your actual medical documents from the notebook
# ============================================================================
@st.cache_resource
def load_knowledge_base():
    from rank_bm25 import BM25Okapi
    kb_path = Path(__file__).parent / "chunked_docs_phase2.json"
    if not kb_path.exists():
        return None, None, None
    with open(kb_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    word_re = re.compile(r'\w+')
    texts = [c['page_content'] for c in chunks]
    tokenized = [word_re.findall(t.lower()) for t in texts]
    bm25 = BM25Okapi(tokenized)
    return chunks, bm25, word_re

KB_CHUNKS, BM25_INDEX, WORD_RE = load_knowledge_base()
RAG_AVAILABLE = KB_CHUNKS is not None

def retrieve_evidence(query, k=5):
    if not RAG_AVAILABLE:
        return [], [], ''
    tokens = WORD_RE.findall(query.lower())
    scores = BM25_INDEX.get_scores(tokens)
    top_idx = scores.argsort()[::-1][:k]
    evidence_blocks, source_lines, retrieved = [], [], []
    for rank, idx in enumerate(top_idx, 1):
        if scores[idx] <= 0:
            continue
        chunk = KB_CHUNKS[idx]
        meta = chunk.get('metadata', {})
        topic = meta.get('topic', 'Unknown')
        source = meta.get('source', 'Unknown')
        url = meta.get('url', '')
        section = meta.get('section_title', 'General')
        date = meta.get('document_date', 'n.d.')
        content = chunk['page_content']
        evidence_blocks.append(f"[{rank}] **{topic}** — {section}\n> {content[:300]}...")
        source_lines.append(f'[{rank}] {source}, "{topic}," section: "{section}," {date}. [Online]. Available: {url}')
        retrieved.append({'rank':rank,'topic':topic,'section':section,'source':source,'url':url,'score':float(scores[idx]),'content':content[:500]})
    return retrieved, evidence_blocks, '\n'.join(source_lines)

# ============================================================================
# TRIAGE — Gemini with real retrieved context from your medical KB
# ============================================================================
TRIAGE_GROUNDED = """You are an Emergency Department triage assistant supporting an ER nurse.

## Rules
- Use the retrieved medical evidence to support your assessment.
- Reference evidence using [1], [2], etc.
- Do NOT provide a diagnosis.

## Urgency Categories (choose EXACTLY ONE)
- Urgent: Immediate or same-day evaluation. Red-flag symptoms.
- Routine: Non-urgent evaluation within days to weeks.
- Self-care: Safe home management. No red or yellow flags.

## Output Format (STRICT)

Urgency: <Urgent / Routine / Self-care>

Confidence: <High / Medium / Low>

Reasoning:
<2-4 sentences referencing evidence with [1], [2], etc.>

Recommendation:
<Specific clinical recommendation>

Next steps:
- <Action 1>
- <Action 2>
- <Action 3>

Patient explanation:
<2-3 simple sentences for the patient. No jargon.>"""

TRIAGE_FALLBACK = """You are an Emergency Department triage assistant.
The knowledge base did not have enough relevant information.
Provide a cautious best-effort triage based only on symptoms.

## Output Format (STRICT)
Urgency: <Urgent / Routine / Self-care>
Confidence: <Low>
Reasoning:
<Include: "Limited evidence was available in the knowledge base for this query.">
Recommendation:
<Clinical recommendation>
Next steps:
- <Action 1>
- <Action 2>
Patient explanation:
<2-3 simple sentences for the patient.>"""

def run_triage(symptoms):
    retrieved, evidence_blocks, sources_text = retrieve_evidence(symptoms, k=5)
    has_evidence = len(retrieved) > 0 and retrieved[0]['score'] > 0.5

    if has_evidence:
        context = '\n\n'.join(f"[{r['rank']}] ({r['topic']} - {r['section']})\n{r['content']}" for r in retrieved)
        user_msg = f'Patient symptoms:\n"""{symptoms}"""\n\nRetrieved medical evidence:\n{context}\n\nClassify urgency using the evidence above.'
        system = TRIAGE_GROUNDED
        mode = 'grounded_rag'
    else:
        user_msg = f'Patient symptoms:\n"""{symptoms}"""'
        system = TRIAGE_FALLBACK
        mode = 'fallback_judgment'

    try:
        from google import genai
        from google.genai import types
        client = genai.Client()
        resp = client.models.generate_content(model='gemini-2.5-flash', contents=user_msg,
            config=types.GenerateContentConfig(system_instruction=system, temperature=0.0,
                max_output_tokens=1200, thinking_config=types.ThinkingConfig(thinking_budget=0)))
        result = parse_triage(resp.text or '')
    except:
        result = run_triage_demo(symptoms)

    result['evidence'] = '\n\n'.join(evidence_blocks) if evidence_blocks else 'No evidence retrieved — knowledge base file not found.'
    result['sources'] = sources_text if sources_text else 'None'
    result['rag_mode'] = mode
    return result

def run_triage_demo(symptoms):
    s = symptoms.lower()
    if any(k in s for k in ['chest pain','stroke','breathing','unconscious','bleeding','suicidal','seizure']):
        return {'urgency':'Urgent','confidence':'High','reasoning':'Red-flag symptoms detected.','recommendation':'Immediate ER evaluation.','next_steps':'- Emergency protocol','patient_explanation':'Your symptoms need immediate attention.'}
    if any(k in s for k in ['cough','burning','back pain','rash','headache','depressed','knee']):
        return {'urgency':'Routine','confidence':'Medium','reasoning':'Non-emergency symptoms.','recommendation':'Schedule appointment.','next_steps':'- Monitor symptoms','patient_explanation':'See a doctor within 1-2 weeks.'}
    return {'urgency':'Self-care','confidence':'Medium','reasoning':'Self-limiting symptoms.','recommendation':'Home management.','next_steps':'- Rest and monitor','patient_explanation':'You can manage this at home.'}

def parse_triage(text):
    result = {'urgency':'Unknown','confidence':'Medium','reasoning':'','recommendation':'','next_steps':'','patient_explanation':'','sources':'','evidence':''}
    if not text: return result
    m = re.search(r'Urgency:\s*(Urgent|Routine|Self-care|Self care)', text, re.IGNORECASE)
    if m:
        u = m.group(1).strip()
        result['urgency'] = 'Self-care' if 'self' in u.lower() else u.capitalize()
    m = re.search(r'Confidence:\s*(High|Medium|Low)', text, re.IGNORECASE)
    if m: result['confidence'] = m.group(1).capitalize()
    for key, pat in [('reasoning',r'Reasoning:\s*\n?(.*?)(?=\n(?:Recommendation|Next|Patient|Sources):|\Z)'),
                     ('recommendation',r'Recommendation:\s*\n?(.*?)(?=\n(?:Next|Patient|Sources):|\Z)'),
                     ('next_steps',r'Next [Ss]teps:\s*\n?(.*?)(?=\n(?:Patient|Sources):|\Z)'),
                     ('patient_explanation',r'Patient [Ee]xplanation:\s*\n?(.*?)(?=\n(?:Sources):|\Z)')]:
        m = re.search(pat, text, re.DOTALL|re.IGNORECASE)
        if m: result[key] = m.group(1).strip()
    return result

# ============================================================================
# AGENTIC BOOKING — reads real schedule from agentic_triage_schedules.xlsx
# ============================================================================
@st.cache_resource
def load_schedules():
    import pandas as pd
    path = Path(__file__).parent / "agentic_triage_schedules.xlsx"
    if not path.exists():
        return None, None
    emergency = pd.read_excel(path, sheet_name='Emergency Schedule')
    routine = pd.read_excel(path, sheet_name='Routine Schedule')
    return emergency, routine

EMERGENCY_DF, ROUTINE_DF = load_schedules()
SCHEDULE_AVAILABLE = EMERGENCY_DF is not None

# Track booked slots in session state (persists during app session)
if 'booked_slots' not in st.session_state:
    st.session_state.booked_slots = set()

def find_available_slot(schedule_df, schedule_type):
    """Find the next available slot that hasn't been booked this session."""
    if schedule_df is None:
        return None
    available = schedule_df[
        (schedule_df['Available'] == 'Available') &
        (~schedule_df['Slot ID'].isin(st.session_state.booked_slots))
    ]
    if available.empty:
        return None
    # Take the first available slot (sorted by date + time)
    slot = available.iloc[0]
    return slot

def book_slot(slot_id):
    """Mark a slot as booked for this session."""
    st.session_state.booked_slots.add(slot_id)

def book_action(case_id, tier, symptoms=''):
    if tier == 'Urgent':
        if SCHEDULE_AVAILABLE:
            slot = find_available_slot(EMERGENCY_DF, 'emergency')
            if slot is not None:
                book_slot(slot['Slot ID'])
                return {
                    'status': 'urgent_referral',
                    'type': 'Immediate ER Admission',
                    'doctor': f"{slot['Doctor']} — On Duty",
                    'time': 'IMMEDIATELY — No appointment needed',
                    'dept': 'Emergency',
                    'room': slot['Room'],
                    'booking_id': slot['Slot ID'],
                    'instructions': (
                        f"Please proceed to {slot['Room']} in the Emergency Room immediately. "
                        f"{slot['Doctor']} is on duty and will see you without an appointment."
                    ),
                }
            return {
                'status': 'urgent_referral', 'type': 'Immediate ER Admission',
                'doctor': 'Next available ER doctor', 'time': 'IMMEDIATELY',
                'dept': 'Emergency', 'room': 'ER Triage',
                'booking_id': f'UR-{uuid.uuid4().hex[:6]}',
                'instructions': 'All ER rooms are currently occupied. Please proceed to ER Triage for immediate assessment.',
            }
        # Fallback if no schedule file
        return {
            'status': 'urgent_referral', 'type': 'Immediate ER Admission',
            'doctor': 'On-duty ER physician',
            'time': 'IMMEDIATELY — No appointment needed',
            'dept': 'Emergency', 'room': f'ER-{random.randint(1,3)}',
            'booking_id': f'UR-{uuid.uuid4().hex[:6]}',
            'instructions': 'Please proceed to the Emergency Room immediately.',
        }

    elif tier == 'Routine':
        if SCHEDULE_AVAILABLE:
            slot = find_available_slot(ROUTINE_DF, 'routine')
            if slot is not None:
                book_slot(slot['Slot ID'])
                return {
                    'status': 'booked',
                    'type': 'Scheduled Appointment',
                    'doctor': slot['Doctor'],
                    'time': f"{slot['Date']} ({slot['Day']}) at {slot['Time']}",
                    'dept': slot['Department'],
                    'room': slot['Room'],
                    'booking_id': slot['Slot ID'],
                    'instructions': (
                        f"Your appointment is confirmed with {slot['Doctor']} "
                        f"on {slot['Date']} ({slot['Day']}) at {slot['Time']} "
                        f"in room {slot['Room']}. Please arrive 15 minutes early."
                    ),
                }
            return {
                'status': 'failed', 'type': 'No Slots Available',
                'doctor': None, 'time': None, 'dept': 'General',
                'room': None, 'booking_id': None,
                'instructions': 'No routine appointment slots are currently available. Please call the clinic to be added to the waitlist.',
            }
        # Fallback
        d = random.randint(2, 10)
        return {
            'status': 'booked', 'type': 'Scheduled Appointment',
            'doctor': 'Dr. Hall (General)', 'dept': 'General',
            'time': (datetime.now() + timedelta(days=d)).strftime('%B %d, %Y at %I:%M %p'),
            'room': f'G-{random.randint(101,104)}',
            'booking_id': f'BK-{uuid.uuid4().hex[:6]}',
            'instructions': 'Please arrive 15 minutes early. Bring ID and insurance.',
        }

    else:  # Self-care
        return {
            'status': 'self_care_issued', 'type': 'Self-Care Guidance',
            'doctor': None, 'time': None, 'dept': None, 'room': None,
            'booking_id': f'SC-{uuid.uuid4().hex[:6]}',
            'instructions': 'You can manage your symptoms at home. Return to the ER or see your doctor if symptoms worsen.',
            'guidance': '- Get plenty of rest\n- Stay hydrated\n- Monitor your symptoms\n- Take over-the-counter medication as needed\n- Return if symptoms worsen or new symptoms develop',
        }
# ============================================================================
# APP
# ============================================================================
st.set_page_config(page_title="Triage Decision Support System", layout="wide")
st.markdown("""<style>div[data-testid="stMetric"]{background:#f8f9fa;padding:12px;border-radius:8px;}</style>""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Dashboard:", ["Patient Dashboard","Nurse Dashboard"])
stats = db_stats()
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Cases:** {stats['total']} total | {stats['pending']} pending")
st.sidebar.markdown(f"**RAG:** {'Connected (' + str(len(KB_CHUNKS)) + ' chunks)' if RAG_AVAILABLE else 'Demo mode — upload chunked_docs_phase2.json'}")
st.sidebar.markdown(f"**Scheduler:** {'Connected' if SCHEDULE_AVAILABLE else 'Demo mode'}")
if SCHEDULE_AVAILABLE:
    st.sidebar.markdown(f"**Booked this session:** {len(st.session_state.booked_slots)}")
st.sidebar.markdown("---")
with st.sidebar.expander("Admin"):
    if st.button("Reset Database", type="secondary"):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM cases")
        conn.commit(); conn.close()
        st.session_state.booked_slots = set()
        st.rerun()

if page == "Patient Dashboard":    st.title("Patient Triage Dashboard")
    tab1, tab2 = st.tabs(["Submit New Case","Check Results"])
    with tab1:
        with st.form("pf"):
            ca, cb = st.columns([1,3])
            with ca: ticket = st.text_input("Ticket Number", placeholder="e.g. 12")
            with cb: symptoms = st.text_area("Describe your symptoms", placeholder="Please describe what you are feeling, when it started, and how severe it is...", height=150)
            submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)
        if submitted:
            if not ticket or not symptoms: st.error("Please fill in both fields.")
            elif len(symptoms.strip()) < 10: st.warning("Please describe your symptoms in more detail.")
            else:
                cid = f"CASE-{uuid.uuid4().hex[:8]}"
                with st.spinner("Analyzing your symptoms..."):
                    tri = run_triage(symptoms)
                    db_insert({'case_id':cid,'ticket_number':ticket,'patient_symptoms':symptoms,'status':'pending','llm_urgency':tri.get('urgency',''),'llm_reasoning':tri.get('reasoning',''),'llm_recommendation':tri.get('recommendation',''),'llm_next_steps':tri.get('next_steps',''),'llm_sources':tri.get('sources',''),'llm_evidence':tri.get('evidence',''),'llm_patient_explanation':tri.get('patient_explanation',''),'llm_confidence':tri.get('confidence',''),'rag_mode':tri.get('rag_mode',''),'created_at':datetime.utcnow().isoformat(),'updated_at':datetime.utcnow().isoformat()})
                st.success("Submitted!"); st.info(f"**Case ID:** `{cid}` — A nurse will review shortly. Check the **Check Results** tab.")
    with tab2:
        chk = st.text_input("Enter your ticket number:", key="chk", placeholder="e.g. 12")
        if chk:
            cases = db_get_by_ticket(chk)
            if not cases: st.info("No cases found.")
            for c in cases:
                if c['status'] == 'reviewed':
                    tier = c.get('final_tier','Unknown')
                    colors = {'Urgent':'red','Routine':'orange','Self-care':'green'}
                    icons = {'Urgent':'🔴','Routine':'🟡','Self-care':'🟢'}
                    st.markdown(f"### {icons.get(tier,'⚪')} Case {c['case_id']}")
                    st.markdown(f"**Final Decision:** :{colors.get(tier,'gray')}[{tier}]")
                    bd = {}
                    try: bd = json.loads(c.get('booking_details','{}') or '{}')
                    except: pass
                    if bd:
                        st.markdown("---"); st.markdown("#### Your Appointment")
                        for fld, lbl in [('type','Type'),('doctor','Doctor'),('time','When'),('dept','Department'),('room','Room')]:
                            if bd.get(fld): st.markdown(f"**{lbl}:** {bd[fld]}")
                        if bd.get('instructions'): st.info(bd['instructions'])
                        if bd.get('guidance'): st.markdown("**Self-Care Guidance:**"); st.markdown(bd['guidance'])
                    if c.get('llm_patient_explanation'): st.markdown("---"); st.markdown("#### What This Means For You"); st.markdown(c['llm_patient_explanation'])
                    if c.get('nurse_notes') and c['nurse_notes'].strip(): st.markdown("---"); st.markdown("#### Nurse Notes"); st.markdown(f"> {c['nurse_notes']}")
                    if c.get('nurse_override_reason') and c['nurse_override_reason'].strip(): st.markdown(f"**Nurse comment:** {c['nurse_override_reason']}")
                    st.markdown("---"); st.markdown("#### Timeline")
                    if c.get('created_at'): st.markdown(f"- **Submitted:** {c['created_at'][:19].replace('T',' ')}")
                    act = c.get('nurse_action',''); lbl = {'approve':'Approved','override_upgrade':'Upgraded','override_downgrade':'Adjusted'}.get(act,act)
                    if c.get('nurse_timestamp'): st.markdown(f"- **Reviewed:** {c['nurse_timestamp'][:19].replace('T',' ')} — {lbl}")
                elif c['status'] == 'pending': st.warning(f"**{c['case_id']}** — Waiting for nurse review...")
                else: st.info(f"**{c['case_id']}** — Processing...")

elif page == "Nurse Dashboard":
    st.title("Nurse Triage Review Dashboard")
    pending = db_get_all(status='pending'); reviewed = db_get_all(status='reviewed')
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Pending",len(pending)); c2.metric("Reviewed",len(reviewed))
    ov = sum(1 for c in reviewed if (c.get('nurse_action') or '').startswith('override'))
    c3.metric("Overrides",ov)
    tv = [c.get('nurse_trust') for c in reviewed if c.get('nurse_trust')]
    c4.metric("Avg Trust", f"{sum(int(v) for v in tv)/len(tv):.1f}/5" if tv else "N/A")

    tab_r, tab_h = st.tabs(["Review Cases","Review History"])
    with tab_r:
        if not pending:
            st.info("No pending cases.")
            if st.button("Refresh", use_container_width=True): st.rerun()
        else:
            opts = {f"{c['case_id']} — Ticket #{c['ticket_number']}":c['case_id'] for c in pending}
            sel = st.selectbox("Select case:",list(opts.keys()))
            case = db_get_one(opts[sel])
            if case:
                st.markdown("---")
                cl, cr = st.columns([3,2])
                with cl:
                    st.subheader("Patient Symptoms")
                    st.text_area("",value=case['patient_symptoms'],height=100,disabled=True,label_visibility="collapsed")
                    st.subheader("AI Triage Assessment")
                    tier = case.get('llm_urgency','Unknown'); conf = case.get('llm_confidence','Unknown')
                    tc = {'Urgent':'red','Routine':'orange','Self-care':'green'}
                    cc = {'High':'green','Medium':'orange','Low':'red'}
                    m1,m2,m3 = st.columns(3)
                    m1.markdown(f"**Urgency:** :{tc.get(tier,'gray')}[{tier}]")
                    m2.markdown(f"**Confidence:** :{cc.get(conf,'gray')}[{conf}]")
                    m3.markdown(f"**Mode:** `{case.get('rag_mode','')}`")
                    if case.get('llm_evidence') and case['llm_evidence'] not in ('','No evidence retrieved — knowledge base file not found.'):
                        st.markdown("**Retrieved Medical Evidence:**")
                        with st.expander("View retrieved passages",expanded=True): st.markdown(case['llm_evidence'])
                    if case.get('llm_reasoning'): st.markdown(f"**Clinical Reasoning:** {case['llm_reasoning']}")
                    if case.get('llm_recommendation'): st.markdown(f"**Recommendation:** {case['llm_recommendation']}")
                    if case.get('llm_next_steps'): st.markdown(f"**Next Steps:** {case['llm_next_steps']}")
                    if case.get('llm_patient_explanation'): st.markdown(f"**Patient Explanation:** _{case['llm_patient_explanation']}_")
                    if case.get('llm_sources') and case['llm_sources'] not in ('None',''):
                        with st.expander("View sources (IEEE citations)"): st.markdown(case['llm_sources'])
                with cr:
                    st.subheader("Your Clinical Decision")
                    tiers = ['Urgent','Routine','Self-care']
                    idx = tiers.index(tier) if tier in tiers else 0
                    nurse_tier = st.selectbox("Final Urgency Tier:",tiers,index=idx)
                    is_ov = nurse_tier != tier
                    if is_ov:
                        d = "upgrading" if tiers.index(nurse_tier)<tiers.index(tier) else "downgrading"
                        st.warning(f"You are **{d}** from {tier} to {nurse_tier}.")
                        ov_reason = st.text_area("Override reason (required):",height=80,placeholder=f"Why are you {d}?")
                    else: ov_reason = ''
                    notes = st.text_area("Clinical notes (optional):",height=80,placeholder="Additional observations...")
                    st.markdown("---"); st.markdown("**Rate this AI assessment:**")
                    clarity = st.slider("Clarity",1,5,3,help="How clear is the reasoning?")
                    useful = st.slider("Usefulness",1,5,3,help="How useful for your decision?")
                    trust = st.slider("Trustworthiness",1,5,3,help="How much do you trust it?")
                    safety = st.slider("Perceived safety",1,5,3,help="How safe is this recommendation?")
                    can = True
                    if is_ov and not (ov_reason or '').strip(): can = False; st.error("Override reason required.")
                    if st.button("Confirm Decision",type="primary",use_container_width=True,disabled=not can):
                        if nurse_tier == tier: act = 'approve'
                        elif tiers.index(nurse_tier)<tiers.index(tier): act = 'override_upgrade'
                        else: act = 'override_downgrade'
                        bk = book_action(case['case_id'],nurse_tier,case.get('patient_symptoms',''))
                        db_update(case['case_id'],{'nurse_tier':nurse_tier,'nurse_action':act,'nurse_notes':notes,'nurse_override_reason':ov_reason,'nurse_timestamp':datetime.utcnow().isoformat(),'nurse_clarity':clarity,'nurse_usefulness':useful,'nurse_trust':trust,'nurse_safety':safety,'final_tier':nurse_tier,'booking_status':bk['status'],'booking_details':json.dumps(bk),'status':'reviewed'})
                        st.success(f"Recorded: **{nurse_tier}** ({act})"); st.balloons(); st.rerun()
    with tab_h:
        if not reviewed: st.info("No reviewed cases yet.")
        else:
            filt = st.selectbox("Filter:",['All','Approved','Overridden'])
            for c in reviewed:
                a = c.get('nurse_action','')
                if filt == 'Approved' and a != 'approve': continue
                if filt == 'Overridden' and not a.startswith('override'): continue
                em = {'approve':'✅','override_upgrade':'⬆️','override_downgrade':'⬇️'}.get(a,'❓')
                lb = {'approve':'Approved','override_upgrade':'Upgraded','override_downgrade':'Downgraded'}.get(a,a)
                with st.expander(f"{em} {c['case_id']} — AI: {c.get('llm_urgency','?')} → Final: {c.get('final_tier','?')} ({lb})"):
                    st.markdown(f"**Symptoms:** {(c.get('patient_symptoms',''))[:200]}...")
                    if c.get('nurse_override_reason'): st.markdown(f"**Override reason:** {c['nurse_override_reason']}")
                    if c.get('nurse_notes'): st.markdown(f"**Notes:** {c['nurse_notes']}")
                    rt = []
                    for r in ['nurse_clarity','nurse_usefulness','nurse_trust','nurse_safety']:
                        v = c.get(r)
                        if v: rt.append(f"{r.replace('nurse_','').title()}: {v}/5")
                    if rt: st.markdown(f"**Ratings:** {' | '.join(rt)}")
