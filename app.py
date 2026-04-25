import streamlit as st
import sqlite3, json, uuid, random, re, os
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
        enriched_symptoms TEXT,
        status TEXT DEFAULT 'processing',
        llm_urgency TEXT, llm_reasoning TEXT, llm_reasoning_original TEXT,
        llm_recommendation TEXT, llm_next_steps TEXT, llm_sources TEXT,
        llm_evidence TEXT, llm_patient_explanation TEXT, llm_confidence TEXT,
        rag_mode TEXT,
        router_complete BOOLEAN, router_questions TEXT, router_answers TEXT,
        evaluator_enhanced BOOLEAN,
        nurse_tier TEXT, nurse_action TEXT, nurse_notes TEXT,
        nurse_override_reason TEXT, nurse_timestamp TEXT, nurse_name TEXT,
        final_tier TEXT, booking_status TEXT, booking_details TEXT,
        booking_agent_decision TEXT,
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
    if status:
        rows = conn.execute('SELECT * FROM cases WHERE status=? ORDER BY created_at DESC', (status,)).fetchall()
    else:
        rows = conn.execute('SELECT * FROM cases ORDER BY created_at DESC').fetchall()
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
# AUTHENTICATION
# ============================================================================
NURSES = {
    'haya':  {'password': '123', 'name': 'Nurse Haya'},
    'malek': {'password': '123', 'name': 'Nurse Malek'},
    'yomna': {'password': '123', 'name': 'Nurse Yomna'},
    'admin': {'password': 'admin2026', 'name': 'Administrator'},
}

def check_nurse_login():
    return st.session_state.get('nurse_name', None)

def nurse_login_form():
    st.markdown("### Nurse Login")
    st.markdown("Enter your credentials to access the review dashboard.")
    with st.form("nurse_login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
    if submitted:
        if username in NURSES and NURSES[username]['password'] == password:
            st.session_state.nurse_name = NURSES[username]['name']
            st.session_state.nurse_username = username
            st.rerun()
        else:
            st.error("Invalid username or password.")

# ============================================================================
# GEMINI CLIENT
# ============================================================================
GEMINI_AVAILABLE = False
try:
    from google import genai
    GEMINI_AVAILABLE = True
except:
    pass

def gemini_call(prompt, system="", temperature=0.0, max_tokens=900):
    from google import genai
    from google.genai import types
    client = genai.Client()
    resp = client.models.generate_content(
        model='gemini-2.5-flash', contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system if system else None,
            temperature=temperature, max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0)))
    return resp.text or ''

# ============================================================================
# AGENTIC AI 1: ROUTER — generates follow-up questions if info is incomplete
# ============================================================================
def router_check_completeness(symptoms):
    if not GEMINI_AVAILABLE:
        return {"complete": True, "questions": []}
    prompt = f'''You are an intake nurse. A patient said: "{symptoms}"

Determine if this gives ENOUGH information for triage.
COMPLETE needs: SPECIFIC symptoms + duration + severity.
INCOMPLETE examples: "I don't feel well", "I'm sick", "I have pain"
COMPLETE examples: "Severe chest pain for 30 minutes radiating to left arm"

If INCOMPLETE, generate 2-3 SHORT follow-up questions.

Respond ONLY in JSON:
{{"has_sufficient_info": true, "questions": []}}
or
{{"has_sufficient_info": false, "questions": ["question 1", "question 2", "question 3"]}}'''
    try:
        raw = gemini_call(prompt, max_tokens=400)
        clean = re.sub(r'```json\s*|```\s*', '', raw).strip()
        result = json.loads(clean)
        return {"complete": result.get("has_sufficient_info", True),
                "questions": result.get("questions", [])}
    except:
        return {"complete": True, "questions": []}

# ============================================================================
# AGENTIC AI 2: EVALUATOR — senior physician review
# ============================================================================
def evaluator_enhance_triage(reasoning, symptoms):
    if not GEMINI_AVAILABLE:
        return reasoning
    prompt = f'''You are a senior emergency physician reviewing a triage assessment.

Patient: "{symptoms}"

Assessment:
{reasoning}

Enhance by:
1. Check urgency appropriateness
2. Add missed red flags
3. Ensure next steps are complete
4. Keep same citation references [1], [2] — do NOT invent sources
5. Do NOT change urgency unless clearly wrong

Output the ENHANCED assessment in the SAME format.'''
    try:
        enhanced = gemini_call(prompt, max_tokens=1500)
        return enhanced if enhanced and len(enhanced) > 50 else reasoning
    except:
        return reasoning

# ============================================================================
# RAG KNOWLEDGE BASE
# ============================================================================
@st.cache_resource
def load_knowledge_base():
    from rank_bm25 import BM25Okapi
    kb_path = Path(__file__).parent / "chunked_docs_phase2.json"
    if not kb_path.exists(): return None, None, None
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
    if not RAG_AVAILABLE: return [], [], ''
    tokens = WORD_RE.findall(query.lower())
    scores = BM25_INDEX.get_scores(tokens)
    top_idx = scores.argsort()[::-1][:k]
    evidence_blocks, source_lines, retrieved = [], [], []
    for rank, idx in enumerate(top_idx, 1):
        if scores[idx] <= 0: continue
        chunk = KB_CHUNKS[idx]; meta = chunk.get('metadata', {})
        topic = meta.get('topic','Unknown'); source = meta.get('source','Unknown')
        url = meta.get('url',''); section = meta.get('section_title','General')
        date = meta.get('document_date','n.d.'); content = chunk['page_content']
        evidence_blocks.append(f"[{rank}] **{topic}** — {section}\n> {content[:300]}...")
        source_lines.append(f'[{rank}] {source}, "{topic}," section: "{section}," {date}. Available: {url}')
        retrieved.append({'rank':rank,'topic':topic,'section':section,'score':float(scores[idx]),'content':content[:500]})
    return retrieved, evidence_blocks, '\n'.join(source_lines)

# ============================================================================
# RAG TRIAGE
# ============================================================================
TRIAGE_SYSTEM = """You are an ER triage assistant. Use retrieved evidence. Reference with [1], [2].
Categories: Urgent / Routine / Self-care.
Output STRICTLY:
Urgency: <tier>
Confidence: <High/Medium/Low>
Reasoning: <2-4 sentences with citations>
Recommendation: <action>
Next steps:
- <step 1>
- <step 2>
Patient explanation: <2-3 simple sentences, no jargon>"""

def run_triage(symptoms):
    retrieved, evidence_blocks, sources_text = retrieve_evidence(symptoms, k=5)
    has_evidence = len(retrieved) > 0 and retrieved[0]['score'] > 0.5
    if has_evidence:
        context = '\n\n'.join(f"[{r['rank']}] ({r['topic']})\n{r['content']}" for r in retrieved)
        user_msg = f'Patient:\n"""{symptoms}"""\n\nEvidence:\n{context}\n\nClassify urgency.'
        mode = 'grounded_rag'
    else:
        user_msg = f'Patient:\n"""{symptoms}"""'
        mode = 'fallback_judgment'
    try:
        raw = gemini_call(user_msg, system=TRIAGE_SYSTEM, max_tokens=1200)
        result = parse_triage(raw)
    except:
        result = triage_demo(symptoms)
    result['evidence'] = '\n\n'.join(evidence_blocks) if evidence_blocks else ''
    result['sources'] = sources_text if sources_text else 'None'
    result['rag_mode'] = mode
    return result

def triage_demo(symptoms):
    s = symptoms.lower()
    if any(k in s for k in ['chest pain','stroke','breathing','unconscious','bleeding','seizure']):
        return {'urgency':'Urgent','confidence':'High','reasoning':'Red-flag symptoms.','recommendation':'Immediate ER.','next_steps':'- Emergency protocol','patient_explanation':'You need immediate attention.'}
    if any(k in s for k in ['cough','burning','back pain','rash','headache','knee']):
        return {'urgency':'Routine','confidence':'Medium','reasoning':'Non-emergency.','recommendation':'Schedule appointment.','next_steps':'- Monitor','patient_explanation':'See a doctor soon.'}
    return {'urgency':'Self-care','confidence':'Medium','reasoning':'Self-limiting.','recommendation':'Home care.','next_steps':'- Rest','patient_explanation':'Manage at home.'}

def parse_triage(text):
    r = {'urgency':'Unknown','confidence':'Medium','reasoning':'','recommendation':'','next_steps':'','patient_explanation':'','sources':'','evidence':''}
    if not text: return r
    m = re.search(r'Urgency:\s*(Urgent|Routine|Self-care|Self care)', text, re.IGNORECASE)
    if m: r['urgency'] = 'Self-care' if 'self' in m.group(1).lower() else m.group(1).capitalize()
    m = re.search(r'Confidence:\s*(High|Medium|Low)', text, re.IGNORECASE)
    if m: r['confidence'] = m.group(1).capitalize()
    for key, pat in [('reasoning',r'Reasoning:\s*\n?(.*?)(?=\n(?:Recommendation|Next|Patient):|\Z)'),
                     ('recommendation',r'Recommendation:\s*\n?(.*?)(?=\n(?:Next|Patient):|\Z)'),
                     ('next_steps',r'Next [Ss]teps:\s*\n?(.*?)(?=\n(?:Patient):|\Z)'),
                     ('patient_explanation',r'Patient [Ee]xplanation:\s*\n?(.*?)(?=\Z)')]:
        m = re.search(pat, text, re.DOTALL|re.IGNORECASE)
        if m: r[key] = m.group(1).strip()
    return r

# ============================================================================
# FULL PIPELINE: Router → Triage → Evaluator
# ============================================================================
def run_pipeline(symptoms):
    router = router_check_completeness(symptoms)
    if not router['complete']:
        return {'status':'incomplete','questions':router['questions']}
    triage = run_triage(symptoms)
    original = triage.get('reasoning','')
    enhanced = False
    if GEMINI_AVAILABLE:
        new_reasoning = evaluator_enhance_triage(original, symptoms)
        if new_reasoning != original:
            triage['reasoning'] = new_reasoning
            enhanced = True
    return {'status':'complete','triage':triage,'original_reasoning':original,
            'evaluator_enhanced':enhanced}

# ============================================================================
# BOOKING (real schedule)
# ============================================================================
@st.cache_resource
def load_schedules():
    import pandas as pd
    path = Path(__file__).parent / "agentic_triage_schedules.xlsx"
    if not path.exists(): return None, None
    return pd.read_excel(path, sheet_name='Emergency Schedule'), pd.read_excel(path, sheet_name='Routine Schedule')

EMERGENCY_DF, ROUTINE_DF = load_schedules()
SCHEDULE_AVAILABLE = EMERGENCY_DF is not None
if 'booked_slots' not in st.session_state:
    st.session_state.booked_slots = set()

def find_slot(df):
    if df is None: return None
    avail = df[(df['Available']=='Available')&(~df['Slot ID'].isin(st.session_state.booked_slots))]
    return avail.iloc[0] if not avail.empty else None

def book_action(cid, tier, symptoms=''):
    if tier == 'Urgent':
        if SCHEDULE_AVAILABLE:
            slot = find_slot(EMERGENCY_DF)
            if slot is not None:
                st.session_state.booked_slots.add(slot['Slot ID'])
                return {'status':'urgent_referral','type':'Immediate ER Admission','doctor':f"{slot['Doctor']} — On Duty",'time':'IMMEDIATELY','dept':'Emergency','room':slot['Room'],'booking_id':slot['Slot ID'],'agent_decision':'emergency_referral','instructions':f"Proceed to {slot['Room']} immediately. {slot['Doctor']} is on duty."}
        return {'status':'urgent_referral','type':'Immediate ER Admission','doctor':'On-duty physician','time':'IMMEDIATELY','dept':'Emergency','room':f'ER-{random.randint(1,3)}','booking_id':f'UR-{uuid.uuid4().hex[:6]}','agent_decision':'emergency_fallback','instructions':'Proceed to Emergency Room immediately.'}
    elif tier == 'Routine':
        if SCHEDULE_AVAILABLE:
            slot = find_slot(ROUTINE_DF)
            if slot is not None:
                st.session_state.booked_slots.add(slot['Slot ID'])
                return {'status':'booked','type':'Scheduled Appointment','doctor':slot['Doctor'],'time':f"{slot['Date']} ({slot['Day']}) at {slot['Time']}",'dept':slot['Department'],'room':slot['Room'],'booking_id':slot['Slot ID'],'agent_decision':'routine_appointment','instructions':f"Appointment with {slot['Doctor']} on {slot['Date']} at {slot['Time']} in {slot['Room']}. Arrive 15 min early."}
        return {'status':'booked','type':'Scheduled Appointment','doctor':'Dr. Ahmed','dept':'General','time':(datetime.now()+timedelta(days=3)).strftime('%B %d at %I:%M %p'),'room':'G-101','booking_id':f'BK-{uuid.uuid4().hex[:6]}','agent_decision':'routine_fallback','instructions':'Arrive 15 minutes early.'}
    return {'status':'self_care_issued','type':'Self-Care Guidance','doctor':None,'time':None,'dept':None,'room':None,'booking_id':f'SC-{uuid.uuid4().hex[:6]}','agent_decision':'self_care','instructions':'Manage at home. Return if worsening.','guidance':'- Rest and hydrate\n- Monitor symptoms\n- OTC medication as needed\n- Return if fever > 38.5C or worsening'}

# ============================================================================
# CUSTOM CSS
# ============================================================================
CUSTOM_CSS = """
<style>
div[data-testid="stMetric"] {background:#f8f9fa;padding:14px;border-radius:10px;border:1px solid #e9ecef;}
.case-card {background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:20px;margin:10px 0;box-shadow:0 1px 3px rgba(0,0,0,0.08);}
.status-badge {display:inline-block;padding:4px 12px;border-radius:20px;font-size:13px;font-weight:600;}
.badge-urgent {background:#fee2e2;color:#dc2626;}
.badge-routine {background:#fef3c7;color:#d97706;}
.badge-selfcare {background:#d1fae5;color:#059669;}
.badge-approve {background:#d1fae5;color:#059669;}
.badge-override {background:#fef3c7;color:#d97706;}
.section-header {font-size:14px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin:16px 0 8px 0;}
.ai-badge {display:inline-block;padding:3px 10px;border-radius:6px;font-size:12px;margin:2px;}
.ai-badge-on {background:#d1fae5;color:#059669;}
.ai-badge-off {background:#f1f5f9;color:#94a3b8;}
</style>
"""

# ============================================================================
# APP LAYOUT
# ============================================================================
st.set_page_config(page_title="Triage Decision Support", layout="wide", initial_sidebar_state="expanded")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Dashboard:", ["Patient","Nurse","Developer"])
stats = db_stats()
st.sidebar.markdown("---")
st.sidebar.caption("SYSTEM STATUS")
st.sidebar.markdown(f"Cases: **{stats['total']}** ({stats['pending']} pending)")
st.sidebar.markdown(f"RAG: **{'ON' if RAG_AVAILABLE else 'OFF'}** {'('+str(len(KB_CHUNKS))+' chunks)' if RAG_AVAILABLE else ''}")
st.sidebar.markdown(f"Scheduler: **{'ON' if SCHEDULE_AVAILABLE else 'OFF'}**")
st.sidebar.markdown(f"Agentic AI: **{'ON' if GEMINI_AVAILABLE else 'OFF'}**")
if SCHEDULE_AVAILABLE:
    st.sidebar.markdown(f"Booked: **{len(st.session_state.booked_slots)}** slots")
st.sidebar.markdown("---")
if st.sidebar.button("Reset All Data", type="secondary", use_container_width=True):
    conn = sqlite3.connect(DB_PATH); conn.execute("DELETE FROM cases"); conn.commit(); conn.close()
    st.session_state.booked_slots = set()
    for k in list(st.session_state.keys()):
        if k.startswith('followup'): del st.session_state[k]
    st.rerun()

# ==================== PATIENT DASHBOARD ======================================
if page == "Patient":
    st.title("Patient Triage Dashboard")
    tab_submit, tab_check = st.tabs(["Submit Case", "Check Results"])

    with tab_submit:
        # Stage 1: Initial submission
        if 'followup_stage' not in st.session_state:
            st.session_state.followup_stage = 'initial'

        if st.session_state.followup_stage == 'initial':
            with st.form("intake"):
                col_t, col_s = st.columns([1,3])
                with col_t:
                    ticket = st.text_input("Ticket #", placeholder="e.g. T1")
                with col_s:
                    symptoms = st.text_area("Describe your symptoms",
                        placeholder="What are you feeling? When did it start? How severe is it?", height=140)
                go = st.form_submit_button("Submit", type="primary", use_container_width=True)

            if go and ticket and symptoms and len(symptoms.strip()) >= 10:
                with st.spinner("Analyzing..."):
                    result = run_pipeline(symptoms)

                if result['status'] == 'incomplete' and result['questions']:
                    # Store state for follow-up
                    st.session_state.followup_stage = 'questions'
                    st.session_state.followup_ticket = ticket
                    st.session_state.followup_symptoms = symptoms
                    st.session_state.followup_questions = result['questions']
                    st.rerun()
                else:
                    # Complete — save case
                    cid = f"CASE-{uuid.uuid4().hex[:8]}"
                    tri = result['triage']
                    db_insert({'case_id':cid,'ticket_number':ticket,'patient_symptoms':symptoms,
                        'enriched_symptoms':symptoms,'status':'pending',
                        'llm_urgency':tri.get('urgency',''),'llm_reasoning':tri.get('reasoning',''),
                        'llm_reasoning_original':result.get('original_reasoning',''),
                        'llm_recommendation':tri.get('recommendation',''),
                        'llm_next_steps':tri.get('next_steps',''),
                        'llm_sources':tri.get('sources',''),'llm_evidence':tri.get('evidence',''),
                        'llm_patient_explanation':tri.get('patient_explanation',''),
                        'llm_confidence':tri.get('confidence',''),'rag_mode':tri.get('rag_mode',''),
                        'router_complete':True,'evaluator_enhanced':result.get('evaluator_enhanced',False),
                        'created_at':datetime.utcnow().isoformat(),'updated_at':datetime.utcnow().isoformat()})
                    st.success(f"Submitted! Case ID: `{cid}`")
                    st.info("A nurse will review your case shortly. Use **Check Results** tab.")
            elif go:
                st.error("Please fill in both fields with enough detail.")

        # Stage 2: Follow-up questions with answer boxes
        elif st.session_state.followup_stage == 'questions':
            st.markdown("### We need a few more details")
            st.markdown(f"**Your initial description:** _{st.session_state.followup_symptoms}_")
            st.markdown("---")

            questions = st.session_state.followup_questions
            with st.form("followup_form"):
                answers = []
                for i, q in enumerate(questions):
                    st.markdown(f"**Question {i+1}:** {q}")
                    ans = st.text_input(f"Your answer:", key=f"fq_{i}", placeholder="Type your answer here...")
                    answers.append(ans)
                    st.markdown("")  # spacing

                col_back, col_submit = st.columns(2)
                with col_back:
                    back = st.form_submit_button("Back", use_container_width=True)
                with col_submit:
                    submit_answers = st.form_submit_button("Submit Answers", type="primary", use_container_width=True)

            if back:
                st.session_state.followup_stage = 'initial'
                st.rerun()

            if submit_answers:
                if all(a.strip() for a in answers):
                    # Build enriched symptoms
                    qa_text = '\n'.join(f"Q: {q}\nA: {a}" for q, a in zip(questions, answers))
                    enriched = f"{st.session_state.followup_symptoms}\n\nAdditional details:\n{qa_text}"

                    with st.spinner("Re-analyzing with additional details..."):
                        result = run_pipeline(enriched)

                    if result['status'] == 'complete':
                        cid = f"CASE-{uuid.uuid4().hex[:8]}"
                        tri = result['triage']
                        db_insert({'case_id':cid,
                            'ticket_number':st.session_state.followup_ticket,
                            'patient_symptoms':st.session_state.followup_symptoms,
                            'enriched_symptoms':enriched,'status':'pending',
                            'llm_urgency':tri.get('urgency',''),'llm_reasoning':tri.get('reasoning',''),
                            'llm_reasoning_original':result.get('original_reasoning',''),
                            'llm_recommendation':tri.get('recommendation',''),
                            'llm_next_steps':tri.get('next_steps',''),
                            'llm_sources':tri.get('sources',''),'llm_evidence':tri.get('evidence',''),
                            'llm_patient_explanation':tri.get('patient_explanation',''),
                            'llm_confidence':tri.get('confidence',''),'rag_mode':tri.get('rag_mode',''),
                            'router_complete':True,'router_questions':json.dumps(questions),
                            'router_answers':json.dumps(answers),
                            'evaluator_enhanced':result.get('evaluator_enhanced',False),
                            'created_at':datetime.utcnow().isoformat(),'updated_at':datetime.utcnow().isoformat()})
                        st.success(f"Submitted! Case ID: `{cid}`")
                        st.info("A nurse will review shortly. Use **Check Results** tab.")
                        st.session_state.followup_stage = 'initial'
                    else:
                        st.warning("Still not enough detail. Please visit the clinic directly.")
                        st.session_state.followup_stage = 'initial'
                else:
                    st.error("Please answer all questions before submitting.")

    with tab_check:
        chk = st.text_input("Enter your ticket number:", placeholder="e.g. T1")
        if chk:
            cases = db_get_by_ticket(chk)
            if not cases:
                st.info("No cases found for this ticket.")
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
                        st.markdown("---")
                        st.markdown("#### Your Appointment")
                        for fld, lbl in [('type','Type'),('doctor','Doctor'),('time','When'),('dept','Department'),('room','Room')]:
                            if bd.get(fld): st.markdown(f"**{lbl}:** {bd[fld]}")
                        if bd.get('instructions'): st.info(bd['instructions'])
                        if bd.get('guidance'): st.markdown(bd['guidance'])
                    if c.get('llm_patient_explanation'):
                        st.markdown("---")
                        st.markdown("#### What This Means For You")
                        st.markdown(c['llm_patient_explanation'])
                    if c.get('nurse_notes') and c['nurse_notes'].strip():
                        st.markdown("---")
                        st.markdown("#### Nurse Notes")
                        st.markdown(f"> {c['nurse_notes']}")
                    if c.get('nurse_override_reason') and c['nurse_override_reason'].strip():
                        st.markdown(f"**Nurse comment:** {c['nurse_override_reason']}")
                    st.markdown("---")
                    st.caption(f"Submitted: {(c.get('created_at',''))[:19].replace('T',' ')} | Reviewed: {(c.get('nurse_timestamp',''))[:19].replace('T',' ')}")
                elif c['status'] == 'pending':
                    st.warning(f"**{c['case_id']}** — Waiting for nurse review...")

# ==================== NURSE DASHBOARD ========================================
elif page == "Nurse":
    nurse = check_nurse_login()
    if not nurse:
        nurse_login_form()
    else:
        st.title("Nurse Triage Review")
        st.markdown(f"Logged in as: **{nurse}**")
        if st.sidebar.button("Logout"):
            del st.session_state['nurse_name']; del st.session_state['nurse_username']; st.rerun()

        pending = db_get_all(status='pending')
        reviewed = db_get_all(status='reviewed')

        # Metrics
        m1,m2,m3 = st.columns(3)
        m1.metric("Pending", len(pending))
        m2.metric("Reviewed", len(reviewed))
        ov = sum(1 for c in reviewed if (c.get('nurse_action') or '').startswith('override'))
        m3.metric("Overrides", ov)

        tab_r, tab_h = st.tabs(["Review Cases","History"])

        with tab_r:
            if not pending:
                st.info("No pending cases.")
                if st.button("Refresh", use_container_width=True): st.rerun()
            else:
                opts = {f"{c['case_id']} — Ticket #{c['ticket_number']}":c['case_id'] for c in pending}
                sel = st.selectbox("Select case:", list(opts.keys()))
                case = db_get_one(opts[sel])

                if case:
                    st.markdown("---")
                    cl, cr = st.columns([3,2])

                    with cl:
                        # Patient info
                        st.markdown('<p class="section-header">Patient Information</p>', unsafe_allow_html=True)
                        st.text_area("Symptoms", value=case['patient_symptoms'], height=80, disabled=True, label_visibility="collapsed")

                        # Show follow-up Q&A if exists
                        if case.get('router_questions'):
                            try:
                                qs = json.loads(case['router_questions'])
                                ans = json.loads(case.get('router_answers','[]'))
                                with st.expander("Follow-up Q&A (from Router)", expanded=False):
                                    for i, (q, a) in enumerate(zip(qs, ans)):
                                        st.markdown(f"**Q{i+1}:** {q}")
                                        st.markdown(f"**A{i+1}:** {a}")
                            except: pass

                        # Agentic AI badges
                        st.markdown('<p class="section-header">Agentic AI Pipeline</p>', unsafe_allow_html=True)
                        bc1, bc2, bc3 = st.columns(3)
                        with bc1:
                            if case.get('router_complete'):
                                st.markdown('<span class="ai-badge ai-badge-on">Router: Complete</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="ai-badge ai-badge-off">Router: Skipped</span>', unsafe_allow_html=True)
                        with bc2:
                            if case.get('evaluator_enhanced'):
                                st.markdown('<span class="ai-badge ai-badge-on">Evaluator: Enhanced</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="ai-badge ai-badge-off">Evaluator: Original</span>', unsafe_allow_html=True)
                        with bc3:
                            st.markdown(f'<span class="ai-badge ai-badge-on">Mode: {case.get("rag_mode","")}</span>', unsafe_allow_html=True)

                        # AI Assessment
                        st.markdown('<p class="section-header">AI Triage Assessment</p>', unsafe_allow_html=True)
                        tier = case.get('llm_urgency','Unknown')
                        conf = case.get('llm_confidence','Unknown')
                        tc = {'Urgent':'red','Routine':'orange','Self-care':'green'}
                        cc = {'High':'green','Medium':'orange','Low':'red'}
                        ac1, ac2 = st.columns(2)
                        ac1.markdown(f"**Urgency:** :{tc.get(tier,'gray')}[{tier}]")
                        ac2.markdown(f"**Confidence:** :{cc.get(conf,'gray')}[{conf}]")

                        # Evaluator before/after
                        if case.get('evaluator_enhanced') and case.get('llm_reasoning_original'):
                            with st.expander("Evaluator: Before vs After", expanded=False):
                                eb, ea = st.columns(2)
                                with eb:
                                    st.caption("ORIGINAL (RAG output)")
                                    st.markdown(case.get('llm_reasoning_original','')[:500])
                                with ea:
                                    st.caption("ENHANCED (Evaluator)")
                                    st.markdown(case.get('llm_reasoning','')[:500])

                        # Evidence
                        if case.get('llm_evidence') and case['llm_evidence'].strip():
                            with st.expander("Retrieved Medical Evidence", expanded=True):
                                st.markdown(case['llm_evidence'])

                        if case.get('llm_reasoning'):
                            st.markdown(f"**Reasoning:** {case['llm_reasoning']}")
                        if case.get('llm_recommendation'):
                            st.markdown(f"**Recommendation:** {case['llm_recommendation']}")
                        if case.get('llm_next_steps'):
                            st.markdown(f"**Next Steps:** {case['llm_next_steps']}")

                        if case.get('llm_sources') and case['llm_sources'] != 'None':
                            with st.expander("Sources (IEEE)"):
                                st.markdown(case['llm_sources'])

                    with cr:
                        st.markdown('<p class="section-header">Your Decision</p>', unsafe_allow_html=True)
                        tiers = ['Urgent','Routine','Self-care']
                        idx = tiers.index(tier) if tier in tiers else 0
                        nurse_tier = st.selectbox("Final Tier:", tiers, index=idx)

                        is_ov = nurse_tier != tier
                        if is_ov:
                            d = "UPGRADING" if tiers.index(nurse_tier) < tiers.index(tier) else "DOWNGRADING"
                            st.warning(f"You are **{d}** from {tier} → {nurse_tier}")
                            ov_reason = st.text_area("Override justification (REQUIRED):", height=100,
                                placeholder=f"Why are you {d.lower()}? This will be logged for audit.")
                        else:
                            ov_reason = ''

                        notes = st.text_area("Clinical notes (optional):", height=80)

                        can = True
                        if is_ov and not (ov_reason or '').strip():
                            can = False
                            st.error("Justification required for overrides.")

                        if st.button("Confirm Decision", type="primary", use_container_width=True, disabled=not can):
                            if nurse_tier == tier: act = 'approve'
                            elif tiers.index(nurse_tier) < tiers.index(tier): act = 'override_upgrade'
                            else: act = 'override_downgrade'
                            bk = book_action(case['case_id'], nurse_tier, case.get('patient_symptoms',''))
                            db_update(case['case_id'], {
                                'nurse_tier':nurse_tier,'nurse_action':act,'nurse_notes':notes,
                                'nurse_name':nurse,'nurse_override_reason':ov_reason,
                                'nurse_timestamp':datetime.utcnow().isoformat(),
                                'final_tier':nurse_tier,'booking_status':bk['status'],
                                'booking_details':json.dumps(bk),
                                'booking_agent_decision':bk.get('agent_decision',''),
                                'status':'reviewed'})
                            st.success(f"Recorded: **{nurse_tier}** ({act})")
                            st.balloons(); st.rerun()

        with tab_h:
            if not reviewed:
                st.info("No reviewed cases.")
            else:
                filt = st.selectbox("Filter:", ['All','Approved','Overridden'])
                for c in reviewed:
                    a = c.get('nurse_action','')
                    if filt == 'Approved' and a != 'approve': continue
                    if filt == 'Overridden' and not a.startswith('override'): continue
                    em = {'approve':'✅','override_upgrade':'⬆️','override_downgrade':'⬇️'}.get(a,'❓')
                    lb = {'approve':'Approved','override_upgrade':'Upgraded','override_downgrade':'Downgraded'}.get(a,a)
                    with st.expander(f"{em} {c['case_id']} — AI: {c.get('llm_urgency','?')} → Nurse: {c.get('final_tier','?')} ({lb})"):
                        st.markdown(f"**Symptoms:** {(c.get('patient_symptoms',''))[:200]}...")
                        if c.get('evaluator_enhanced'): st.caption("Evaluator: Enhanced")
                        if c.get('nurse_override_reason'): st.markdown(f"**Override reason:** {c['nurse_override_reason']}")
                        if c.get('nurse_notes'): st.markdown(f"**Notes:** {c['nurse_notes']}")
                        st.caption(f"Nurse: {c.get('nurse_name','')} | {(c.get('nurse_timestamp',''))[:19]}")

# ==================== DEVELOPER DASHBOARD ====================================
elif page == "Developer":
    st.title("Developer Dashboard")
    st.caption("Audit trail for all processed cases — AI vs Nurse comparison")

    all_cases = db_get_all()
    if not all_cases:
        st.info("No cases in the database.")
    else:
        # Summary metrics
        total = len(all_cases)
        reviewed = [c for c in all_cases if c['status'] == 'reviewed']
        pending = [c for c in all_cases if c['status'] == 'pending']
        overrides = [c for c in reviewed if (c.get('nurse_action') or '').startswith('override')]
        upgrades = [c for c in overrides if c.get('nurse_action') == 'override_upgrade']
        downgrades = [c for c in overrides if c.get('nurse_action') == 'override_downgrade']

        d1,d2,d3,d4,d5 = st.columns(5)
        d1.metric("Total Cases", total)
        d2.metric("Reviewed", len(reviewed))
        d3.metric("Pending", len(pending))
        d4.metric("Overrides", len(overrides))
        d5.metric("Override Rate", f"{len(overrides)/len(reviewed)*100:.0f}%" if reviewed else "N/A")

        # Override breakdown
        if overrides:
            st.markdown("---")
            st.markdown("### Override Analysis")
            oc1, oc2 = st.columns(2)
            oc1.metric("Upgrades (nurse said MORE urgent)", len(upgrades))
            oc2.metric("Downgrades (nurse said LESS urgent)", len(downgrades))

        # Case-by-case comparison
        st.markdown("---")
        st.markdown("### Case-by-Case Comparison")

        view_filter = st.selectbox("Show:", ['All','Reviewed','Pending','Overrides Only'])

        if view_filter == 'Reviewed':
            display = reviewed
        elif view_filter == 'Pending':
            display = pending
        elif view_filter == 'Overrides Only':
            display = overrides
        else:
            display = all_cases

        for c in display:
            ai_tier = c.get('llm_urgency','—')
            nurse_tier = c.get('final_tier','—')
            action = c.get('nurse_action','pending')
            agree = ai_tier == nurse_tier

            # Color code
            if action == 'approve':
                border = '#059669'  # green
            elif action.startswith('override'):
                border = '#d97706'  # orange
            else:
                border = '#94a3b8'  # gray

            st.markdown(f"""<div style="border-left:4px solid {border};padding:12px 16px;margin:8px 0;background:#fafafa;border-radius:0 8px 8px 0;">
                <b>{c['case_id']}</b> — Ticket #{c.get('ticket_number','')} — <i>{c['status']}</i>
            </div>""", unsafe_allow_html=True)

            with st.expander(f"Details: {c['case_id']}", expanded=False):
                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    st.caption("AI DECISION")
                    st.markdown(f"**Urgency:** {ai_tier}")
                    st.markdown(f"**Confidence:** {c.get('llm_confidence','—')}")
                    st.markdown(f"**Mode:** {c.get('rag_mode','—')}")
                    if c.get('evaluator_enhanced'):
                        st.caption("Evaluator: Enhanced")
                with cc2:
                    st.caption("NURSE DECISION")
                    st.markdown(f"**Final Tier:** {nurse_tier}")
                    st.markdown(f"**Action:** {action}")
                    st.markdown(f"**Nurse:** {c.get('nurse_name','—')}")
                with cc3:
                    st.caption("COMPARISON")
                    if c['status'] != 'reviewed':
                        st.markdown("*Pending review*")
                    elif agree:
                        st.markdown("**Agreement**")
                    else:
                        st.markdown(f"**Disagreement:** {ai_tier} → {nurse_tier}")

                if c.get('nurse_override_reason'):
                    st.markdown(f"**Override justification:** {c['nurse_override_reason']}")
                if c.get('nurse_notes'):
                    st.markdown(f"**Notes:** {c['nurse_notes']}")

                st.caption(f"Symptoms: {(c.get('patient_symptoms',''))[:150]}...")

        # Export
        st.markdown("---")
        if st.button("Export All Cases as JSON", use_container_width=True):
            export = json.dumps(all_cases, indent=2, default=str)
            st.download_button("Download JSON", data=export, file_name="triage_cases_export.json",
                             mime="application/json", use_container_width=True)
