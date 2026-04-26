import streamlit as st
import sqlite3, json, uuid, random, re, os
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================================
# DATABASE — auto-migration, never DROP TABLE
# ============================================================================
DB_PATH = "triage_hil.db"

SCHEMA_COLS = [
    ('case_id','TEXT PRIMARY KEY'),('ticket_number','TEXT'),('patient_symptoms','TEXT'),
    ('enriched_symptoms','TEXT'),('status','TEXT DEFAULT "processing"'),
    ('llm_urgency','TEXT'),('llm_reasoning','TEXT'),('llm_reasoning_original','TEXT'),
    ('llm_recommendation','TEXT'),('llm_next_steps','TEXT'),('llm_sources','TEXT'),
    ('llm_evidence','TEXT'),('llm_patient_explanation','TEXT'),('llm_confidence','TEXT'),
    ('rag_mode','TEXT'),('router_complete','BOOLEAN'),('router_questions','TEXT'),
    ('router_answers','TEXT'),('evaluator_enhanced','BOOLEAN'),
    ('nurse_tier','TEXT'),('nurse_action','TEXT'),('nurse_notes','TEXT'),
    ('nurse_override_reason','TEXT'),('nurse_timestamp','TEXT'),('nurse_name','TEXT'),
    ('final_tier','TEXT'),('booking_status','TEXT'),('booking_details','TEXT'),
    ('booking_agent_decision','TEXT'),('created_at','TEXT'),('updated_at','TEXT'),
]

def init_db():
    conn = sqlite3.connect(DB_PATH)
    col_defs = ', '.join(f'{n} {t}' for n,t in SCHEMA_COLS)
    conn.execute(f"CREATE TABLE IF NOT EXISTS cases ({col_defs})")
    # Auto-migrate: add missing columns without dropping anything
    cur = conn.execute("PRAGMA table_info(cases)")
    existing = {row[1] for row in cur.fetchall()}
    for name, typ in SCHEMA_COLS:
        if name not in existing:
            base_type = typ.split()[0]
            try: conn.execute(f"ALTER TABLE cases ADD COLUMN {name} {base_type}")
            except: pass
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
        rows = conn.execute('SELECT * FROM cases WHERE status=? ORDER BY created_at DESC',(status,)).fetchall()
    else:
        rows = conn.execute('SELECT * FROM cases ORDER BY created_at DESC').fetchall()
    conn.close(); return [dict(r) for r in rows]

def db_get_one(cid):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    row = conn.execute('SELECT * FROM cases WHERE case_id=?',(cid,)).fetchone()
    conn.close(); return dict(row) if row else None

def db_get_by_ticket(t):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT * FROM cases WHERE ticket_number=? ORDER BY created_at DESC',(t,)).fetchall()
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
    'haya':{'password':'123','name':'Nurse Haya'},
    'malek':{'password':'123','name':'Nurse Malek'},
    'yomna':{'password':'123','name':'Nurse Yomna'},
    'admin':{'password':'admin2026','name':'Administrator'},
}

def check_login():
    return st.session_state.get('nurse_name')

def login_form():
    st.markdown("### Nurse Login")
    with st.form("login"):
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        go = st.form_submit_button("Login", type="primary", use_container_width=True)
    if go:
        if user in NURSES and NURSES[user]['password'] == pw:
            st.session_state.nurse_name = NURSES[user]['name']
            st.session_state.nurse_username = user
            st.rerun()
        else:
            st.error("Invalid credentials.")

# ============================================================================
# GEMINI
# ============================================================================
GEMINI_OK = False
try:
    from google import genai; GEMINI_OK = True
except: pass

def llm(prompt, system="", temp=0.0, tokens=900):
    from google import genai
    from google.genai import types
    r = genai.Client().models.generate_content(
        model='gemini-2.5-flash', contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system or None, temperature=temp,
            max_output_tokens=tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0)))
    return r.text or ''

# ============================================================================
# AGENTIC ROUTER — adaptive multi-round questioning
# ============================================================================
def router_assess(symptoms, previous_qa=None):
    """Assess completeness. If incomplete, generate targeted follow-ups based on what's known."""
    if not GEMINI_OK:
        return {"ready": True, "questions": []}
    history = ""
    if previous_qa:
        history = "\n\nPrevious follow-up Q&A:\n" + "\n".join(
            f"Q: {q}\nA: {a}" for q, a in previous_qa)
    prompt = f'''You are an experienced intake nurse assessing if you have enough info to triage.

Patient said: "{symptoms}"{history}

RULES:
- READY means you have: specific symptom(s) + duration OR severity + relevant context
- If NOT ready, ask 1-2 SHORT targeted questions about what's STILL missing
- Questions must be DIFFERENT from any already asked
- Focus on: location, duration, severity, associated symptoms, medical history
- If the patient has given 3+ rounds of answers, you likely have enough — mark ready

Respond ONLY in JSON:
{{"ready": true, "questions": []}}
or
{{"ready": false, "questions": ["question 1", "question 2"]}}'''
    try:
        raw = llm(prompt, tokens=300)
        clean = re.sub(r'```json\s*|```\s*', '', raw).strip()
        r = json.loads(clean)
        return {"ready": r.get("ready", True), "questions": r.get("questions", [])}
    except:
        return {"ready": True, "questions": []}

# ============================================================================
# EVALUATOR — senior physician review
# ============================================================================
def evaluator(reasoning, symptoms):
    if not GEMINI_OK: return reasoning
    prompt = f'''Senior emergency physician reviewing triage:

Patient: "{symptoms}"

Assessment:
{reasoning}

Enhance: check urgency, add missed red flags, complete next steps.
Keep same [1],[2] citations. Do NOT invent sources. Do NOT change urgency unless clearly wrong.
Output enhanced assessment in SAME format.'''
    try:
        e = llm(prompt, tokens=1500)
        return e if e and len(e) > 50 else reasoning
    except:
        return reasoning

# ============================================================================
# RAG KB
# ============================================================================
@st.cache_resource
def load_kb():
    from rank_bm25 import BM25Okapi
    p = Path(__file__).parent / "chunked_docs_phase2.json"
    if not p.exists(): return None, None, None
    with open(p,'r',encoding='utf-8') as f: chunks = json.load(f)
    wr = re.compile(r'\w+')
    tok = [wr.findall(c['page_content'].lower()) for c in chunks]
    return chunks, BM25Okapi(tok), wr

KB, BM25, WR = load_kb()
RAG_ON = KB is not None

def retrieve(query, k=5):
    if not RAG_ON: return [],[],'',[]
    tok = WR.findall(query.lower())
    sc = BM25.get_scores(tok)
    top = sc.argsort()[::-1][:k]
    ev, src, ret = [],[],[]
    for rk, ix in enumerate(top, 1):
        if sc[ix] <= 0: continue
        c = KB[ix]; m = c.get('metadata',{})
        tp=m.get('topic','?'); sr=m.get('source','?'); url=m.get('url','')
        sec=m.get('section_title','General'); dt=m.get('document_date','n.d.')
        txt = c['page_content']
        ev.append(f"[{rk}] **{tp}** — {sec}\n> {txt[:300]}...")
        src.append(f'[{rk}] {sr}, "{tp}," section: "{sec}," {dt}. Available: {url}')
        ret.append({'rank':rk,'topic':tp,'section':sec,'score':float(sc[ix]),'content':txt[:500]})
    return ret, ev, '\n'.join(src)

# ============================================================================
# TRIAGE
# ============================================================================
SYS_TRIAGE = """ER triage assistant. Use retrieved evidence, cite [1],[2].
Categories: Urgent / Routine / Self-care.
Output:
Urgency: <tier>
Confidence: <High/Medium/Low>
Reasoning: <2-4 sentences with citations>
Recommendation: <action>
Next steps:
- <step>
Patient explanation: <2-3 simple sentences>"""

def triage(symptoms):
    ret, ev, sources = retrieve(symptoms, k=5)
    has = len(ret) > 0 and ret[0]['score'] > 0.5
    if has:
        ctx = '\n\n'.join(f"[{r['rank']}] ({r['topic']})\n{r['content']}" for r in ret)
        msg = f'Patient:\n"""{symptoms}"""\n\nEvidence:\n{ctx}\n\nClassify urgency.'
        mode = 'grounded_rag'
    else:
        msg = f'Patient:\n"""{symptoms}"""'; mode = 'fallback_judgment'
    try:
        raw = llm(msg, system=SYS_TRIAGE, tokens=1200)
        r = parse(raw)
    except:
        r = demo_triage(symptoms)
    r['evidence'] = '\n\n'.join(ev) if ev else ''
    r['sources'] = sources or 'None'
    r['rag_mode'] = mode
    return r

def demo_triage(s):
    sl = s.lower()
    if any(k in sl for k in ['chest pain','stroke','breathing','unconscious','bleeding','seizure']):
        return {'urgency':'Urgent','confidence':'High','reasoning':'Red-flag symptoms.','recommendation':'Immediate ER.','next_steps':'- Emergency protocol','patient_explanation':'You need immediate attention.'}
    if any(k in sl for k in ['cough','burning','back pain','rash','headache','knee']):
        return {'urgency':'Routine','confidence':'Medium','reasoning':'Non-emergency.','recommendation':'Schedule appointment.','next_steps':'- Monitor','patient_explanation':'See a doctor soon.'}
    return {'urgency':'Self-care','confidence':'Medium','reasoning':'Self-limiting.','recommendation':'Home care.','next_steps':'- Rest','patient_explanation':'Manage at home.'}

def parse(text):
    r = {'urgency':'Unknown','confidence':'Medium','reasoning':'','recommendation':'','next_steps':'','patient_explanation':''}
    if not text: return r
    m = re.search(r'Urgency:\s*(Urgent|Routine|Self-care|Self care)', text, re.I)
    if m: r['urgency'] = 'Self-care' if 'self' in m.group(1).lower() else m.group(1).capitalize()
    m = re.search(r'Confidence:\s*(High|Medium|Low)', text, re.I)
    if m: r['confidence'] = m.group(1).capitalize()
    for k, p in [('reasoning',r'Reasoning:\s*\n?(.*?)(?=\n(?:Recommendation|Next|Patient):|\Z)'),
                  ('recommendation',r'Recommendation:\s*\n?(.*?)(?=\n(?:Next|Patient):|\Z)'),
                  ('next_steps',r'Next [Ss]teps:\s*\n?(.*?)(?=\n(?:Patient):|\Z)'),
                  ('patient_explanation',r'Patient [Ee]xplanation:\s*\n?(.*?)(?=\Z)')]:
        m = re.search(p, text, re.DOTALL|re.I)
        if m: r[k] = m.group(1).strip()
    return r

# ============================================================================
# PIPELINE: Router (adaptive) → Triage → Evaluator
# ============================================================================
def run_pipeline(symptoms, prev_qa=None):
    # Router: check if enough info (adaptive based on previous Q&A)
    router = router_assess(symptoms, prev_qa)
    if not router['ready'] and router['questions']:
        return {'status':'incomplete','questions':router['questions']}
    # Triage
    tri = triage(symptoms)
    orig = tri.get('reasoning','')
    # Evaluator
    enhanced = False
    if GEMINI_OK:
        new_r = evaluator(orig, symptoms)
        if new_r != orig:
            tri['reasoning'] = new_r; enhanced = True
    return {'status':'complete','triage':tri,'original_reasoning':orig,'evaluator_enhanced':enhanced}

# ============================================================================
# BOOKING — always future dates
# ============================================================================
@st.cache_resource
def load_sched():
    import pandas as pd
    p = Path(__file__).parent / "agentic_triage_schedules.xlsx"
    if not p.exists(): return None, None
    return pd.read_excel(p, sheet_name='Emergency Schedule'), pd.read_excel(p, sheet_name='Routine Schedule')

EM_DF, RT_DF = load_sched()
SCHED_ON = EM_DF is not None
if 'booked' not in st.session_state: st.session_state.booked = set()

def find_slot(df):
    if df is None: return None
    av = df[(df['Available']=='Available')&(~df['Slot ID'].isin(st.session_state.booked))]
    return av.iloc[0] if not av.empty else None

def future_date(slot_date, slot_day, slot_time):
    """Ensure appointment date is always in the future."""
    today = datetime.now()
    try:
        orig = datetime.strptime(str(slot_date)[:10], '%Y-%m-%d')
        if orig.date() < today.date():
            # Shift forward by whole weeks until it's in the future
            days_diff = (today.date() - orig.date()).days
            weeks_ahead = (days_diff // 7) + 1
            new = orig + timedelta(weeks=weeks_ahead)
            return new.strftime('%B %d, %Y') + f" ({slot_day}) at {slot_time}"
        return orig.strftime('%B %d, %Y') + f" ({slot_day}) at {slot_time}"
    except:
        return f"{slot_date} ({slot_day}) at {slot_time}"

def book(cid, tier, symptoms=''):
    if tier == 'Urgent':
        if SCHED_ON:
            s = find_slot(EM_DF)
            if s is not None:
                st.session_state.booked.add(s['Slot ID'])
                return {'status':'urgent_referral','type':'Immediate ER Admission',
                    'doctor':f"{s['Doctor']} — On Duty",'time':'IMMEDIATELY — No appointment needed',
                    'dept':'Emergency','room':s['Room'],'booking_id':s['Slot ID'],
                    'agent_decision':'emergency_referral',
                    'instructions':f"Proceed to {s['Room']} immediately. {s['Doctor']} is on duty."}
        return {'status':'urgent_referral','type':'Immediate ER Admission',
            'doctor':'On-duty physician','time':'IMMEDIATELY','dept':'Emergency',
            'room':f'ER-{random.randint(1,3)}','booking_id':f'UR-{uuid.uuid4().hex[:6]}',
            'agent_decision':'emergency_fallback',
            'instructions':'Proceed to Emergency Room immediately.'}
    elif tier == 'Routine':
        if SCHED_ON:
            s = find_slot(RT_DF)
            if s is not None:
                st.session_state.booked.add(s['Slot ID'])
                ft = future_date(s['Date'], s['Day'], s['Time'])
                return {'status':'booked','type':'Scheduled Appointment',
                    'doctor':s['Doctor'],'time':ft,'dept':s['Department'],
                    'room':s['Room'],'booking_id':s['Slot ID'],
                    'agent_decision':'routine_appointment',
                    'instructions':f"Appointment with {s['Doctor']} on {ft} in {s['Room']}. Arrive 15 min early."}
        d = random.randint(2,7)
        ft = (datetime.now()+timedelta(days=d)).strftime('%B %d, %Y at %I:%M %p')
        return {'status':'booked','type':'Scheduled Appointment','doctor':'Dr. Ahmed',
            'dept':'General','time':ft,'room':'G-101','booking_id':f'BK-{uuid.uuid4().hex[:6]}',
            'agent_decision':'routine_fallback',
            'instructions':f'Appointment on {ft}. Arrive 15 min early.'}
    return {'status':'self_care_issued','type':'Self-Care Guidance','doctor':None,'time':None,
        'dept':None,'room':None,'booking_id':f'SC-{uuid.uuid4().hex[:6]}',
        'agent_decision':'self_care',
        'instructions':'Manage at home. Return if worsening.',
        'guidance':'- Rest and hydrate\n- Monitor symptoms\n- OTC medication as needed\n- Return if fever > 38.5°C or worsening'}

# ============================================================================
# CUSTOM CSS
# ============================================================================
CSS = """<style>
:root {--primary:#028090;--success:#059669;--warning:#d97706;--danger:#dc2626;--muted:#64748b;--bg:#f8fafc;--card:#ffffff;--border:#e2e8f0;}
div[data-testid="stMetric"]{background:var(--card);padding:16px;border-radius:12px;border:1px solid var(--border);box-shadow:0 1px 2px rgba(0,0,0,.04);}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,.06);}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600;margin:2px 4px;}
.badge-green{background:#d1fae5;color:#059669;}.badge-orange{background:#fef3c7;color:#d97706;}
.badge-red{background:#fee2e2;color:#dc2626;}.badge-gray{background:#f1f5f9;color:#64748b;}
.badge-blue{background:#dbeafe;color:#2563eb;}
.section-label{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin:20px 0 8px 0;}
.progress-bar{height:4px;background:#e2e8f0;border-radius:2px;margin:8px 0;}
.progress-fill{height:4px;background:var(--primary);border-radius:2px;transition:width .3s;}
.compare-card{background:#f8fafc;border:1px solid var(--border);border-radius:8px;padding:14px;margin:4px 0;}
</style>"""

# ============================================================================
# APP
# ============================================================================
st.set_page_config(page_title="Triage Decision Support", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

# Session state defaults
for key in ['followup_stage','followup_ticket','followup_symptoms','followup_qa','followup_round']:
    if key not in st.session_state:
        st.session_state[key] = 'initial' if key == 'followup_stage' else ([] if key == 'followup_qa' else (0 if key == 'followup_round' else ''))

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Patient","Nurse","Developer"], label_visibility="collapsed")
stats = db_stats()
st.sidebar.markdown("---")
st.sidebar.markdown(f'<p class="section-label">System Status</p>', unsafe_allow_html=True)
col_s1, col_s2 = st.sidebar.columns(2)
col_s1.markdown(f"**Cases:** {stats['total']}")
col_s2.markdown(f"**Pending:** {stats['pending']}")
st.sidebar.markdown(f'<span class="badge {"badge-green" if RAG_ON else "badge-gray"}">RAG {"ON" if RAG_ON else "OFF"}</span>'
    f' <span class="badge {"badge-green" if SCHED_ON else "badge-gray"}">Scheduler {"ON" if SCHED_ON else "OFF"}</span>'
    f' <span class="badge {"badge-green" if GEMINI_OK else "badge-gray"}">AI {"ON" if GEMINI_OK else "OFF"}</span>',
    unsafe_allow_html=True)
st.sidebar.markdown("---")
if st.sidebar.button("Reset All Data", use_container_width=True):
    conn = sqlite3.connect(DB_PATH); conn.execute("DELETE FROM cases"); conn.commit(); conn.close()
    st.session_state.booked = set()
    st.session_state.followup_stage = 'initial'
    st.session_state.followup_qa = []
    st.session_state.followup_round = 0
    st.rerun()

# ======================== PATIENT ============================================
if page == "Patient":
    st.title("Patient Triage Dashboard")
    tab_new, tab_check = st.tabs(["Submit Case","Check Results"])

    with tab_new:
        stage = st.session_state.followup_stage

        if stage == 'initial':
            st.markdown('<div class="card">', unsafe_allow_html=True)
            with st.form("intake"):
                c1, c2 = st.columns([1,3])
                with c1: ticket = st.text_input("Ticket #", placeholder="T1")
                with c2: symptoms = st.text_area("Describe your symptoms",
                    placeholder="What are you feeling? When did it start? How severe?", height=140)
                go = st.form_submit_button("Submit", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if go and ticket and symptoms and len(symptoms.strip()) >= 10:
                with st.spinner("AI is analyzing your symptoms..."):
                    result = run_pipeline(symptoms)
                if result['status'] == 'incomplete' and result['questions']:
                    st.session_state.followup_stage = 'questions'
                    st.session_state.followup_ticket = ticket
                    st.session_state.followup_symptoms = symptoms
                    st.session_state.followup_qa = []
                    st.session_state.followup_round = 1
                    st.session_state.followup_current_qs = result['questions']
                    st.rerun()
                elif result['status'] == 'complete':
                    cid = f"CASE-{uuid.uuid4().hex[:8]}"
                    tri = result['triage']
                    db_insert({'case_id':cid,'ticket_number':ticket,'patient_symptoms':symptoms,
                        'enriched_symptoms':symptoms,'status':'pending',
                        'llm_urgency':tri.get('urgency',''),'llm_reasoning':tri.get('reasoning',''),
                        'llm_reasoning_original':result.get('original_reasoning',''),
                        'llm_recommendation':tri.get('recommendation',''),
                        'llm_next_steps':tri.get('next_steps',''),'llm_sources':tri.get('sources',''),
                        'llm_evidence':tri.get('evidence',''),'llm_patient_explanation':tri.get('patient_explanation',''),
                        'llm_confidence':tri.get('confidence',''),'rag_mode':tri.get('rag_mode',''),
                        'router_complete':True,'evaluator_enhanced':result.get('evaluator_enhanced',False),
                        'created_at':datetime.utcnow().isoformat(),'updated_at':datetime.utcnow().isoformat()})
                    st.success(f"Submitted! Case ID: `{cid}`")
                    st.info("A nurse will review your case shortly.")
            elif go:
                st.error("Please fill in both fields with sufficient detail.")

        elif stage == 'questions':
            # Progress indicator
            round_num = st.session_state.followup_round
            st.markdown(f'<p class="section-label">Assessment in progress — Round {round_num}</p>', unsafe_allow_html=True)
            progress = min(round_num / 3 * 100, 90)
            st.markdown(f'<div class="progress-bar"><div class="progress-fill" style="width:{progress}%"></div></div>', unsafe_allow_html=True)

            st.markdown(f"**Your description:** _{st.session_state.followup_symptoms}_")

            # Show previous Q&A if any
            if st.session_state.followup_qa:
                with st.expander(f"Previous answers ({len(st.session_state.followup_qa)} questions answered)", expanded=False):
                    for q, a in st.session_state.followup_qa:
                        st.markdown(f"**Q:** {q}")
                        st.markdown(f"**A:** {a}")

            st.markdown("---")
            st.markdown("### We need a few more details")
            questions = st.session_state.get('followup_current_qs', [])

            with st.form(f"followup_{round_num}"):
                answers = []
                for i, q in enumerate(questions):
                    st.markdown(f'<div class="card" style="padding:14px">', unsafe_allow_html=True)
                    st.markdown(f"**Question {i+1}:** {q}")
                    a = st.text_input("Your answer:", key=f"fqa_{round_num}_{i}",
                        placeholder="Type your answer here...", label_visibility="collapsed")
                    answers.append(a)
                    st.markdown('</div>', unsafe_allow_html=True)

                fc1, fc2 = st.columns(2)
                with fc1: back = st.form_submit_button("Start Over", use_container_width=True)
                with fc2: submit = st.form_submit_button("Submit Answers", type="primary", use_container_width=True)

            if back:
                st.session_state.followup_stage = 'initial'
                st.session_state.followup_qa = []
                st.session_state.followup_round = 0
                st.rerun()

            if submit:
                if all(a.strip() for a in answers):
                    # Accumulate Q&A
                    new_qa = list(zip(questions, answers))
                    all_qa = st.session_state.followup_qa + new_qa

                    # Build enriched symptoms
                    qa_text = '\n'.join(f"Q: {q}\nA: {a}" for q, a in all_qa)
                    enriched = f"{st.session_state.followup_symptoms}\n\nAdditional details:\n{qa_text}"

                    with st.spinner("Re-analyzing with your additional details..."):
                        result = run_pipeline(enriched, prev_qa=all_qa)

                    if result['status'] == 'incomplete' and result['questions'] and round_num < 3:
                        # Another round of questions (max 3 rounds)
                        st.session_state.followup_qa = all_qa
                        st.session_state.followup_round = round_num + 1
                        st.session_state.followup_current_qs = result['questions']
                        st.rerun()
                    else:
                        # Either complete OR max rounds reached — proceed with best effort
                        if result['status'] != 'complete':
                            result = run_pipeline(enriched)  # Force triage even if router says incomplete
                            if result['status'] != 'complete':
                                # Final fallback: run triage directly
                                tri_result = triage(enriched)
                                result = {'status':'complete','triage':tri_result,
                                    'original_reasoning':tri_result.get('reasoning',''),
                                    'evaluator_enhanced':False}

                        cid = f"CASE-{uuid.uuid4().hex[:8]}"
                        tri = result['triage']
                        db_insert({'case_id':cid,
                            'ticket_number':st.session_state.followup_ticket,
                            'patient_symptoms':st.session_state.followup_symptoms,
                            'enriched_symptoms':enriched,'status':'pending',
                            'llm_urgency':tri.get('urgency',''),'llm_reasoning':tri.get('reasoning',''),
                            'llm_reasoning_original':result.get('original_reasoning',''),
                            'llm_recommendation':tri.get('recommendation',''),
                            'llm_next_steps':tri.get('next_steps',''),'llm_sources':tri.get('sources',''),
                            'llm_evidence':tri.get('evidence',''),
                            'llm_patient_explanation':tri.get('patient_explanation',''),
                            'llm_confidence':tri.get('confidence',''),'rag_mode':tri.get('rag_mode',''),
                            'router_complete':True,'router_questions':json.dumps([q for q,_ in all_qa]),
                            'router_answers':json.dumps([a for _,a in all_qa]),
                            'evaluator_enhanced':result.get('evaluator_enhanced',False),
                            'created_at':datetime.utcnow().isoformat(),'updated_at':datetime.utcnow().isoformat()})
                        st.success(f"Submitted! Case ID: `{cid}`")
                        st.info("A nurse will review shortly.")
                        st.session_state.followup_stage = 'initial'
                        st.session_state.followup_qa = []
                        st.session_state.followup_round = 0
                else:
                    st.error("Please answer all questions.")

    with tab_check:
        chk = st.text_input("Enter your ticket number:", placeholder="e.g. T1")
        if chk:
            cases = db_get_by_ticket(chk)
            if not cases: st.info("No cases found.")
            for c in cases:
                tier = c.get('final_tier', c.get('llm_urgency',''))
                if c['status'] == 'reviewed':
                    colors = {'Urgent':'red','Routine':'orange','Self-care':'green'}
                    icons = {'Urgent':'🔴','Routine':'🟡','Self-care':'🟢'}
                    st.markdown(f'<div class="card">', unsafe_allow_html=True)
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
                        st.markdown("---"); st.markdown("#### What This Means For You")
                        st.markdown(c['llm_patient_explanation'])
                    if c.get('nurse_notes') and c['nurse_notes'].strip():
                        st.markdown("---"); st.markdown("#### Nurse Notes")
                        st.markdown(f"> {c['nurse_notes']}")
                    if c.get('nurse_override_reason') and c['nurse_override_reason'].strip():
                        st.markdown(f"**Nurse comment:** {c['nurse_override_reason']}")
                    st.caption(f"Submitted: {(c.get('created_at',''))[:16].replace('T',' ')} | Reviewed: {(c.get('nurse_timestamp','') or '')[:16].replace('T',' ')}")
                    st.markdown('</div>', unsafe_allow_html=True)
                elif c['status'] == 'pending':
                    st.warning(f"**{c['case_id']}** — Waiting for nurse review...")

# ======================== NURSE =============================================
elif page == "Nurse":
    nurse = check_login()
    if not nurse:
        login_form()
    else:
        st.title("Nurse Triage Review")
        st.markdown(f"Logged in as: **{nurse}**")
        if st.sidebar.button("Logout"):
            for k in ['nurse_name','nurse_username']:
                if k in st.session_state: del st.session_state[k]
            st.rerun()

        pending = db_get_all(status='pending')
        reviewed = db_get_all(status='reviewed')
        m1,m2,m3 = st.columns(3)
        m1.metric("Pending",len(pending)); m2.metric("Reviewed",len(reviewed))
        m3.metric("Overrides",sum(1 for c in reviewed if (c.get('nurse_action') or '').startswith('override')))

        tab_r, tab_h = st.tabs(["Review","History"])

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
                        st.markdown('<p class="section-label">Patient Information</p>', unsafe_allow_html=True)
                        st.markdown(f'<div class="card" style="padding:14px"><b>Symptoms:</b><br>{case["patient_symptoms"]}</div>', unsafe_allow_html=True)

                        # Show follow-up Q&A
                        if case.get('router_questions'):
                            try:
                                qs = json.loads(case['router_questions'])
                                ans = json.loads(case.get('router_answers','[]'))
                                if qs:
                                    with st.expander(f"Follow-up Q&A ({len(qs)} questions)", expanded=False):
                                        for i,(q,a) in enumerate(zip(qs,ans)):
                                            st.markdown(f"**Q{i+1}:** {q}\n**A{i+1}:** {a}")
                            except: pass

                        # Pipeline badges
                        st.markdown('<p class="section-label">AI Pipeline</p>', unsafe_allow_html=True)
                        badges = ""
                        badges += f'<span class="badge badge-green">Router: Complete</span>' if case.get('router_complete') else '<span class="badge badge-gray">Router: Skipped</span>'
                        badges += f' <span class="badge badge-green">Evaluator: Enhanced</span>' if case.get('evaluator_enhanced') else ' <span class="badge badge-gray">Evaluator: Original</span>'
                        badges += f' <span class="badge badge-blue">Mode: {case.get("rag_mode","")}</span>'
                        st.markdown(badges, unsafe_allow_html=True)

                        # AI Assessment
                        st.markdown('<p class="section-label">AI Triage Assessment</p>', unsafe_allow_html=True)
                        tier = case.get('llm_urgency','Unknown')
                        conf = case.get('llm_confidence','Unknown')
                        tc = {'Urgent':'badge-red','Routine':'badge-orange','Self-care':'badge-green'}
                        cc = {'High':'badge-green','Medium':'badge-orange','Low':'badge-red'}
                        st.markdown(f'<span class="badge {tc.get(tier,"badge-gray")}">Urgency: {tier}</span>'
                            f' <span class="badge {cc.get(conf,"badge-gray")}">Confidence: {conf}</span>', unsafe_allow_html=True)

                        # Evaluator comparison
                        if case.get('evaluator_enhanced') and case.get('llm_reasoning_original'):
                            with st.expander("Evaluator: Before vs After"):
                                b1,b2 = st.columns(2)
                                with b1:
                                    st.markdown('<div class="compare-card">', unsafe_allow_html=True)
                                    st.caption("ORIGINAL"); st.markdown(case.get('llm_reasoning_original','')[:500])
                                    st.markdown('</div>', unsafe_allow_html=True)
                                with b2:
                                    st.markdown('<div class="compare-card">', unsafe_allow_html=True)
                                    st.caption("ENHANCED"); st.markdown(case.get('llm_reasoning','')[:500])
                                    st.markdown('</div>', unsafe_allow_html=True)

                        if case.get('llm_evidence'):
                            with st.expander("Retrieved Medical Evidence", expanded=True):
                                st.markdown(case['llm_evidence'])
                        if case.get('llm_reasoning'): st.markdown(f"**Reasoning:** {case['llm_reasoning']}")
                        if case.get('llm_recommendation'): st.markdown(f"**Recommendation:** {case['llm_recommendation']}")
                        if case.get('llm_next_steps'): st.markdown(f"**Next Steps:** {case['llm_next_steps']}")
                        if case.get('llm_sources') and case['llm_sources'] != 'None':
                            with st.expander("Sources (IEEE)"): st.markdown(case['llm_sources'])

                    with cr:
                        st.markdown('<p class="section-label">Your Decision</p>', unsafe_allow_html=True)
                        tiers = ['Urgent','Routine','Self-care']
                        idx = tiers.index(tier) if tier in tiers else 0
                        nurse_tier = st.selectbox("Final Tier:", tiers, index=idx)

                        is_ov = nurse_tier != tier
                        if is_ov:
                            d = "UPGRADING" if tiers.index(nurse_tier) < tiers.index(tier) else "DOWNGRADING"
                            st.warning(f"**{d}:** {tier} → {nurse_tier}")
                            ov_reason = st.text_area("Justification (REQUIRED):", height=100,
                                placeholder=f"Why {d.lower()}? Logged for audit.")
                        else:
                            ov_reason = ''

                        notes = st.text_area("Clinical notes (optional):", height=80)

                        can = not is_ov or bool((ov_reason or '').strip())
                        if is_ov and not can: st.error("Justification required.")

                        if st.button("Confirm Decision", type="primary", use_container_width=True, disabled=not can):
                            act = 'approve' if nurse_tier == tier else ('override_upgrade' if tiers.index(nurse_tier) < tiers.index(tier) else 'override_downgrade')
                            bk = book(case['case_id'], nurse_tier, case.get('patient_symptoms',''))
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
            if not reviewed: st.info("No history yet.")
            else:
                filt = st.selectbox("Filter:",['All','Approved','Overridden'])
                for c in reviewed:
                    a = c.get('nurse_action','')
                    if filt == 'Approved' and a != 'approve': continue
                    if filt == 'Overridden' and not a.startswith('override'): continue
                    em = {'approve':'✅','override_upgrade':'⬆️','override_downgrade':'⬇️'}.get(a,'❓')
                    lb = {'approve':'Approved','override_upgrade':'Upgraded','override_downgrade':'Downgraded'}.get(a,a)
                    with st.expander(f"{em} {c['case_id']} — AI: {c.get('llm_urgency','?')} → Nurse: {c.get('final_tier','?')} ({lb})"):
                        st.markdown(f"**Symptoms:** {(c.get('patient_symptoms',''))[:200]}...")
                        if c.get('nurse_override_reason'): st.markdown(f"**Justification:** {c['nurse_override_reason']}")
                        if c.get('nurse_notes'): st.markdown(f"**Notes:** {c['nurse_notes']}")
                        st.caption(f"Nurse: {c.get('nurse_name','')} | {(c.get('nurse_timestamp','') or '')[:16]}")

# ======================== DEVELOPER ==========================================
elif page == "Developer":
    st.title("Developer Dashboard")
    st.caption("Audit trail — AI vs Nurse decision comparison")

    all_cases = db_get_all()
    if not all_cases:
        st.info("No cases yet.")
    else:
        reviewed = [c for c in all_cases if c['status'] == 'reviewed']
        overrides = [c for c in reviewed if (c.get('nurse_action') or '').startswith('override')]
        ups = [c for c in overrides if c.get('nurse_action') == 'override_upgrade']
        downs = [c for c in overrides if c.get('nurse_action') == 'override_downgrade']

        d1,d2,d3,d4,d5 = st.columns(5)
        d1.metric("Total",len(all_cases)); d2.metric("Reviewed",len(reviewed))
        d3.metric("Pending",len(all_cases)-len(reviewed))
        d4.metric("Overrides",len(overrides))
        d5.metric("Rate",f"{len(overrides)/max(len(reviewed),1)*100:.0f}%")

        if overrides:
            st.markdown("---")
            st.markdown("### Override Breakdown")
            o1,o2 = st.columns(2)
            o1.metric("Upgrades (nurse → more urgent)",len(ups))
            o2.metric("Downgrades (nurse → less urgent)",len(downs))

        st.markdown("---")
        st.markdown("### Case Comparison")
        vf = st.selectbox("Show:",['All','Reviewed','Overrides Only'])
        display = overrides if vf == 'Overrides Only' else (reviewed if vf == 'Reviewed' else all_cases)

        for c in display:
            ai = c.get('llm_urgency','—'); nr = c.get('final_tier','—')
            act = c.get('nurse_action','pending')
            agree = ai == nr and c['status'] == 'reviewed'
            border = '#059669' if agree else ('#d97706' if act.startswith('override') else '#94a3b8')

            st.markdown(f'<div style="border-left:4px solid {border};padding:10px 16px;margin:6px 0;background:#fafafa;border-radius:0 8px 8px 0;">'
                f'<b>{c["case_id"]}</b> — Ticket #{c.get("ticket_number","")} — <code>{c["status"]}</code></div>', unsafe_allow_html=True)

            with st.expander(f"Details: {c['case_id']}"):
                x1,x2,x3 = st.columns(3)
                with x1:
                    st.markdown('<div class="compare-card">', unsafe_allow_html=True)
                    st.caption("AI DECISION")
                    st.markdown(f"**Urgency:** {ai}")
                    st.markdown(f"**Confidence:** {c.get('llm_confidence','—')}")
                    st.markdown(f"**Mode:** {c.get('rag_mode','—')}")
                    if c.get('evaluator_enhanced'): st.caption("Evaluator: Enhanced")
                    st.markdown('</div>', unsafe_allow_html=True)
                with x2:
                    st.markdown('<div class="compare-card">', unsafe_allow_html=True)
                    st.caption("NURSE DECISION")
                    st.markdown(f"**Tier:** {nr}"); st.markdown(f"**Action:** {act}")
                    st.markdown(f"**Nurse:** {c.get('nurse_name','—')}")
                    st.markdown('</div>', unsafe_allow_html=True)
                with x3:
                    st.markdown('<div class="compare-card">', unsafe_allow_html=True)
                    st.caption("RESULT")
                    if c['status'] != 'reviewed': st.markdown("*Pending*")
                    elif agree: st.markdown('**Agreement** ✅')
                    else: st.markdown(f"**Disagreement:** {ai} → {nr}")
                    st.markdown('</div>', unsafe_allow_html=True)
                if c.get('nurse_override_reason'):
                    st.markdown(f"**Override justification:** {c['nurse_override_reason']}")
                if c.get('nurse_notes'):
                    st.markdown(f"**Notes:** {c['nurse_notes']}")
                # Booking info
                try:
                    bd = json.loads(c.get('booking_details','{}') or '{}')
                    if bd.get('type'):
                        st.markdown(f"**Booking:** {bd['type']} — {bd.get('doctor','')} — {bd.get('time','')}")
                except: pass
                st.caption(f"Symptoms: {(c.get('patient_symptoms',''))[:120]}...")

        st.markdown("---")
        if st.button("Export All as JSON", use_container_width=True):
            st.download_button("Download", data=json.dumps(all_cases, indent=2, default=str),
                file_name="triage_export.json", mime="application/json", use_container_width=True)
