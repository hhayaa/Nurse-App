import streamlit as st
import sqlite3, json, uuid, random, re, os
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================================
# DATABASE -- auto-migration, NEVER drops data
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
        prompt_chain_complete BOOLEAN, prompt_chain_questions TEXT, prompt_chain_answers TEXT,
        prompt_chain_rounds INTEGER, gate_decision TEXT,
        agent_action_trace TEXT, agent_confidence TEXT, agent_plan TEXT,
        evaluator_enhanced BOOLEAN,
        nurse_tier TEXT, nurse_action TEXT, nurse_notes TEXT,
        nurse_override_reason TEXT, nurse_timestamp TEXT, nurse_name TEXT,
        final_tier TEXT, booking_status TEXT, booking_details TEXT,
        booking_agent_decision TEXT,
        created_at TEXT, updated_at TEXT)""")
    cur = conn.execute("PRAGMA table_info(cases)")
    existing = {row[1] for row in cur.fetchall()}
    new_cols = [('enriched_symptoms','TEXT'),('llm_reasoning_original','TEXT'),
        ('prompt_chain_complete','BOOLEAN'),('prompt_chain_questions','TEXT'),
        ('prompt_chain_answers','TEXT'),('prompt_chain_rounds','INTEGER'),
        ('gate_decision','TEXT'),('agent_action_trace','TEXT'),
        ('agent_confidence','TEXT'),('agent_plan','TEXT'),
        ('evaluator_enhanced','BOOLEAN'),('booking_agent_decision','TEXT'),
        ('router_complete','BOOLEAN'),('router_questions','TEXT'),('router_answers','TEXT')]
    for col, typ in new_cols:
        if col not in existing:
            try: conn.execute(f'ALTER TABLE cases ADD COLUMN {col} {typ}')
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

NURSES = {
    'haya':  {'password': '123', 'name': 'Haya'},
    'malek': {'password': '123', 'name': 'Malek'},
    'yomna': {'password': '123', 'name': 'Yomna'},
    'admin': {'password': 'admin2026', 'name': 'Administrator'},
}

def check_nurse_login():
    return st.session_state.get('nurse_name', None)

def nurse_login_form():
    st.title("Nurse Login")
    st.markdown("Please enter your credentials to access the triage review dashboard.")
    with st.form("nurse_login"):
        username = st.text_input("Username", placeholder="e.g. nurse_name")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
    if submitted:
        if username in NURSES and NURSES[username]['password'] == password:
            st.session_state.nurse_name = NURSES[username]['name']
            st.session_state.nurse_username = username
            st.rerun()
        else:
            st.error("Invalid username or password.")

GEMINI_OK = False
try:
    from google import genai; GEMINI_OK = True
except: pass

def gemini_call(prompt, system="", temperature=0.0, max_tokens=900):
    from google import genai
    from google.genai import types
    client = genai.Client()
    resp = client.models.generate_content(model='gemini-2.5-flash', contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system if system else None,
            temperature=temperature, max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0)))
    return resp.text or ''

# ============================================================================
# TOOL 1: CHECK_RED_FLAGS -- deterministic emergency keyword scan
# ============================================================================
EMERGENCY_KEYWORDS = [
    'chest pain', 'cannot breathe', 'can not breathe', 'difficulty breathing',
    'unconscious', 'seizure', 'stroke', 'bleeding heavily', 'heavy bleeding',
    'suicidal', 'suicide', 'overdose', 'choking', 'not breathing',
    'heart attack', 'severe allergic', 'anaphylaxis', 'collapsed',
    'unresponsive', 'severe burn', 'stabbed', 'shot', 'gunshot',
]

def tool_check_red_flags(symptoms):
    """Tool: Scan for emergency keywords. Returns list of matched flags."""
    found = []
    symptoms_lower = symptoms.lower()
    for kw in EMERGENCY_KEYWORDS:
        if kw in symptoms_lower:
            found.append(kw)
    return found

# ============================================================================
# TOOL 2: ASK_FOLLOWUP -- prompt chaining for vague inputs
# ============================================================================
MAX_CHAIN_ROUNDS = 3

def tool_ask_followup(symptoms, previous_qa=None):
    """Tool: Generate follow-up questions for vague patient inputs."""
    if not GEMINI_OK:
        return {"ready": True, "questions": []}

    chain_context = ""
    if previous_qa and len(previous_qa) > 0:
        chain_context = "\n\nPrevious follow-up answers:\n"
        chain_context += "\n".join(f"Q: {q}\nA: {a}" for q, a in previous_qa)

    prompt = f'''You are an ER intake nurse. Decide if a patient gave enough info to triage.

Patient said: "{symptoms}"{chain_context}

IMPORTANT: You are NOT filling out a form. Only ask if the input is genuinely VAGUE.

READY -- examples: "bad cough for 3 days", "fever and rash since yesterday", "chest pain radiating to left arm"
NOT READY -- examples: "I feel sick", "I need help", "I have pain"

DECISION RULES:
- If the patient described at least ONE specific symptom with ANY context, mark READY.
- If the patient gave ONLY vague words with NO specific symptom at all, mark NOT READY.
- "I feel sick" = NOT READY. "I feel sick and have a cough" = READY.
- "Pain" alone = NOT READY. "Back pain for 3 days" = READY.
If NOT READY, ask only 1-2 SHORT questions about what is critically missing.
Questions must be DIFFERENT from any already asked.

Respond ONLY in JSON:
{{"ready": true, "questions": []}}
or
{{"ready": false, "questions": ["question 1", "question 2"]}}'''

    try:
        raw = gemini_call(prompt, max_tokens=300)
        clean = re.sub(r'```json\s*|```\s*', '', raw).strip()
        r = json.loads(clean)
        return {"ready": r.get("ready", True), "questions": r.get("questions", [])}
    except:
        return {"ready": True, "questions": []}

# ============================================================================
# TOOL 3: SEARCH_KB -- BM25 retrieval from medical knowledge base
# ============================================================================
@st.cache_resource
def load_knowledge_base():
    from rank_bm25 import BM25Okapi
    kb_path = Path(__file__).parent / "chunked_docs_phase2.json"
    if not kb_path.exists(): return None, None, None
    with open(kb_path, 'r', encoding='utf-8') as f: chunks = json.load(f)
    word_re = re.compile(r'\w+')
    texts = [c['page_content'] for c in chunks]
    tokenized = [word_re.findall(t.lower()) for t in texts]
    bm25 = BM25Okapi(tokenized)
    return chunks, bm25, word_re

KB_CHUNKS, BM25_INDEX, WORD_RE = load_knowledge_base()
RAG_AVAILABLE = KB_CHUNKS is not None

def tool_search_kb(query, k=5):
    """Tool: Search medical knowledge base using BM25."""
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
        evidence_blocks.append(f"[{rank}] **{topic}** -- {section}\n> {content[:300]}...")
        source_lines.append(f'[{rank}] {source}, "{topic}," section: "{section}," {date}. [Online]. Available: {url}')
        retrieved.append({'rank':rank,'topic':topic,'section':section,'score':float(scores[idx]),'content':content[:500]})
    return retrieved, evidence_blocks, '\n'.join(source_lines)

# ============================================================================
# TOOL 4: ASSESS_URGENCY -- RAG-grounded LLM triage
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
Recommendation: <Clinical recommendation>
Next steps:
- <Action 1>
- <Action 2>
Patient explanation: <2-3 simple sentences for the patient.>"""

def tool_assess_urgency(symptoms):
    """Tool: Classify urgency using RAG evidence + Gemini."""
    retrieved, evidence_blocks, sources_text = tool_search_kb(symptoms, k=5)
    has_evidence = len(retrieved) > 0 and retrieved[0]['score'] > 0.5
    if has_evidence:
        context = '\n\n'.join(f"[{r['rank']}] ({r['topic']} - {r['section']})\n{r['content']}" for r in retrieved)
        user_msg = f'Patient symptoms:\n"""{symptoms}"""\n\nRetrieved medical evidence:\n{context}\n\nClassify urgency using the evidence above.'
        system = TRIAGE_GROUNDED; mode = 'grounded_rag'
    else:
        user_msg = f'Patient symptoms:\n"""{symptoms}"""'
        system = TRIAGE_FALLBACK; mode = 'fallback_judgment'
    try:
        raw = gemini_call(user_msg, system=system, max_tokens=1200)
        result = parse_triage(raw)
    except:
        result = run_triage_demo(symptoms)
    result['evidence'] = '\n\n'.join(evidence_blocks) if evidence_blocks else 'No evidence retrieved.'
    result['sources'] = sources_text if sources_text else 'None'
    result['rag_mode'] = mode
    return result

def run_triage_demo(symptoms):
    s = symptoms.lower()
    if any(k in s for k in ['chest pain','stroke','breathing','unconscious','bleeding','suicidal','seizure']):
        return {'urgency':'Urgent','confidence':'High','reasoning':'Red-flag symptoms.','recommendation':'Immediate ER.','next_steps':'- Emergency protocol','patient_explanation':'Your symptoms need immediate attention.'}
    if any(k in s for k in ['cough','burning','back pain','rash','headache','depressed','knee']):
        return {'urgency':'Routine','confidence':'Medium','reasoning':'Non-emergency.','recommendation':'Schedule appointment.','next_steps':'- Monitor symptoms','patient_explanation':'See a doctor within 1-2 weeks.'}
    return {'urgency':'Self-care','confidence':'Medium','reasoning':'Self-limiting.','recommendation':'Home management.','next_steps':'- Rest and monitor','patient_explanation':'You can manage this at home.'}

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
# AGENT CONFIDENCE CALIBRATION (Idea 2)
# The agent reports HOW confident it is, WHAT would change its mind,
# and WHAT the nurse should specifically watch for.
# ============================================================================
def agent_confidence_check(symptoms, urgency, evidence):
    """Agent reports calibrated confidence + guidance for nurse."""
    if not GEMINI_OK:
        return {"confidence_pct": 50, "would_change_if": "N/A", "nurse_watch_for": "N/A"}
    prompt = f'''You just classified a patient as {urgency}.

Patient: "{symptoms}"
Evidence: {(evidence or "None")[:300]}

Answer THREE questions honestly:
1. CONFIDENCE (0-100): How confident are you in this classification?
2. WOULD CHANGE IF: What ONE piece of information would make you change the urgency?
3. NURSE SHOULD CHECK: What should the nurse specifically verify?

Respond ONLY in JSON:
{{"confidence_pct": 85, "would_change_if": "patient reports chest tightness", "nurse_watch_for": "check vitals and family cardiac history"}}'''
    try:
        raw = gemini_call(prompt, max_tokens=300)
        clean = re.sub(r'```json\s*|```\s*', '', raw).strip()
        return json.loads(clean)
    except:
        return {"confidence_pct": 50, "would_change_if": "N/A", "nurse_watch_for": "N/A"}

# ============================================================================
# TRIAGE AGENT -- autonomous tool selection, self-reflection, confidence
#
# This is the CORE agentic component. Unlike a fixed pipeline, the agent
# DECIDES its own plan based on the patient's input:
#
#   1. CHECK_RED_FLAGS on every input (safety first)
#   2. If emergency -> fast-track to ASSESS_URGENCY (skip questions)
#   3. If vague -> ASK_FOLLOWUP (prompt chaining)
#   4. If clear -> SEARCH_KB + ASSESS_URGENCY
#   5. SELF_REFLECT on its own assessment (can trigger re-triage)
#   6. CONFIDENCE_CHECK (calibrated % + nurse guidance)
#
# Every decision is logged in an action trace visible to the nurse.
# ============================================================================
def is_vague(text):
    """Heuristic: is the input too vague for reliable triage?"""
    words = text.lower().split()
    if len(words) < 4:
        return True
    symptom_words = ['pain', 'ache', 'fever', 'cough', 'bleeding', 'swollen',
        'rash', 'nausea', 'vomiting', 'dizzy', 'headache', 'burn',
        'breathing', 'chest', 'stomach', 'back', 'throat', 'ear',
        'diarrhea', 'fatigue', 'weakness', 'numbness', 'tingling',
        'sore', 'cramp', 'itch', 'bruise', 'fracture', 'sprain']
    return not any(w in text.lower() for w in symptom_words)

def run_triage_agent(symptoms, previous_qa=None):
    """
    Autonomous triage agent with tool selection + self-reflection + confidence.

    The agent has 4 tools and DECIDES which to use:
      - CHECK_RED_FLAGS: emergency keyword scan
      - ASK_FOLLOWUP: prompt chaining for vague inputs
      - SEARCH_KB + ASSESS_URGENCY: RAG triage
      - SELF_REFLECT: reviews its own assessment, can trigger re-triage

    Returns full action trace so the nurse sees WHY it chose each action.
    """
    action_trace = []
    enriched = symptoms
    if previous_qa:
        qa_text = '\n'.join(f"Q: {q}\nA: {a}" for q, a in previous_qa)
        enriched = f"{symptoms}\n\nAdditional details:\n{qa_text}"

    round_num = len(previous_qa) if previous_qa else 0

    # ---- STEP 1: Always check red flags first (safety) ----
    red_flags = tool_check_red_flags(enriched)
    action_trace.append({
        "step": 1, "tool": "CHECK_RED_FLAGS",
        "reason": "Safety scan -- always runs first",
        "result": f"Found: {', '.join(red_flags)}" if red_flags else "No red flags detected"
    })

    # ---- STEP 2: Agent decides next action based on red flag results ----
    if red_flags:
        # EMERGENCY PATH: skip follow-up, go straight to triage
        action_trace.append({
            "step": 2, "tool": "DECISION",
            "reason": f"Emergency detected ({red_flags[0]}) -- skipping follow-up, fast-tracking to triage",
            "plan": "emergency_fast_track"
        })
        tri = tool_assess_urgency(enriched)
        action_trace.append({
            "step": 3, "tool": "ASSESS_URGENCY",
            "reason": "Emergency fast-track triage",
            "result": f"Urgency: {tri.get('urgency','Unknown')}"
        })
        # Skip self-reflection for emergencies -- speed matters
        conf = agent_confidence_check(enriched, tri.get('urgency',''), tri.get('evidence',''))
        action_trace.append({
            "step": 4, "tool": "CONFIDENCE_CHECK",
            "reason": "Calibrate confidence for nurse",
            "result": f"{conf.get('confidence_pct',50)}% confident"
        })
        return {
            'status': 'complete', 'triage': tri,
            'action_trace': action_trace, 'agent_plan': 'emergency_fast_track',
            'original_reasoning': tri.get('reasoning',''),
            'evaluator_enhanced': False,
            'gate_decision': f"emergency_bypass:{red_flags[0]}",
            'agent_confidence': conf,
        }

    # No emergency -- check if input is vague
    input_is_vague = is_vague(enriched)

    if input_is_vague and round_num < MAX_CHAIN_ROUNDS:
        # VAGUE PATH: need more info from patient
        chain_result = tool_ask_followup(enriched, previous_qa)
        if not chain_result['ready'] and chain_result['questions']:
            action_trace.append({
                "step": 2, "tool": "ASK_FOLLOWUP",
                "reason": "Input too vague -- need specific symptoms before triage",
                "result": f"Asking {len(chain_result['questions'])} question(s)"
            })
            return {
                'status': 'need_more_info',
                'questions': chain_result['questions'],
                'action_trace': action_trace,
                'agent_plan': 'gather_info',
                'gate_decision': 'need_more_info',
            }

    # Also check if prompt chain wants to ask (even if heuristic says not vague)
    if round_num < MAX_CHAIN_ROUNDS and not input_is_vague:
        chain_result = tool_ask_followup(enriched, previous_qa)
        if not chain_result['ready'] and chain_result['questions']:
            action_trace.append({
                "step": 2, "tool": "ASK_FOLLOWUP",
                "reason": "Prompt chain detected missing critical info",
                "result": f"Asking {len(chain_result['questions'])} question(s)"
            })
            return {
                'status': 'need_more_info',
                'questions': chain_result['questions'],
                'action_trace': action_trace,
                'agent_plan': 'gather_info',
                'gate_decision': 'need_more_info',
            }

    # STANDARD PATH or MAX ROUNDS: proceed to triage
    gate_reason = 'chain_ready' if round_num == 0 else (
        'max_rounds_reached' if round_num >= MAX_CHAIN_ROUNDS else 'chain_ready')

    action_trace.append({
        "step": 2, "tool": "DECISION",
        "reason": "Sufficient info available -- proceeding to RAG triage",
        "plan": "standard_triage"
    })

    # ---- STEP 3: RAG triage ----
    tri = tool_assess_urgency(enriched)
    original_reasoning = tri.get('reasoning', '')
    action_trace.append({
        "step": 3, "tool": "ASSESS_URGENCY",
        "reason": "RAG-grounded triage classification",
        "result": f"Urgency: {tri.get('urgency','Unknown')} | Mode: {tri.get('rag_mode','')}"
    })

    # ---- STEP 4: Self-reflection (can trigger re-triage) ----
    evaluator_ran = False
    if GEMINI_OK:
        reflect_prompt = f'''You are a senior emergency physician. You just triaged this patient as {tri.get("urgency","Unknown")}.

Patient: "{enriched}"
Your reasoning: "{original_reasoning}"

SELF-CHECK -- answer honestly:
1. Is the urgency level correct for these symptoms?
2. Did you miss any red flags or critical details?
3. Is the reasoning well-supported by evidence?

If CONFIDENT the assessment is correct:
{{"confident": true, "enhanced_reasoning": "your improved version of the reasoning"}}

If NOT CONFIDENT (something seems wrong):
{{"confident": false, "concern": "what worries you", "suggested_urgency": "Urgent or Routine or Self-care"}}

Respond ONLY in JSON.'''
        try:
            raw = gemini_call(reflect_prompt, max_tokens=500)
            clean = re.sub(r'```json\s*|```\s*', '', raw).strip()
            reflection = json.loads(clean)
        except:
            reflection = {"confident": True, "enhanced_reasoning": original_reasoning}

        if reflection.get('confident'):
            if reflection.get('enhanced_reasoning') and len(reflection['enhanced_reasoning']) > 50:
                tri['reasoning'] = reflection['enhanced_reasoning']
                evaluator_ran = True
            action_trace.append({
                "step": 4, "tool": "SELF_REFLECT",
                "reason": "Senior physician review -- assessment approved",
                "result": "Confident -- reasoning enhanced"
            })
        else:
            concern = reflection.get('concern', 'unclear assessment')
            suggested = reflection.get('suggested_urgency', '')
            action_trace.append({
                "step": 4, "tool": "SELF_REFLECT",
                "reason": f"NOT confident -- concern: {concern}",
                "result": f"Triggering re-triage (suggested: {suggested})"
            })
            # RE-TRIAGE with self-correction guidance
            correction_input = f"{enriched}\n\n[AGENT SELF-CORRECTION]: Previous assessment flagged as potentially wrong. Concern: {concern}. Consider urgency: {suggested}"
            tri = tool_assess_urgency(correction_input)
            evaluator_ran = True
            action_trace.append({
                "step": 5, "tool": "RE_TRIAGE",
                "reason": f"Self-correction based on concern: {concern}",
                "result": f"New urgency: {tri.get('urgency','Unknown')}"
            })

    # ---- STEP 5/6: Confidence calibration ----
    conf = agent_confidence_check(enriched, tri.get('urgency',''), tri.get('evidence',''))
    action_trace.append({
        "step": len(action_trace) + 1, "tool": "CONFIDENCE_CHECK",
        "reason": "Calibrate confidence and generate nurse guidance",
        "result": f"{conf.get('confidence_pct',50)}% | Watch: {conf.get('nurse_watch_for','N/A')}"
    })

    return {
        'status': 'complete', 'triage': tri,
        'action_trace': action_trace,
        'agent_plan': 'standard_triage',
        'original_reasoning': original_reasoning,
        'evaluator_enhanced': evaluator_ran,
        'gate_decision': gate_reason,
        'agent_confidence': conf,
    }

# ============================================================================
# CROSS-CASE LEARNING AGENT (Idea 3)
# Analyzes patterns across ALL reviewed cases to detect systematic biases.
# This answers the professor's question: "what new knowledge?"
# ============================================================================
def agent_cross_case_analysis():
    """Cross-case learning agent: finds patterns in AI vs nurse decisions."""
    reviewed = db_get_all(status='reviewed')
    if len(reviewed) < 3:
        return {"status": "insufficient_data", "message": "Need at least 3 reviewed cases"}
    case_summaries = []
    for c in reviewed:
        case_summaries.append(
            f"- AI: {c.get('llm_urgency','?')}, Nurse: {c.get('final_tier','?')}, "
            f"Action: {c.get('nurse_action','?')}, "
            f"Override reason: {c.get('nurse_override_reason','none')}, "
            f"Symptoms: {(c.get('patient_symptoms',''))[:80]}")
    prompt = f'''You are a clinical AI researcher analyzing triage decision patterns.

{len(reviewed)} reviewed cases (AI vs nurse decisions):
{chr(10).join(case_summaries)}

Analyze:
1. AGREEMENT_RATE: percentage AI and nurse agreed
2. BIAS_PATTERN: when they disagree, is there a consistent direction?
3. FAILURE_MODES: what case types does the AI get wrong?
4. IMPROVEMENT: what ONE change would reduce disagreements?
5. NOVEL_FINDING: any unexpected or counterintuitive pattern?

Respond ONLY in JSON:
{{"agreement_rate": "X%", "bias_pattern": "description", "failure_modes": ["mode1","mode2"], "improvement": "suggestion", "novel_finding": "finding or null"}}'''
    try:
        raw = gemini_call(prompt, max_tokens=600)
        clean = re.sub(r'```json\s*|```\s*', '', raw).strip()
        return json.loads(clean)
    except:
        return {"status": "error", "message": "Analysis failed"}

# ============================================================================
# HARD CODED BOOKING -- reads real schedule, always future dates
# ============================================================================
@st.cache_resource
def load_schedules():
    import pandas as pd
    path = Path(__file__).parent / "agentic_triage_schedules.xlsx"
    if not path.exists(): return None, None
    return pd.read_excel(path, sheet_name='Emergency Schedule'), pd.read_excel(path, sheet_name='Routine Schedule')

EMERGENCY_DF, ROUTINE_DF = load_schedules()
SCHEDULE_AVAILABLE = EMERGENCY_DF is not None
if 'booked_slots' not in st.session_state: st.session_state.booked_slots = set()

def future_date(slot_date, slot_day, slot_time):
    today = datetime.now()
    try:
        orig = datetime.strptime(str(slot_date)[:10], '%Y-%m-%d')
        if orig.date() < today.date():
            days_diff = (today.date() - orig.date()).days
            weeks_ahead = (days_diff // 7) + 1
            new_date = orig + timedelta(weeks=weeks_ahead)
            return new_date.strftime('%B %d, %Y') + f" ({slot_day}) at {slot_time}"
        return orig.strftime('%B %d, %Y') + f" ({slot_day}) at {slot_time}"
    except:
        return f"{slot_date} ({slot_day}) at {slot_time}"

def find_available_slot(schedule_df):
    if schedule_df is None: return None
    available = schedule_df[(schedule_df['Available']=='Available')&(~schedule_df['Slot ID'].isin(st.session_state.booked_slots))]
    return available.iloc[0] if not available.empty else None

def book_action(case_id, tier, symptoms=''):
    if tier == 'Urgent':
        if SCHEDULE_AVAILABLE:
            slot = find_available_slot(EMERGENCY_DF)
            if slot is not None:
                st.session_state.booked_slots.add(slot['Slot ID'])
                return {'status':'urgent_referral','type':'Immediate ER Admission','doctor':f"{slot['Doctor']} -- On Duty",'time':'IMMEDIATELY -- No appointment needed','dept':'Emergency','room':slot['Room'],'booking_id':slot['Slot ID'],'agent_decision':'emergency_referral','instructions':f"Proceed to {slot['Room']} immediately. {slot['Doctor']} is on duty."}
        return {'status':'urgent_referral','type':'Immediate ER Admission','doctor':'On-duty ER physician','time':'IMMEDIATELY','dept':'Emergency','room':f'ER-{random.randint(1,3)}','booking_id':f'UR-{uuid.uuid4().hex[:6]}','agent_decision':'emergency_fallback','instructions':'Proceed to Emergency Room immediately.'}
    elif tier == 'Routine':
        if SCHEDULE_AVAILABLE:
            slot = find_available_slot(ROUTINE_DF)
            if slot is not None:
                st.session_state.booked_slots.add(slot['Slot ID'])
                ft = future_date(slot['Date'], slot['Day'], slot['Time'])
                return {'status':'booked','type':'Scheduled Appointment','doctor':slot['Doctor'],'time':ft,'dept':slot['Department'],'room':slot['Room'],'booking_id':slot['Slot ID'],'agent_decision':'routine_appointment','instructions':f"Appointment with {slot['Doctor']} on {ft} in {slot['Room']}. Arrive 15 minutes early."}
        d = random.randint(2,10)
        return {'status':'booked','type':'Scheduled Appointment','doctor':'Dr. Hall','dept':'General','time':(datetime.now()+timedelta(days=d)).strftime('%B %d, %Y at %I:%M %p'),'room':f'G-{random.randint(101,104)}','booking_id':f'BK-{uuid.uuid4().hex[:6]}','agent_decision':'routine_fallback','instructions':'Arrive 15 minutes early.'}
    else:
        return {'status':'self_care_issued','type':'Self-Care Guidance','doctor':None,'time':None,'dept':None,'room':None,'booking_id':f'SC-{uuid.uuid4().hex[:6]}','agent_decision':'self_care','instructions':'Manage at home. Return if worsening.','guidance':'- Rest and hydrate\n- Monitor symptoms\n- OTC medication as needed\n- Return if fever > 38.5C or worsening'}

# ============================================================================
# STREAMLIT APP
# ============================================================================
st.set_page_config(page_title="Triage Decision Support System", layout="wide")
st.markdown("""<style>div[data-testid="stMetric"]{background:#f8f9fa;padding:12px;border-radius:8px;}</style>""", unsafe_allow_html=True)

if 'fu_stage' not in st.session_state: st.session_state.fu_stage = 'initial'
if 'fu_qa' not in st.session_state: st.session_state.fu_qa = []
if 'fu_round' not in st.session_state: st.session_state.fu_round = 0
if 'fu_ticket' not in st.session_state: st.session_state.fu_ticket = ''
if 'fu_symptoms' not in st.session_state: st.session_state.fu_symptoms = ''
if 'fu_questions' not in st.session_state: st.session_state.fu_questions = []

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Dashboard:", ["Patient Dashboard","Nurse Dashboard","Developer Dashboard"])
stats = db_stats()
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Cases:** {stats['total']} total | {stats['pending']} pending")
st.sidebar.markdown(f"**📚 RAG:** {'✅ Connected (' + str(len(KB_CHUNKS)) + ' chunks)' if RAG_AVAILABLE else '❌ Demo mode'}")
st.sidebar.markdown(f"**📅 Scheduler:** {'✅ Connected' if SCHEDULE_AVAILABLE else '❌ Demo mode'}")
st.sidebar.markdown(f"**🤖 Agentic AI:** {'✅ Enabled (Agent + Tools + Self-Reflect)' if GEMINI_OK else '❌ Disabled'}")
if SCHEDULE_AVAILABLE: st.sidebar.markdown(f"**Booked this session:** {len(st.session_state.booked_slots)}")
st.sidebar.markdown("---")
with st.sidebar.expander("Admin"):
    if st.button("Reset Database", type="secondary"):
        conn = sqlite3.connect(DB_PATH); conn.execute("DELETE FROM cases"); conn.commit(); conn.close()
        st.session_state.booked_slots = set(); st.session_state.fu_stage = 'initial'
        st.session_state.fu_qa = []; st.session_state.fu_round = 0; st.rerun()

# ======================== PATIENT DASHBOARD ==================================
if page == "Patient Dashboard":
    st.title("🏥 Patient Triage Dashboard")
    tab1, tab2 = st.tabs(["Submit New Case","Check Results"])
    with tab1:
        if st.session_state.fu_stage == 'initial':
            with st.form("pf"):
                ca, cb = st.columns([1,3])
                with ca:
                    ticket = st.text_input("Ticket Number", placeholder="e.g. T0")
                with cb: symptoms = st.text_area("Describe your symptoms", placeholder="Please describe what you are feeling, when it started, and how severe it is...", height=150)
                submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)
            if submitted:
                if not ticket or not symptoms: st.error("Please fill in both fields.")
                elif len(symptoms.strip()) < 10: st.warning("Please describe your symptoms in more detail.")
                else:
                    with st.spinner("🔍 AI agent is analyzing your symptoms..."):
                        result = run_triage_agent(symptoms)
                    if result['status'] == 'need_more_info' and result.get('questions'):
                        st.session_state.fu_stage = 'followup'; st.session_state.fu_ticket = ticket
                        st.session_state.fu_symptoms = symptoms; st.session_state.fu_qa = []
                        st.session_state.fu_round = 1; st.session_state.fu_questions = result['questions']
                        st.rerun()
                    elif result['status'] == 'complete':
                        cid = f"CASE-{uuid.uuid4().hex[:8]}"; tri = result['triage']
                        conf = result.get('agent_confidence', {})
                        db_insert({'case_id':cid,'ticket_number':ticket,'patient_symptoms':symptoms,'enriched_symptoms':symptoms,'status':'pending','llm_urgency':tri.get('urgency',''),'llm_reasoning':tri.get('reasoning',''),'llm_reasoning_original':result.get('original_reasoning',''),'llm_recommendation':tri.get('recommendation',''),'llm_next_steps':tri.get('next_steps',''),'llm_sources':tri.get('sources',''),'llm_evidence':tri.get('evidence',''),'llm_patient_explanation':tri.get('patient_explanation',''),'llm_confidence':tri.get('confidence',''),'rag_mode':tri.get('rag_mode',''),'prompt_chain_complete':True,'prompt_chain_rounds':0,'gate_decision':result.get('gate_decision',''),'agent_action_trace':json.dumps(result.get('action_trace',[])),'agent_confidence':json.dumps(conf),'agent_plan':result.get('agent_plan',''),'evaluator_enhanced':result.get('evaluator_enhanced',False),'created_at':datetime.utcnow().isoformat(),'updated_at':datetime.utcnow().isoformat()})
                        st.success("✅ Submitted!"); st.info(f"**Case ID:** `{cid}` -- A nurse will review shortly.")

        elif st.session_state.fu_stage == 'followup':
            round_num = st.session_state.fu_round
            st.markdown(f"### We need a few more details (Round {round_num}/{MAX_CHAIN_ROUNDS})")
            st.markdown(f"**Your initial description:** _{st.session_state.fu_symptoms}_")
            if st.session_state.fu_qa:
                with st.expander(f"Previous answers ({len(st.session_state.fu_qa)} answered)", expanded=False):
                    for q, a in st.session_state.fu_qa:
                        st.markdown(f"**Q:** {q}"); st.markdown(f"**A:** {a}"); st.markdown("")
            st.markdown("---")
            questions = st.session_state.fu_questions
            with st.form(f"followup_{round_num}"):
                answers = []
                for i, q in enumerate(questions):
                    st.markdown(f"**Question {i+1}:** {q}")
                    ans = st.text_input("Your answer:", key=f"fqa_{round_num}_{i}", placeholder="Type your answer here...")
                    answers.append(ans); st.markdown("")
                fc1, fc2 = st.columns(2)
                with fc1: back_btn = st.form_submit_button("Start Over", use_container_width=True)
                with fc2: submit_btn = st.form_submit_button("Submit Answers", type="primary", use_container_width=True)
            if back_btn:
                st.session_state.fu_stage = 'initial'; st.session_state.fu_qa = []; st.session_state.fu_round = 0; st.rerun()
            if submit_btn:
                if not all(a.strip() for a in answers): st.error("Please answer all questions.")
                else:
                    new_pairs = list(zip(questions, answers))
                    all_qa = st.session_state.fu_qa + new_pairs
                    qa_text = '\n'.join(f"Q: {q}\nA: {a}" for q, a in all_qa)
                    enriched = f"{st.session_state.fu_symptoms}\n\nAdditional details:\n{qa_text}"
                    with st.spinner("🔄 Re-analyzing with your additional details..."):
                        result = run_triage_agent(enriched, previous_qa=all_qa)
                    if result['status'] == 'need_more_info' and result.get('questions') and round_num < MAX_CHAIN_ROUNDS:
                        st.session_state.fu_qa = all_qa; st.session_state.fu_round = round_num + 1
                        st.session_state.fu_questions = result['questions']; st.rerun()
                    else:
                        if result['status'] != 'complete':
                            tri = tool_assess_urgency(enriched)
                            result = {'status':'complete','triage':tri,'original_reasoning':tri.get('reasoning',''),'evaluator_enhanced':False,'gate_decision':'max_rounds_forced','action_trace':[],'agent_confidence':{},'agent_plan':'forced'}
                        cid = f"CASE-{uuid.uuid4().hex[:8]}"; tri = result['triage']
                        conf = result.get('agent_confidence', {})
                        db_insert({'case_id':cid,'ticket_number':st.session_state.fu_ticket,'patient_symptoms':st.session_state.fu_symptoms,'enriched_symptoms':enriched,'status':'pending','llm_urgency':tri.get('urgency',''),'llm_reasoning':tri.get('reasoning',''),'llm_reasoning_original':result.get('original_reasoning',''),'llm_recommendation':tri.get('recommendation',''),'llm_next_steps':tri.get('next_steps',''),'llm_sources':tri.get('sources',''),'llm_evidence':tri.get('evidence',''),'llm_patient_explanation':tri.get('patient_explanation',''),'llm_confidence':tri.get('confidence',''),'rag_mode':tri.get('rag_mode',''),'prompt_chain_complete':True,'prompt_chain_questions':json.dumps([q for q,_ in all_qa]),'prompt_chain_answers':json.dumps([a for _,a in all_qa]),'prompt_chain_rounds':round_num,'gate_decision':result.get('gate_decision',''),'agent_action_trace':json.dumps(result.get('action_trace',[])),'agent_confidence':json.dumps(conf),'agent_plan':result.get('agent_plan',''),'evaluator_enhanced':result.get('evaluator_enhanced',False),'created_at':datetime.utcnow().isoformat(),'updated_at':datetime.utcnow().isoformat()})
                        st.success("✅ Submitted!"); st.info(f"**Case ID:** `{cid}` -- A nurse will review shortly.")
                        st.session_state.fu_stage = 'initial'; st.session_state.fu_qa = []; st.session_state.fu_round = 0

    with tab2:
        chk = st.text_input("Enter your ticket number:", key="chk", placeholder="e.g. T1234")
        if chk:
            cases = db_get_by_ticket(chk)
            if not cases: st.info("No cases found.")
            for c in cases:
                if c['status'] == 'reviewed':
                    tier = c.get('final_tier','Unknown'); colors = {'Urgent':'red','Routine':'orange','Self-care':'green'}; icons = {'Urgent':'🔴','Routine':'🟡','Self-care':'🟢'}
                    st.markdown(f"### {icons.get(tier,'⚪')} Case {c['case_id']}"); st.markdown(f"**Final Decision:** :{colors.get(tier,'gray')}[{tier}]")
                    bd = {}
                    try: bd = json.loads(c.get('booking_details','{}') or '{}')
                    except: pass
                    if bd:
                        st.markdown("---"); st.markdown("#### 📅 Your Appointment")
                        for fld, lbl in [('type','Type'),('doctor','Doctor'),('time','When'),('dept','Department'),('room','Room')]:
                            if bd.get(fld): st.markdown(f"**{lbl}:** {bd[fld]}")
                        if bd.get('instructions'): st.info(bd['instructions'])
                        if bd.get('guidance'): st.markdown("**Self-Care Guidance:**"); st.markdown(bd['guidance'])
                    if c.get('llm_patient_explanation'): st.markdown("---"); st.markdown("#### What This Means For You"); st.markdown(c['llm_patient_explanation'])
                    # Pipeline trace
                    st.markdown("---"); st.markdown("#### 🔄 Pipeline Trace")
                    steps = ["📝 Submitted"]
                    if c.get('prompt_chain_rounds') and int(c.get('prompt_chain_rounds',0)) > 0:
                        steps.append(f"🔗 Follow-up ({c['prompt_chain_rounds']} rounds)")
                    gd = c.get('gate_decision','')
                    if gd:
                        ge = '🚨' if 'emergency' in gd else '✅'
                        steps.append(f"{ge} Gate: {gd}")
                    steps.append(f"🏥 AI: {c.get('llm_urgency','?')}")
                    if c.get('evaluator_enhanced'): steps.append("🔬 Self-Reflected")
                    steps.append(f"👩\u200d⚕️ Nurse: {c.get('final_tier','?')}")
                    st.markdown(" → ".join(steps))
                    if c.get('nurse_notes') and c['nurse_notes'].strip(): st.markdown("---"); st.markdown("#### Nurse Notes"); st.markdown(f"> {c['nurse_notes']}")
                elif c['status'] == 'pending': st.warning(f"**{c['case_id']}** -- Waiting for nurse review...")

# ======================== NURSE DASHBOARD ====================================
elif page == "Nurse Dashboard":
    nurse = check_nurse_login()
    if not nurse:
        nurse_login_form()
    else:
        st.title("👩\u200d⚕️ Nurse Triage Review Dashboard")
        st.markdown(f"Logged in as: **{nurse}**")
        if st.sidebar.button("Logout"):
            del st.session_state['nurse_name']; del st.session_state['nurse_username']; st.rerun()

        pending = db_get_all(status='pending'); reviewed = db_get_all(status='reviewed')
        c1,c2,c3 = st.columns(3)
        c1.metric("Pending",len(pending)); c2.metric("Reviewed",len(reviewed))
        ov = sum(1 for c in reviewed if (c.get('nurse_action') or '').startswith('override'))
        c3.metric("Overrides",ov)
        tab_r, tab_h = st.tabs(["Review Cases","Review History"])
        with tab_r:
            if not pending:
                st.info("No pending cases.")
                if st.button("Refresh", use_container_width=True): st.rerun()
            else:
                opts = {f"{c['case_id']} -- Ticket #{c['ticket_number']}":c['case_id'] for c in pending}
                sel = st.selectbox("Select case:", list(opts.keys())); case = db_get_one(opts[sel])
                if case:
                    st.markdown("---"); cl, cr = st.columns([3,2])
                    with cl:
                        st.subheader("Patient Symptoms")
                        st.text_area("",value=case['patient_symptoms'],height=100,disabled=True,label_visibility="collapsed")
                        pq = case.get('prompt_chain_questions') or case.get('router_questions')
                        pa = case.get('prompt_chain_answers') or case.get('router_answers')
                        if pq:
                            try:
                                qs = json.loads(pq); ans = json.loads(pa or '[]')
                                if qs:
                                    rounds = case.get('prompt_chain_rounds', len(qs))
                                    with st.expander(f"🔗 Prompt Chain Q&A ({len(qs)} questions, {rounds} round(s))", expanded=False):
                                        for i,(q,a) in enumerate(zip(qs,ans)): st.markdown(f"**Q{i+1}:** {q}"); st.markdown(f"**A{i+1}:** {a}"); st.markdown("")
                            except: pass
                        st.subheader("AI Triage Assessment")
                        tier = case.get('llm_urgency','Unknown'); conf = case.get('llm_confidence','Unknown')
                        tc = {'Urgent':'red','Routine':'orange','Self-care':'green'}; cc = {'High':'green','Medium':'orange','Low':'red'}
                        m1,m2,m3 = st.columns(3)
                        m1.markdown(f"**Urgency:** :{tc.get(tier,'gray')}[{tier}]"); m2.markdown(f"**Confidence:** :{cc.get(conf,'gray')}[{conf}]"); m3.markdown(f"**Mode:** `{case.get('rag_mode','')}`")
                        # Agent confidence meter
                        if case.get('agent_confidence'):
                            try:
                                ac = json.loads(case['agent_confidence'])
                                pct = ac.get('confidence_pct', 50)
                                st.progress(min(pct, 100) / 100, text=f"🤖 Agent confidence: {pct}%")
                                if ac.get('would_change_if') and ac['would_change_if'] != 'N/A':
                                    st.info(f"🔍 **Would change if:** {ac['would_change_if']}")
                                if ac.get('nurse_watch_for') and ac['nurse_watch_for'] != 'N/A':
                                    st.warning(f"👩\u200d⚕️ **Nurse should check:** {ac['nurse_watch_for']}")
                            except: pass
                        # Agent action trace
                        if case.get('agent_action_trace'):
                            try:
                                trace = json.loads(case['agent_action_trace'])
                                if trace:
                                    with st.expander(f"🤖 Agent Decision Trace ({len(trace)} steps)", expanded=False):
                                        for step in trace:
                                            tool = step.get('tool','')
                                            tool_icons = {'CHECK_RED_FLAGS':'🚨','ASK_FOLLOWUP':'💬','ASSESS_URGENCY':'🏥','SELF_REFLECT':'🔬','CONFIDENCE_CHECK':'📊','DECISION':'🧠','RE_TRIAGE':'🔄'}
                                            icon = tool_icons.get(tool, '▶️')
                                            st.markdown(f"**{icon} Step {step.get('step','')}: {tool}**")
                                            if step.get('reason'): st.markdown(f"  _Reason: {step['reason']}_")
                                            if step.get('result'): st.markdown(f"  Result: {step['result']}")
                                            if step.get('plan'): st.markdown(f"  Plan: `{step['plan']}`")
                            except: pass
                        # Gate + Evaluator badges
                        gate_d = case.get('gate_decision','')
                        if gate_d:
                            gate_labels = {'chain_ready':'✅ Chain confirmed ready','emergency_bypass':'🚨 Emergency bypass','max_rounds_reached':'⏱️ Max rounds reached','max_rounds_forced':'⏱️ Max rounds forced'}
                            matched = gate_d
                            for k,v in gate_labels.items():
                                if k in gate_d: matched = v; break
                            st.info(f"Gate: {matched}")
                        if case.get('evaluator_enhanced'):
                            st.success("🔬 Self-Reflection: Assessment enhanced after agent review")
                            if case.get('llm_reasoning_original'):
                                with st.expander("View Self-Reflection Changes (Before/After)", expanded=False):
                                    col_b, col_a = st.columns(2)
                                    with col_b: st.markdown("**📝 Original RAG Output:**"); st.text_area("",value=(case.get('llm_reasoning_original',''))[:500],height=200,disabled=True,label_visibility="collapsed",key=f"before_{case['case_id']}")
                                    with col_a: st.markdown("**✨ Enhanced after Self-Reflection:**"); st.text_area("",value=(case.get('llm_reasoning',''))[:500],height=200,disabled=True,label_visibility="collapsed",key=f"after_{case['case_id']}")
                        if case.get('llm_evidence') and case['llm_evidence'] not in ('','No evidence retrieved.'):
                            st.markdown("**📚 Retrieved Medical Evidence:**")
                            with st.expander("View retrieved passages",expanded=True): st.markdown(case['llm_evidence'])
                        if case.get('llm_reasoning'): st.markdown(f"**Clinical Reasoning:** {case['llm_reasoning']}")
                        if case.get('llm_recommendation'): st.markdown(f"**Recommendation:** {case['llm_recommendation']}")
                        if case.get('llm_next_steps'): st.markdown(f"**Next Steps:** {case['llm_next_steps']}")
                        if case.get('llm_patient_explanation'): st.markdown(f"**Patient Explanation:** _{case['llm_patient_explanation']}_")
                        if case.get('llm_sources') and case['llm_sources'] not in ('None',''):
                            with st.expander("View sources (IEEE citations)"): st.markdown(case['llm_sources'])
                    with cr:
                        st.subheader("Your Clinical Decision")
                        tiers = ['Urgent','Routine','Self-care']; idx = tiers.index(tier) if tier in tiers else 0
                        nurse_tier = st.selectbox("Final Urgency Tier:",tiers,index=idx)
                        is_ov = nurse_tier != tier
                        if is_ov:
                            d = "upgrading" if tiers.index(nurse_tier)<tiers.index(tier) else "downgrading"
                            st.warning(f"You are **{d}** from {tier} to {nurse_tier}.")
                            ov_reason = st.text_area("Override reason (required):",height=80,placeholder=f"Why are you {d}?")
                        else: ov_reason = ''
                        notes = st.text_area("Clinical notes (optional):",height=80,placeholder="Additional observations...")
                        can = True
                        if is_ov and not (ov_reason or '').strip(): can = False; st.error("Override reason required.")
                        if st.button("Confirm Decision",type="primary",use_container_width=True,disabled=not can):
                            if nurse_tier == tier: act = 'approve'
                            elif tiers.index(nurse_tier)<tiers.index(tier): act = 'override_upgrade'
                            else: act = 'override_downgrade'
                            bk = book_action(case['case_id'],nurse_tier,case.get('patient_symptoms',''))
                            db_update(case['case_id'],{'nurse_tier':nurse_tier,'nurse_action':act,'nurse_notes':notes,'nurse_name':nurse,'nurse_override_reason':ov_reason,'nurse_timestamp':datetime.utcnow().isoformat(),'final_tier':nurse_tier,'booking_status':bk['status'],'booking_details':json.dumps(bk),'booking_agent_decision':bk.get('agent_decision',''),'status':'reviewed'})
                            st.success(f"✅ Recorded: **{nurse_tier}** ({act})"); st.balloons(); st.rerun()
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
                    with st.expander(f"{em} {c['case_id']} -- AI: {c.get('llm_urgency','?')} -> Final: {c.get('final_tier','?')} ({lb})"):
                        st.markdown(f"**Symptoms:** {(c.get('patient_symptoms',''))[:200]}...")
                        if c.get('agent_plan'): st.markdown(f"**🤖 Agent plan:** `{c['agent_plan']}`")
                        if c.get('evaluator_enhanced'): st.markdown("**🔬 Self-Reflection:** Enhanced")
                        if c.get('gate_decision'): st.markdown(f"**Gate:** {c['gate_decision']}")
                        if c.get('booking_agent_decision'): st.markdown(f"**📅 Booking:** {c['booking_agent_decision']}")
                        if c.get('nurse_override_reason'): st.markdown(f"**Override reason:** {c['nurse_override_reason']}")
                        if c.get('nurse_notes'): st.markdown(f"**Notes:** {c['nurse_notes']}")
                        st.caption(f"Nurse: {c.get('nurse_name','')} | {(c.get('nurse_timestamp','') or '')[:19]}")

# ======================== DEVELOPER DASHBOARD ================================
elif page == "Developer Dashboard":
    st.title("🔧 Developer Dashboard")
    st.caption("Audit trail -- AI vs Nurse decisions, agent traces, override analysis, cross-case learning")
    all_cases = db_get_all()
    if not all_cases: st.info("No cases yet.")
    else:
        reviewed_cases = [c for c in all_cases if c['status'] == 'reviewed']
        pending_cases = [c for c in all_cases if c['status'] == 'pending']
        override_cases = [c for c in reviewed_cases if (c.get('nurse_action') or '').startswith('override')]
        upgrade_cases = [c for c in override_cases if c.get('nurse_action') == 'override_upgrade']
        downgrade_cases = [c for c in override_cases if c.get('nurse_action') == 'override_downgrade']
        d1,d2,d3,d4,d5 = st.columns(5)
        d1.metric("Total",len(all_cases)); d2.metric("Reviewed",len(reviewed_cases)); d3.metric("Pending",len(pending_cases))
        d4.metric("Overrides",len(override_cases))
        d5.metric("Override Rate",f"{len(override_cases)/max(len(reviewed_cases),1)*100:.0f}%")

        # Triage distribution
        if reviewed_cases:
            st.markdown("---"); st.markdown("### 📈 Triage Distribution")
            tier_counts = {}
            for c in reviewed_cases:
                t = c.get('final_tier', 'Unknown')
                tier_counts[t] = tier_counts.get(t, 0) + 1
            tcols = st.columns(len(tier_counts))
            tier_icons = {'Urgent':'🔴','Routine':'🟡','Self-care':'🟢'}
            for i, (t, cnt) in enumerate(tier_counts.items()):
                with tcols[i]: st.metric(f"{tier_icons.get(t,'⚪')} {t}", cnt)

        if override_cases:
            st.markdown("---"); st.markdown("### ⚖️ Override Analysis")
            o1,o2 = st.columns(2); o1.metric("⬆️ Upgrades",len(upgrade_cases)); o2.metric("⬇️ Downgrades",len(downgrade_cases))
            if upgrade_cases:
                with st.expander("View upgrade justifications"):
                    for c in upgrade_cases: st.markdown(f"**{c['case_id']}:** {c.get('llm_urgency','?')} -> {c.get('final_tier','?')}"); st.markdown(f"> {c.get('nurse_override_reason','No reason')}"); st.markdown("")
            if downgrade_cases:
                with st.expander("View downgrade justifications"):
                    for c in downgrade_cases: st.markdown(f"**{c['case_id']}:** {c.get('llm_urgency','?')} -> {c.get('final_tier','?')}"); st.markdown(f"> {c.get('nurse_override_reason','No reason')}"); st.markdown("")

        # Cross-Case Learning Agent
        st.markdown("---"); st.markdown("### 🧠 Cross-Case Learning Agent")
        st.caption("AI analyzes patterns across all reviewed cases to detect biases and generate new knowledge")
        if st.button("🔍 Run Cross-Case Analysis", use_container_width=True):
            if len(reviewed_cases) < 3:
                st.warning("Need at least 3 reviewed cases for meaningful analysis.")
            else:
                with st.spinner("🧠 Agent is analyzing decision patterns across all cases..."):
                    analysis = agent_cross_case_analysis()
                if analysis.get('status') == 'error' or analysis.get('status') == 'insufficient_data':
                    st.error(analysis.get('message', 'Analysis failed.'))
                else:
                    a1, a2 = st.columns(2)
                    with a1: st.metric("📊 Agreement Rate", analysis.get('agreement_rate', 'N/A'))
                    with a2: st.metric("📋 Cases Analyzed", len(reviewed_cases))
                    if analysis.get('bias_pattern'):
                        st.markdown(f"**Bias Pattern:** {analysis['bias_pattern']}")
                    if analysis.get('failure_modes'):
                        st.markdown("**AI Failure Modes:**")
                        for fm in analysis['failure_modes']:
                            st.markdown(f"  - {fm}")
                    if analysis.get('improvement'):
                        st.success(f"**💡 Suggested Improvement:** {analysis['improvement']}")
                    if analysis.get('novel_finding') and analysis['novel_finding'] != 'null':
                        st.info(f"**🔬 Novel Finding:** {analysis['novel_finding']}")

        st.markdown("---"); st.markdown("### 📊 Case-by-Case Comparison")
        vf = st.selectbox("Show:",['All','Reviewed Only','Pending Only','Overrides Only'])
        if vf == 'Reviewed Only': display = reviewed_cases
        elif vf == 'Pending Only': display = pending_cases
        elif vf == 'Overrides Only': display = override_cases
        else: display = all_cases
        for c in display:
            ai_t = c.get('llm_urgency','--'); nr_t = c.get('final_tier','--')
            act = c.get('nurse_action','pending'); is_rev = c['status'] == 'reviewed'
            agree = ai_t == nr_t and is_rev
            emoji = '✅' if agree else ('⚠️' if act.startswith('override') else '⏳')
            with st.expander(f"{emoji} {c['case_id']} -- Ticket #{c.get('ticket_number','')} -- {c['status']}"):
                x1,x2,x3 = st.columns(3)
                with x1:
                    st.markdown("**🤖 AI Decision**")
                    st.markdown(f"Urgency: **{ai_t}**")
                    st.markdown(f"Confidence: {c.get('llm_confidence','--')}")
                    st.markdown(f"Mode: {c.get('rag_mode','--')}")
                    st.markdown(f"Plan: `{c.get('agent_plan','--')}`")
                    st.markdown(f"Gate: {c.get('gate_decision','--')}")
                    st.markdown(f"Self-Reflect: {'✅' if c.get('evaluator_enhanced') else '—'}")
                    if c.get('agent_confidence'):
                        try:
                            ac = json.loads(c['agent_confidence'])
                            st.markdown(f"Agent conf: {ac.get('confidence_pct','?')}%")
                        except: pass
                with x2:
                    st.markdown("**👩\u200d⚕️ Nurse Decision**")
                    st.markdown(f"Tier: **{nr_t}**" if is_rev else "_Pending_")
                    st.markdown(f"Action: {act}" if is_rev else "")
                    st.markdown(f"Nurse: {c.get('nurse_name','--')}" if is_rev else "")
                with x3:
                    st.markdown("**Result**")
                    st.markdown("✅ Agreement" if agree else (f"⚠️ {ai_t} -> {nr_t}" if is_rev else "⏳ Awaiting"))
                if c.get('nurse_override_reason'): st.markdown(f"**Override:** {c['nurse_override_reason']}")
                if c.get('nurse_notes'): st.markdown(f"**Notes:** {c['nurse_notes']}")
                try:
                    bd = json.loads(c.get('booking_details','{}') or '{}')
                    if bd and bd.get('type'): st.markdown(f"**📅 Booking:** {bd['type']} -- {bd.get('doctor','N/A')} -- {bd.get('time','N/A')}")
                except: pass
                st.caption(f"Symptoms: {(c.get('patient_symptoms',''))[:120]}...")
        st.markdown("---")
        export_data = json.dumps(all_cases, indent=2, default=str)
        st.download_button("📥 Export All Cases as JSON", data=export_data, file_name=f"triage_export_{datetime.now().strftime('%Y%m%d')}.json", mime="application/json", use_container_width=True)
