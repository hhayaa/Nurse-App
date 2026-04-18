
import re
import os

# ============================================================================
# DEMO MODE: Returns pre-computed triage for demonstration purposes.
# Replace this with your actual RAG pipeline for production use.
#
# To connect your real RAG pipeline:
# 1. Copy your ChromaDB folder here
# 2. Import your functions from the notebook
# 3. Replace run_triage() below
# ============================================================================

def parse_triage_output(answer_text: str) -> dict:
    result = {'urgency': 'Unknown', 'reasoning': '', 'recommendation': '',
              'next_steps': '', 'sources': ''}
    if not answer_text:
        return result
    urg_match = re.search(r'Urgency:\s*(Urgent|Routine|Self-care|Self care)',
                          answer_text, re.IGNORECASE)
    if urg_match:
        urg = urg_match.group(1).strip()
        result['urgency'] = 'Self-care' if 'self' in urg.lower() else urg.capitalize()
    sections = re.split(r'\n(?=(?:Urgency|Reasoning|Recommendation|Next [Ss]teps|Sources):)',
                        answer_text)
    for section in sections:
        s = section.strip()
        if s.lower().startswith('reasoning:'):
            result['reasoning'] = s.split(':', 1)[1].strip()
        elif s.lower().startswith('recommendation:'):
            result['recommendation'] = s.split(':', 1)[1].strip()
        elif s.lower().startswith('next steps:') or s.lower().startswith('next steps:'):
            result['next_steps'] = s.split(':', 1)[1].strip()
        elif s.lower().startswith('sources:'):
            result['sources'] = s.split(':', 1)[1].strip()
    return result


def run_triage(symptoms: str) -> dict:
    """
    Main entry point. Replace this function body with your actual RAG call.

    Expected return format:
    {
        'urgency': 'Urgent' | 'Routine' | 'Self-care',
        'reasoning': '...',
        'recommendation': '...',
        'next_steps': '...',
        'sources': '...',
        'rag_mode': 'grounded_rag' | 'fallback' | 'demo',
    }
    """
    # --- OPTION A: Use Gemini (standalone) ---
    try:
        from google import genai
        from google.genai import types

        SYSTEM = (
            'You are an ER triage assistant. Given patient symptoms, classify urgency.\n'
            'Categories: Urgent / Routine / Self-care\n'
            'Output format:\nUrgency: <tier>\n\nReasoning:\n<explanation>\n\n'
            'Recommendation:\n<action>\n\nNext steps:\n- <bullets>\n\nSources:\nNone'
        )
        client = genai.Client()
        resp = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f'Patient symptoms:\n"""{symptoms}"""',
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM, temperature=0.0,
                max_output_tokens=900,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        parsed = parse_triage_output(resp.text or '')
        parsed['rag_mode'] = 'standalone_gemini'
        return parsed
    except Exception as e:
        # Fallback to demo mode if no API key
        pass

    # --- OPTION B: Demo mode (no API needed) ---
    symptoms_lower = symptoms.lower()
    urgent_kw = ['chest pain', 'stroke', 'breathing', 'unconscious', 'bleeding heavily',
                 'swelling throat', 'suicidal', 'seizure']
    routine_kw = ['cough weeks', 'burning urin', 'back pain', 'rash', 'headache',
                  'depressed', 'knee pain', 'heartburn']

    if any(kw in symptoms_lower for kw in urgent_kw):
        tier = 'Urgent'
        reasoning = 'Red-flag symptoms detected requiring immediate evaluation.'
        rec = 'Immediate ER evaluation'
    elif any(kw in symptoms_lower for kw in routine_kw):
        tier = 'Routine'
        reasoning = 'Symptoms suggest non-emergency condition needing clinical follow-up.'
        rec = 'Schedule appointment within 1-2 weeks'
    else:
        tier = 'Self-care'
        reasoning = 'Symptoms appear self-limiting with no red or yellow flags.'
        rec = 'Home management with monitoring'

    return {
        'urgency': tier,
        'reasoning': reasoning,
        'recommendation': rec,
        'next_steps': '- Monitor symptoms\n- Return if worsening',
        'sources': 'None (demo mode)',
        'rag_mode': 'demo',
    }
