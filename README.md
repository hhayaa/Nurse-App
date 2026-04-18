# Triage HIL — Streamlit App

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Usage
1. Open the app in your browser
2. **Patient Dashboard**: Enter ticket number + symptoms, submit
3. Switch to **Nurse Dashboard** in the sidebar
4. Review the LLM assessment, pick final tier, add notes, confirm
5. Switch back to **Patient Dashboard**, enter ticket number to see result

## For the Pilot User Study
- Each participant reviews 10-15 pre-loaded cases
- Rate each on Clarity, Usefulness, Trust, Safety (1-5 Likert)
- Export results from the SQLite database: `triage_hil.db`

## API Key
Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` as an environment variable:
```bash
export GEMINI_API_KEY=your-key-here
streamlit run app.py
```

If no API key is set, the app runs in demo mode (keyword-based triage).
