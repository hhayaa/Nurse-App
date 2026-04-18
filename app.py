import streamlit as st
import sys
from pathlib import Path

# Fix: add the app's own directory to Python's path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Triage Decision Support", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Dashboard:", ["Patient Dashboard", "Nurse Dashboard"])

if page == "Patient Dashboard":
    from pages import patient
    patient.render()
elif page == "Nurse Dashboard":
    from pages import nurse
    nurse.render()
