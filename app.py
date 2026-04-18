
import streamlit as st

st.set_page_config(page_title="Triage Decision Support", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Dashboard:", ["Patient Dashboard", "Nurse Dashboard"])

if page == "Patient Dashboard":
    import pages.patient as patient
    patient.render()
elif page == "Nurse Dashboard":
    import pages.nurse as nurse
    nurse.render()
