
import streamlit as st

# Configure the dashboard
st.set_page_config(
    page_title="NO2 Prediction Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Dashboard heading
st.title("Air Quality NO₂ Prediction Dashboard")

st.write(
    "Advanced Data Science Project – "
    "ASE Sensor Analysis and Machine Learning"
)

st.divider()

st.header("Project Overview")

st.write(
    "This dashboard explores NO₂ concentrations "
    "recorded by ASE sensors at the SUD3 and QDP "
    "monitoring stations."
)

st.info("Sensor comparisons and model results will be added next.")
