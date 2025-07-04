import streamlit as st

st.set_page_config(page_title="EEG Explorer", layout="wide")
st.title("🧠 Welcome to EEG Explorer")

st.markdown("""
Choose one of the tools from the left sidebar:

- **MindMonitor Viewer** for analyzing exported Muse EEG from Mind Monitor.
- **Muse Raw CSV Viewer** for processing raw CSV recordings with filtering and band power analysis.
""")
