import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal import butter, filtfilt
from datetime import datetime
import io

st.set_page_config(page_title="EEG Muse 2 Dashboard", layout="wide")
st.title("🧠 EEG Dashboard - Muse 2 (4 Channels)")

uploaded_csv = st.file_uploader("📤 Upload Filtered EEG CSV", type="csv")

def bandpass_filter(data, lowcut, highcut, fs=256, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 100),
}

fs = 256
channels = ["TP9", "AF7", "AF8", "TP10"]

def clean_number(val):
    try:
        cleaned = val.replace(' ', '').replace(',', '')
        return float(cleaned)
    except:
        return np.nan

if uploaded_csv:
    df = pd.read_csv(uploaded_csv, delimiter=';')
    df.columns = df.columns.str.strip()

    for ch in channels:
        df[ch] = df[ch].astype(str).apply(clean_number)

    st.success("File successfully uploaded!")
    st.write("Detected columns:", df.columns.tolist())

    st.subheader("📈 EEG Band-Pass Signals (Per Channel)")
    for ch in channels:
        signal = df[ch].dropna().to_numpy()
        n = len(signal)
        time = np.linspace(0.0, n / fs, n)

        fig = go.Figure()
        for band_name, (lowcut, highcut) in bands.items():
            filtered = bandpass_filter(signal, lowcut, highcut, fs)
            fig.add_trace(go.Scatter(
                x=time,
                y=filtered,
                mode='lines',
                name=band_name
            ))

        fig.update_layout(
            title=f"{ch} - Filtered Band Signals",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude (μV)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    # === Band Power Summary ===
    st.subheader("📊 Band Power Summary (Mean Amplitude per Channel)")

    band_summary = []
    for band_name, (lowcut, highcut) in bands.items():
        row = {'Band': band_name}
        for ch in channels:
            signal = df[ch].dropna().to_numpy()
            filtered = bandpass_filter(signal, lowcut, highcut, fs)
            row[ch] = np.mean(np.abs(filtered))
        band_summary.append(row)

    summary_df = pd.DataFrame(band_summary)
    st.dataframe(summary_df.set_index("Band").style.background_gradient(cmap="viridis", axis=1))

    # === Bar Charts per Band ===
    st.subheader("📉 Band Power Comparison (Bar)")

    for idx, row in summary_df.iterrows():
        fig = go.Figure()
        for ch in channels:
            fig.add_trace(go.Bar(
                x=[row[ch]],
                y=[ch],
                orientation='h',
                name=ch
            ))

        fig.update_layout(
            title=f"{row['Band']} Power Across Channels",
            xaxis_title="Mean Amplitude (μV)",
            height=300,
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

    # === Gauge Charts ===
    st.subheader("🕹️ Average Band Gauges (Across All Channels)")
    cols = st.columns(len(bands))

    for i, (band_name, _) in enumerate(bands.items()):
        avg_val = summary_df.iloc[i][channels].mean()
        with cols[i]:
            st.markdown(f"**{band_name}**")
            st.plotly_chart(go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_val,
                title={'text': band_name},
                gauge={'axis': {'range': [None, avg_val * 2]}}
            )), use_container_width=True)

    # === File Download Section ===
    st.subheader("📁 Download Filtered EEG Data")

    custom_name = st.text_input("📌 Enter your preferred filename (without .csv):", value="MyEEG")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_filename = f"{custom_name}_OtakQu_Muse_filtered_{timestamp}.csv"

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="⬇️ Download Cleaned EEG CSV",
        data=csv_data,
        file_name=final_filename,
        mime="text/csv"
    )

else:
    st.info("📥 Please upload a Muse 2 EEG CSV file to begin.")
