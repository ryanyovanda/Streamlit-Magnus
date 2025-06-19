import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt

st.set_page_config(page_title="EEG Muse 2 Dashboard", layout="wide")
st.title("🧠 EEG Muse 2 Dashboard - Mind Monitor Style")

# --- Sidebar Navigation ---
page = st.sidebar.radio("📁 Navigation", ["Dashboard", "Details & Downloads"])

def clean_number(val):
    try:
        return float(str(val).replace(' ', '').replace(',', ''))
    except:
        return np.nan

uploaded_csv = st.file_uploader("📤 Upload Filtered EEG CSV", type="csv")

bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 100),
}
band_cols = list(bands.keys())
channels = ["TP9", "AF7", "AF8", "TP10"]
fs = 256

region_map = {
    "Left": ["TP9", "AF7"],
    "Right": ["TP10", "AF8"],
    "Front": ["AF7", "AF8"],
    "Back": ["TP9", "TP10"],
}

def bandpass_filter(data, lowcut, highcut, fs=256, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    df.columns = df.columns.str.strip()

    for ch in channels:
        df[ch] = df[ch].astype(str).apply(clean_number)

    def compute_region_stats(df, bands, regions):
        abs_data = []
        rel_rows = []
        for band, (lowcut, highcut) in bands.items():
            band_abs = {}
            band_rel = {"Band": band.capitalize()}
            total_power = 0
            band_power = {}
            for ch in channels:
                signal = df[ch].dropna().to_numpy()
                filtered = bandpass_filter(signal, lowcut, highcut, fs)
                abs_val = np.mean(np.abs(filtered))
                band_power[ch] = abs_val
                band_abs[ch] = abs_val
                total_power += abs_val
            for ch in channels:
                band_rel[ch] = band_power[ch] / total_power if total_power else 0
            band_abs['Band'] = band.capitalize()
            abs_data.append(band_abs)
            rel_rows.append(band_rel)
        return pd.DataFrame(abs_data), pd.DataFrame(rel_rows)

    abs_df, rel_df = compute_region_stats(df, bands, region_map)

    # --- PAGE 1: DASHBOARD ---
    if page == "Dashboard":
        st.subheader("📈 Absolute Brain Waves (Simulated 30s)")
        time_axis = pd.date_range("2025-06-18 15:10", periods=30, freq='1S')
        line_data = pd.DataFrame({"Time": time_axis})

        for band, (lowcut, highcut) in bands.items():
            band_vals = []
            for ch in channels:
                signal = df[ch].dropna().to_numpy()
                filtered = bandpass_filter(signal, lowcut, highcut, fs)
                band_vals.append(np.mean(np.abs(filtered)))
            avg = np.mean(band_vals)
            noise = np.random.normal(0, avg * 0.1, 30)
            line_data[band] = avg + noise

        fig_line = go.Figure()
        color_map = {
            'delta': 'red', 'theta': 'purple',
            'alpha': 'green', 'beta': 'blue', 'gamma': 'orange'
        }
        for band in band_cols:
            fig_line.add_trace(go.Scatter(
                x=line_data['Time'], y=line_data[band],
                mode='lines', name=band.capitalize(),
                line=dict(color=color_map[band])
            ))
        fig_line.update_layout(
            title="Mind Monitor - Absolute Brain Waves",
            xaxis_title="Time",
            yaxis_title="Absolute Power",
            height=400
        )
        st.plotly_chart(fig_line, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Left vs Right")
            fig_lr = go.Figure()
            for band in band_cols:
                left_val = abs_df.loc[abs_df['Band'] == band.capitalize(), ['TP9', 'AF7']].mean(axis=1).values[0]
                right_val = abs_df.loc[abs_df['Band'] == band.capitalize(), ['TP10', 'AF8']].mean(axis=1).values[0]
                fig_lr.add_trace(go.Bar(y=[band.capitalize()], x=[left_val], name='Left', orientation='h', marker_color='blue'))
                fig_lr.add_trace(go.Bar(y=[band.capitalize()], x=[-right_val], name='Right', orientation='h', marker_color='red'))
            fig_lr.update_layout(barmode='relative', height=350, xaxis=dict(title="Absolute Power", zeroline=True))
            st.plotly_chart(fig_lr, use_container_width=True)

        with col2:
            st.markdown("### 🧠 Front vs Back")
            fig_fb = go.Figure()
            for band in band_cols:
                front_val = abs_df.loc[abs_df['Band'] == band.capitalize(), ['AF7', 'AF8']].mean(axis=1).values[0]
                back_val = abs_df.loc[abs_df['Band'] == band.capitalize(), ['TP9', 'TP10']].mean(axis=1).values[0]
                fig_fb.add_trace(go.Bar(y=[band.capitalize()], x=[front_val], name='Front', orientation='h', marker_color='green'))
                fig_fb.add_trace(go.Bar(y=[band.capitalize()], x=[-back_val], name='Back', orientation='h', marker_color='orange'))
            fig_fb.update_layout(barmode='relative', height=350, xaxis=dict(title="Absolute Power", zeroline=True))
            st.plotly_chart(fig_fb, use_container_width=True)

        # --- GAUGE SECTION FIXED ---
        st.markdown("### 🕹️ Average Band Power Gauges")
        abs_df["Total"] = abs_df[channels].sum(axis=1)
        total_abs_power = abs_df["Total"].sum()
        abs_df["Relative"] = abs_df["Total"] / total_abs_power * 100

        cols = st.columns(5)
        for i, band in enumerate(band_cols):
            with cols[i]:
                st.markdown(f"**{band.capitalize()}**")
                left = abs_df.loc[i, ['TP9', 'AF7']].mean()
                right = abs_df.loc[i, ['TP10', 'AF8']].mean()
                rel_percent = abs_df.loc[i, "Relative"]

                scaled_value = (left + right) / 1000  

                st.plotly_chart(go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=scaled_value,
                    number={'valueformat': '.2f'},
                    title={'text': "Abs"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}}
                )), use_container_width=True, key=f"{band}_abs")

                st.plotly_chart(go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=rel_percent,
                    number={'suffix': '%', 'valueformat': '.1f'},
                    title={'text': "Rel"},
                    gauge={'axis': {'range': [0, 100]}}
                )), use_container_width=True, key=f"{band}_rel")

    # --- PAGE 2: DETAILS & DOWNLOAD ---
    elif page == "Details & Downloads":
        st.header("📊 Channel Power Table")
        st.subheader("Absolute Power per Channel")
        st.dataframe(abs_df, use_container_width=True)

        st.subheader("Relative Power per Channel")
        st.dataframe(rel_df, use_container_width=True)

        st.markdown("### 💾 Download Data")
        st.download_button(
            label="📥 Download Absolute EEG Stats CSV",
            data=abs_df.to_csv(index=False).encode("utf-8"),
            file_name="eeg_absolute_stats.csv",
            mime="text/csv"
        )
        st.download_button(
            label="📥 Download Relative EEG Stats CSV",
            data=rel_df.to_csv(index=False).encode("utf-8"),
            file_name="eeg_relative_stats.csv",
            mime="text/csv"
        )

else:
    st.info("📥 Please upload your filtered EEG CSV file to start.")
