import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from datetime import datetime
import io

# --- Config ---
st.set_page_config(page_title="EEG Muse 2 Dashboard", layout="wide")
st.title("🧠 Peep Brain Dashboard Muse")

# --- Sidebar Navigation ---
page = st.sidebar.radio("📁 Navigation", ["Dashboard", "Details & Downloads"])

# --- Constants ---
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

# --- Functions ---
def clean_number(val):
    try:
        return float(str(val).replace(' ', '').replace(',', ''))
    except:
        return np.nan

def bandpass_filter(data, lowcut, highcut, fs=256, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def compute_region_stats(df):
    abs_data = []  # absolute power table
    rel_data = []  # relative power table

    for ch in channels:
        signal = df[ch].dropna().to_numpy()
        ch_abs = {"Channel": ch}
        ch_rel = {"Channel": ch}
        band_values = {}

        # hitung abs value per band
        for band, (lowcut, highcut) in bands.items():
            filtered = bandpass_filter(signal, lowcut, highcut, fs)
            abs_val = np.mean(np.abs(filtered))
            band_values[band.capitalize()] = abs_val
            ch_abs[band.capitalize()] = abs_val

        # total absolute power for that channel
        total_power = sum(band_values.values())
        for band in band_values:
            ch_rel[band] = band_values[band] / total_power if total_power else 0

        abs_data.append(ch_abs)
        rel_data.append(ch_rel)

    abs_df = pd.DataFrame(abs_data)
    rel_df = pd.DataFrame(rel_data)
    return abs_df, rel_df


# --- Upload and Process File ---
uploaded_csv = st.file_uploader("📤 Upload Raw EEG CSV", type="csv")

if uploaded_csv:
    raw_df = pd.read_csv(uploaded_csv, delimiter=';')
    raw_df.columns = raw_df.columns.str.strip()

    for ch in channels:
        raw_df[ch] = raw_df[ch].astype(str).apply(clean_number)

    filtered_df = raw_df.copy()

    abs_df, rel_df = compute_region_stats(filtered_df)

    # --- Pages ---
    if page == "Dashboard":
        st.subheader("📈 Absolute Brain Waves")
        time_axis = pd.date_range("2025-06-18 15:10", periods=300, freq='1S')
        line_data = pd.DataFrame({"Time": time_axis})

        for band, (lowcut, highcut) in bands.items():
            band_vals = []
            for ch in channels:
                signal = filtered_df[ch].dropna().to_numpy()
                filtered = bandpass_filter(signal, lowcut, highcut, fs)
                band_vals.append(np.mean(np.abs(filtered)))
            avg = np.mean(band_vals)
            noise = np.random.normal(0, avg * 0.1, 300)
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
            title="Absolute Brain Waves",
            xaxis_title="Time",
            yaxis_title="Absolute Power",
            height=400
        )
        st.plotly_chart(fig_line, use_container_width=True)

       # --- Region Plots ---
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Left vs Right")
            fig_lr = go.Figure()

            # Hitung nilai maksimum untuk scale simetris
            max_val_lr = max([
                abs(abs_df.set_index("Channel").loc[region_map["Left"], band.capitalize()].mean()) if not pd.isna(abs_df.set_index("Channel").loc[region_map["Left"], band.capitalize()].mean()) else 0
                for band in band_cols
            ] + [
                abs(abs_df.set_index("Channel").loc[region_map["Right"], band.capitalize()].mean()) if not pd.isna(abs_df.set_index("Channel").loc[region_map["Right"], band.capitalize()].mean()) else 0
                for band in band_cols
            ])

            for i, band in enumerate(band_cols):
                left_val = abs_df.set_index("Channel").loc[region_map["Left"], band.capitalize()].mean()
                right_val = abs_df.set_index("Channel").loc[region_map["Right"], band.capitalize()].mean()

                left_val = 0 if pd.isna(left_val) else left_val
                right_val = 0 if pd.isna(right_val) else right_val

                fig_lr.add_trace(go.Bar(
                    y=[band.capitalize()],
                    x=[-left_val],
                    name='Left' if i == 0 else None,
                    orientation='h',
                    marker_color='blue',
                    showlegend=(i == 0)
                ))

                fig_lr.add_trace(go.Bar(
                    y=[band.capitalize()],
                    x=[right_val],
                    name='Right' if i == 0 else None,
                    orientation='h',
                    marker_color='red',
                    showlegend=(i == 0)
                ))

            fig_lr.update_layout(
                barmode='relative',
                height=350,
                xaxis=dict(title="Absolute Power", zeroline=True, range=[-max_val_lr * 1.1, max_val_lr * 1.1]),
                legend=dict(title="Region", orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_lr, use_container_width=True)

        with col2:
            st.markdown("### 🧠 Front vs Back")
            fig_fb = go.Figure()

            max_val_fb = max([
                abs(abs_df.set_index("Channel").loc[region_map["Front"], band.capitalize()].mean()) if not pd.isna(abs_df.set_index("Channel").loc[region_map["Front"], band.capitalize()].mean()) else 0
                for band in band_cols
            ] + [
                abs(abs_df.set_index("Channel").loc[region_map["Back"], band.capitalize()].mean()) if not pd.isna(abs_df.set_index("Channel").loc[region_map["Back"], band.capitalize()].mean()) else 0
                for band in band_cols
            ])

            for i, band in enumerate(band_cols):
                front_val = abs_df.set_index("Channel").loc[region_map["Front"], band.capitalize()].mean()
                back_val = abs_df.set_index("Channel").loc[region_map["Back"], band.capitalize()].mean()

                front_val = 0 if pd.isna(front_val) else front_val
                back_val = 0 if pd.isna(back_val) else back_val

                fig_fb.add_trace(go.Bar(
                    y=[band.capitalize()],
                    x=[-front_val],
                    name='Front' if i == 0 else None,
                    orientation='h',
                    marker_color='blue',
                    showlegend=(i == 0)
                ))

                fig_fb.add_trace(go.Bar(
                    y=[band.capitalize()],
                    x=[back_val],
                    name='Back' if i == 0 else None,
                    orientation='h',
                    marker_color='red',
                    showlegend=(i == 0)
                ))

            fig_fb.update_layout(
                barmode='relative',
                height=350,
                xaxis=dict(title="Absolute Power", zeroline=True, range=[-max_val_fb * 1.1, max_val_fb * 1.1]),
                legend=dict(title="Region", orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_fb, use_container_width=True)

        # --- Gauges ---
        st.markdown("### 🕹️ Average Band Power Gauges")
        abs_df["Total"] = abs_df[list(band.capitalize() for band in band_cols)].sum(axis=1)
        total_abs_power = abs_df["Total"].sum()
        abs_df["Relative"] = abs_df["Total"] / total_abs_power * 100

        cols = st.columns(5)
        for i, band in enumerate(band_cols):
            with cols[i]:
                st.markdown(f"**{band.capitalize()}**")
                left = abs_df.set_index("Channel").loc[region_map["Left"], band.capitalize()].mean()
                right = abs_df.set_index("Channel").loc[region_map["Right"], band.capitalize()].mean()
                rel_percent = rel_df.set_index("Channel")[band.capitalize()].mean() * 100

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

    elif page == "Details & Downloads":
        st.header("📊 Channel Power Table")
        st.subheader("Absolute Power per Channel")
        st.dataframe(abs_df, use_container_width=True)

        st.subheader("Relative Power per Channel")
        st.dataframe(rel_df, use_container_width=True)

        st.markdown("### 📝 Enter Your Name for File Export")
        user_name = st.text_input("Name (used for filename)", value="", placeholder="e.g. Ryan")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        input_filename = uploaded_csv.name.rsplit('.', 1)[0] if uploaded_csv else "EEG"

        if user_name.strip() == "":
            st.warning("⚠️ Please enter your name to enable download.")
        else:
            base_name = f"{user_name}_OtakQu_Muse_{timestamp}"

            st.markdown("### 💾 Download Processed EEG Data")
            st.download_button(
                label="⬇️ Download Filtered EEG CSV",
                data=filtered_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{base_name}.csv",
                mime="text/csv"
            )

            st.download_button(
                label="⬇️ Download Absolute Power CSV",
                data=abs_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{base_name}_ABS.csv",
                mime="text/csv"
            )

            st.download_button(
                label="⬇️ Download Relative Power CSV",
                data=rel_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{base_name}_REL.csv",
                mime="text/csv"
            )

else:
    st.info("📥 Please upload your raw Muse 2 EEG CSV file to begin.")