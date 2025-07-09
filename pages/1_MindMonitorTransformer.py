import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- Config ---
st.set_page_config(page_title="EEG Muse 2 Dashboard", layout="wide")
st.title("🧠 Peep Brain Dashboard Muse")

# --- Sidebar Navigation ---
page = st.sidebar.radio("📁 Navigation", ["Dashboard", "Details & Downloads"])

# --- Constants ---
bands = {
    "delta": "Delta",
    "theta": "Theta",
    "alpha": "Alpha",
    "beta": "Beta",
    "gamma": "Gamma",
}
band_cols = list(bands.keys())
channels = ["TP9", "AF7", "AF8", "TP10"]

region_map = {
    "Left": ["TP9", "AF7"],
    "Right": ["TP10", "AF8"],
    "Front": ["AF7", "AF8"],
    "Back": ["TP9", "TP10"],
}

# --- Functions ---
def compute_region_stats(df):
    abs_data = []
    rel_data = []

    for ch in channels:
        ch_abs = {"channel": ch}
        ch_rel = {"channel": ch}
        band_values = {}

        for band in band_cols:
            col_name = f"{bands[band]}_{ch}"
            if col_name in df.columns:
                val = df[col_name].dropna().mean()
            else:
                val = np.nan
            band_values[band] = val
            ch_abs[band] = val

        total_power = sum([v for v in band_values.values() if not np.isnan(v)])
        for band in band_cols:
            val = band_values[band]
            ch_rel[band] = val / total_power if total_power and not np.isnan(val) else 0

        abs_data.append(ch_abs)
        rel_data.append(ch_rel)

    return pd.DataFrame(abs_data), pd.DataFrame(rel_data)

# --- Upload and Process File ---
uploaded_csv = st.file_uploader("📤 Upload MindMonitor EEG CSV", type="csv")

if uploaded_csv:
    raw_df = pd.read_csv(uploaded_csv)
    raw_df.columns = raw_df.columns.str.strip()

    abs_df, rel_df = compute_region_stats(raw_df)

    if page == "Dashboard":
        st.subheader("📈 Absolute Brain Waves (Simulated)")
        time_axis = pd.date_range("2025-06-18 15:10", periods=300, freq='1S')
        line_data = pd.DataFrame({"Time": time_axis})

        skipped_bands = []

        for band in band_cols:
            band_vals = []
            for ch in channels:
                col = f"{bands[band]}_{ch}"
                if col in raw_df.columns:
                    val = raw_df[col].dropna().mean()
                    if not np.isnan(val):
                        band_vals.append(val)

            if not band_vals:
                skipped_bands.append(band)
                continue  # skip plotting this band

            avg = np.nanmean(band_vals)
            if not np.isfinite(avg) or avg <= 0:
                avg = 1e-6  # fallback to safe value
            noise = np.random.normal(0, avg * 0.1, 300)

            if np.isnan(avg) or avg == 0:
                avg = 1e-6
            noise = np.random.normal(0, avg * 0.1, 300)
            line_data[band] = avg + noise

        if skipped_bands:
            st.warning(f"⚠️ Skipped bands due to missing data: {', '.join(skipped_bands)}")

        fig_line = go.Figure()
        color_map = {
            'delta': 'red', 'theta': 'purple',
            'alpha': 'green', 'beta': 'blue', 'gamma': 'orange'
        }
        for band in band_cols:
            if band in line_data.columns:
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

        # --- Region Comparison ---
        col1, col2 = st.columns(2)

        def render_region_plot(region1, region2, title, key):
            fig = go.Figure()
            try:
                max_val = max([
                    abs(abs_df.set_index("channel").loc[region_map[region1], band].mean()) for band in band_cols
                ] + [
                    abs(abs_df.set_index("channel").loc[region_map[region2], band].mean()) for band in band_cols
                ])
            except:
                st.error(f"Unable to render {title} due to missing data.")
                return

            for i, band in enumerate(band_cols):
                try:
                    val1 = abs_df.set_index("channel").loc[region_map[region1], band].mean()
                    val2 = abs_df.set_index("channel").loc[region_map[region2], band].mean()
                    fig.add_trace(go.Bar(
                        y=[band], x=[-val1], name=region1 if i == 0 else None,
                        orientation='h', marker_color='blue', showlegend=(i == 0)
                    ))
                    fig.add_trace(go.Bar(
                        y=[band], x=[val2], name=region2 if i == 0 else None,
                        orientation='h', marker_color='red', showlegend=(i == 0)
                    ))
                except:
                    continue

            fig.update_layout(
                barmode='relative',
                height=350,
                xaxis=dict(title="Absolute Power", zeroline=True, range=[-max_val * 1.1, max_val * 1.1]),
                legend=dict(title="Region", orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                title=title
            )
            st.plotly_chart(fig, use_container_width=True, key=key)

        with col1:
            render_region_plot("Left", "Right", "📊 Left vs Right", "lr")
        with col2:
            render_region_plot("Front", "Back", "🧠 Front vs Back", "fb")

        # --- Gauges ---
        st.markdown("### 🕹️ Average Band Power Gauges")
        abs_df["total"] = abs_df[band_cols].sum(axis=1)
        total_abs_power = abs_df["total"].sum()
        abs_df["relative"] = abs_df["total"] / total_abs_power * 100

        cols = st.columns(5)
        for i, band in enumerate(band_cols):
            with cols[i]:
                st.markdown(f"**{band}**")
                try:
                    left = abs_df.set_index("channel").loc[region_map["Left"], band].mean()
                    right = abs_df.set_index("channel").loc[region_map["Right"], band].mean()
                    rel_percent = rel_df.set_index("channel")[band].mean() * 100
                    scaled_value = (left + right)

                    st.plotly_chart(go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=scaled_value,
                        number={'valueformat': '.2f'},
                        title={'text': "Abs"},
                        gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "green"}}
                    )), use_container_width=True, key=f"{band}_abs")

                    st.plotly_chart(go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=rel_percent,
                        number={'suffix': '%', 'valueformat': '.1f'},
                        title={'text': "Rel"},
                        gauge={'axis': {'range': [0, 100]}}
                    )), use_container_width=True, key=f"{band}_rel")
                except:
                    st.warning(f"{band} gauge could not be rendered due to missing values.")

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
            base_name = f"{user_name}_OtakQu_MindMonitor_{timestamp}"

            st.markdown("### 💾 Download Processed EEG Data")
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
    st.info("📥 Please upload your EEG MindMonitor CSV file to begin.")
