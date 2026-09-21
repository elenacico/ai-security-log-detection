import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from ai_explainer import generate_incident_report
from csv_normalizer import normalize_csv#
from scipy.stats import percentileofscore



st.set_page_config(page_title="Dual-Engine SIEM Platform", layout="wide")
st.title("🛡️ Modular SIEM & Anomaly Detection Dashboard")

# sidebar
live_mode = st.sidebar.checkbox("📡 Enable Live Streaming Mode")
df_raw = None

# --- Data Ingestion & Normalization ---
if live_mode:
    st_autorefresh(interval=2000, key="siem_live_stream_refresh")
    if os.path.exists("live_stream.csv"):
        try:
            raw_data = pd.read_csv("live_stream.csv")
            df_raw = normalize_csv(raw_data)  # Normalize first
        except Exception:
            pass
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Log Dataset (CSV)", type=["csv"]
    )
    if uploaded_file is not None:
        raw_data = pd.read_csv(uploaded_file)
        df_raw = normalize_csv(raw_data)  # Normalize first

# --- Schema Auto-Detection ---
if df_raw is not None and not df_raw.empty:
    # Now that headers are normalized, check if Network Telemetry fields exist
    if "Protocol" in df_raw.columns or "Packet_Size_Bytes" in df_raw.columns:
        engine_type = "Network Telemetry Engine"
    else:
        engine_type = "Host SSH Engine"


    st.sidebar.info(f"Detected Engine: **{engine_type}**")
    st.subheader(f"📋 Live Ingest Stream ({len(df_raw):,} Events Processed)")

    # Display newest records first
    st.dataframe(df_raw.tail(15).iloc[::-1], use_container_width=True)

    # --- 4. Network Telemetry Engine Execution ---
    if engine_type == "Network Telemetry Engine":
        try:
            model = joblib.load("isolation_forest_network.pkl")

            df = df_raw.copy()





            # Generate missing numeric metrics if not present
            if "bytes_per_ms" not in df.columns:
                df["bytes_per_ms"] = np.round(
                    df["Packet_Size_Bytes"] / (df["Connection_Duration_ms"] + 1), 4
                )
                df["failed_login_rate"] = np.round(
                    df["Failed_Logins"] / ((df["Connection_Duration_ms"] / 1000) + 1), 4
                )
                df["geo_failed_interaction"] = df["Geo_Distance_km"] * (df["Failed_Logins"] + 1)

            # Apply dummy encoding for protocol if raw categorical protocol column exists
            if "Protocol" in df.columns:
                df = pd.get_dummies(df, columns=["Protocol"], prefix="proto", dtype=int)

            # Retrieve exact features expected by the trained Isolation Forest
            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
            else:
                # Fallback if model was trained without feature names
                base_features = [
                    "Packet_Size_Bytes", "Connection_Duration_ms", "Failed_Logins",
                    "Geo_Distance_km", "bytes_per_ms", "failed_login_rate", "geo_failed_interaction"
                ]
                expected_features = base_features + ["proto_TCP", "proto_UDP", "proto_ICMP"]

            # 1. Fill missing trained features with 0
            for col in expected_features:
                if col not in df.columns:
                    df[col] = 0

            # 2. Slice and order matrix strictly by expected_features (ignoring extra unseen cols like proto_ARP)
            X = df[expected_features]

             # Calculate risk score as a percentile rank of anomaly severity
            raw_scores = model.decision_function(X)

            # Invert scores so lower raw values = higher risk percentile
            anomaly_severity = -raw_scores
            df["risk_score"] = np.round([
                percentileofscore(anomaly_severity, score) for score in anomaly_severity
            ], 1)

            # Compute decision function scores on aligned matrix
            raw_scores = model.decision_function(X)
            s_min, s_max = raw_scores.min(), raw_scores.max()
            df["risk_score"] = np.round((1 - (raw_scores - s_min) / (s_max - s_min + 1e-9)) * 100, 1)

            col1, col2 = st.columns(2)
            col1.metric("Total Streamed Sessions", f"{len(df):,}")
            col2.metric("Flagged Network Anomalies", f"{(df['risk_score'] >= 80).sum():,}")

            fig = px.scatter(
                df,
                x=df.index,
                y="bytes_per_ms",
                color="risk_score",
                color_continuous_scale="Reds",
                title="Real-Time Throughput Velocity vs Event Sequence",
            )
            st.plotly_chart(fig, use_container_width=True)

            # AI Threat Analysis Trigger
            st.subheader("🤖 Gemini AI Threat Remediation")
            high_risk = df[df["risk_score"] >= 80]
            if not high_risk.empty:
                selected_idx = st.selectbox("Select Flagged Event Index", high_risk.index[::-1])
                event_data = high_risk.loc[selected_idx]

                if st.button("Generate AI Mitigation Brief"):
                    payload = {
                        "Packet_Size_Bytes": int(event_data["Packet_Size_Bytes"]),
                        "Connection_Duration_ms": int(event_data["Connection_Duration_ms"]),
                        "bytes_per_ms": float(event_data["bytes_per_ms"]),
                        "Geo_Distance_km": int(event_data["Geo_Distance_km"]),
                        "risk_score": float(event_data["risk_score"]),
                    }
                    report = generate_incident_report(engine_type, payload)
                    st.info(f"**Analyst Summary:** {report.get('summary')}")
                    st.code(report.get("command"), language="bash")
            else:
                st.success("No high-risk network anomalies detected in active stream.")

        except Exception as e:
            st.error(f"Error processing live network stream: {e}")


    # --- 5. Host SSH Engine Execution ---
    else:
        try:
            model = joblib.load("isolation_forest_ssh.pkl")
            df = df_raw.copy()


            # --- Load Model ---
            model = joblib.load("isolation_forest_ssh.pkl")

            # --- Declare expected_features ---
            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
            else:
                expected_features = [
                    "time_delta_sec",
                    "failed_count_5m",
                    "total_count_5m",
                    "failure_ratio_5m",
                ]

            # --- Align Data with Model Schema ---
            for col in expected_features:
                if col not in df.columns:
                    df[col] = 0

            # --- Build Feature Matrix X & Predict ---
            X = df[expected_features]
            raw_scores = model.decision_function(X)

            if "failed_count_5m" not in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
                df["time_delta_sec"] = df.groupby("source_ip")["timestamp"].diff().dt.total_seconds().fillna(0)
                df = df.set_index("timestamp")
                df["failed_count_5m"] = df.groupby("source_ip")["status"].transform(
                    lambda x: (x == "failed").astype(int).rolling("5min").sum()
                ).fillna(0)
                df["total_count_5m"] = df.groupby("source_ip")["status"].transform(
                    lambda x: x.rolling("5min").count()
                ).fillna(0)
                df["failure_ratio_5m"] = np.round(
                    df["failed_count_5m"] / (df["total_count_5m"] + 1e-9), 4
                )
                df = df.reset_index()

            features = ["time_delta_sec", "failed_count_5m", "total_count_5m", "failure_ratio_5m"]

            raw_scores = model.decision_function(df[features])
            s_min, s_max = raw_scores.min(), raw_scores.max()
            df["risk_score"] = np.round((1 - (raw_scores - s_min) / (s_max - s_min + 1e-9)) * 100, 1)

            col1, col2 = st.columns(2)
            col1.metric("Total Streamed Events", f"{len(df):,}")
            col2.metric("Flagged SSH Anomalies", f"{(df['risk_score'] >= 80).sum():,}")

            fig = px.scatter(
                df, x=df.index, y="failed_count_5m", color="risk_score",
                color_continuous_scale="Reds", title="SSH Failed Logins (5m Window) vs Event Index"
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing SSH stream: {e}")

else:
    if live_mode:
        st.warning("⏳ Live stream initialized. Waiting for incoming events from `04_log_simulator.py`...")
    else:
        st.info("👆 Please upload a CSV file or check 'Enable Live Streaming Mode' in the sidebar.")
