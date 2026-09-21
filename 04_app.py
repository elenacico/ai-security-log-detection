import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest
from streamlit_autorefresh import st_autorefresh
from ai_explainer import generate_incident_report
from csv_normalizer import normalize_csv#
from scipy.stats import percentileofscore


def get_scoring_model(pretrained_path, df, candidate_features, force_retrain):
    """loads the pretrained isolation forest for this engine, but falls back
    to fitting a fresh one on this dataset's own real (non-constant)
    features whenever the pretrained model's expected features don't carry
    real signal here (e.g. a dataset missing fields the model was trained
    on, which normalize_csv fills with constant placeholders). returns
    (model, features_used, was_retrained)."""
    pretrained = joblib.load(pretrained_path)
    expected = (
        list(pretrained.feature_names_in_)
        if hasattr(pretrained, "feature_names_in_")
        else candidate_features
    )
    degenerate = [
        f for f in expected if f in df.columns and df[f].nunique(dropna=False) <= 1
    ]

    if not force_retrain and not degenerate:
        return pretrained, expected, False

    usable = [
        f
        for f in candidate_features
        if f in df.columns and df[f].nunique(dropna=False) > 1
    ]
    if not usable:
        # nothing usable to retrain on - fall back to the pretrained model
        return pretrained, expected, False

    fresh_model = IsolationForest(
        n_estimators=100, contamination="auto", random_state=42, n_jobs=-1
    )
    fresh_model.fit(df[usable])
    return fresh_model, usable, True


st.set_page_config(page_title="Dual-Engine SIEM Platform", layout="wide")
st.title("Modular SIEM & Anomaly Detection Dashboard")

# sidebar
live_mode = st.sidebar.checkbox("Enable Live Streaming Mode")
df_raw = None
normalize_warnings = []

# data ingestion & normalization
if live_mode:
    st_autorefresh(interval=2000, key="siem_live_stream_refresh")
    if os.path.exists("live_stream.csv"):
        try:
            raw_data = pd.read_csv("live_stream.csv")
            df_raw, normalize_warnings = normalize_csv(raw_data)  # normalize first
        except Exception:
            pass
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Log Dataset (CSV)", type=["csv"]
    )
    if uploaded_file is not None:
        raw_data = pd.read_csv(uploaded_file)
        df_raw, normalize_warnings = normalize_csv(raw_data)  # normalize first

# schema auto-detection
if df_raw is not None and not df_raw.empty:
    # now that headers are normalized, check if network telemetry fields exist
    if "Protocol" in df_raw.columns or "Packet_Size_Bytes" in df_raw.columns:
        engine_type = "Network Telemetry Engine"
    else:
        engine_type = "Host SSH Engine"


    st.sidebar.info(f"Detected Engine: **{engine_type}**")

    if normalize_warnings:
        st.warning(
            "This dataset has no real data for: **"
            + ", ".join(normalize_warnings)
            + "**. These fields were filled with constant placeholder values, "
            "which flattens the corresponding model features and can make "
            "anomaly detection significantly less sensitive on this dataset."
        )

    st.subheader(f"Live Ingest Stream ({len(df_raw):,} Events Processed)")

    # display newest records first
    st.dataframe(df_raw.tail(15).iloc[::-1], use_container_width=True)

    # network telemetry engine execution
    if engine_type == "Network Telemetry Engine":
        try:
            df = df_raw.copy()

            # generate missing numeric metrics if not present
            if "bytes_per_ms" not in df.columns:
                df["bytes_per_ms"] = np.round(
                    df["Packet_Size_Bytes"] / (df["Connection_Duration_ms"] + 1), 4
                )
                df["failed_login_rate"] = np.round(
                    df["Failed_Logins"] / ((df["Connection_Duration_ms"] / 1000) + 1), 4
                )
                df["geo_failed_interaction"] = df["Geo_Distance_km"] * (df["Failed_Logins"] + 1)

            # apply dummy encoding for protocol if raw categorical protocol column exists
            if "Protocol" in df.columns:
                df = pd.get_dummies(df, columns=["Protocol"], prefix="proto", dtype=int)

            # candidate features for a fresh fit, if the pretrained model's
            # expected features turn out to be constant/placeholder here
            candidate_features = [
                "Packet_Size_Bytes", "Connection_Duration_ms", "Failed_Logins",
                "Geo_Distance_km", "bytes_per_ms", "failed_login_rate", "geo_failed_interaction",
            ] + [c for c in df.columns if c.startswith("proto_")]

            model, expected_features, retrained = get_scoring_model(
                "isolation_forest_network.pkl", df, candidate_features, bool(normalize_warnings)
            )

            # fill any still-missing expected features with 0 (extra unseen
            # protocol categories etc.)
            for col in expected_features:
                if col not in df.columns:
                    df[col] = 0

            # slice and order matrix strictly by expected_features
            X = df[expected_features]

            if retrained:
                st.info(
                    "The pretrained network model's features didn't carry real "
                    "signal for this dataset, so a fresh Isolation Forest was fit "
                    f"on this dataset's own features: **{', '.join(expected_features)}**"
                )

            # calculate risk score as a percentile rank of anomaly severity
            raw_scores = model.decision_function(X)

            # invert scores so lower raw values = higher risk percentile
            anomaly_severity = -raw_scores
            df["risk_score"] = np.round([
                percentileofscore(anomaly_severity, score) for score in anomaly_severity
            ], 1)

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

            # ai threat analysis trigger
            st.subheader("Gemini AI Threat Remediation")
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


    # host ssh engine execution
    else:
        try:
            df = df_raw.copy()

            # compute rolling ssh features from raw columns first, before
            # any placeholder-fill runs, so real signal isn't overwritten
            # with zeros
            if "failed_count_5m" not in df.columns and {"timestamp", "source_ip", "status"} <= set(df.columns):
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
                df["time_delta_sec"] = df.groupby("source_ip")["timestamp"].diff().dt.total_seconds().fillna(0)
                df = df.set_index("timestamp")
                df["failed_count_5m"] = df.groupby("source_ip")["status"].transform(
                    lambda x: (x.astype(str).str.lower() != "success").astype(int).rolling("5min").sum()
                ).fillna(0)
                df["total_count_5m"] = df.groupby("source_ip")["status"].transform(
                    lambda x: x.rolling("5min").count()
                ).fillna(0)
                df["failure_ratio_5m"] = np.round(
                    df["failed_count_5m"] / (df["total_count_5m"] + 1e-9), 4
                )
                df = df.reset_index()

            candidate_features = ["time_delta_sec", "failed_count_5m", "total_count_5m", "failure_ratio_5m"]

            model, expected_features, retrained = get_scoring_model(
                "isolation_forest_ssh.pkl", df, candidate_features, bool(normalize_warnings)
            )

            # fill any still-missing expected features with 0
            for col in expected_features:
                if col not in df.columns:
                    df[col] = 0

            if retrained:
                st.info(
                    "The pretrained SSH model's features didn't carry real "
                    "signal for this dataset, so a fresh Isolation Forest was fit "
                    f"on this dataset's own features: **{', '.join(expected_features)}**"
                )

            raw_scores = model.decision_function(df[expected_features])
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
        st.warning("Live stream initialized. Waiting for incoming events from `06_log_simulator.py`...")
    else:
        st.info("Please upload a CSV file or check 'Enable Live Streaming Mode' in the sidebar.")
