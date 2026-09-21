import sys
import numpy as np
import pandas as pd
from csv_normalizer import normalize_csv


def preprocess_ssh(df):
    """processes host-level ssh authentication logs using rolling time windows."""
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # calculate time deltas
    df["time_delta_sec"] = (
        df.groupby("source_ip")["timestamp"].diff().dt.total_seconds().fillna(0)
    )

    # any status other than a successful login counts as a failed/suspicious
    # attempt (auth_fail, connection_refused, timed out, etc. - the literal
    # string "failed" never appears in real log data, so matching on it
    # left this column constant at 0 for every dataset)
    df["is_failed"] = (df["status"].astype(str).str.lower() != "success").astype(int)

    # 5-minute rolling window aggregation
    df = df.set_index("timestamp")
    df["failed_count_5m"] = (
        df.groupby("source_ip")["is_failed"]
        .transform(lambda x: x.rolling("5min").sum())
        .fillna(0)
    )
    df["total_count_5m"] = (
        df.groupby("source_ip")["status"]
        .transform(lambda x: x.rolling("5min").count())
        .fillna(0)
    )
    df["failure_ratio_5m"] = np.round(
        df["failed_count_5m"] / (df["total_count_5m"] + 1e-9), 4
    )
    df = df.reset_index()

    output_path = "data/processed_ssh_logs.csv"
    df.to_csv(output_path, index=False)
    print(
        f"successfully processed ssh auth logs ({len(df):,} rows) -> '{output_path}'"
    )


def preprocess_network(df):
    """processes session-level network telemetry logs."""
    # encode protocol (tcp, udp, icmp)
    df = pd.get_dummies(df, columns=["Protocol"], prefix="proto", dtype=int)

    # connection-level threat metrics
    df["bytes_per_ms"] = np.round(
        df["Packet_Size_Bytes"] / (df["Connection_Duration_ms"] + 1), 4
    )
    df["failed_login_rate"] = np.round(
        df["Failed_Logins"] / ((df["Connection_Duration_ms"] / 1000) + 1), 4
    )
    df["geo_failed_interaction"] = df["Geo_Distance_km"] * (
        df["Failed_Logins"] + 1
    )

    output_path = "data/processed_network_logs.csv"
    df.to_csv(output_path, index=False)
    print(
        f"successfully processed network telemetry logs ({len(df):,} rows) -> '{output_path}'"
    )



def main(file_path):
    raw_df = pd.read_csv(file_path)

    # standardize column structure first
    df, warnings = normalize_csv(raw_df)
    for field in warnings:
        print(f"warning: '{field}' not found in source data - filled with a constant placeholder.")

    # proceed with feature extraction on standardized column names
    if "protocol" in df.columns:
        print("Processing normalized network telemetry...")
        preprocess_network(df)
    elif "timestamp" in df.columns and "source_ip" in df.columns:
        print("Processing normalized host SSH logs...")
        preprocess_ssh(df)
    else:
        raise ValueError("Could not map dataset to known operational schema.")


if __name__ == "__main__":
    target_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "cybersecurity_network_logs.csv"
    )
    main(target_file)