import os
import sys
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# Ensure visuals folder exists
os.makedirs("visuals", exist_ok=True)


def train_ssh_engine(df):
    """Trains, serializes, and evaluates the SSH Auth Isolation Forest model."""
    feature_cols = [
        "time_delta_sec",
        "failed_count_5m",
        "total_count_5m",
        "failure_ratio_5m",
    ]
    train_normal = df[df["label"] == "normal"][feature_cols]

    model = IsolationForest(
        n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1
    )
    model.fit(train_normal)

    # Serialize model to disk
    model_filename = "isolation_forest_ssh.pkl"
    joblib.dump(model, model_filename)
    print(f"✓ Model successfully serialized -> '{model_filename}'")

    # Predict & Compare
    df["model_raw_score"] = model.predict(df[feature_cols])
    df["predicted_anomaly"] = df["model_raw_score"].map({-1: 1, 1: 0})
    df["true_anomaly"] = (df["label"] != "normal").astype(int)

    return df, feature_cols, "failed_count_5m", "visuals/ssh_anomaly_plot.png"


def train_network_engine(df):
    """Trains, serializes, and evaluates the Network Telemetry Isolation Forest model."""
    # Dynamically select numerical and encoded protocol features
    base_features = [
        "Packet_Size_Bytes",
        "Connection_Duration_ms",
        "Failed_Logins",
        "Geo_Distance_km",
        "bytes_per_ms",
        "failed_login_rate",
        "geo_failed_interaction",
    ]
    proto_features = [col for col in df.columns if col.startswith("proto_")]
    feature_cols = base_features + proto_features

    train_normal = df[df["Is_Malicious"] == 0][feature_cols]

    model = IsolationForest(
        n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1
    )
    model.fit(train_normal)

    # Serialize model to disk
    model_filename = "isolation_forest_network.pkl"
    joblib.dump(model, model_filename)
    print(f"✓ Model successfully serialized -> '{model_filename}'")

    # Predict & Compare
    df["model_raw_score"] = model.predict(df[feature_cols])
    df["predicted_anomaly"] = df["model_raw_score"].map({-1: 1, 1: 0})
    df["true_anomaly"] = df["Is_Malicious"].astype(int)

    return df, feature_cols, "bytes_per_ms", "visuals/network_anomaly_plot.png"


def main(file_path):
    df = pd.read_csv(file_path)

    # Schema Detection & Training Routing
    if "Is_Malicious" in df.columns:
        print(f"=== Training Network Telemetry Engine ({file_path}) ===")
        df, features, y_plot_col, plot_path = train_network_engine(df)
        plot_title = (
            "Network Telemetry Anomaly Detection (Throughput Velocity)"
        )
    elif "label" in df.columns:
        print(f"=== Training SSH Auth Engine ({file_path}) ===")
        df, features, y_plot_col, plot_path = train_ssh_engine(df)
        plot_title = (
            "SSH Log Anomaly Detection (5-Minute Rolling Failed Logins)"
        )
    else:
        raise ValueError(f"Unrecognized dataset schema in '{file_path}'.")

    # Metrics Evaluation
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(df["true_anomaly"], df["predicted_anomaly"]))

    print("\n=== Classification Report ===")
    print(
        classification_report(
            df["true_anomaly"],
            df["predicted_anomaly"],
            target_names=["Normal (0)", "Anomaly (1)"],
        )
    )

    # Save Scatter Plot Visual
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        x=range(len(df)),
        y=y_plot_col,
        hue="predicted_anomaly",
        data=df,
        palette={0: "blue", 1: "red"},
        alpha=0.6,
        s=25,
    )
    plt.title(plot_title, fontsize=14)
    plt.xlabel("Log Entry Index (Chronological)", fontsize=12)
    plt.ylabel(y_plot_col, fontsize=12)
    plt.legend(title="Prediction", labels=["Normal (0)", "Anomaly (1)"])
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Visual plot saved -> '{plot_path}'")


if __name__ == "__main__":
    input_file = (
        sys.argv[1] if len(sys.argv) > 1 else "processed_network_logs.csv"
    )
    main(input_file)
