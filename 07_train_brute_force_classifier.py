import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# positive class = attempts that are actually part of a login-based attack.
# "config_anomaly" is a different failure mode (not a login attack), so it's
# excluded from the target and left out of train/test entirely
BRUTE_FORCE_LABELS = {"brute_force", "brute_force_connection_issue"}


def build_actor_features(df: pd.DataFrame) -> pd.DataFrame:
    """aggregates raw ssh log rows into per-actor, rolling-window behavioral
    features. each output row is still keyed to one raw event, but its
    feature values describe the actor's (source_ip's / username's) behavior
    over the trailing 5-minute window as of that event - this is what
    actually distinguishes brute-forcing from a single unusual packet."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["is_failed"] = (df["status"].astype(str).str.lower() != "success").astype(int)

    # rolling .apply requires numeric input even with a custom function, so
    # string columns are factorized to integer codes purely for the
    # nunique-in-window computation below
    df["username_code"] = df["username"].astype("category").cat.codes
    df["source_ip_code"] = df["source_ip"].astype("category").cat.codes

    df = df.set_index("timestamp")

    # per source_ip window features (brute force / password spraying)
    by_ip = df.groupby("source_ip")
    df["attempts_5m"] = by_ip["is_failed"].transform(lambda s: s.rolling("5min").count())
    df["failed_5m"] = by_ip["is_failed"].transform(lambda s: s.rolling("5min").sum())
    df["failure_ratio_5m"] = np.round(df["failed_5m"] / (df["attempts_5m"] + 1e-9), 4)
    df["distinct_usernames_5m"] = by_ip["username_code"].transform(
        lambda s: s.rolling("5min").apply(lambda w: len(set(w)), raw=True)
    )
    df["time_delta_sec"] = by_ip["is_failed"].transform(
        lambda s: s.index.to_series().diff().dt.total_seconds()
    ).fillna(0)

    # per username window feature (credential stuffing / distributed attack)
    by_user = df.groupby("username")
    df["distinct_source_ips_5m"] = by_user["source_ip_code"].transform(
        lambda s: s.rolling("5min").apply(lambda w: len(set(w)), raw=True)
    )

    df = df.reset_index()
    df[["attempts_5m", "failed_5m", "distinct_usernames_5m", "distinct_source_ips_5m"]] = df[
        ["attempts_5m", "failed_5m", "distinct_usernames_5m", "distinct_source_ips_5m"]
    ].fillna(0)

    return df


FEATURE_COLS = [
    "time_delta_sec",
    "attempts_5m",
    "failed_5m",
    "failure_ratio_5m",
    "distinct_usernames_5m",
    "distinct_source_ips_5m",
]


def main(file_path):
    raw = pd.read_csv(file_path)
    raw = raw[raw["label"] != "config_anomaly"].reset_index(drop=True)

    df = build_actor_features(raw)
    df["is_brute_force"] = df["label"].isin(BRUTE_FORCE_LABELS).astype(int)

    # chronological split - train on the earlier 70% of the timeline, test
    # on the later 30%, so the evaluation reflects predicting attacks the
    # model hasn't seen the shape of yet, not just interpolating
    split_idx = int(len(df) * 0.70)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"train: {len(train_df):,} rows ({train_df['is_brute_force'].mean():.1%} brute force)")
    print(f"test:  {len(test_df):,} rows ({test_df['is_brute_force'].mean():.1%} brute force)\n")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(train_df[FEATURE_COLS], train_df["is_brute_force"])

    preds = clf.predict(test_df[FEATURE_COLS])

    print("random forest: confusion matrix")
    print(confusion_matrix(test_df["is_brute_force"], preds))
    print("\nrandom forest: classification report")
    print(
        classification_report(
            test_df["is_brute_force"], preds, target_names=["Normal (0)", "Brute Force (1)"]
        )
    )

    print("feature importances")
    for feat, imp in sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:<24} {imp:.3f}")

    # benchmark against the current approach: isolation forest on the
    # same features, unsupervised, for a side-by-side comparison
    iso = IsolationForest(n_estimators=100, contamination="auto", random_state=42, n_jobs=-1)
    iso.fit(train_df[FEATURE_COLS])
    iso_preds = np.where(iso.predict(test_df[FEATURE_COLS]) == -1, 1, 0)

    print("\nisolation forest baseline: confusion matrix")
    print(confusion_matrix(test_df["is_brute_force"], iso_preds))
    print("\nisolation forest baseline: classification report")
    print(
        classification_report(
            test_df["is_brute_force"], iso_preds, target_names=["Normal (0)", "Brute Force (1)"]
        )
    )

    model_filename = "brute_force_classifier_ssh.pkl"
    joblib.dump(clf, model_filename)
    print(f"\nmodel serialized -> '{model_filename}'")


if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "data/ssh_logs.csv"
    main(target_file)
