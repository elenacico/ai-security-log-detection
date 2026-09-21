import numpy as np
import pandas as pd

# positive class = attempts that are actually part of a login-based attack.
# "config_anomaly" is a different failure mode (not a login attack), so it's
# excluded from the target and left out of train/test entirely
BRUTE_FORCE_LABELS = {"brute_force", "brute_force_connection_issue"}

FEATURE_COLS = [
    "time_delta_sec",
    "attempts_5m",
    "failed_5m",
    "failure_ratio_5m",
    "distinct_usernames_5m",
    "distinct_source_ips_5m",
]


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
