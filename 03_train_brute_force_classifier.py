import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from brute_force_features import BRUTE_FORCE_LABELS, FEATURE_COLS, build_actor_features


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
    target_file = sys.argv[1] if len(sys.argv) > 1 else "data/ssh_anomaly_dataset.csv"
    main(target_file)
