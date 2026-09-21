import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# load full dataset and slice out the final 30% as unseen test traffic
df_full = pd.read_csv('processed_ssh_logs.csv')
split_idx = int(len(df_full) * 0.70)

# Save the 30% holdout split as a new file
df_unseen = df_full.iloc[split_idx:].copy().reset_index(drop=True)
df_unseen.to_csv('unseen_ssh_logs.csv', index=False)
print(f"Created 'unseen_ssh_logs.csv' with {len(df_unseen):,} new log entries.")

# 2. Load serialized model from disk
model = joblib.load('isolation_forest_ssh.pkl')

# 3. Run predictions on unseen data
feature_cols = ['time_delta_sec', 'failed_count_5m', 'total_count_5m', 'failure_ratio_5m']
raw_preds = model.predict(df_unseen[feature_cols])
df_unseen['predicted_anomaly'] = np.where(raw_preds == -1, 1, 0)
df_unseen['true_anomaly'] = (df_unseen['label'] != 'normal').astype(int)

# 4. Evaluate metrics on unseen traffic
print("\n=== Unseen Test Data Confusion Matrix ===")
print(confusion_matrix(df_unseen['true_anomaly'], df_unseen['predicted_anomaly']))

print("\n=== Unseen Test Data Classification Report ===")
print(classification_report(
    df_unseen['true_anomaly'],
    df_unseen['predicted_anomaly'],
    target_names=['Normal (0)', 'Anomaly (1)']
))

# 5. Save plot for unseen data
plt.figure(figsize=(10, 5))
sns.scatterplot(
    x=range(len(df_unseen)),
    y='failed_count_5m',
    hue='predicted_anomaly',
    data=df_unseen,
    palette={0: 'blue', 1: 'red'},
    alpha=0.6,
    s=25
)
plt.title('Anomaly Detection on Unseen Holdout Data (12,548 Rows)')
plt.xlabel('Unseen Log Sequence Index')
plt.ylabel('Failed Logins in 5-Min Window')
plt.tight_layout()
plt.savefig('unseen_data_anomalies.png', dpi=300)
plt.close()

print("Plot saved to 'unseen_data_anomalies.png'!")
