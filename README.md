# AI Security Log Detection

A dual-engine SIEM anomaly detection project: Isolation Forest models for network
telemetry and SSH authentication logs, a supervised Random Forest classifier for
brute-force detection, and a Streamlit dashboard for live/uploaded log analysis
with Gemini-generated incident reports.

## Data Sources

This project uses publicly available datasets from Kaggle. Credit to the original
authors:

### Training

- **SSH authentication training data** (`data/ssh_logs.csv`)
  [SSH Anomaly Dataset](https://www.kaggle.com/datasets/mdwiraputradananjaya/ssh-anomaly-dataset)
  by [Md Wira Putra Dananjaya](https://www.kaggle.com/mdwiraputradananjaya)
  (renamed from `ssh_anomaly_dataset.csv`)

- **Network telemetry training data** (`data/cybersecurity_network_logs.csv`)
  [Cybersecurity Network Logs - Anomaly Detection](https://www.kaggle.com/datasets/aatmaca/cybersecurity-network-logs-anomaly-detection)
  by [aatmaca](https://www.kaggle.com/aatmaca)

### Testing

- [Cybersecurity Network Logs - Anomaly Detection](https://www.kaggle.com/datasets/aatmaca/cybersecurity-network-logs-anomaly-detection)
  by [aatmaca](https://www.kaggle.com/aatmaca)
- [Real-Time Network Traffic Dataset for IDS](https://www.kaggle.com/datasets/rajashrichaudhari1/real-time-network-traffic-dataset-for-ids)
  by [Rajashri Chaudhari](https://www.kaggle.com/rajashrichaudhari1) - used to stress-test
  generalization to out-of-distribution traffic

## Pipeline

Run in order to reproduce the trained models from raw data:

1. `01_preprocess.py` - normalizes a raw CSV and engineers rolling-window features
2. `02_train_isolation_forest.py` - trains the network/SSH Isolation Forest models
3. `03_train_brute_force_classifier.py` - trains the supervised SSH brute-force
   classifier directly from raw logs
4. `04_app.py` - the Streamlit dashboard (the models above are already trained and
   committed, so this can be run on its own)

`csv_normalizer.py`, `ai_explainer.py`, and `brute_force_features.py` are shared
modules imported by the scripts above, not run directly.

`scripts/` holds optional utilities not part of the core pipeline: a live-traffic
generator, a holdout-evaluation script, an API connectivity smoke test, and a
synthetic SSH log generator.

## Setup

```
pip install -r requirements.txt
streamlit run 04_app.py
```

To enable AI-generated incident reports, copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and add your own Gemini API key (or set `GEMINI_API_KEY`
as an environment variable instead). `secrets.toml` is gitignored, so your key is
never committed.
