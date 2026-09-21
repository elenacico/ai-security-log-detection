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

## Setup

```
pip install -r requirements.txt
streamlit run 04_app.py
```

To enable AI-generated incident reports, copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and add your own Gemini API key (or set `GEMINI_API_KEY`
as an environment variable instead). `secrets.toml` is gitignored, so your key is
never committed.
