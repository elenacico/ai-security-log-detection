# AI Security Log Detection

A dual-engine SIEM anomaly detection project: Isolation Forest models for network
telemetry and SSH authentication logs, a supervised Random Forest classifier for
brute-force detection, and a Streamlit dashboard for live/uploaded log analysis
with Gemini-generated incident reports.

## Data Sources

This project uses publicly available datasets from Kaggle. Credit to the original
authors:

- **Network telemetry training data** (`data/cybersecurity_network_logs.csv`)
  [Cybersecurity Network Logs - Anomaly Detection](https://www.kaggle.com/datasets/aatmaca/cybersecurity-network-logs-anomaly-detection)
  by [aatmaca](https://www.kaggle.com/aatmaca)

- **SSH authentication training data** (`data/ssh_logs.csv`)
  [SSH Anomaly Dataset](https://www.kaggle.com/datasets/mdwiraputradananjaya/ssh-anomaly-dataset)
  by [Md Wira Putra Dananjaya](https://www.kaggle.com/mdwiraputradananjaya)

Two additional public datasets were used during development to validate the
models against traffic they weren't trained on (not included in this repo):

- [Cybersecurity Intrusion Detection Dataset](https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset)
  by [Dinesh Naveen Kumar Samudrala](https://www.kaggle.com/dnkumars)
- [Real-Time Network Traffic Dataset for IDS](https://www.kaggle.com/datasets/rajashrichaudhari1/real-time-network-traffic-dataset-for-ids)
  by [Rajashri Chaudhari](https://www.kaggle.com/rajashrichaudhari1) - used to stress-test
  generalization to out-of-distribution traffic

## Setup

```
pip install -r requirements.txt
streamlit run 04_app.py
```

Set `GEMINI_API_KEY` in `.streamlit/secrets.toml` or as an environment variable to
enable AI-generated incident reports.
