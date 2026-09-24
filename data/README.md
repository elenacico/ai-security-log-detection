# data/

This folder is where training/testing data lives when you run the project. The
datasets themselves aren't included in this repository - see the main
[README](../README.md#data-sources) for the license and terms of the original
sources before downloading.

## Required files

Download the two datasets and place them here with these exact names:

- **`cybersecurity_network_logs.csv`** - network telemetry training data
  ([source](https://www.kaggle.com/datasets/aatmaca/cybersecurity-network-logs-anomaly-detection)).
  Expected columns: `Protocol`, `Packet_Size_Bytes`, `Connection_Duration_ms`,
  `Failed_Logins`, `Geo_Distance_km`, `Is_Malicious`.

- **`ssh_anomaly_dataset.csv`** - SSH authentication training data
  ([source](https://www.kaggle.com/datasets/mdwiraputradananjaya/ssh-anomaly-dataset)).
  Expected columns: `timestamp`, `source_ip`, `username`, `event_type`, `status`,
  `label`, `detail`.

If your own dataset uses different column names, `csv_normalizer.py` will try to
map common synonyms automatically (see its `COLUMN_ALIASES`).

## Generated files

Everything else in this folder is produced by running the pipeline and doesn't
need to be downloaded or committed:

- `processed_network_logs.csv`, `processed_ssh_logs.csv` - written by
  `01_preprocess.py`
- `unseen_ssh_logs.csv` - written by `scripts/evaluate_holdout.py`
- `live_stream.csv` - written by `scripts/log_simulator.py` for the dashboard's
  live-mode demo
