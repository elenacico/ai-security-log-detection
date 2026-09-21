import pandas as pd
import numpy as np
import time
import os

# Source dataset to pull realistic rows from
SOURCE_CSV = "data/cybersecurity_network_logs.csv"
STREAM_OUTPUT = "live_stream.csv"

def init_stream():
    """Reads source dataset and prepares an empty live stream file with headers."""
    df_src = pd.read_csv(SOURCE_CSV)
    # Create empty streaming file with same columns
    df_src.iloc[:0].to_csv(STREAM_OUTPUT, index=False)
    print(f"✓ Initialized live stream buffer: '{STREAM_OUTPUT}'")
    return df_src

def start_simulation(df_src, interval_sec=1.5):
    """Appends 1 random row to live_stream.csv every `interval_sec` seconds."""
    print(f"🚀 Live Log Generator running (Appending every {interval_sec}s)... Press Ctrl+C to stop.")

    try:
        while True:
            # 90% chance normal row, 10% chance artificial threat spike
            sample_row = df_src.sample(1).copy()

            if np.random.rand() < 0.10:
                # Inject artificial exfiltration/scan attack pattern
                sample_row['Packet_Size_Bytes'] = np.random.randint(50000, 150000)
                sample_row['Connection_Duration_ms'] = np.random.randint(1, 50)
                sample_row['Failed_Logins'] = np.random.randint(15, 50)

            # Append to stream file
            sample_row.to_csv(STREAM_OUTPUT, mode='a', header=False, index=False)
            print(f"[STREAM EVENT] Appended row -> Bytes: {sample_row['Packet_Size_Bytes'].values[0]} | Duration: {sample_row['Connection_Duration_ms'].values[0]}ms")

            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped.")

if __name__ == "__main__":
    if os.path.exists(SOURCE_CSV):
        df_source = init_stream()
        start_simulation(df_source)
    else:
        print(f"Error: Could not find base dataset '{SOURCE_CSV}'.")
