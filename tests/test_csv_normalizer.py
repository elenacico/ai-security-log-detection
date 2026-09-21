import os
import pandas as pd
from csv_normalizer import normalize_csv
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_pipeline():
    # Test 1: Known synonyms (Layer 1)
    df1 = pd.DataFrame({
        'src_ip': ['10.0.0.1'],
        'event_time': ['2026-03-01 10:00:00'],
        'bytes_sent': [500],
        'login_failures': [2],
        'proto': ['tcp']
    })

    # Test 2: Generic / Unlabeled column names (Layer 2)
    df2 = pd.DataFrame({
        'col_a': ['192.168.1.50'],
        'col_b': ['2026-03-01 10:05:00']
    })

    # Test 3: Unknown / Vendor-specific headers (Layer 3)
    df3 = pd.DataFrame({
        'cisco_sa': ['172.16.0.10'],
        'syslog_time': ['2026-03-01 10:10:00'],
        'auth_fail_count': [5]
    })

    print("\n--- Test 1: Alias Dictionary (Layer 1) ---")
    res1 = normalize_csv(df1)
    print("Mapped Columns:", list(res1.columns))

    print("\n--- Test 2: Content Inspection (Layer 2) ---")
    res2 = normalize_csv(df2)
    print("Mapped Columns:", list(res2.columns))

    print("\n--- Test 3: Gemini Fallback (Layer 3) ---")
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping Layer 3 test: GEMINI_API_KEY environment variable not set.")
    else:
        res3 = normalize_csv(df3)
        print("Mapped Columns:", list(res3.columns))

if __name__ == "__main__":
    test_pipeline()
