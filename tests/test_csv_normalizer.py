import os
import pandas as pd
from csv_normalizer import normalize_csv
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_pipeline():
    # test 1: known synonyms (layer 1)
    df1 = pd.DataFrame({
        'src_ip': ['10.0.0.1'],
        'event_time': ['2026-03-01 10:00:00'],
        'bytes_sent': [500],
        'login_failures': [2],
        'proto': ['tcp']
    })

    # test 2: generic / unlabeled column names (layer 2)
    df2 = pd.DataFrame({
        'col_a': ['192.168.1.50'],
        'col_b': ['2026-03-01 10:05:00']
    })

    # test 3: unknown / vendor-specific headers (layer 3)
    df3 = pd.DataFrame({
        'cisco_sa': ['172.16.0.10'],
        'syslog_time': ['2026-03-01 10:10:00'],
        'auth_fail_count': [5]
    })

    print("\ntest 1: alias dictionary (layer 1)")
    res1, warnings1 = normalize_csv(df1)
    print("Mapped Columns:", list(res1.columns))
    print("Warnings:", warnings1)

    print("\ntest 2: content inspection (layer 2)")
    res2, warnings2 = normalize_csv(df2)
    print("Mapped Columns:", list(res2.columns))
    print("Warnings:", warnings2)

    print("\ntest 3: gemini fallback (layer 3)")
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping Layer 3 test: GEMINI_API_KEY environment variable not set.")
    else:
        res3, warnings3 = normalize_csv(df3)
        print("Mapped Columns:", list(res3.columns))
        print("Warnings:", warnings3)

if __name__ == "__main__":
    test_pipeline()