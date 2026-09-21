import numpy as np
import pandas as pd

np.random.seed(42)

records = []
current_time = pd.Timestamp("2026-03-20 10:00:00")

normal_ips = ["192.168.1.10", "192.168.1.15", "192.168.1.20"]
attacker_ips = ["45.142.120.5", "185.220.101.4"]
users = ["root", "admin", "deploy", "alex", "ubuntu"]

for i in range(40):
    is_attack = 15 <= i < 30

    if is_attack:
        delta = int(np.random.randint(1, 3))
        current_time += pd.Timedelta(seconds=delta)
        ip = str(np.random.choice(attacker_ips))
        user = str(np.random.choice(users))
        status = "FAIL"
        failed_count_5m = int(np.random.randint(25, 60))
        total_count_5m = int(failed_count_5m + np.random.randint(0, 3))
        label = "brute_force"
    else:
        delta = int(np.random.randint(20, 120))
        current_time += pd.Timedelta(seconds=delta)
        ip = str(np.random.choice(normal_ips))
        user = str(np.random.choice(["alex", "ubuntu"]))
        status = "SUCCESS" if np.random.rand() > 0.1 else "FAIL"
        failed_count_5m = int(np.random.randint(0, 2))
        total_count_5m = int(failed_count_5m + np.random.randint(1, 4))
        label = "normal"

    fail_ratio = round(failed_count_5m / total_count_5m, 2)

    records.append(
        {
            "timestamp": str(current_time),
            "source_ip": ip,
            "username": user,
            "status": status,
            "time_delta_sec": float(delta),
            "failed_count_5m": failed_count_5m,
            "total_count_5m": total_count_5m,
            "failure_ratio_5m": fail_ratio,
            "label": label,
        }
    )

df = pd.DataFrame(records)
df.to_csv("unseen_ssh_logs.csv", index=False)
print("Saved 40 synthetic log entries to 'unseen_ssh_logs.csv'!")