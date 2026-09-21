import pandas as pd

# dataset
df = pd.read_csv('data/cybersecurity_network_logs.csv') 

print("=== Dataset Overview ===")
print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}\n")

print("=== Column Names ===")
print(df.columns.tolist(), "\n")

print("=== First 5 Rows ===")
print(df.head(), "\n")

print("=== Column Data Types & Missing Values ===")
df.info()
