import json
import os
import re
from google import genai
import pandas as pd

# 1. Target canonical schema definition and synonym dictionary
CANONICAL_KEYS = [
    "source_ip",
    "timestamp",
    "bytes",
    "failed_logins",
    "protocol",
    "duration",
    "geo_distance",
    "status",
    "username",
]

COLUMN_ALIASES = {
    "source_ip": [
        "source_ip",
        "src_ip",
        "ip_src",
        "ip.src",  # Wireshark / TShark
        "client_ip",
        "src",
        "source_address",
        "cisco_sa_ip",
        "remote_addr",
    ],
    "timestamp": [
        "timestamp",
        "datetime",
        "time",
        "@timestamp",
        "event_time",
        "date",
        "log_time",
        "frame.time_epoch",  # Wireshark / TShark
        "frame.time",
        "epoch",
    ],
    "bytes": [
        "packet_size_bytes",
        "bytes",
        "packet_size",
        "size",
        "length",
        "bytes_sent",
        "payload_size",
        "frame.len",  # Wireshark / TShark
        "ip.len",
    ],
    "failed_logins": [
        "failed_logins",
        "failed_count",
        "login_failures",
        "failed_attempts",
        "is_failed",
    ],
    "protocol": [
        "protocol",
        "proto",
        "transport_protocol",
        "net_proto",
        "frame.protocols",  # Wireshark / TShark
        "ip.proto",
    ],
    "duration": [
        "connection_duration_ms",
        "duration",
        "duration_ms",
        "conn_time",
        "session_duration",
    ],
    "geo_distance": [
        "geo_distance_km",
        "geo_distance",
        "distance",
        "geo_dist_km",
    ],
    "status": ["status", "event_type", "action", "auth_result"],
    "username": ["username", "user", "account", "login_user"],
}

IP_REGEX = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"


# --- LAYER 1: Fast Alias Dictionary ---
def _apply_alias_map(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rename_map = {}
    for canonical_name, aliases in COLUMN_ALIASES.items():
        for col in df.columns:
            if col.lower().strip() in [a.lower() for a in aliases]:
                rename_map[col] = canonical_name
                break

    df = df.rename(columns=rename_map)
    unmapped = [key for key in CANONICAL_KEYS if key not in df.columns]
    return df, unmapped


# --- LAYER 2: Data Content Inspection (Regex & Type Checks) ---
def _apply_content_heuristics(
    df: pd.DataFrame, unmapped_keys: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    rename_map = {}
    remaining_unmapped = list(unmapped_keys)

    for col in df.columns:
        if col in CANONICAL_KEYS:
            continue

        sample_vals = df[col].dropna().astype(str).head(10).tolist()
        if not sample_vals:
            continue

        # Check for IP Address pattern
        if "source_ip" in remaining_unmapped:
            if all(re.match(IP_REGEX, str(v).strip()) for v in sample_vals):
                rename_map[col] = "source_ip"
                remaining_unmapped.remove("source_ip")
                continue

        # Check for Timestamp pattern
        if "timestamp" in remaining_unmapped:
            try:
                pd.to_datetime(sample_vals, errors="raise")
                rename_map[col] = "timestamp"
                remaining_unmapped.remove("timestamp")
                continue
            except (ValueError, TypeError):
                pass

    df = df.rename(columns=rename_map)
    return df, remaining_unmapped


# --- LAYER 3: Gemini Zero-Shot Schema Mapper ---
def _apply_gemini_fallback(
    df: pd.DataFrame, unmapped_keys: list[str]
) -> pd.DataFrame:
    api_key = os.getenv("GEMINI_API_KEY")
    if not unmapped_keys or not api_key:
        return df

    client = genai.Client(api_key=api_key)
    sample_data = df.head(2).to_dict(orient="records")

    prompt = f"""
    You are a cybersecurity log normalizer. Map the unmapped original CSV columns to these missing target canonical keys: {unmapped_keys}.

    Original CSV Headers: {list(df.columns)}
    Sample Data (First 2 Rows): {json.dumps(sample_data, default=str)}

    Instructions:
    - Determine which unmapped original column corresponds to which missing canonical key.
    - Return ONLY a valid JSON object mapping original_column_name -> canonical_key.
    - Do not include markdown formatting or explanation.
    Example: {{"cisco_sa": "source_ip", "event_dt": "timestamp"}}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        mapping = json.loads(response.text or "{}")

        valid_map = {
            k: v
            for k, v in mapping.items()
            if k in df.columns and v in unmapped_keys
        }
        df = df.rename(columns=valid_map)
    except Exception:
        pass

    return df


# --- MAIN NORMALIZATION FUNCTION ---
def normalize_csv(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Run 3-Layer Schema Discovery
    df, unmapped = _apply_alias_map(df)
    if unmapped:
        df, unmapped = _apply_content_heuristics(df, unmapped)
    if unmapped:
        df = _apply_gemini_fallback(df, unmapped)

    # 2. Reconcile Canonical Names to Feature Engine Expectations
    column_reconciliation = {
        "bytes": "Packet_Size_Bytes",
        "duration": "Connection_Duration_ms",
        "failed_logins": "Failed_Logins",
        "geo_distance": "Geo_Distance_km",
        "protocol": "Protocol",
    }
    df = df.rename(columns=column_reconciliation)

    # 3. Handle Unix Epoch Timestamps (e.g., Wireshark 1772179636)
    if "timestamp" in df.columns:
        # If timestamp is purely numeric epoch, convert using unit='s'
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # 4. Clean Wireshark Protocol Strings (e.g. "Ether / IP / TCP" -> "TCP")
    if "Protocol" in df.columns:
        df["Protocol"] = (
            df["Protocol"]
            .astype(str)
            .apply(
                lambda x: (
                    x.split("/")[-1].strip().split(" ")[0] if "/" in x else x
                )
            )
        )

    # 5. Fill Missing Default Telemetry Features
    if "Connection_Duration_ms" not in df.columns:
        df["Connection_Duration_ms"] = 1.0
    if "Failed_Logins" not in df.columns:
        df["Failed_Logins"] = 0
    if "Geo_Distance_km" not in df.columns:
        df["Geo_Distance_km"] = 0

    return df
