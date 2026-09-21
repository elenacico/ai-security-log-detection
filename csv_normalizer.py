import json
import os
import re
from google import genai
import pandas as pd

# 1. target canonical schema definition and synonym dictionary
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
        "ip.src",  # wireshark / tshark
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
        "frame.time_epoch",  # wireshark / tshark
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
        "frame.len",  # wireshark / tshark
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
        "frame.protocols",  # wireshark / tshark
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


def _extract_protocol(proto_str: str) -> str:
    """pulls the transport protocol out of a tshark-style frame.protocols
    summary string. the protocol always follows the 'ip' layer segment;
    trailing segments (packet flags, 'raw', 'padding', etc.) are noise and
    must not be mistaken for the protocol name."""
    if "/" not in proto_str:
        return proto_str
    segments = [s.strip() for s in proto_str.split("/")]
    for i, seg in enumerate(segments):
        if seg.upper() == "IP" and i + 1 < len(segments):
            return segments[i + 1].split(" ")[0]
    return segments[-1].split(" ")[0]


# layer 1: fast alias dictionary
def _apply_alias_map(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rename_map = {}
    for canonical_name, aliases in COLUMN_ALIASES.items():
        # if a column already has the canonical name verbatim, that
        # requirement is already satisfied - don't rename a different
        # aliased column onto it too, or both end up sharing the same
        # name (e.g. an ssh log with both "event_type" and "status": the
        # latter already is the canonical "status" column)
        if canonical_name in df.columns:
            continue
        for col in df.columns:
            if col in rename_map:
                continue
            if col.lower().strip() in [a.lower() for a in aliases]:
                rename_map[col] = canonical_name
                break

    df = df.rename(columns=rename_map)
    unmapped = [key for key in CANONICAL_KEYS if key not in df.columns]
    return df, unmapped


# layer 2: data content inspection (regex & type checks)
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

        # check for ip address pattern
        if "source_ip" in remaining_unmapped:
            if all(re.match(IP_REGEX, str(v).strip()) for v in sample_vals):
                rename_map[col] = "source_ip"
                remaining_unmapped.remove("source_ip")
                continue

        # check for timestamp pattern
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


# layer 3: gemini zero-shot schema mapper
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
            model="gemini-3.6-flash",
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


# main normalization function
def normalize_csv(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """returns (normalized_df, warnings), where warnings lists which
    telemetry fields were not present in the source data and were filled
    with constant placeholders (see step 5 below)."""
    # 1. run 3-layer schema discovery
    df, unmapped = _apply_alias_map(df)
    if unmapped:
        df, unmapped = _apply_content_heuristics(df, unmapped)
    if unmapped:
        df = _apply_gemini_fallback(df, unmapped)

    # 2. reconcile canonical names to feature engine expectations
    column_reconciliation = {
        "bytes": "Packet_Size_Bytes",
        "duration": "Connection_Duration_ms",
        "failed_logins": "Failed_Logins",
        "geo_distance": "Geo_Distance_km",
        "protocol": "Protocol",
    }
    df = df.rename(columns=column_reconciliation)

    # 3. handle unix epoch timestamps (e.g., wireshark 1772179636)
    if "timestamp" in df.columns:
        # if timestamp is purely numeric epoch, convert using unit='s'
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # 4. clean wireshark protocol strings (e.g. "ether / ip / tcp" -> "tcp",
    #    "ether / ip / tcp 1.2.3.4:https > ... pa / raw" -> "tcp")
    if "Protocol" in df.columns:
        df["Protocol"] = df["Protocol"].astype(str).apply(_extract_protocol)

    # 5. fill missing default telemetry features
    # these are placeholder values, not real signal - a dataset missing these
    # fields (e.g. a raw packet capture with no session/login data) will have
    # constant values here, which flattens the corresponding model features.
    # only relevant to network telemetry - an ssh log has no business getting
    # these columns bolted on
    warnings = []
    is_network_schema = "Protocol" in df.columns or "Packet_Size_Bytes" in df.columns
    if is_network_schema:
        if "Connection_Duration_ms" not in df.columns:
            df["Connection_Duration_ms"] = 1.0
            warnings.append("Connection_Duration_ms")
        if "Failed_Logins" not in df.columns:
            df["Failed_Logins"] = 0
            warnings.append("Failed_Logins")
        if "Geo_Distance_km" not in df.columns:
            df["Geo_Distance_km"] = 0
            warnings.append("Geo_Distance_km")

    return df, warnings