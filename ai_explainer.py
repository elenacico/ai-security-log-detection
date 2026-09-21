import json
import os
from google import genai
from google.genai import types


def generate_incident_report(engine_type: str, telemetry: dict) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {
            "summary": "GEMINI_API_KEY environment variable not found.",
            "command": "# Please set GEMINI_API_KEY in your environment",
        }

    # Initialize the official Gemini client
    client = genai.Client(api_key=api_key)

    if engine_type == "Host SSH Engine":
        prompt = f"""
        You are a Tier-1 SOC Security Analyst. Analyze this host SSH anomaly:
        - Target IP: {telemetry.get('source_ip', 'Unknown')}
        - Peak Failed Logins (5m): {telemetry.get('failed_count_5m', 0)}
        - Targeted Usernames: {telemetry.get('usernames', [])}
        - Risk Score: {telemetry.get('risk_score', 0)}/100

        Provide your response strictly in JSON format with these exact keys:
        1. "summary": A 2-sentence plain-English breakdown of the brute-force attempt.
        2. "command": The exact Linux UFW command to block this IP (e.g., sudo ufw deny from <IP>).
        """
    else:
        prompt = f"""
        You are a Tier-1 SOC Security Analyst. Analyze this network telemetry anomaly:
        - Packet Size: {telemetry.get('Packet_Size_Bytes', 0)} Bytes
        - Duration: {telemetry.get('Connection_Duration_ms', 0)} ms
        - Throughput: {telemetry.get('bytes_per_ms', 0)} bytes/ms
        - Geo Distance: {telemetry.get('Geo_Distance_km', 0)} km
        - Risk Score: {telemetry.get('risk_score', 0)}/100

        Provide your response strictly in JSON format with these exact keys:
        1. "summary": A 2-sentence plain-English breakdown of the network anomaly (e.g., exfiltration, port scan, DDoS).
        2. "command": The exact Linux iptables command to drop traffic for this threat (e.g., sudo iptables -A INPUT -p tcp --dport 80 -j DROP).
        """

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )
        return json.loads(response.text or "{}")
    except Exception as e:
        return {
            "summary": f"Gemini API execution error: {str(e)}",
            "command": "# Unable to generate automated block rule.",
        }
