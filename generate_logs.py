#!/usr/bin/env python3

import csv
import argparse
import datetime as dt
import random
from typing import Dict, Tuple, Optional

SESSION_TIMEOUT_SECONDS = 600


def parse_timestamp(ts: str) -> dt.datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(ts, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized timestamp format: {ts!r}")


def format_timestamp(d: dt.datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def random_session_id(length: int = 12) -> str:
    hex_chars = "0123456789abcdef"
    return "".join(random.choice(hex_chars) for _ in range(length))


def random_password(min_len: int = 6, max_len: int = 12) -> str:
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
    length = random.randint(min_len, max_len)
    return "".join(random.choice(chars) for _ in range(length))


def format_log_message(row: Dict[str, str]) -> str:
    event_type = row.get("event_type", "").strip()
    status = row.get("status", "").strip()
    username = row.get("username", "").strip()
    source_ip = row.get("source_ip", "").strip()
    detail = (row.get("detail") or "").strip()

    if event_type == "Accepted password":
        pwd = random_password()
        return f"login attempt [{username}/{pwd}] succeeded"

    if event_type == "Failed password":
        pwd = random_password()
        return f"login attempt [{username}/{pwd}] failed"

    if event_type == "Command executed":
        cmd = status or "unknown"
        return f"CMD: {cmd}"

    if event_type == "Disconnected":
        seconds = random.randint(1, 300)
        return f"Connection lost after {seconds} seconds"

    if event_type == "Connection error":
        human_reason = status.replace("_", " ") if status else "error"
        return f"Connection error from {source_ip}: {human_reason}"

    if event_type == "Configuration Anomaly":
        extra = detail or status or "anomaly detected"
        return f"Configuration anomaly detected for user {username} from {source_ip}: {extra}"

    return f"Unknown event '{event_type}' for user {username} from {source_ip} (status={status})"


def generate_logs(input_csv: str, output_log: str) -> None:
    with open(input_csv, newline="") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)

    rows.sort(key=lambda r: r["timestamp"])

    sessions: Dict[Tuple[str, str], Tuple[str, dt.datetime]] = {}

    with open(output_log, "w") as f_out:
        for row in rows:
            ts_str = row.get("timestamp", "")
            if not ts_str:
                continue

            ts = parse_timestamp(ts_str)

            key = (row.get("source_ip", "").strip(), row.get("username", "").strip())

            if key in sessions:
                session_id, last_ts = sessions[key]
                if (ts - last_ts).total_seconds() > SESSION_TIMEOUT_SECONDS:
                    session_id = random_session_id()
            else:
                session_id = random_session_id()

            sessions[key] = (session_id, ts)

            ts_out = format_timestamp(ts)
            message = format_log_message(row)

            detail = (row.get("label") or "").strip()

            line = f"{ts_out} {session_id} {message};{detail}\n"
            f_out.write(line)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate audit-style SSH logs from ssh_anomaly_dataset.csv"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="ssh_anomaly_dataset.csv",
        help="Path to input CSV (default: ssh_anomaly_dataset.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="generated_audit.log",
        help="Path to output log file (default: generated_audit.log)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    generate_logs(args.input, args.output)


if __name__ == "__main__":
    main()
