#!/usr/bin/env python3
import argparse
from pathlib import Path
from datetime import datetime
import re
from typing import Optional, Tuple

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\b")

def parse_ts(line: str) -> Optional[datetime]:
    """
    Zwraca datetime jeśli na początku linii jest ISO timestamp zakończony 'Z',
    w przeciwnym razie None.
    """
    if not line or not line.strip():
        return None

    m = TS_RE.match(line)
    if not m:
        return None

    ts = m.group(1)
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None

def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two audit logs into one file (concat or timestamp sort).")
    parser.add_argument("--a", default="audit_bruteforce.log", help="First input file (default: audit_bruteforce.log)")
    parser.add_argument("--b", default="generated_audit.log", help="Second input file (default: generated_audit.log)")
    parser.add_argument("-o", "--output", default="merged_audit.log", help="Output file (default: merged_audit.log)")
    parser.add_argument("--mode", choices=["sort", "concat"], default="sort",
                        help="sort = merge and sort by timestamp (default), concat = just append B after A")
    parser.add_argument("--keep-empty", action="store_true",
                        help="Keep empty/whitespace-only lines (default: skip them)")

    args = parser.parse_args()

    a_path = Path(args.a)
    b_path = Path(args.b)
    out_path = Path(args.output)

    if args.mode == "concat":
        with out_path.open("w", encoding="utf-8") as out:
            for p in (a_path, b_path):
                with p.open("r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if (not args.keep_empty) and (not line.strip()):
                            continue
                        out.write(line)
        return

    entries: list[Tuple[datetime, int, str]] = []
    idx = 0

    for p in (a_path, b_path):
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if (not args.keep_empty) and (not line.strip()):
                    continue

                ts = parse_ts(line)
                if ts is None:
                    ts = datetime.max

                entries.append((ts, idx, line))
                idx += 1

    entries.sort(key=lambda x: (x[0], x[1]))

    with out_path.open("w", encoding="utf-8") as out:
        for _, __, line in entries:
            out.write(line)

if __name__ == "__main__":
    main()
