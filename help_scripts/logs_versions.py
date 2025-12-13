#!/usr/bin/env python3
import argparse
from pathlib import Path

def strip_after_semicolon(line: str) -> str:
    newline = "\n" if line.endswith("\n") else ""
    core = line[:-1] if newline else line

    if not core.strip():
        return line 

    if ";" not in core:
        return core + newline

    left = core.split(";", 1)[0]
    return left + newline

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create two versions of a log: with labels (original) and without (strip everything after ';')."
    )
    parser.add_argument("-i", "--input", default="merged_audit.log", help="Input log file")
    parser.add_argument("--with-labels", default="merged_with_labels.log", help="Output file with labels (original)")
    parser.add_argument("--no-labels", default="merged_no_labels.log", help="Output file without labels")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_with = Path(args.with_labels)
    out_no = Path(args.no_labels)

    with in_path.open("r", encoding="utf-8", errors="replace") as f_in, \
         out_with.open("w", encoding="utf-8") as f_with, \
         out_no.open("w", encoding="utf-8") as f_no:

        for line in f_in:
            f_with.write(line)
            f_no.write(strip_after_semicolon(line))

if __name__ == "__main__":
    main()
