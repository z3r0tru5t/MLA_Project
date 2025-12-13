#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

LABEL_SEP = ";"


def split_labeled_line(line: str) -> Tuple[str, Optional[str]]:
    line = line.rstrip("\n")
    if LABEL_SEP in line:
        base, maybe_label = line.rsplit(LABEL_SEP, 1)
        if maybe_label and (" " not in maybe_label) and ("\t" not in maybe_label):
            return base, maybe_label.strip()
    return line, None


def parse_base_line(base_line: str):
    """
    2023-...Z <session_id> <message...>
    """
    base_line = base_line.strip()
    if not base_line:
        return None
    parts = base_line.split(" ", 2)
    if len(parts) < 3:
        return None
    ts, session, msg = parts[0], parts[1], parts[2]
    return ts, session, msg


def load_ground_truth_sessions(labeled_path: str) -> Dict[str, str]:
    counts: Dict[str, Dict[str, int]] = {}
    with open(labeled_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            base, label = split_labeled_line(raw)
            if not label:
                continue
            parsed = parse_base_line(base)
            if not parsed:
                continue
            _, session, _ = parsed
            counts.setdefault(session, {})
            counts[session][label] = counts[session].get(label, 0) + 1

    gt: Dict[str, str] = {}
    for session, lab_counts in counts.items():
        gt[session] = max(lab_counts.items(), key=lambda kv: kv[1])[0]
    return gt


def load_unlabeled_sessions(unlabeled_path: str) -> Dict[str, List[str]]:
    sessions: Dict[str, List[str]] = {}
    with open(unlabeled_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            parsed = parse_base_line(raw)
            if not parsed:
                continue
            ts, session, msg = parsed
            sessions.setdefault(session, [])
            sessions[session].append(f"{ts} {msg}")
    return sessions


def make_prompt(session_id: str, events: List[str], max_events: int, max_chars: int) -> str:
    cut = events[:max_events]
    text = "SESSION_ID: " + session_id + "\nEVENTS:\n" + "\n".join(cut)
    text = text[:max_chars]

    return f"""Classify this Cowrie SSH session as brute_force or not.

Return ONLY JSON:
{{"is_bruteforce": true}} or {{"is_bruteforce": false}}

SESSION:
{text}
"""


def parse_model_json(output_text: str) -> Optional[bool]:

    m = re.search(r"\{.*?\}", output_text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if "is_bruteforce" in obj and isinstance(obj["is_bruteforce"], bool):
            return obj["is_bruteforce"]
    except Exception:
        return None
    return None


def load_model(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        dtype=torch.float16, 
    )
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def classify_session(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    max_length: int,
) -> str:

    messages = [
        {"role": "system", "content": "You are a security analyst. Output only JSON."},
        {"role": "user", "content": prompt},
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length, 
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False, 
        eos_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)


    return decoded[len(chat_text):].strip() if decoded.startswith(chat_text) else decoded.strip()


def metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return acc, prec, rec, f1


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unlabeled", required=True)
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--label-name", default="brute_force")
    ap.add_argument("--limit-sessions", type=int, default=0)
    ap.add_argument("--cache", default="qwen_pred_cache.jsonl")


    ap.add_argument("--max-events", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--max-length", type=int, default=1024)


    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--save-raw", action="store_true", help="zapisz pelny raw output (duze pliki)")

    args = ap.parse_args()

    gt = load_ground_truth_sessions(args.labeled)
    sessions = load_unlabeled_sessions(args.unlabeled)

    tokenizer, model = load_model(args.model)


    cache: Dict[str, dict] = {}
    if args.cache and os.path.exists(args.cache):
        with open(args.cache, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "session" in obj:
                    cache[obj["session"]] = obj

    gt_sessions = [sid for sid in sessions.keys() if sid in gt]
    if args.limit_sessions and args.limit_sessions > 0:
        gt_sessions = gt_sessions[:args.limit_sessions]

    tp = fp = tn = fn = 0

    all_rows: List[dict] = []
    mismatch_rows: List[dict] = []

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_fp = open(args.cache, "a", encoding="utf-8") if args.cache else None

    for sid in gt_sessions:
        y_true = (gt[sid] == args.label_name)

        if sid in cache:
            y_pred = bool(cache[sid].get("is_bruteforce", False))
            raw_out = cache[sid].get("raw", "")
        else:
            prompt = make_prompt(sid, sessions[sid], args.max_events, args.max_chars)
            raw_out = classify_session(
                tokenizer, model, prompt,
                max_new_tokens=args.max_new_tokens,
                max_length=args.max_length,
            )
            pred = parse_model_json(raw_out)
            y_pred = bool(pred) if pred is not None else False

            rec_obj = {"session": sid, "is_bruteforce": y_pred, "raw": raw_out}
            cache[sid] = rec_obj
            if cache_fp:
                cache_fp.write(json.dumps(rec_obj, ensure_ascii=False) + "\n")
                cache_fp.flush()


        if y_true and y_pred:
            tp += 1
            status = "TP"
        elif (not y_true) and y_pred:
            fp += 1
            status = "FP"
        elif (not y_true) and (not y_pred):
            tn += 1
            status = "TN"
        else:
            fn += 1
            status = "FN"

        raw_to_save = raw_out if args.save_raw else (raw_out[:500] + ("..." if len(raw_out) > 500 else ""))

        row = {
            "session": sid,
            "true_label": gt[sid],
            "pred_is_bruteforce": int(y_pred),
            "true_is_bruteforce": int(y_true),
            "status": status,
            "raw_preview": raw_to_save,
        }
        all_rows.append(row)
        if status in ("FP", "FN"):
            mismatch_rows.append(row)

    if cache_fp:
        cache_fp.close()

    acc, prec, rec, f1 = metrics(tp, fp, tn, fn)

    print("\n=== Wyniki ===")
    print(f"Sesje ocenione: {len(gt_sessions)}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")

    if mismatch_rows:
        print("\n=== Błędne sesje (pierwsze 20) ===")
        for r in mismatch_rows[:20]:
            pred_txt = "brute_force" if r["pred_is_bruteforce"] else "not"
            print(f"- session={r['session']} true={r['true_label']} pred={pred_txt}")


    results_csv = out_dir / "results.csv"
    mismatches_csv = out_dir / "mismatches.csv"
    summary_json = out_dir / "summary.json"

    fieldnames = ["session", "true_label", "true_is_bruteforce", "pred_is_bruteforce", "status", "raw_preview"]
    write_csv(results_csv, all_rows, fieldnames)
    write_csv(mismatches_csv, mismatch_rows, fieldnames)

    summary = {
        "sessions_evaluated": len(gt_sessions),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "model": args.model,
        "label_name": args.label_name,
        "max_events": args.max_events,
        "max_chars": args.max_chars,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nZapisano:\n- {results_csv}\n- {mismatches_csv}\n- {summary_json}")


if __name__ == "__main__":
    main()
