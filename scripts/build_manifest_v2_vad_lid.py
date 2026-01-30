"""
Build a v2 manifest by adding LID info and selecting target clips.

Rules:
  - Drop rows with lang == "nospeech"
  - Prefer all non-English rows, then fill with English to target_clips
"""

import argparse
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Build manifest v2 with LID filtering")
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--lid_json_path", type=str, required=True)
    parser.add_argument("--target_clips", type=int, default=20000)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_prefix", type=str, default="manifest_v2")
    return parser.parse_args()


def load_manifest_rows(manifest_path: str):
    rows = []
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def normalize_lid_key(value: str) -> str:
    if not isinstance(value, str):
        return ""
    if value.startswith("/data/"):
        value = value[len("/data/") :]
    if value.endswith(".mp3"):
        value = value[: -len(".mp3")]
    if "/" in value:
        value = value.split("/")[-1]
    return value


def extract_key_from_entry(entry: dict) -> str:
    for field in ("key", "vad_key", "id", "uid", "filepath", "path", "audio_filepath", "audio_path", "audio", "file"):
        if field in entry:
            return normalize_lid_key(entry.get(field))
    return ""


def extract_lang_from_entry(entry: dict):
    for field in ("prediction", "lang", "language"):
        if field in entry:
            val = entry.get(field)
            if isinstance(val, dict):
                return val.get("label") or val.get("lang") or val.get("language")
            return val
    return None


def load_lid_map(lid_json_path: str, print_every: int = 1_000_000):
    lid_map = {}
    _, ext = os.path.splitext(lid_json_path)
    try:
        print_every = int(os.environ.get("LID_PRINT_EVERY", print_every))
    except ValueError:
        print_every = 1_000_000

    def _add_entry(key: str, lang) -> None:
        if not key or lang is None:
            return
        lid_map[key] = str(lang).lower().strip()

    if ext.lower() == ".jsonl":
        with open(lid_json_path, "r", encoding="utf-8") as f:
            line_count = 0
            for line in f:
                if not line.strip():
                    continue
                line_count += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                key = extract_key_from_entry(obj)
                lang = extract_lang_from_entry(obj)
                _add_entry(key, lang)
                if line_count % print_every == 0:
                    print(f"LID loaded lines: {line_count} | entries: {len(lid_map)}")
        return lid_map

    try:
        with open(lid_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = None
        with open(lid_json_path, "r", encoding="utf-8") as f:
            line_count = 0
            for line in f:
                if not line.strip():
                    continue
                line_count += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                key = extract_key_from_entry(obj)
                lang = extract_lang_from_entry(obj)
                _add_entry(key, lang)
                if line_count % print_every == 0:
                    print(f"LID loaded lines: {line_count} | entries: {len(lid_map)}")
        return lid_map

    if isinstance(data, dict):
        for raw_key, value in data.items():
            key = normalize_lid_key(raw_key)
            if isinstance(value, dict):
                lang = extract_lang_from_entry(value)
            else:
                lang = value
            _add_entry(key, lang)
        return lid_map

    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            key = extract_key_from_entry(entry)
            lang = extract_lang_from_entry(entry)
            _add_entry(key, lang)
        return lid_map

    return lid_map


def main():
    args = parse_args()
    out_dir = os.environ.get("UPS_ARTIFACT_DIR", "./artifacts")
    os.makedirs(out_dir, exist_ok=True)

    rows = load_manifest_rows(args.manifest_path)
    if not rows:
        raise ValueError(f"No rows found in {args.manifest_path}")

    lid_map = load_lid_map(args.lid_json_path)
    if not lid_map:
        raise ValueError(f"No entries found in {args.lid_json_path}")

    rng = random.Random(args.seed)

    stats = {
        "kept_total": 0,
        "kept_non_en": 0,
        "kept_en": 0,
        "dropped_nospeech": 0,
        "dropped_missing_lid": 0,
        "lang_histogram": {},
    }

    non_en_rows = []
    en_rows = []

    for row in rows:
        key = normalize_lid_key(row.get("vad_key"))
        if not key:
            stats["dropped_missing_lid"] += 1
            continue
        lang = lid_map.get(key)
        if lang is None:
            stats["dropped_missing_lid"] += 1
            continue

        lang = str(lang).lower().strip()
        if lang == "nospeech":
            stats["dropped_nospeech"] += 1
            continue

        row_out = dict(row)
        row_out["lang"] = lang
        stats["lang_histogram"][lang] = stats["lang_histogram"].get(lang, 0) + 1

        if lang == "en":
            en_rows.append(row_out)
        else:
            non_en_rows.append(row_out)

    if args.shuffle:
        rng.shuffle(non_en_rows)
        rng.shuffle(en_rows)

    selected = []
    for row in non_en_rows:
        if len(selected) >= args.target_clips:
            break
        selected.append(row)
    for row in en_rows:
        if len(selected) >= args.target_clips:
            break
        selected.append(row)

    stats["kept_total"] = len(selected)
    stats["kept_non_en"] = sum(1 for r in selected if r.get("lang") != "en")
    stats["kept_en"] = sum(1 for r in selected if r.get("lang") == "en")

    out_manifest = os.path.join(out_dir, f"{args.out_prefix}.jsonl")
    out_stats = os.path.join(out_dir, f"{args.out_prefix}_stats.json")

    with open(out_manifest, "w") as f:
        for row in selected:
            f.write(json.dumps(row) + "\n")

    with open(out_stats, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Done. Wrote {len(selected)} rows to {out_manifest}")
    print(f"Wrote stats to {out_stats}")


if __name__ == "__main__":
    main()
