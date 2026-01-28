"""
Stream LID results from Hugging Face and filter to a given manifest's vad_key set.

Example:
  python scripts/build_lid_for_manifest.py \
    --manifest_path /content/drive/MyDrive/ups_artifacts/manifest_v1_small2000.jsonl \
    --out_prefix manifest_v1
"""

import argparse
import json
import os
import time

import requests

LID_URL = "https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/lang_id_results.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Build LID map for a manifest from streamed LID results")
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--out_prefix", type=str, default="manifest_v1")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--print_every", type=int, default=200000)
    return parser.parse_args()


def load_manifest_keys(manifest_path: str) -> set[str]:
    keys = set()
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("vad_key")
            if key:
                keys.add(key)
    return keys


def canonical_id(filepath: str) -> str:
    if filepath.startswith("/data/"):
        filepath = filepath[len("/data/") :]
    if filepath.endswith(".mp3"):
        filepath = filepath[: -len(".mp3")]
    return filepath


def main():
    args = parse_args()
    out_dir = os.environ.get("UPS_ARTIFACT_DIR", "./artifacts")
    os.makedirs(out_dir, exist_ok=True)

    manifest_keys = load_manifest_keys(args.manifest_path)
    if not manifest_keys:
        raise ValueError(f"No vad_key entries found in {args.manifest_path}")

    token = args.hf_token or os.environ.get("HF_TOKEN")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    lid_map = {}
    lang_histogram = {}
    nospeech_variants = {"nospeech", "no_speech", "no-speech"}

    start_time = time.time()
    line_count = 0

    with requests.get(LID_URL, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line_count += 1
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            filepath = row.get("filepath")
            pred = row.get("prediction")
            if not filepath or pred is None:
                continue

            key = canonical_id(filepath)
            if key not in manifest_keys:
                if line_count % args.print_every == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"Lines: {line_count} | found: {len(lid_map)}/{len(manifest_keys)} | "
                        f"elapsed_sec: {elapsed:.1f}"
                    )
                continue

            pred_norm = str(pred).lower().strip()
            lid_map[key] = pred_norm
            lang_histogram[pred_norm] = lang_histogram.get(pred_norm, 0) + 1

            if pred_norm in nospeech_variants:
                pass

            if line_count % args.print_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"Lines: {line_count} | found: {len(lid_map)}/{len(manifest_keys)} | "
                    f"elapsed_sec: {elapsed:.1f}"
                )

            if len(lid_map) >= len(manifest_keys):
                print("All manifest keys found, stopping early.")
                break

    found = len(lid_map)
    missing = len(manifest_keys) - found
    nospeech_count = sum(count for lang, count in lang_histogram.items() if lang in nospeech_variants)

    lid_path = os.path.join(out_dir, f"{args.out_prefix}_lid_for_manifest.json")
    stats_path = os.path.join(out_dir, f"{args.out_prefix}_lid_stats.json")

    with open(lid_path, "w") as f:
        json.dump(lid_map, f)

    stats = {
        "total_manifest_keys": len(manifest_keys),
        "found": found,
        "missing": missing,
        "nospeech_count": nospeech_count,
        "lang_histogram": lang_histogram,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    elapsed = time.time() - start_time
    print(f"Done. Found {found}/{len(manifest_keys)} keys in {elapsed:.1f}s")
    print(f"Wrote: {lid_path}")
    print(f"Wrote: {stats_path}")


if __name__ == "__main__":
    main()
