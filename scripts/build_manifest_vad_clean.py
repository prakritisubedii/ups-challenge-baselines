"""
Build a clean training manifest from the VAD JSONL stream.

Example:
  UPS_ARTIFACT_DIR=/content/drive/MyDrive/ups_artifacts \
    python scripts/build_manifest_vad_clean.py --target_clips 20000

Progress logging prints every N seen entries (default: 50,000).
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime

import requests

VAD_URL = "https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/vad_results.jsonl"
SAMPLE_RATE = 16000
TEN_SECONDS_SAMPLES = 10 * SAMPLE_RATE


def parse_args():
    parser = argparse.ArgumentParser(description="Build VAD-clean manifest from JSONL stream.")
    parser.add_argument("--min_speech_ratio", type=float, default=0.30)
    parser.add_argument("--min_segment_sec", type=float, default=10.0)
    parser.add_argument("--min_trainable_sec", type=float, default=30.0)
    parser.add_argument("--target_clips", type=int, default=20000)
    parser.add_argument("--max_per_tar", type=int, default=200)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--print_every", type=int, default=50000)
    return parser.parse_args()


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_manifest(args):
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    artifact_dir = os.environ.get("UPS_ARTIFACT_DIR", "./artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    ratio_tag = str(args.min_speech_ratio).replace(".", "p")
    seg_tag = str(args.min_segment_sec).replace(".", "p")
    train_tag = str(args.min_trainable_sec).replace(".", "p")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    manifest_name = (
        f"manifest_vad_clean_ratio{ratio_tag}_seg{seg_tag}_"
        f"train{train_tag}_n{args.target_clips}_{timestamp}.jsonl"
    )
    manifest_path = os.path.join(artifact_dir, manifest_name)

    stats_name = f"manifest_stats_{timestamp}.json"
    stats_path = os.path.join(artifact_dir, stats_name)

    seen_entries = 0
    kept = 0

    drops_by_reason = {
        "no_timestamps": 0,
        "bad_duration": 0,
        "low_ratio": 0,
        "short_segment": 0,
        "low_trainable": 0,
        "tar_cap": 0,
        "json_error": 0,
    }

    tar_counts = {}
    unique_tars = set()

    total_speech_sec = 0.0
    total_duration_sec = 0.0
    total_trainable_sec = 0.0

    sum_speech_ratio = 0.0
    sum_duration_min = 0.0
    sum_max_segment_sec = 0.0

    try:
        resp = requests.get(VAD_URL, headers=headers, stream=True, timeout=args.timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"Error: failed to download VAD JSONL: {exc}", file=sys.stderr)
        return 1

    with open(manifest_path, "w") as out_f:
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                drops_by_reason["json_error"] += 1
                continue

            if not isinstance(obj, dict):
                drops_by_reason["json_error"] += 1
                continue

            for vad_key, payload in obj.items():
                seen_entries += 1

                if seen_entries % args.print_every == 0:
                    total_speech_hours = total_speech_sec / 3600.0
                    total_duration_hours = total_duration_sec / 3600.0
                    total_trainable_hours = total_trainable_sec / 3600.0
                    print(
                        f"seen={seen_entries} kept={kept} "
                        f"drops={drops_by_reason} unique_tars={len(unique_tars)} "
                        f"speech_hr={total_speech_hours:.2f} "
                        f"duration_hr={total_duration_hours:.2f} "
                        f"trainable_10s_hr={total_trainable_hours:.2f}"
                    )

                if kept >= args.target_clips:
                    break

                timestamps = payload.get("timestamps") if isinstance(payload, dict) else None
                duration = safe_float(payload.get("duration"), default=0.0) if isinstance(payload, dict) else 0.0
                tar_number = payload.get("tar_number") if isinstance(payload, dict) else None

                if not timestamps:
                    drops_by_reason["no_timestamps"] += 1
                    continue

                if duration <= 0:
                    drops_by_reason["bad_duration"] += 1
                    continue

                speech_samples = 0
                max_segment_samples = 0
                trainable_10s_seconds = 0.0

                for seg in timestamps:
                    if not isinstance(seg, dict):
                        continue
                    start = int(seg.get("start", 0))
                    end = int(seg.get("end", 0))
                    seg_len = max(0, end - start)
                    speech_samples += seg_len
                    max_segment_samples = max(max_segment_samples, seg_len)
                    if seg_len >= TEN_SECONDS_SAMPLES:
                        trainable_10s_seconds += math.floor(seg_len / TEN_SECONDS_SAMPLES) * 10

                speech_seconds = speech_samples / SAMPLE_RATE
                speech_ratio = speech_seconds / duration if duration > 0 else 0.0
                max_segment_seconds = max_segment_samples / SAMPLE_RATE if max_segment_samples > 0 else 0.0

                if speech_ratio < args.min_speech_ratio:
                    drops_by_reason["low_ratio"] += 1
                    continue

                if max_segment_seconds < args.min_segment_sec:
                    drops_by_reason["short_segment"] += 1
                    continue

                if trainable_10s_seconds < args.min_trainable_sec:
                    drops_by_reason["low_trainable"] += 1
                    continue

                tar_count = tar_counts.get(tar_number, 0)
                if tar_count >= args.max_per_tar:
                    drops_by_reason["tar_cap"] += 1
                    continue

                tar_counts[tar_number] = tar_count + 1
                unique_tars.add(tar_number)

                entry = {
                    "vad_key": str(vad_key),
                    "tar_number": safe_int(tar_number),
                    "duration_sec": float(duration),
                    "speech_sec": float(speech_seconds),
                    "speech_ratio": float(speech_ratio),
                    "n_segments": int(len(timestamps)),
                    "max_segment_sec": float(max_segment_seconds),
                    "trainable_10s_sec": float(trainable_10s_seconds),
                }

                out_f.write(json.dumps(entry) + "\n")

                kept += 1
                total_speech_sec += speech_seconds
                total_duration_sec += duration
                total_trainable_sec += trainable_10s_seconds

                sum_speech_ratio += speech_ratio
                sum_duration_min += duration / 60.0
                sum_max_segment_sec += max_segment_seconds

            if kept >= args.target_clips:
                break

    mean_speech_ratio = (sum_speech_ratio / kept) if kept > 0 else 0.0
    mean_duration_min = (sum_duration_min / kept) if kept > 0 else 0.0
    mean_max_segment_sec = (sum_max_segment_sec / kept) if kept > 0 else 0.0

    stats = {
        "params": {
            "min_speech_ratio": args.min_speech_ratio,
            "min_segment_sec": args.min_segment_sec,
            "min_trainable_sec": args.min_trainable_sec,
            "target_clips": args.target_clips,
            "max_per_tar": args.max_per_tar,
            "timeout": args.timeout,
            "print_every": args.print_every,
        },
        "seen_entries": seen_entries,
        "kept": kept,
        "drops_by_reason": drops_by_reason,
        "unique_tars": len(unique_tars),
        "total_speech_hours_kept": total_speech_sec / 3600.0,
        "total_duration_hours_kept": total_duration_sec / 3600.0,
        "total_trainable_hours_10s_kept": total_trainable_sec / 3600.0,
        "mean_speech_ratio": mean_speech_ratio,
        "mean_duration_min": mean_duration_min,
        "mean_max_segment_sec": mean_max_segment_sec,
        "manifest_path": manifest_path,
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\nDone.")
    print(f"Manifest: {manifest_path}")
    print(f"Stats: {stats_path}")
    return 0


def main():
    args = parse_args()
    return build_manifest(args)


if __name__ == "__main__":
    raise SystemExit(main())
