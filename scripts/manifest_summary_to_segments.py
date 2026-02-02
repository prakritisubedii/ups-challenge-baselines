#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter


def _stable_int_from_key(key: str) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _select_window_indices(n_windows: int, max_windows: int, seed: int, key: str):
    if n_windows <= max_windows:
        return list(range(n_windows))
    rng = random.Random(seed + _stable_int_from_key(key))
    idxs = rng.sample(range(n_windows), max_windows)
    return sorted(idxs)


def _iter_windows(duration_sec: float, segment_sec: float, min_seg_sec: float):
    if duration_sec <= 0 or segment_sec <= 0 or min_seg_sec <= 0:
        return []
    n_full = int(math.floor(duration_sec / segment_sec))
    windows = []
    for i in range(n_full):
        start = i * segment_sec
        end = start + segment_sec
        windows.append((start, end))
    # Optional tail window if long enough but shorter than segment_sec.
    tail_start = n_full * segment_sec
    tail_len = duration_sec - tail_start
    if tail_len >= min_seg_sec and tail_start < duration_sec:
        windows.append((tail_start, min(tail_start + segment_sec, duration_sec)))
    return windows


def main():
    parser = argparse.ArgumentParser(
        description="Convert summary-style VAD+LID manifest into fixed-length segments."
    )
    parser.add_argument("--in_manifest", required=True, help="Path to input summary JSONL.")
    parser.add_argument("--out_manifest", required=True, help="Path to output segments JSONL.")
    parser.add_argument("--segment_sec", type=float, default=10.0, help="Segment length in seconds.")
    parser.add_argument(
        "--min_seg_sec",
        type=float,
        default=None,
        help="Minimum segment length; defaults to segment_sec.",
    )
    parser.add_argument(
        "--max_windows_per_key",
        type=int,
        default=5,
        help="Maximum windows to emit per key.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducible sampling.")
    args = parser.parse_args()

    if args.min_seg_sec is None:
        args.min_seg_sec = args.segment_sec

    input_rows = 0
    output_segments = 0
    dropped_rows = 0
    lid_counter = Counter()
    seg_lengths = []

    with open(args.in_manifest, "r", encoding="utf-8") as f_in, open(
        args.out_manifest, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            input_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped_rows += 1
                continue

            key = row.get("vad_key")
            lid = row.get("lang")
            dur = row.get("duration_sec")
            if not key or not lid or dur is None or float(dur) <= 0:
                dropped_rows += 1
                continue

            try:
                dur = float(dur)
            except (TypeError, ValueError):
                dropped_rows += 1
                continue

            windows = _iter_windows(dur, args.segment_sec, args.min_seg_sec)
            if not windows:
                dropped_rows += 1
                continue

            idxs = _select_window_indices(len(windows), args.max_windows_per_key, args.seed, key)
            tar_number = row.get("tar_number")

            for i in idxs:
                start, end = windows[i]
                seg = {
                    "key": key,
                    "start_sec": float(f"{start:.3f}"),
                    "end_sec": float(f"{end:.3f}"),
                    "lid": lid,
                    "tar_number": tar_number,
                    "chunk_id": f"{key}__{start:.3f}__{end:.3f}",
                }
                f_out.write(json.dumps(seg, ensure_ascii=True) + "\n")
                output_segments += 1
                lid_counter[lid] += 1
                seg_lengths.append(end - start)

    print("Report")
    print(f"  input_rows: {input_rows}")
    print(f"  output_segments: {output_segments}")
    print(f"  dropped_rows: {dropped_rows}")
    print("  top_lids:")
    for lid, count in lid_counter.most_common(10):
        print(f"    {lid}: {count}")
    if seg_lengths:
        mean = statistics.mean(seg_lengths)
        stdev = statistics.pstdev(seg_lengths)
        print("  segment_length_sec:")
        print(f"    min: {min(seg_lengths):.6f}")
        print(f"    mean: {mean:.6f}")
        print(f"    max: {max(seg_lengths):.6f}")
        print(f"    std: {stdev:.6f}")


if __name__ == "__main__":
    main()
