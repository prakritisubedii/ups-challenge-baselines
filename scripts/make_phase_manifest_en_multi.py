#!/usr/bin/env python3
"""
Usage example:
  python scripts/make_phase_manifest_en_multi.py \
    --in_manifest data/manifest.jsonl \
    --out_dir out/phase1 \
    --n_en 60000 \
    --n_multi 40000 \
    --max_pct 0.10 \
    --exclude_tars data/exclude_tars.txt \
    --seed 1337
"""

import argparse
import json
import math
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create EN and MULTI manifests for Phase-1 precompute"
    )
    parser.add_argument("--in_manifest", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_en", type=int, required=True)
    parser.add_argument("--n_multi", type=int, required=True)
    parser.add_argument("--max_pct", type=float, required=True)
    parser.add_argument("--exclude_tars", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def load_exclude_tars(path: str | None):
    int_set: set[int] = set()
    unique_keys: set[int] = set()
    if path is None:
        return int_set, 0

    with open(path, "r") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
                    val = int(raw)
                    int_set.add(val)
                    unique_keys.add(val)
            except ValueError:
                pass
    return int_set, len(unique_keys)


def tar_is_excluded(tar_int: int | None, int_set: set[int]) -> bool:
    if tar_int is None:
        return False
    return tar_int in int_set


def count_by_lid(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        lang = row.get("lang") or row.get("lid") or "MISSING"
        counts[lang] = counts.get(lang, 0) + 1
    return counts


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


def main() -> None:
    args = parse_args()
    in_manifest = Path(args.in_manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    int_exclude, exclude_id_count = load_exclude_tars(args.exclude_tars)

    total_rows = 0
    skipped_excluded = 0
    en_rows: list[dict] = []
    multi_rows: list[dict] = []

    with in_manifest.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            tar = row.get("tar_number")
            try:
                tar_int = int(tar) if tar is not None else None
            except (TypeError, ValueError):
                tar_int = None
            if tar_is_excluded(tar_int, int_exclude):
                skipped_excluded += 1
                continue

            lang = row.get("lang") or row.get("lid") or "MISSING"
            if lang == "en":
                en_rows.append(row)
            elif lang != "nospeech":
                multi_rows.append(row)

    rng_en = random.Random(args.seed)
    rng_en.shuffle(en_rows)
    en_selected = en_rows[: args.n_en]
    if len(en_selected) < args.n_en:
        print(
            f"Warning: only {len(en_selected)} EN rows available (target {args.n_en})."
        )

    cap = int(math.floor(args.max_pct * args.n_multi))
    if cap < 1:
        cap = 1
    rng_multi = random.Random(args.seed + 1)
    rng_multi.shuffle(multi_rows)

    multi_selected: list[dict] = []
    multi_counts: dict[str, int] = {}
    selected_indices: set[int] = set()

    for idx, row in enumerate(multi_rows):
        if len(multi_selected) >= args.n_multi:
            break
        lang = row.get("lang") or row.get("lid") or "MISSING"
        if multi_counts.get(lang, 0) < cap:
            multi_selected.append(row)
            multi_counts[lang] = multi_counts.get(lang, 0) + 1
            selected_indices.add(idx)

    relaxed_cap = cap
    if len(multi_selected) < args.n_multi:
        buffer = max(1, int(math.floor(0.05 * cap)))
        relaxed_cap = cap + buffer
        print(
            f"Warning: only {len(multi_selected)} MULTI rows after cap={cap}. "
            f"Relaxing to cap={relaxed_cap}."
        )
        for idx, row in enumerate(multi_rows):
            if len(multi_selected) >= args.n_multi:
                break
            if idx in selected_indices:
                continue
            lang = row.get("lang") or row.get("lid") or "MISSING"
            if multi_counts.get(lang, 0) < relaxed_cap:
                multi_selected.append(row)
                multi_counts[lang] = multi_counts.get(lang, 0) + 1
                selected_indices.add(idx)

    if len(multi_selected) < args.n_multi:
        print(
            f"Warning: only {len(multi_selected)} MULTI rows available "
            f"(target {args.n_multi})."
        )

    out_en = out_dir / "manifest_en.jsonl"
    out_multi = out_dir / "manifest_multi.jsonl"
    write_jsonl(out_en, en_selected)
    write_jsonl(out_multi, multi_selected)

    stats = {
        "n_en_target": args.n_en,
        "n_multi_target": args.n_multi,
        "n_en": len(en_selected),
        "n_multi": len(multi_selected),
        "cap": cap,
        "relaxed_cap": relaxed_cap,
        "max_pct": args.max_pct,
        "seed": args.seed,
        "counts_en_by_lid": count_by_lid(en_selected),
        "counts_multi_by_lid": count_by_lid(multi_selected),
        "excluded_tar_id_count": exclude_id_count,
        "rows_skipped_excluded": skipped_excluded,
        "total_rows_scanned": total_rows,
    }
    stats_path = out_dir / "phase_manifest_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
        f.write("\n")

    print("Summary")
    print(f"  total_rows_scanned: {total_rows}")
    print(f"  excluded_tar_id_count: {exclude_id_count}")
    print(f"  rows_skipped_excluded: {skipped_excluded}")
    print(f"  en_selected: {len(en_selected)} -> {out_en}")
    print(f"  multi_selected: {len(multi_selected)} -> {out_multi}")
    print(f"  cap: {cap}")
    if relaxed_cap != cap:
        print(f"  relaxed_cap: {relaxed_cap}")
    print(f"  stats: {stats_path}")


if __name__ == "__main__":
    main()
