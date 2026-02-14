"""
Build a local VAD segments cache for a specific manifest batch.

Output JSONL schema (one line per key):
  {"vad_key":"<key>","timestamps":[{"start":int,"end":int}, ...]}

The timestamps are sample indices at 16kHz from the upstream VAD stream.
"""

import argparse
import json
import os
import sys

import requests

VAD_URL = "https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/vad_results.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Build per-manifest VAD segments cache JSONL.")
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--print_every", type=int, default=50000)
    return parser.parse_args()


def load_required_keys(manifest_path: str) -> set[str]:
    keys: set[str] = set()
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            key = obj.get("vad_key") or obj.get("key") or obj.get("__key__")
            if key:
                keys.add(str(key))
    return keys


def build_cache(args) -> int:
    required_keys = load_required_keys(args.manifest_path)
    needed = len(required_keys)
    if needed == 0:
        print(f"No keys found in manifest: {args.manifest_path}", file=sys.stderr)
        return 1

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    try:
        resp = requests.get(VAD_URL, headers=headers, stream=True, timeout=args.timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"Error: failed to download VAD JSONL: {exc}", file=sys.stderr)
        return 1

    found = 0
    scanned = 0
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w") as out_f:
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            scanned += 1
            if args.print_every > 0 and scanned % args.print_every == 0:
                print(f"scanned={scanned} found={found}/{needed}", flush=True)
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if not required_keys:
                break

            matched_keys = [k for k in obj.keys() if k in required_keys]
            for key in matched_keys:
                payload = obj.get(key)
                timestamps = payload.get("timestamps") if isinstance(payload, dict) else None
                if not isinstance(timestamps, list):
                    timestamps = []
                cleaned = []
                for seg in timestamps:
                    if not isinstance(seg, dict):
                        continue
                    try:
                        start = int(seg.get("start", 0))
                        end = int(seg.get("end", 0))
                    except (TypeError, ValueError):
                        continue
                    if end > start:
                        cleaned.append({"start": start, "end": end})
                out_f.write(json.dumps({"vad_key": key, "timestamps": cleaned}) + "\n")
                required_keys.remove(key)
                found += 1
            if not required_keys:
                break

    missing = needed - found
    print(f"needed_keys={needed}", flush=True)
    print(f"found_keys={found}", flush=True)
    print(f"missing_keys={missing}", flush=True)
    print(f"cache_path={args.out_path}", flush=True)
    if missing > 0:
        sample_missing = sorted(list(required_keys))[:10]
        print(f"missing_sample={sample_missing}", flush=True)
    return 0


def main():
    args = parse_args()
    return build_cache(args)


if __name__ == "__main__":
    raise SystemExit(main())
