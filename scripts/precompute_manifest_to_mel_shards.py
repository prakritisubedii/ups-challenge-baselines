"""
Precompute log-mel features for a v2 manifest into shard .pt files.

Output:
  - <out_dir>/shards/shard-00000.pt ... list of dicts with mel tensors
  - <out_dir>/manifest_features.jsonl ... metadata without mel
  - <out_dir>/stats.json
  - <out_dir>/precompute_config.json
"""

import argparse
import concurrent.futures as futures
import glob
import json
import os
import sys
import time

import torch
import webdataset as wds
from torchcodec.decoders import AudioDecoder

# Make sure Python can import from this repo
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

SAMPLE_RATE_DEFAULT = 16000


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute log-mel shards from a manifest v2 JSONL")
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--examples_per_shard", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=400)
    parser.add_argument("--hop_length", type=int, default=160)
    return parser.parse_args()


def load_manifest_entries(manifest_path: str) -> list[dict]:
    entries = []
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def tar_url_for_number(tar_number: str, hf_token: str | None):
    tar_number = str(tar_number).zfill(6)
    if int(tar_number) <= 5000:
        base = "https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/audio"
    else:
        base = "https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/audio2"
    url = f"{base}/{tar_number}.tar?download=True"
    if hf_token is None:
        raise ValueError("HF_TOKEN is not set")
    token_header = f"Authorization:Bearer {hf_token}"
    return f"pipe:curl -s -L {url} -H {token_header}"


def fetch_mp3_bytes_from_url(url: str, key: str):
    dataset = (
        wds.WebDataset([url], shardshuffle=False)
        .to_tuple("mp3", "__key__", "__url__", handler=wds.handlers.ignore_and_continue)
    )
    for mp3_bytes, sample_key, _ in dataset:
        if sample_key == key:
            return mp3_bytes
    return None


def fetch_mp3_bytes(tar_number: int, key: str, hf_token: str | None):
    url = tar_url_for_number(str(tar_number), hf_token)
    return fetch_mp3_bytes_from_url(url, key)


def create_mel_filterbank(sr: int, n_fft: int, n_mels: int, f_min: float = 0.0, f_max: float | None = None):
    if f_max is None:
        f_max = sr / 2.0

    def hz_to_mel(freq_hz):
        return 2595.0 * torch.log10(torch.tensor(1.0) + freq_hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    m_min = hz_to_mel(torch.tensor(f_min))
    m_max = hz_to_mel(torch.tensor(f_max))
    m_points = torch.linspace(m_min, m_max, n_mels + 2)
    hz_points = mel_to_hz(m_points)
    bin_freqs = torch.floor((n_fft + 1) * hz_points / sr).long()

    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for i in range(n_mels):
        left = bin_freqs[i].item()
        center = bin_freqs[i + 1].item()
        right = bin_freqs[i + 2].item()
        if center == left or right == center:
            continue
        for j in range(left, center):
            fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            fb[i, j] = (right - j) / (right - center)
    return fb


def waveform_to_log_mel(waveform: torch.Tensor, sr: int, n_fft: int = 400, hop_length: int = 160, n_mels: int = 80):
    # waveform: [B, T]
    try:
        import torchaudio

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )(waveform)
        log_mel = torch.log(mel + 1e-6)
        return log_mel
    except Exception:
        stft = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft),
            return_complex=True,
        )
        power = stft.abs() ** 2
        fb = create_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels).to(power.device)
        mel = torch.matmul(fb, power)
        log_mel = torch.log(mel + 1e-6)
        return log_mel


def get_value(row: dict, keys: list[str], default=None):
    for k in keys:
        if k in row:
            return row.get(k)
    return default


def build_chunk_id(row: dict) -> str:
    chunk_id = get_value(row, ["chunk_id", "id", "uid"])
    if chunk_id:
        return str(chunk_id)
    key = get_value(row, ["key", "vad_key", "__key__"])
    start_sec = get_value(row, ["start_sec", "start"])
    end_sec = get_value(row, ["end_sec", "end"])
    if key is None or start_sec is None or end_sec is None:
        return ""
    return f"{key}_{start_sec}_{end_sec}"


def extract_segment_seconds(row: dict):
    start_sec = get_value(row, ["start_sec", "start"])
    end_sec = get_value(row, ["end_sec", "end"])
    if start_sec is None or end_sec is None:
        return None, None
    return float(start_sec), float(end_sec)


def process_entry(entry: dict, sr: int, n_fft: int, hop_length: int, n_mels: int, hf_token: str | None) -> dict | None:
    key = get_value(entry, ["key", "vad_key", "__key__"])
    if key is None:
        raise ValueError("Missing key/vad_key")

    chunk_id = build_chunk_id(entry)
    if not chunk_id:
        raise ValueError("Missing chunk_id and cannot derive from key/start/end")

    start_sec, end_sec = extract_segment_seconds(entry)
    if start_sec is None or end_sec is None:
        raise ValueError("Missing start/end seconds")

    url = get_value(entry, ["url", "tar_url"])
    tar_number = get_value(entry, ["tar_number"])

    mp3_bytes = None
    if url:
        mp3_bytes = fetch_mp3_bytes_from_url(url, key)
    elif tar_number is not None:
        mp3_bytes = fetch_mp3_bytes(int(tar_number), key, hf_token)

    if mp3_bytes is None:
        raise ValueError("Failed to fetch mp3 bytes")

    decoder = AudioDecoder(source=mp3_bytes, sample_rate=sr, num_channels=1)
    duration = decoder.metadata.duration_seconds_from_header
    if duration is None or duration <= 0:
        raise ValueError("Invalid duration")

    start_sec = max(0.0, min(start_sec, duration))
    end_sec = max(start_sec, min(end_sec, duration))

    samples = decoder.get_samples_played_in_range(start_sec, end_sec)
    chunk = samples.data.squeeze(0)

    waveform = chunk.unsqueeze(0)
    log_mel = waveform_to_log_mel(waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel = log_mel.squeeze(0).float().cpu()

    out = {
        "chunk_id": chunk_id,
        "key": key,
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "sr": int(sr),
        "mel": mel,
    }
    if url:
        out["url"] = url
    lid = get_value(entry, ["lid", "lang"])
    if lid is not None:
        out["lid"] = lid
    return out


def load_processed_chunk_ids(features_path: str) -> set[str]:
    processed = set()
    if not os.path.exists(features_path):
        return processed
    with open(features_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_id = obj.get("chunk_id")
            if chunk_id:
                processed.add(str(chunk_id))
    return processed


def next_shard_index(shards_dir: str) -> int:
    existing = sorted(glob.glob(os.path.join(shards_dir, "shard-*.pt")))
    if not existing:
        return 0
    last = os.path.basename(existing[-1])
    try:
        return int(last.replace("shard-", "").replace(".pt", "")) + 1
    except ValueError:
        return len(existing)


def main():
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN")

    os.makedirs(args.out_dir, exist_ok=True)
    shards_dir = os.path.join(args.out_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    entries = load_manifest_entries(args.manifest_path)
    if not entries:
        raise ValueError(f"No entries found in {args.manifest_path}")

    features_path = os.path.join(args.out_dir, "manifest_features.jsonl")
    processed = load_processed_chunk_ids(features_path) if args.resume else set()

    filtered = []
    for entry in entries:
        chunk_id = build_chunk_id(entry)
        if not chunk_id:
            continue
        if chunk_id in processed:
            continue
        filtered.append(entry)

    if args.max_examples > 0:
        filtered = filtered[: args.max_examples]

    if not filtered:
        print("No new entries to process.", flush=True)
        return

    shard_idx = next_shard_index(shards_dir)
    current_shard = []
    processed_count = 0
    skipped_count = len(processed)
    error_count = 0

    start_time = time.time()
    print(
        f"Processing {len(filtered)} entries with {args.num_workers} workers "
        f"(resume={bool(args.resume)}, skipped={skipped_count})",
        flush=True,
    )

    with futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_entry = {
            executor.submit(
                process_entry,
                entry,
                args.sr,
                args.n_fft,
                args.hop_length,
                args.n_mels,
                hf_token,
            ): entry
            for entry in filtered
        }

        for idx, future in enumerate(futures.as_completed(future_to_entry), start=1):
            try:
                result = future.result()
            except Exception:
                result = None
                error_count += 1

            if result is None:
                continue

            current_shard.append(result)
            processed_count += 1

            meta = {
                "chunk_id": result["chunk_id"],
                "key": result["key"],
                "start_sec": result["start_sec"],
                "end_sec": result["end_sec"],
                "sr": result["sr"],
            }
            if "url" in result:
                meta["url"] = result["url"]
            if "lid" in result:
                meta["lid"] = result["lid"]

            with open(features_path, "a") as f:
                f.write(json.dumps(meta) + "\n")

            if len(current_shard) >= args.examples_per_shard:
                shard_path = os.path.join(shards_dir, f"shard-{shard_idx:05d}.pt")
                torch.save(current_shard, shard_path)
                print(
                    f"Wrote {len(current_shard)} examples to {shard_path} "
                    f"(processed={processed_count}, errors={error_count})",
                    flush=True,
                )
                shard_idx += 1
                current_shard = []

            if idx % 200 == 0:
                elapsed = time.time() - start_time
                print(
                    f"Progress: {idx}/{len(filtered)} futures | "
                    f"processed={processed_count} errors={error_count} elapsed_sec={elapsed:.1f}",
                    flush=True,
                )

    if current_shard:
        shard_path = os.path.join(shards_dir, f"shard-{shard_idx:05d}.pt")
        torch.save(current_shard, shard_path)
        print(
            f"Wrote {len(current_shard)} examples to {shard_path} "
            f"(processed={processed_count}, errors={error_count})",
            flush=True,
        )

    elapsed = time.time() - start_time
    stats = {
        "total_entries": len(entries),
        "processed_new": processed_count,
        "skipped_existing": skipped_count,
        "errors": error_count,
        "elapsed_sec": elapsed,
    }
    stats_path = os.path.join(args.out_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    cfg = vars(args)
    cfg_path = os.path.join(args.out_dir, "precompute_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Done. Stats: {stats_path}", flush=True)
    print(f"Wrote config: {cfg_path}", flush=True)


if __name__ == "__main__":
    main()
