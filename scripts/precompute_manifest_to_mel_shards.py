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
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
import tarfile
from multiprocessing import Manager, get_context
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import torch
import torchaudio
import webdataset as wds
try:
    from torchcodec.decoders import AudioDecoder
except (ImportError, RuntimeError):
    AudioDecoder = None
    logging.warning("torchcodec unavailable, falling back to torchaudio")

# Make sure Python can import from this repo
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

SAMPLE_RATE_DEFAULT = 16000
_PROCESSED_CHUNK_IDS: set[str] | None = None
_CACHE_HIT = None
_CACHE_MISS = None
_CACHE_LOCK = None


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute log-mel shards from a manifest v2 JSONL")
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--examples_per_shard", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument(
        "--chunk_sec",
        type=float,
        default=10.0,
        help="Chunk length in seconds when segment boundaries are not provided; default 10.0s (recommended for UPS).",
    )
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=400)
    parser.add_argument("--hop_length", type=int, default=160)
    parser.add_argument("--tar_cache_dir", type=str, default="/content/ups_cache/tars")
    parser.add_argument("--timeout_sec", type=float, default=25.0)
    parser.add_argument(
        "--force_ffmpeg_decode",
        type=int,
        default=1,
        help="Use ffmpeg CLI to decode audio before torchaudio load (default: 1).",
    )
    parser.add_argument(
        "--submit_batch_size",
        type=int,
        default=200,
        help="Number of entries to submit to each process-pool batch.",
    )
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument("--write_stats_every", type=int, default=200)
    return parser.parse_args()


def load_manifest_entries(manifest_path: str) -> tuple[list[dict], int]:
    entries = []
    line_count = 0
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_count += 1
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries, line_count


from ups_challenge.audio_precompute import tar_url_for_number, fetch_mp3_bytes, waveform_to_log_mel


def fetch_mp3_bytes_from_url(url: str, key: str):
    dataset = (
        wds.WebDataset([url], shardshuffle=False)
        .to_tuple("mp3", "__key__", "__url__", handler=wds.handlers.ignore_and_continue)
    )
    for mp3_bytes, sample_key, _ in dataset:
        if sample_key == key:
            return mp3_bytes
    return None


def run_with_timeout(fn, timeout_sec: float):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        return future.result(timeout=timeout_sec)


def read_mp3_bytes_from_local_tar(tar_path: str, key: str) -> bytes | None:
    target = f"{key}.mp3"
    try:
        with tarfile.open(tar_path, "r") as tar:
            direct = f"data/{target}"
            try:
                member = tar.getmember(direct)
                extracted = tar.extractfile(member)
                if extracted is not None:
                    return extracted.read()
            except KeyError:
                pass

            members = tar.getmembers()
            for member in members:
                name = member.name
                if name.endswith(f"/{target}"):
                    extracted = tar.extractfile(member)
                    if extracted is not None:
                        return extracted.read()
            for member in members:
                name = member.name
                if name.endswith(target):
                    extracted = tar.extractfile(member)
                    if extracted is not None:
                        return extracted.read()
    except Exception:
        return None
    return None


def get_value(row: dict, keys: list[str], default=None):
    for k in keys:
        if k in row:
            return row.get(k)
    return default


def build_chunk_id(key: str, tar_number: int | None, chunk_index: int) -> str:
    tar_label = "na" if tar_number is None else str(tar_number)
    return f"{key}__t{tar_label}__i{chunk_index}"


def _get_segment_bounds(entry: dict):
    start = get_value(entry, ["start_sec", "start"])
    end = get_value(entry, ["end_sec", "end"])
    if start is None or end is None:
        return None
    try:
        start_f = float(start)
        end_f = float(end)
    except (TypeError, ValueError):
        return None
    if end_f <= start_f:
        return None
    return start_f, end_f


def decode_with_ffmpeg_to_wav_then_load(
    in_path: str, target_sr: int = SAMPLE_RATE_DEFAULT, mono: bool = True
) -> tuple[torch.Tensor, int]:
    fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        in_path,
    ]
    if mono:
        cmd.extend(["-ac", "1"])
    cmd.extend(["-ar", str(target_sr), "-f", "wav", tmp_wav])

    try:
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required for audio decoding but was not found in PATH") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(f"ffmpeg decode failed for {in_path}: {stderr}") from exc

        try:
            return torchaudio.load(tmp_wav)
        except Exception as exc:
            stderr = (result.stderr or "").strip()
            suffix = f" ffmpeg_stderr={stderr}" if stderr else ""
            raise RuntimeError(f"failed to load ffmpeg wav output for {in_path}.{suffix}") from exc
    finally:
        if os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass


def load_audio(
    path: str,
    target_sr: int = SAMPLE_RATE_DEFAULT,
    mono: bool = True,
    force_ffmpeg_decode: bool = True,
) -> torch.Tensor:
    lower_path = path.lower()
    is_wav = lower_path.endswith(".wav") or lower_path.endswith(".wave")
    if force_ffmpeg_decode or not is_wav:
        waveform, source_sr = decode_with_ffmpeg_to_wav_then_load(path, target_sr=target_sr, mono=mono)
    else:
        try:
            waveform, source_sr = torchaudio.load(path)
        except Exception as exc:
            raise RuntimeError(f"torchaudio.load failed for {path}: {exc}") from exc

    waveform = waveform.to(torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, source_sr, target_sr)
    return waveform.squeeze(0).contiguous()


def process_entry(
    entry: dict,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    hf_token: str | None,
    chunk_sec: float,
    timeout_sec: float,
    tar_cache_dir: str,
    force_ffmpeg_decode: bool,
) -> list[dict]:
    global AudioDecoder
    key = get_value(entry, ["vad_key", "key", "__key__"])
    if key is None:
        raise ValueError("Missing vad_key")

    url = get_value(entry, ["url", "tar_url"])
    tar_number = get_value(entry, ["tar_number"])

    mp3_bytes = None
    if url:
        mp3_bytes = run_with_timeout(lambda: fetch_mp3_bytes_from_url(url, key), timeout_sec)
    elif tar_number is not None:
        tar_path = os.path.join(tar_cache_dir, f"{int(tar_number):06d}.tar")
        if os.path.exists(tar_path) and os.path.getsize(tar_path) > 0:
            mp3_bytes = read_mp3_bytes_from_local_tar(tar_path, key)
            if _CACHE_HIT is not None and _CACHE_MISS is not None:
                if mp3_bytes is not None:
                    if _CACHE_LOCK:
                        with _CACHE_LOCK:
                            _CACHE_HIT.value += 1
                    else:
                        _CACHE_HIT.value += 1
                else:
                    if _CACHE_LOCK:
                        with _CACHE_LOCK:
                            _CACHE_MISS.value += 1
                    else:
                        _CACHE_MISS.value += 1
        else:
            if _CACHE_HIT is not None and _CACHE_MISS is not None:
                if _CACHE_LOCK:
                    with _CACHE_LOCK:
                        _CACHE_MISS.value += 1
                else:
                    _CACHE_MISS.value += 1

        if mp3_bytes is None:
            mp3_bytes = run_with_timeout(
                lambda: fetch_mp3_bytes(int(tar_number), key, hf_token), timeout_sec
            )
    else:
        raise ValueError("Missing tar_number and url")

    if mp3_bytes is None:
        raise ValueError("Failed to fetch mp3 bytes")

    lid = get_value(entry, ["lid", "lang"])
    results = []
    temp_mp3_path = None

    def _get_temp_mp3_path() -> str:
        nonlocal temp_mp3_path
        if temp_mp3_path is None:
            fd, path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            with open(path, "wb") as f:
                f.write(mp3_bytes)
            temp_mp3_path = path
        return temp_mp3_path

    def _decode_with_torchaudio() -> torch.Tensor:
        try:
            return load_audio(
                _get_temp_mp3_path(),
                target_sr=sr,
                mono=True,
                force_ffmpeg_decode=force_ffmpeg_decode,
            )
        except RuntimeError as exc:
            logging.error(
                f"audio decode failure key={key} tar_number={tar_number}: {exc}",
                exc_info=False,
            )
            raise

    try:
        segment_bounds = _get_segment_bounds(entry)
        if segment_bounds is not None:
            start_sec, end_sec = segment_bounds
            expected = int(round((end_sec - start_sec) * sr))
            if expected <= 0:
                raise ValueError("Invalid segment duration")
            chunk_id = entry.get("chunk_id") or f"{key}__{start_sec:.3f}__{end_sec:.3f}"

            if _PROCESSED_CHUNK_IDS and chunk_id in _PROCESSED_CHUNK_IDS:
                return []

            chunk = None
            if AudioDecoder is not None:
                try:
                    def _decode_segment():
                        decoder = AudioDecoder(source=mp3_bytes, sample_rate=sr, num_channels=1)
                        return decoder.get_samples_played_in_range(start_sec, end_sec)

                    samples = run_with_timeout(_decode_segment, timeout_sec)
                    chunk = samples.data.squeeze(0)
                except RuntimeError:
                    AudioDecoder = None
                    logging.warning("torchcodec unavailable, falling back to torchaudio")

            if chunk is None:
                waveform_full = run_with_timeout(_decode_with_torchaudio, timeout_sec)
                start_idx = int(round(start_sec * sr))
                end_idx = int(round(end_sec * sr))
                chunk = waveform_full[start_idx:end_idx]

            if chunk.shape[-1] < expected:
                pad = expected - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, pad))
            elif chunk.shape[-1] > expected:
                chunk = chunk[..., :expected]

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
            if lid is not None:
                out["lid"] = lid
            if tar_number is not None:
                out["tar_number"] = int(tar_number)

            return [out]

        duration = None
        if AudioDecoder is not None:
            try:
                def _get_duration():
                    decoder = AudioDecoder(source=mp3_bytes, sample_rate=sr, num_channels=1)
                    return decoder.metadata.duration_seconds_from_header

                duration = run_with_timeout(_get_duration, timeout_sec)
            except RuntimeError:
                AudioDecoder = None
                logging.warning("torchcodec unavailable, falling back to torchaudio")

        waveform_full = None
        if AudioDecoder is None:
            waveform_full = run_with_timeout(_decode_with_torchaudio, timeout_sec)
            duration = waveform_full.shape[-1] / sr

        if duration is None or duration <= 0:
            raise ValueError("Invalid duration")

        chunk_count = max(1, int(math.ceil(duration / chunk_sec)))
        expected = int(chunk_sec * sr)

        for idx in range(chunk_count):
            start_sec = idx * chunk_sec
            end_sec = min(start_sec + chunk_sec, duration)
            chunk_id = build_chunk_id(str(key), tar_number if tar_number is not None else None, idx)

            if _PROCESSED_CHUNK_IDS and chunk_id in _PROCESSED_CHUNK_IDS:
                continue

            if AudioDecoder is not None:
                try:
                    def _decode_chunk():
                        decoder = AudioDecoder(source=mp3_bytes, sample_rate=sr, num_channels=1)
                        return decoder.get_samples_played_in_range(start_sec, end_sec)

                    samples = run_with_timeout(_decode_chunk, timeout_sec)
                    chunk = samples.data.squeeze(0)
                except RuntimeError:
                    AudioDecoder = None
                    logging.warning("torchcodec unavailable, falling back to torchaudio")
                    if waveform_full is None:
                        waveform_full = run_with_timeout(_decode_with_torchaudio, timeout_sec)
                    start_idx = int(round(start_sec * sr))
                    end_idx = int(round(end_sec * sr))
                    chunk = waveform_full[start_idx:end_idx]
            else:
                if waveform_full is None:
                    waveform_full = run_with_timeout(_decode_with_torchaudio, timeout_sec)
                start_idx = int(round(start_sec * sr))
                end_idx = int(round(end_sec * sr))
                chunk = waveform_full[start_idx:end_idx]

            if chunk.shape[-1] < expected:
                pad = expected - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, pad))

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
            if lid is not None:
                out["lid"] = lid
            if tar_number is not None:
                out["tar_number"] = int(tar_number)

            results.append(out)

        return results
    finally:
        if temp_mp3_path and os.path.exists(temp_mp3_path):
            try:
                os.remove(temp_mp3_path)
            except OSError:
                pass


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


def _init_worker(processed_chunk_ids: set[str], cache_hit, cache_miss, cache_lock):
    global _PROCESSED_CHUNK_IDS, _CACHE_HIT, _CACHE_MISS, _CACHE_LOCK
    _PROCESSED_CHUNK_IDS = processed_chunk_ids
    _CACHE_HIT = cache_hit
    _CACHE_MISS = cache_miss
    _CACHE_LOCK = cache_lock


def main():
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN")

    os.makedirs(args.out_dir, exist_ok=True)
    shards_dir = os.path.join(args.out_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    entries, line_count = load_manifest_entries(args.manifest_path)
    if not entries:
        raise ValueError(f"No entries found in {args.manifest_path}")

    features_path = os.path.join(args.out_dir, "manifest_features.jsonl")
    processed = load_processed_chunk_ids(features_path) if args.resume else set()
    errors_path = os.path.join(args.out_dir, "errors.jsonl")

    entries_to_process = []
    for entry in entries:
        key = get_value(entry, ["vad_key", "key", "__key__"])
        if key is None:
            continue
        entries_to_process.append(entry)

    remaining_count = len(entries_to_process)
    processed_loaded = len(processed)

    print(f"manifest_path: {args.manifest_path}", flush=True)
    print(f"out_dir: {args.out_dir}", flush=True)
    print(f"manifest_exists: {os.path.exists(args.manifest_path)}", flush=True)
    print(f"manifest_lines_read: {line_count}", flush=True)
    print(f"resume_enabled: {bool(args.resume)}", flush=True)
    print(f"processed_chunk_ids_loaded: {processed_loaded}", flush=True)
    print(f"remaining_to_process: {remaining_count}", flush=True)
    sample_entry = entries_to_process[0] if entries_to_process else None
    segment_mode = bool(sample_entry and _get_segment_bounds(sample_entry))
    print(f"segment_mode_detected: {segment_mode}", flush=True)
    print(f"timeout_sec: {args.timeout_sec}", flush=True)
    print(f"tar_cache_dir: {args.tar_cache_dir} cache_hit=0 cache_miss=0", flush=True)

    shard_idx = next_shard_index(shards_dir)
    buffer = []
    processed_count = 0
    skipped_count = len(processed)
    error_count = 0
    chunk_total = 0
    attempted = 0
    last_chunk_id = None
    last_tar_number = None

    start_time = time.time()
    print(
        f"Processing {len(entries_to_process)} entries with {args.num_workers} workers "
        f"(resume={bool(args.resume)}, skipped={skipped_count})",
        flush=True,
    )

    def _log_error(meta: dict, error_type: str, error_msg: str):
        err = {
            "chunk_id": meta.get("chunk_id"),
            "key": meta.get("key"),
            "tar_number": meta.get("tar_number"),
            "start_sec": meta.get("start_sec"),
            "end_sec": meta.get("end_sec"),
            "lid": meta.get("lid"),
            "error_type": error_type,
            "error_msg": error_msg,
        }
        with open(errors_path, "a") as f:
            f.write(json.dumps(err) + "\n")

    def _flush_buffer():
        nonlocal buffer, shard_idx
        while len(buffer) >= args.examples_per_shard:
            shard_path = os.path.join(shards_dir, f"shard-{shard_idx:05d}.pt")
            torch.save(buffer[: args.examples_per_shard], shard_path)
            print(
                f"Wrote {args.examples_per_shard} examples to {shard_path} "
                f"(processed={processed_count}, errors={error_count})",
                flush=True,
            )
            buffer = buffer[args.examples_per_shard :]
            shard_idx += 1

    def _write_partial_stats():
        stats_partial = {
            "attempted": attempted,
            "ok_count": processed_count,
            "error_count": error_count,
            "last_chunk_id": last_chunk_id,
        }
        stats_partial_path = os.path.join(args.out_dir, "stats_partial.json")
        with open(stats_partial_path, "w") as f:
            json.dump(stats_partial, f, indent=2)

    early_stop = False
    cache_hit = None
    cache_miss = None

    if args.num_workers <= 1:
        print("running in SEQUENTIAL mode", flush=True)
        _init_worker(processed, None, None, None)
        for idx, entry in enumerate(entries_to_process, start=1):
            attempted += 1
            try:
                results = process_entry(
                    entry,
                    args.sr,
                    args.n_fft,
                    args.hop_length,
                    args.n_mels,
                    hf_token,
                    args.chunk_sec,
                    args.timeout_sec,
                    args.tar_cache_dir,
                    bool(args.force_ffmpeg_decode),
                )
            except Exception as exc:
                results = None
                error_count += 1
                key = get_value(entry, ["vad_key", "key", "__key__"])
                tar_number = get_value(entry, ["tar_number"])
                lid = get_value(entry, ["lid", "lang"])
                seg = _get_segment_bounds(entry)
                start_sec, end_sec = (seg if seg else (None, None))
                chunk_id = entry.get("chunk_id")
                if not chunk_id and key is not None and seg is not None:
                    chunk_id = f"{key}__{start_sec:.3f}__{end_sec:.3f}"
                _log_error(
                    {
                        "chunk_id": chunk_id,
                        "key": key,
                        "tar_number": tar_number,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "lid": lid,
                    },
                    type(exc).__name__,
                    str(exc),
                )

            if not results:
                if args.print_every > 0 and attempted % args.print_every == 0:
                    elapsed = time.time() - start_time
                    print(
                        "Progress: "
                        f"attempted={attempted} ok={processed_count} errors={error_count} "
                        f"last_chunk_id={last_chunk_id} last_tar_number={last_tar_number} "
                        f"elapsed_sec={elapsed:.1f}",
                        flush=True,
                    )
                continue

            for result in results:
                chunk_total += 1
                buffer.append(result)
                processed_count += 1

                last_chunk_id = result.get("chunk_id")
                last_tar_number = result.get("tar_number")

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
                if "tar_number" in result:
                    meta["tar_number"] = result["tar_number"]

                with open(features_path, "a") as f:
                    f.write(json.dumps(meta) + "\n")

                _flush_buffer()

                if args.write_stats_every > 0 and processed_count % args.write_stats_every == 0:
                    _write_partial_stats()

                if args.max_examples > 0 and processed_count >= args.max_examples:
                    early_stop = True
                    break

            if args.print_every > 0 and attempted % args.print_every == 0:
                elapsed = time.time() - start_time
                print(
                    "Progress: "
                    f"attempted={attempted} ok={processed_count} errors={error_count} "
                    f"last_chunk_id={last_chunk_id} last_tar_number={last_tar_number} "
                    f"elapsed_sec={elapsed:.1f}",
                    flush=True,
                )
            if early_stop:
                break
    else:
        manager = Manager()
        cache_hit = manager.Value("i", 0)
        cache_miss = manager.Value("i", 0)
        cache_lock = manager.Lock()
        spawn_ctx = get_context("spawn")
        submit_batch_size = max(1, int(args.submit_batch_size))

        batch_start = 0
        while batch_start < len(entries_to_process) and not early_stop:
            batch_entries = entries_to_process[batch_start : batch_start + submit_batch_size]
            batch_remaining = list(batch_entries)

            while batch_remaining and not early_stop:
                pool_broken = False
                with futures.ProcessPoolExecutor(
                    max_workers=args.num_workers,
                    mp_context=spawn_ctx,
                    initializer=_init_worker,
                    initargs=(processed, cache_hit, cache_miss, cache_lock),
                ) as executor:
                    future_to_entry = {}
                    batch_remaining_set = {id(entry) for entry in batch_remaining}
                    for entry in batch_remaining:
                        try:
                            future = executor.submit(
                                process_entry,
                                entry,
                                args.sr,
                                args.n_fft,
                                args.hop_length,
                                args.n_mels,
                                hf_token,
                                args.chunk_sec,
                                args.timeout_sec,
                                args.tar_cache_dir,
                                bool(args.force_ffmpeg_decode),
                            )
                        except BrokenProcessPool:
                            attempted += 1
                            error_count += 1
                            key = get_value(entry, ["vad_key", "key", "__key__"])
                            tar_number = get_value(entry, ["tar_number"])
                            lid = get_value(entry, ["lid", "lang"])
                            seg = _get_segment_bounds(entry)
                            start_sec, end_sec = (seg if seg else (None, None))
                            chunk_id = entry.get("chunk_id")
                            if not chunk_id and key is not None and seg is not None:
                                chunk_id = f"{key}__{start_sec:.3f}__{end_sec:.3f}"
                            _log_error(
                                {
                                    "chunk_id": chunk_id,
                                    "key": key,
                                    "tar_number": tar_number,
                                    "start_sec": start_sec,
                                    "end_sec": end_sec,
                                    "lid": lid,
                                },
                                "BrokenProcessPool",
                                "worker crashed; restarting pool",
                            )
                            batch_remaining_set.discard(id(entry))
                            pool_broken = True
                            for fut in future_to_entry:
                                if not fut.done():
                                    fut.cancel()
                            break
                        future_to_entry[future] = entry

                    if pool_broken:
                        continue

                    for future in futures.as_completed(future_to_entry):
                        entry = future_to_entry[future]
                        attempted += 1
                        try:
                            results = future.result()
                        except BrokenProcessPool:
                            results = None
                            error_count += 1
                            key = get_value(entry, ["vad_key", "key", "__key__"])
                            tar_number = get_value(entry, ["tar_number"])
                            lid = get_value(entry, ["lid", "lang"])
                            seg = _get_segment_bounds(entry)
                            start_sec, end_sec = (seg if seg else (None, None))
                            chunk_id = entry.get("chunk_id")
                            if not chunk_id and key is not None and seg is not None:
                                chunk_id = f"{key}__{start_sec:.3f}__{end_sec:.3f}"
                            _log_error(
                                {
                                    "chunk_id": chunk_id,
                                    "key": key,
                                    "tar_number": tar_number,
                                    "start_sec": start_sec,
                                    "end_sec": end_sec,
                                    "lid": lid,
                                },
                                "BrokenProcessPool",
                                "worker crashed; restarting pool",
                            )
                            batch_remaining_set.discard(id(entry))
                            for fut in future_to_entry:
                                if not fut.done():
                                    fut.cancel()
                            pool_broken = True
                            break
                        except Exception as exc:
                            results = None
                            error_count += 1
                            key = get_value(entry, ["vad_key", "key", "__key__"])
                            tar_number = get_value(entry, ["tar_number"])
                            lid = get_value(entry, ["lid", "lang"])
                            seg = _get_segment_bounds(entry)
                            start_sec, end_sec = (seg if seg else (None, None))
                            chunk_id = entry.get("chunk_id")
                            if not chunk_id and key is not None and seg is not None:
                                chunk_id = f"{key}__{start_sec:.3f}__{end_sec:.3f}"
                            _log_error(
                                {
                                    "chunk_id": chunk_id,
                                    "key": key,
                                    "tar_number": tar_number,
                                    "start_sec": start_sec,
                                    "end_sec": end_sec,
                                    "lid": lid,
                                },
                                type(exc).__name__,
                                str(exc),
                            )

                        batch_remaining_set.discard(id(entry))

                        if not results:
                            if args.print_every > 0 and attempted % args.print_every == 0:
                                elapsed = time.time() - start_time
                                print(
                                    "Progress: "
                                    f"attempted={attempted} ok={processed_count} errors={error_count} "
                                    f"last_chunk_id={last_chunk_id} last_tar_number={last_tar_number} "
                                    f"elapsed_sec={elapsed:.1f}",
                                    flush=True,
                                )
                            continue

                        for result in results:
                            chunk_total += 1
                            buffer.append(result)
                            processed_count += 1

                            last_chunk_id = result.get("chunk_id")
                            last_tar_number = result.get("tar_number")

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
                            if "tar_number" in result:
                                meta["tar_number"] = result["tar_number"]

                            with open(features_path, "a") as f:
                                f.write(json.dumps(meta) + "\n")

                            _flush_buffer()

                            if args.write_stats_every > 0 and processed_count % args.write_stats_every == 0:
                                _write_partial_stats()

                            if args.max_examples > 0 and processed_count >= args.max_examples:
                                early_stop = True
                                break

                        if args.print_every > 0 and attempted % args.print_every == 0:
                            elapsed = time.time() - start_time
                            print(
                                "Progress: "
                                f"attempted={attempted} ok={processed_count} errors={error_count} "
                                f"last_chunk_id={last_chunk_id} last_tar_number={last_tar_number} "
                                f"elapsed_sec={elapsed:.1f}",
                                flush=True,
                            )

                        if early_stop:
                            for fut in future_to_entry:
                                if not fut.done():
                                    fut.cancel()
                            break

                if pool_broken and not early_stop:
                    batch_remaining = [entry for entry in batch_remaining if id(entry) in batch_remaining_set]
                    continue
                batch_remaining = [entry for entry in batch_remaining if id(entry) in batch_remaining_set]
                if not batch_remaining:
                    break

            batch_start += submit_batch_size

    if buffer:
        shard_path = os.path.join(shards_dir, f"shard-{shard_idx:05d}.pt")
        torch.save(buffer, shard_path)
        print(
            f"Wrote {len(buffer)} examples to {shard_path} "
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
        "chunks_emitted": chunk_total,
        "attempted": attempted,
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
    cache_hit_value = cache_hit.value if cache_hit is not None else 0
    cache_miss_value = cache_miss.value if cache_miss is not None else 0
    print(f"cache_hit: {cache_hit_value} cache_miss: {cache_miss_value}", flush=True)


if __name__ == "__main__":
    main()
