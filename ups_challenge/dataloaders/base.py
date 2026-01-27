# ups_challenge/dataloaders/base.py

import os
import random

import torch
import webdataset as wds
from torchcodec.decoders import AudioDecoder

from .urls import build_urls
from .vad_cache import load_cache, get_vad_segments_for_key
from .lid_cache import load_cache as load_lid_cache, get_lid_prediction_for_key

# ---- Simple global cache so we don't reload the JSON on every sample ----
_VAD_CACHE = None
_LID_CACHE = None


def _get_vad_cache(cache_path: str):
    global _VAD_CACHE
    if _VAD_CACHE is None:
        _VAD_CACHE = load_cache(cache_path)
    return _VAD_CACHE


def _get_lid_cache(cache_path: str):
    global _LID_CACHE
    if _LID_CACHE is None:
        _LID_CACHE = load_lid_cache(cache_path)
    return _LID_CACHE


def decode_and_normalize(
    sample,
    target_sr=16000,
    chunk_sec=10.0,
    max_chunks_per_example=16,
    shuffle_chunks=False,
    use_lid=True,
    lid_cache_path="./data/lid_cache.json",
    use_vad=True,
    vad_cache_path="./data/vad_cache.json",
    # --- strict filtering knobs ---
    min_file_speech_ratio=0.25,      # skip files that are mostly non-speech
    require_full_speech_chunk=True,  # chunk must fit entirely inside a VAD speech segment
):
    """
    sample comes from .to_tuple('mp3', '__key__', '__url__')
    so it's (mp3_bytes, key, url).

    We:
      - decode mp3 (streamed chunks)
      - return:
          input_values: [N_chunks, chunk_samples]
          attention_mask: same shape
          keys: list length N_chunks
          urls: list length N_chunks
    """
    mp3_bytes, key, url = sample
    hf_token = os.environ.get("HF_TOKEN")

    if use_lid:
        pred = get_lid_prediction_for_key(
            key,
            _get_lid_cache(lid_cache_path),
            lid_cache_path,
            hf_token,
        )
        if pred == "nospeech":
            return None

    chunk_samples = int(chunk_sec * target_sr)

    decoder = AudioDecoder(source=mp3_bytes, sample_rate=target_sr, num_channels=1)
    duration = decoder.metadata.duration_seconds_from_header

    # If duration is missing/invalid, skip
    if duration is None or duration <= 0:
        return None

    # --- VAD lookup (speech regions) ---
    vad_segments = []
    if use_vad:
        vad_cache = _get_vad_cache(vad_cache_path)
        vad_segments = get_vad_segments_for_key(key, vad_cache, vad_cache_path, hf_token)

        # File-level speech ratio filter
        total_samples = int(duration * target_sr)
        if total_samples <= 0:
            return None

        speech_samples = 0
        for seg in vad_segments:
            speech_samples += max(0, int(seg["end"]) - int(seg["start"]))

        speech_ratio = speech_samples / total_samples

        # Skip files that are mostly not speech (often music/silence)
        if speech_ratio < min_file_speech_ratio:
            return None

    # ---- If short file (<= chunk), stream entire file as one chunk ----
    if duration <= chunk_sec:
        samples = decoder.get_samples_played_in_range(0.0, duration)
        chunk = samples.data.squeeze(0)

        # pad to exact chunk length
        if chunk.shape[-1] < chunk_samples:
            pad = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        batch_wave = torch.stack([chunk])
        attention_mask = torch.ones_like(batch_wave, dtype=torch.long)

        num_chunks = batch_wave.shape[0]
        return {
            "input_values": batch_wave,
            "attention_mask": attention_mask,
            "keys": [key] * num_chunks,
            "urls": [url] * num_chunks,
        }

    # ---- Choose chunk start times ----
    max_start_sec = duration - chunk_sec
    if max_start_sec <= 0:
        return None

    start_times = []

    if use_vad and len(vad_segments) > 0:
        # Only use VAD segments long enough to fit a full chunk
        eligible = []
        for seg in vad_segments:
            seg_len = int(seg["end"]) - int(seg["start"])
            if seg_len >= chunk_samples:
                eligible.append(seg)

        # If we require full speech chunks but none exist, skip file
        if require_full_speech_chunk and len(eligible) == 0:
            return None

        # Sample chunk starts fully inside eligible segments
        for _ in range(max_chunks_per_example):
            seg = random.choice(eligible)
            seg_start_samp = int(seg["start"])
            seg_end_samp = int(seg["end"])

            # pick start so [start, start+chunk_samples] stays inside segment
            max_start_samp = seg_end_samp - chunk_samples
            if max_start_samp < seg_start_samp:
                continue  # safety

            start_samp = random.randint(seg_start_samp, max_start_samp)
            start_sec = start_samp / target_sr

            # also clamp to file boundary (safety)
            start_sec = max(0.0, min(start_sec, max_start_sec))
            start_times.append(start_sec)

        if len(start_times) == 0:
            return None

    else:
        # fallback: random anywhere
        start_times = [random.uniform(0.0, max_start_sec) for _ in range(max_chunks_per_example)]

    # ---- Stream each chunk ----
    output_chunks = []
    for start_sec in start_times:
        end_sec = start_sec + chunk_sec
        samples = decoder.get_samples_played_in_range(start_sec, end_sec)

        chunk = samples.data.squeeze(0)

        # pad end-of-file short outputs
        if chunk.shape[-1] < chunk_samples:
            pad = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        output_chunks.append(chunk)

    if len(output_chunks) == 0:
        return None

    # ---- Shuffle chunks (optional) ----
    if shuffle_chunks:
        random.shuffle(output_chunks)

    # ---- Stack into tensors ----
    batch_wave = torch.stack(output_chunks)
    attention_mask = torch.ones_like(batch_wave, dtype=torch.long)

    num_chunks = batch_wave.shape[0]
    return {
        "input_values": batch_wave,
        "attention_mask": attention_mask,
        "keys": [key] * num_chunks,
        "urls": [url] * num_chunks,
    }


def collate_fn(batch: list):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    input_values = [b["input_values"] for b in batch]
    attention_masks = [b["attention_mask"] for b in batch]

    # flatten per-chunk keys/urls
    keys = [k for b in batch for k in b["keys"]]
    urls = [u for b in batch for u in b["urls"]]

    return {
        "input_values": torch.cat(input_values, dim=0),
        "attention_mask": torch.cat(attention_masks, dim=0),
        "keys": keys,
        "urls": urls,
    }


def build_wds_dataset(langs: list = [], index_path: str = "./data/lid_index.pkl"):
    """
    Build a WebDataset dataset for the given languages.
    If langs is empty, all languages are included.
    """
    urls = build_urls(langs, index_path=index_path)
    return (
        wds.WebDataset(
            urls,
            shardshuffle=False,
        )
        .to_tuple("mp3", "__key__", "__url__", handler=wds.handlers.ignore_and_continue)
        .map(decode_and_normalize)
    )
