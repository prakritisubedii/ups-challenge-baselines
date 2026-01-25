import random
import os
import torch
import webdataset as wds
from torchcodec.decoders import AudioDecoder

from .urls import build_urls
from .vad_cache import load_cache, get_vad_segments_for_key


def decode_and_normalize(
    sample,
    target_sr=16000,
    chunk_sec=10.0,
    max_chunks_per_example=16,
    shuffle_chunks=False,
    use_vad= True,
    vad_cache_path="./data/vad.cache.json",
):
    """
    sample comes from .to_tuple('mp3', '__key__', '__url__')
    so it's (mp3_bytes, key, url).

    We:
      - decode mp3 using torchaudio
      - resample to default_sample_rate
      - convert to mono
      - return a dict

    Any samples that fail to decode are logged and skipped.
    """
    mp3_bytes, key, url = sample
    chunk_samples = int(chunk_sec * target_sr)

    output_chunks = []

    decoder = AudioDecoder(source=mp3_bytes, sample_rate=target_sr, num_channels=1)

    duration = decoder.metadata.duration_seconds_from_header
    
    # --- VAD lookup (speech regions) ---
    hf_token = os.environ.get("HF_TOKEN")
    vad_cache = load_cache(vad_cache_path) if use_vad else {}
    vad_segments = []
    if use_vad:
        vad_segments = get_vad_segments_for_key(key, vad_cache, vad_cache_path, hf_token)

    # ---- 2) If short file, stream entire audio ----
    if duration <= chunk_sec:
        samples = decoder.get_samples_played_in_range(0.0, duration)
        chunk = samples.data
        chunk = chunk.squeeze(0)

        # pad to exact chunk length
        if chunk.shape[-1] < chunk_samples:
            pad = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        output_chunks.append(chunk)
        batch_wave = torch.stack(output_chunks)
        attention_mask = torch.ones_like(batch_wave, dtype=torch.long)
        num_chunks = batch_wave.shape[0]
        return {
            "input_values": batch_wave,                 # [N_chunks, chunk_samples]
            "attention_mask": attention_mask,           # same shape
            "keys": [key] * num_chunks,                 # one key per chunk
            "urls": [url] * num_chunks,                 # one url per chunk
        }



    # ---- 3) Choose random chunk start times (in seconds) ----
    max_start_sec = duration - chunk_sec

    start_times = []

    if use_vad and len(vad_segments) > 0:
        # Pick chunk starts that overlap speech segments
        for _ in range(max_chunks_per_example):
            seg = random.choice(vad_segments)
            seg_start = seg["start"] / target_sr
            seg_end = seg["end"] / target_sr

            # pick a random time inside the speech segment
            t = random.uniform(seg_start, seg_end)

            # choose chunk start so [start, start+chunk_sec] overlaps speech time t
            start_sec = t - (chunk_sec / 2.0)
            start_sec = max(0.0, min(start_sec, max_start_sec))
            start_times.append(start_sec)
    else:
        # fallback: old behavior (random anywhere)
        start_times = [
            random.uniform(0.0, max_start_sec) for _ in range(max_chunks_per_example)
        ]


    # ---- 4) Stream each chunk ----
    for start_sec in start_times:
        end_sec = start_sec + chunk_sec

        samples = decoder.get_samples_played_in_range(start_sec, end_sec)

        chunk = samples.data
        chunk = chunk.squeeze(0)
        # Pad end-of-file short outputs
        if chunk.shape[-1] < chunk_samples:
            pad = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        output_chunks.append(chunk)

    # ---- 5) Shuffle chunks across examples ----
    if shuffle_chunks:
        random.shuffle(output_chunks)

    # ---- 6) Stack into batch tensors ----
    batch_wave = torch.stack(output_chunks)

    attention_mask = torch.ones_like(batch_wave, dtype=torch.long)

    num_chunks = batch_wave.shape[0]
    return {
        "input_values": batch_wave,                 # [N_chunks, chunk_samples]
        "attention_mask": attention_mask,           # same shape
        "keys": [key] * num_chunks,                 # one key per chunk
        "urls": [url] * num_chunks,                 # one url per chunk
    }



def collate_fn(batch: list):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    input_values = [b["input_values"] for b in batch]
    attention_masks = [b["attention_mask"] for b in batch]

    # b["keys"] is already a list, so we flatten them
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
    Args:
        langs (list): List of language codes to include. If empty, all languages are included.
        index_path (str): Path to the language ID index folder.
    Returns:
        wds.WebDataset: The constructed WebDataset.
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
