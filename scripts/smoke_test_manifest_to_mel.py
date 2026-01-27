"""
Smoke test a VAD-clean manifest by decoding a few entries and computing log-mel.

Example:
  python scripts/smoke_test_manifest_to_mel.py \
    --manifest_path /content/drive/MyDrive/ups_artifacts/manifest_v1_small2000.jsonl
"""

import argparse
import json
import os
import random
import sys

import torch
import webdataset as wds
from torchcodec.decoders import AudioDecoder

# Make sure Python can import from this repo
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

from ups_challenge.dataloaders.vad_cache import load_cache as load_vad_cache, get_vad_segments_for_key

SAMPLE_RATE_DEFAULT = 16000


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test manifest -> waveform -> log-mel")
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="/content/drive/MyDrive/ups_artifacts/manifest_v1_small2000.jsonl",
    )
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--chunk_sec", type=float, default=10.0)
    parser.add_argument("--sr", type=int, default=16000)
    return parser.parse_args()


def load_manifest_entries(manifest_path: str):
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


def fetch_mp3_bytes(tar_number: int, key: str, hf_token: str | None):
    url = tar_url_for_number(str(tar_number), hf_token)
    dataset = (
        wds.WebDataset([url], shardshuffle=False)
        .to_tuple("mp3", "__key__", "__url__", handler=wds.handlers.ignore_and_continue)
    )
    for mp3_bytes, sample_key, _ in dataset:
        if sample_key == key:
            return mp3_bytes
    return None


def select_chunk_start(duration: float, chunk_sec: float, vad_segments: list[dict] | None, sr: int):
    if duration <= chunk_sec:
        return 0.0
    max_start = duration - chunk_sec

    if vad_segments:
        eligible = []
        for seg in vad_segments:
            seg_len = int(seg["end"]) - int(seg["start"])
            if seg_len >= int(chunk_sec * sr):
                eligible.append(seg)
        if eligible:
            seg = random.choice(eligible)
            seg_start = int(seg["start"])
            seg_end = int(seg["end"])
            max_start_samp = seg_end - int(chunk_sec * sr)
            if max_start_samp >= seg_start:
                start_samp = random.randint(seg_start, max_start_samp)
                return max(0.0, min(start_samp / sr, max_start))

    return random.uniform(0.0, max_start)


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


def main():
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN is not set")

    entries = load_manifest_entries(args.manifest_path)
    if not entries:
        raise ValueError(f"No entries found in {args.manifest_path}")

    vad_cache = load_vad_cache("./data/vad_cache.json")

    for batch_idx in range(args.num_batches):
        batch_entries = random.sample(entries, k=min(args.batch_size, len(entries)))

        waveforms = []
        keys = []

        for entry in batch_entries:
            key = entry.get("vad_key")
            tar_number = entry.get("tar_number")
            if key is None or tar_number is None:
                continue

            mp3_bytes = fetch_mp3_bytes(tar_number, key, hf_token)
            if mp3_bytes is None:
                continue

            decoder = AudioDecoder(source=mp3_bytes, sample_rate=args.sr, num_channels=1)
            duration = decoder.metadata.duration_seconds_from_header
            if duration is None or duration <= 0:
                continue

            vad_segments = get_vad_segments_for_key(key, vad_cache, "./data/vad_cache.json", hf_token)
            start_sec = select_chunk_start(duration, args.chunk_sec, vad_segments, args.sr)
            end_sec = min(start_sec + args.chunk_sec, duration)
            samples = decoder.get_samples_played_in_range(start_sec, end_sec)
            chunk = samples.data.squeeze(0)

            expected = int(args.chunk_sec * args.sr)
            if chunk.shape[-1] < expected:
                pad = expected - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, pad))

            waveforms.append(chunk)
            keys.append(key)

        if not waveforms:
            print(f"Batch {batch_idx + 1}: no valid waveforms")
            continue

        batch_wave = torch.stack(waveforms, dim=0)
        log_mel = waveform_to_log_mel(batch_wave, sr=args.sr)

        print(f"\nBatch {batch_idx + 1}/{args.num_batches}")
        print("Keys:", keys[:3])
        print("Waveform shape:", tuple(batch_wave.shape))
        print("Log-mel shape:", tuple(log_mel.shape))
        print("Waveform min/max/mean:", batch_wave.min().item(), batch_wave.max().item(), batch_wave.mean().item())
        print("Log-mel min/max/mean:", log_mel.min().item(), log_mel.max().item(), log_mel.mean().item())
        print("Waveform has NaNs:", torch.isnan(batch_wave).any().item())
        print("Log-mel has NaNs:", torch.isnan(log_mel).any().item())


if __name__ == "__main__":
    main()
