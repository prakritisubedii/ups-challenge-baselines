"""
Tiny masked log-mel reconstruction trainer for a quick end-to-end sanity check.

Example:
  python scripts/train_logmel_masked_recon_tiny.py \
    --manifest_path /content/drive/MyDrive/ups_artifacts/manifest_v1_small2000.jsonl
"""

import argparse
import json
import os
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass

import torch
import webdataset as wds
from torchcodec.decoders import AudioDecoder

# Make sure Python can import from this repo
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

from ups_challenge.dataloaders.vad_cache import load_cache as load_vad_cache, get_vad_segments_for_key

SAMPLE_RATE_DEFAULT = 16000


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny masked log-mel reconstruction trainer")
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="/content/drive/MyDrive/ups_artifacts/manifest_v1_small2000.jsonl",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--chunk_sec", type=float, default=10.0)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--suppress_warnings", action=argparse.BooleanOptionalAction, default=True)
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
    waveform = waveform.float()
    device = waveform.device
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=torch.hann_window(n_fft, device=device),
        return_complex=True,
    )
    power = stft.abs() ** 2
    fb = create_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels).to(device)
    mel = torch.matmul(fb, power)
    log_mel = torch.log(mel + 1e-6)
    return log_mel


class TinyConvAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def mask_mel(log_mel: torch.Tensor, mask_ratio: float):
    # log_mel: [B, 80, T]
    bsz, n_mels, t_len = log_mel.shape
    masked = log_mel.clone()
    mask = torch.zeros_like(log_mel, dtype=torch.bool)

    num_mask = max(1, int(t_len * mask_ratio))
    for b in range(bsz):
        idx = torch.randperm(t_len, device=log_mel.device)[:num_mask]
        mask[b, :, idx] = True
        masked[b, :, idx] = 0.0

    return masked, mask


def build_waveform_sample(entries, hf_token, vad_cache, args, max_tries=16):
    tries = 0
    while tries < max_tries:
        tries += 1
        entry = random.choice(entries)
        key = entry.get("vad_key")
        tar_number = entry.get("tar_number")
        if key is None or tar_number is None:
            continue

        try:
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

            return chunk
        except Exception:
            continue
    return None


class ManifestWaveformIterable(torch.utils.data.IterableDataset):
    def __init__(self, entries, hf_token, vad_cache, args, max_tries=16):
        super().__init__()
        self.entries = entries
        self.hf_token = hf_token
        self.vad_cache = vad_cache
        self.args = args
        self.max_tries = max_tries

    def __iter__(self):
        while True:
            sample = build_waveform_sample(
                self.entries,
                hf_token=self.hf_token,
                vad_cache=self.vad_cache,
                args=self.args,
                max_tries=self.max_tries,
            )
            if sample is None:
                continue
            yield sample


def collate_waveforms(samples):
    samples = [s for s in samples if s is not None]
    if not samples:
        return None
    return torch.stack(samples, dim=0)


@dataclass
class TrainSummary:
    steps: int
    final_loss: float
    avg_loss_last_50: float
    time_elapsed_sec: float


def save_checkpoint(out_dir, step, model, optimizer, args):
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_step{step:04d}.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "args": vars(args),
        },
        ckpt_path,
    )


def main():
    args = parse_args()
    if args.suppress_warnings:
        warnings.filterwarnings("ignore", category=FutureWarning)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.out_dir is None:
        args.out_dir = os.environ.get("UPS_ARTIFACT_DIR", "./artifacts")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN is not set")

    entries = load_manifest_entries(args.manifest_path)
    if not entries:
        raise ValueError(f"No entries found in {args.manifest_path}")

    vad_cache = load_vad_cache("./data/vad_cache.json")

    device = torch.device(args.device)
    model = TinyConvAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    dataset = ManifestWaveformIterable(
        entries,
        hf_token=hf_token,
        vad_cache=vad_cache,
        args=args,
        max_tries=max(16, args.batch_size * 4),
    )
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "collate_fn": collate_waveforms,
    }
    if args.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
        dataloader_kwargs["persistent_workers"] = args.persistent_workers
    loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    loader_iter = iter(loader)

    losses = []
    start_time = time.time()
    for step in range(1, args.steps + 1):
        t_step_start = time.time()
        batch_wave = next(loader_iter)
        t_after_data = time.time()

        if batch_wave is None:
            if step % args.log_every == 0:
                print(f"Step {step}: skipped (no valid batch)")
            continue

        batch_wave = batch_wave.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            log_mel = waveform_to_log_mel(batch_wave, sr=args.sr)  # [B, 80, T]
            masked_mel, mask = mask_mel(log_mel, args.mask_ratio)
            target = log_mel.unsqueeze(1)
            masked_in = masked_mel.unsqueeze(1)
            pred = model(masked_in)

            mask_4d = mask.unsqueeze(1)
            diff = (pred - target) ** 2
            loss = (diff * mask_4d).sum() / mask_4d.sum().clamp(min=1)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_val = float(loss.detach().cpu().item())
        losses.append(loss_val)
        t_after_compute = time.time()

        if step % args.log_every == 0:
            time_data = t_after_data - t_step_start
            time_compute = t_after_compute - t_after_data
            step_time = t_after_compute - t_step_start
            print(
                f"Step {step}/{args.steps} - loss: {loss_val:.6f} | "
                f"time_data: {time_data:.3f}s time_compute: {time_compute:.3f}s step_time: {step_time:.3f}s"
            )

        if step % args.save_every == 0:
            save_checkpoint(args.out_dir, step, model, optimizer, args)

    final_step = args.steps
    save_checkpoint(args.out_dir, final_step, model, optimizer, args)

    avg_last_50 = float(sum(losses[-50:]) / max(1, len(losses[-50:])))
    final_loss = float(losses[-1]) if losses else float("nan")
    elapsed = time.time() - start_time

    summary = TrainSummary(
        steps=final_step,
        final_loss=final_loss,
        avg_loss_last_50=avg_last_50,
        time_elapsed_sec=elapsed,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)

    print(f"Done. Final loss: {final_loss:.6f}. Summary: {summary_path}")


if __name__ == "__main__":
    main()
