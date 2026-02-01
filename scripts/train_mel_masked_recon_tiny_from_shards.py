"""
Tiny masked log-mel reconstruction trainer using precomputed mel shards.

Example:
  python scripts/train_mel_masked_recon_tiny_from_shards.py \
    --shards_dir ./artifacts/shards --out_dir ./artifacts/train_from_shards
"""

import argparse
import glob
import json
import os
import random
import time
from dataclasses import asdict, dataclass

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny masked log-mel reconstruction trainer (from shards)")
    parser.add_argument("--shards_dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mask_ratio", type=float, default=0.65)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


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


class ShardMelIterable(torch.utils.data.IterableDataset):
    def __init__(self, shard_files: list[str]):
        super().__init__()
        self.shard_files = list(shard_files)

    def __iter__(self):
        rng = random.Random()
        while True:
            shard_files = list(self.shard_files)
            rng.shuffle(shard_files)
            for shard_path in shard_files:
                try:
                    samples = torch.load(shard_path, map_location="cpu")
                except Exception:
                    continue
                if not isinstance(samples, list):
                    continue
                rng.shuffle(samples)
                for sample in samples:
                    if not isinstance(sample, dict):
                        continue
                    mel = sample.get("mel")
                    if mel is None:
                        continue
                    mel = mel.float().cpu()
                    if mel.ndim != 2:
                        continue
                    yield mel


def collate_mels(samples):
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
    shard_files = sorted(glob.glob(os.path.join(args.shards_dir, "shard-*.pt")))
    if not shard_files:
        raise ValueError(f"No shard-*.pt files found in {args.shards_dir}")

    print(f"num_shard_files: {len(shard_files)}", flush=True)
    try:
        first_samples = torch.load(shard_files[0], map_location="cpu")
        if isinstance(first_samples, list) and first_samples:
            sample = first_samples[0]
            mel = sample.get("mel") if isinstance(sample, dict) else None
            print(
                f"sample_keys: {sorted(list(sample.keys())) if isinstance(sample, dict) else 'N/A'}",
                flush=True,
            )
            print(f"sample_mel_shape: {tuple(mel.shape) if isinstance(mel, torch.Tensor) else None}", flush=True)
    except Exception:
        pass

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.", flush=True)
        args.device = "cpu"

    device = torch.device(args.device)
    model = TinyConvAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    dataset = ShardMelIterable(shard_files)
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_mels,
        "pin_memory": True,
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
        batch_mel = next(loader_iter)
        t_after_data = time.time()

        if batch_mel is None:
            if step % args.log_every == 0:
                print(f"Step {step}: skipped (no valid batch)", flush=True)
            continue

        batch_mel = batch_mel.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            log_mel = batch_mel  # [B, 80, 201]
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
                f"time_data: {time_data:.3f}s time_compute: {time_compute:.3f}s step_time: {step_time:.3f}s",
                flush=True,
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

    print(f"Done. Final loss: {final_loss:.6f}. Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
