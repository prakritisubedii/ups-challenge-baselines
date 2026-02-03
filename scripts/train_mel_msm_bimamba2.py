"""
Train masked spectrogram modeling (MSM) on precomputed mel shards using Bi-Mamba2.

Example:
  python scripts/train_mel_msm_bimamba2.py \
    --precompute_dir /content/drive/MyDrive/ups_precompute/cached_now_full_mel_20260202_220041 \
    --save_dir ./artifacts/msm_bimamba2 \
    --num_steps 200 \
    --batch_size 8
"""

import os

# Safety: disable torch compile/dynamo before importing torch.
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import argparse
import glob
import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn.functional as F

try:
    from ssd.modules import BiMamba2
except Exception as exc:
    raise ImportError(
        "Failed to import BiMamba2 from ssd.modules. "
        "Ensure the ssd package is installed and available on PYTHONPATH."
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Masked spectrogram modeling with Bi-Mamba2 (from shards)")
    parser.add_argument("--precompute_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.30)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=500)
    parser.add_argument("--val_shards", type=int, default=5)
    parser.add_argument("--val_every", type=int, default=200)
    parser.add_argument("--val_batches", type=int, default=5)
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle_shards", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle_within_shard", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


class BiMambaMSM(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj_in = torch.nn.Linear(80, d_model)
        self.backbone = BiMamba2(
            d_model=d_model,
            d_state=64,
            d_conv=7,
            expand=2,
            use_mem_eff_path=False,
        )
        self.proj_out = torch.nn.Linear(d_model, 80)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.backbone(x)
        return self.proj_out(x)


class ShardStream:
    def __init__(self, shard_files: list[str], shuffle_shards: bool, shuffle_within_shard: bool, seed: int):
        self.shard_files = list(shard_files)
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.rng = random.Random(seed)

    def __iter__(self):
        while True:
            shard_files = list(self.shard_files)
            if self.shuffle_shards:
                self.rng.shuffle(shard_files)
            for shard_path in shard_files:
                try:
                    samples = torch.load(shard_path, map_location="cpu")
                except Exception:
                    continue
                if not isinstance(samples, list):
                    continue
                if self.shuffle_within_shard:
                    self.rng.shuffle(samples)
                for sample in samples:
                    if not isinstance(sample, dict):
                        continue
                    mel = sample.get("mel")
                    if mel is None or not isinstance(mel, torch.Tensor):
                        continue
                    if mel.ndim != 2 or mel.shape[0] != 80 or mel.shape[1] < 1:
                        continue
                    yield mel.float().cpu()


def sample_stream(shard_files: list[str], shuffle: bool, shuffle_within_shard: bool, seed: int):
    return ShardStream(
        shard_files=shard_files,
        shuffle_shards=shuffle,
        shuffle_within_shard=shuffle_within_shard,
        seed=seed,
    )


def collate_batch(samples: list[torch.Tensor]):
    samples = [s for s in samples if s is not None]
    if not samples:
        return None
    seqs = [s.t().contiguous() for s in samples]  # [T, 80]
    lengths = [s.shape[0] for s in seqs]
    max_len = max(lengths)

    batch_size = len(seqs)
    x = torch.zeros(batch_size, max_len, 80, dtype=torch.float32)
    pad_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, seq in enumerate(seqs):
        cur_len = seq.shape[0]
        x[i, :cur_len] = seq
        pad_mask[i, :cur_len] = True
    return x, pad_mask, torch.tensor(lengths, dtype=torch.long)


def apply_time_mask(x: torch.Tensor, pad_mask: torch.Tensor, mask_ratio: float, rng: torch.Generator):
    batch_size, max_len, _ = x.shape
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=x.device)
    for i in range(batch_size):
        li = int(pad_mask[i].sum().item())
        if li < 1:
            continue
        k = max(1, int(mask_ratio * li))
        idx = torch.randperm(li, generator=rng, device=x.device)[:k]
        mask[i, idx] = True
    valid_mask = mask & pad_mask
    x_masked = x.clone()
    x_masked[valid_mask] = 0.0
    return x_masked, valid_mask


@dataclass
class TrainLog:
    step: int
    loss: float
    val_loss: Optional[float]
    elapsed_sec: float


def save_checkpoint(save_dir: str, step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, cfg: dict):
    ckpt_path = os.path.join(save_dir, f"ckpt_step_{step}.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "cfg": cfg,
        },
        ckpt_path,
    )


@torch.no_grad()
def run_val(
    proj_in: torch.nn.Module,
    backbone: torch.nn.Module,
    proj_out: torch.nn.Module,
    val_shard_files: list[str],
    batch_size: int,
    mask_ratio: float,
    device: torch.device,
    freq: int = 80,
    seed: int,
    val_batches: int,
):
    if not val_shard_files:
        return None
    if freq != 80:
        raise ValueError(f"Expected freq=80, got {freq}")

    proj_in.eval()
    backbone.eval()
    proj_out.eval()

    stream = sample_stream(
        shard_files=val_shard_files,
        shuffle=False,
        shuffle_within_shard=False,
        seed=seed,
    )
    loader = torch.utils.data.DataLoader(
        stream,
        batch_size=batch_size,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    loader_iter = iter(loader)

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    losses = []
    attempts = 0
    while len(losses) < val_batches and attempts < val_batches * 3:
        attempts += 1
        batch = next(loader_iter)
        if batch is None:
            continue
        x, pad_mask, _lengths = batch
        x = x.to(device, non_blocking=True)
        pad_mask = pad_mask.to(device, non_blocking=True)

        x_masked, valid_mask = apply_time_mask(x, pad_mask, mask_ratio, rng)
        pred = proj_out(backbone(proj_in(x_masked)))
        loss = F.mse_loss(pred[valid_mask], x[valid_mask])
        losses.append(float(loss.detach().cpu().item()))

    proj_in.train()
    backbone.train()
    proj_out.train()

    if not losses:
        return None
    return sum(losses) / len(losses)


def main():
    args = parse_args()
    shard_dir = os.path.join(args.precompute_dir, "shards")
    shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard-*.pt")))
    if not shard_files:
        raise ValueError(f"No shard-*.pt files found in {shard_dir}")

    if args.val_shards <= 0:
        val_shard_files = []
        train_shard_files = shard_files
    elif args.val_shards >= len(shard_files):
        print(
            f"WARNING: val_shards ({args.val_shards}) >= total shards ({len(shard_files)}); "
            "disabling validation.",
            flush=True,
        )
        val_shard_files = []
        train_shard_files = shard_files
    else:
        val_shard_files = shard_files[-args.val_shards :]
        train_shard_files = shard_files[: -args.val_shards]
    print(f"train shards: {len(train_shard_files)} | val shards: {len(val_shard_files)}", flush=True)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, training on CPU will be slow.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    cfg = {
        **vars(args),
        "shard_dir": shard_dir,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    cfg_path = os.path.join(args.save_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    model = BiMambaMSM(d_model=args.d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt.get("model_state", {}))
        optimizer.load_state_dict(ckpt.get("optimizer_state", {}))
        start_step = int(ckpt.get("step", 0))
        print(f"Resumed from {args.resume_from} at step {start_step}", flush=True)
    else:
        start_step = 0

    stream = sample_stream(
        shard_files=train_shard_files,
        shuffle=args.shuffle_shards,
        shuffle_within_shard=args.shuffle_within_shard,
        seed=args.seed,
    )

    loader = torch.utils.data.DataLoader(
        stream,
        batch_size=args.batch_size,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    loader_iter = iter(loader)

    log_path = os.path.join(args.save_dir, "train_log.jsonl")
    start_time = time.time()
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    for step in range(start_step + 1, args.num_steps + 1):
        batch = next(loader_iter)
        if batch is None:
            if step % args.log_every == 0:
                print(f"Step {step}: skipped (empty batch)", flush=True)
            continue

        x, pad_mask, _lengths = batch
        x = x.to(device, non_blocking=True)
        pad_mask = pad_mask.to(device, non_blocking=True)

        x_masked, valid_mask = apply_time_mask(x, pad_mask, args.mask_ratio, rng)
        pred = model(x_masked)
        loss = F.mse_loss(pred[valid_mask], x[valid_mask])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().cpu().item())
        elapsed = time.time() - start_time

        if step % args.log_every == 0:
            if val_shard_files and step % args.val_every == 0:
                val_loss = run_val(
                    model.proj_in,
                    model.backbone,
                    model.proj_out,
                    val_shard_files=val_shard_files,
                    batch_size=args.batch_size,
                    mask_ratio=args.mask_ratio,
                    device=device,
                    seed=args.seed,
                    val_batches=args.val_batches,
                )
            else:
                val_loss = None
            log_payload = {
                "step": step,
                "loss": loss_val,
                "val_loss": val_loss,
                "elapsed_sec": elapsed,
            }
            print(log_payload, flush=True)
        else:
            val_loss = None

        log_entry = TrainLog(step=step, loss=loss_val, val_loss=val_loss, elapsed_sec=elapsed)
        with open(log_path, "a") as f:
            f.write(json.dumps(asdict(log_entry)) + "\n")

        if step % args.ckpt_every == 0:
            save_checkpoint(args.save_dir, step, model, optimizer, cfg)

    save_checkpoint(args.save_dir, args.num_steps, model, optimizer, cfg)


if __name__ == "__main__":
    main()
