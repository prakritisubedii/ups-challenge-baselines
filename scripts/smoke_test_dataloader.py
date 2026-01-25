import os
import sys
import soundfile as sf

# Make sure Python can import from this repo
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

def main():
    print("HF_TOKEN set:", os.environ.get("HF_TOKEN") is not None)

    from ups_challenge.dataloaders.base import build_wds_dataset, collate_fn

    num_samples_to_save = 10
    batch_size = 10  # we want 10 clips at once

    ds = build_wds_dataset()
    batched = ds.batched(batch_size, collation_fn=collate_fn)
    batch = next(iter(batched))

    print("\n✅ Got one batch!")
    print("Batch keys:", list(batch.keys()))
    print("input_values shape:", tuple(batch["input_values"].shape))

    sr = 16000
    out_dir = "audio_samples"
    os.makedirs(out_dir, exist_ok=True)

    for i in range(num_samples_to_save):
        x = batch["input_values"][i].cpu()
        out_path = os.path.join(out_dir, f"sample_{i+1:02d}.wav")
        sf.write(out_path, x.numpy(), sr)
        print("Saved:", out_path)

    print("\nDone. Saved 10 audio files in:", out_dir)

if __name__ == "__main__":
    main()
