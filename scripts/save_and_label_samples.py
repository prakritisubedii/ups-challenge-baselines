import os
import sys
import time
import soundfile as sf

# Make sure Python can import from this repo
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

def main():
    print("HF_TOKEN set:", os.environ.get("HF_TOKEN") is not None)

    from ups_challenge.dataloaders.base import build_wds_dataset, collate_fn

    num_samples_to_save = 10
    batch_size_files = 2
    sr = 16000

    ds = build_wds_dataset()
    batched = ds.batched(batch_size_files, collation_fn=collate_fn)
    batch = next(iter(batched))

    keys = batch.get("keys", None)

    # Unique folder each run (no overwrite)
    out_dir = f"audio_samples_run_{int(time.time())}"
    os.makedirs(out_dir, exist_ok=True)

    n = min(num_samples_to_save, batch["input_values"].shape[0])

    map_path = os.path.join(out_dir, "mapping.txt")
    with open(map_path, "w") as f:
        for i in range(n):
            x = batch["input_values"][i].cpu()
            wav_name = f"sample_{i+1:02d}.wav"
            out_path = os.path.join(out_dir, wav_name)
            sf.write(out_path, x.numpy(), sr)

            k = keys[i] if keys is not None else "UNKNOWN_KEY"
            f.write(f"{wav_name}\t{k}\n")
            print("Saved:", out_path, "| key:", k)

    print("\nSaved mapping file:", map_path)
    print("✅ Done. Folder:", out_dir)

if __name__ == "__main__":
    main()
