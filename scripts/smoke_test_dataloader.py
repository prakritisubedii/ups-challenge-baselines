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

    # IMPORTANT:
    # build_wds_dataset() returns multiple chunks per example (default up to 16)
    # so "batch_size" here is number of source files, but output may be > 10 chunks.
    batch_size = 2  # start small so we don't accidentally get 32+ chunks

    ds = build_wds_dataset()
    batched = ds.batched(batch_size, collation_fn=collate_fn)
    batch = next(iter(batched))

    print("\n✅ Got one batch!")
    print("Batch keys:", list(batch.keys()))
    print("input_values shape:", tuple(batch["input_values"].shape))

    keys = batch.get("keys", None)
    if keys is not None:
        print("Number of keys:", len(keys))
        print("First 3 keys:", keys[:3])

    sr = 16000
    out_dir = "audio_samples"
    os.makedirs(out_dir, exist_ok=True)

    # Save min(num_samples_to_save, available_chunks)
    n = min(num_samples_to_save, batch["input_values"].shape[0])

    # Save a text file mapping wav -> key
    map_path = os.path.join(out_dir, "mapping.txt")
    with open(map_path, "w") as f:
        for i in range(n):
            x = batch["input_values"][i].cpu()
            out_path = os.path.join(out_dir, f"sample_{i+1:02d}.wav")
            sf.write(out_path, x.numpy(), sr)

            k = keys[i] if keys is not None else "UNKNOWN_KEY"
            f.write(f"sample_{i+1:02d}.wav\t{k}\n")

            print("Saved:", out_path, "| key:", k)

    print("\nSaved mapping file:", map_path)
    print("Done. Saved", n, "audio files in:", out_dir)

if __name__ == "__main__":
    main()
