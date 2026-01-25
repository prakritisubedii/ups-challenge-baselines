import os
import sys
import soundfile as sf

# Make sure Python can import from this repo
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

def main():
    print("HF_TOKEN set:", os.environ.get("HF_TOKEN") is not None)

    from ups_challenge.dataloaders.base import build_wds_dataset, collate_fn

    batch_size = 2  # small is fine for testing

    # Build dataset (streaming)
    ds = build_wds_dataset()

    # Turn it into batches
    batched = ds.batched(batch_size, collation_fn=collate_fn)

    # Grab the first batch
    batch = next(iter(batched))

    print("\n✅ Got one batch!")
    print("Batch type:", type(batch))

    if isinstance(batch, dict):
        print("Batch keys:", list(batch.keys()))
        for k, v in batch.items():
            try:
                print(f" - {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
            except Exception:
                print(f" - {k}: type={type(v)}")
    else:
        print("Batch is not a dict. Type:", type(batch))
        return  # stop here, because below we assume dict format

    # ---- Save the first audio sample so we can listen ----
    x = batch["input_values"][0].cpu()

    print("\nAudio stats (first sample):")
    print(" - min:", float(x.min()))
    print(" - max:", float(x.max()))
    print(" - mean:", float(x.mean()))

    out_path = "sample.wav"
    sr = 16000
    sf.write(out_path, x.numpy(), sr)
    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()
