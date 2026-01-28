import os
import sys

# Make sure Python can import from this repo
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

def main():
    print("HF_TOKEN set:", os.environ.get("HF_TOKEN") is not None)

    from ups_challenge.dataloaders.base import build_wds_dataset, collate_fn

    # small batch for quick debugging
    batch_size_files = 2
    max_batches = 2600  # ~5200 samples if nothing is filtered; adjust as needed

    ds = build_wds_dataset()
    batched = ds.batched(batch_size_files, collation_fn=collate_fn)
    batch = None
    for i, b in enumerate(batched, start=1):
        batch = b
        if i % 200 == 0:
            print(f"Processed {i} batches...")
        if i >= max_batches:
            break

    if batch is None:
        raise RuntimeError("No batch was produced by the dataloader.")

    print("\n✅ Got one batch!")
    print("Batch keys:", list(batch.keys()))
    print("input_values shape:", tuple(batch["input_values"].shape))
    print("attention_mask shape:", tuple(batch["attention_mask"].shape))

    keys = batch.get("keys")
    urls = batch.get("urls")

    if keys is not None:
        print("Number of keys:", len(keys))
        print("First 3 keys:", keys[:3])
    else:
        print("No keys found in batch.")

    if urls is not None:
        print("First 1 url:", urls[0])
    else:
        print("No urls found in batch.")

    # quick sanity check: how many unique source files in this batch?
    if keys is not None:
        print("Unique file keys in this batch:", len(set(keys)))

if __name__ == "__main__":
    main()
