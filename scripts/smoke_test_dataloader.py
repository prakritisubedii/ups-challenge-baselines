import os
import sys

# Make sure Python can import from this repo
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

def main():
    print("HF_TOKEN set:", os.environ.get("HF_TOKEN") is not None)

    from ups_challenge.dataloaders.base import build_wds_dataset, collate_fn

    # Simple settings for a first test
    batch_size = 2

    # Build dataset (streaming)
    ds = build_wds_dataset()

    # Turn it into batches
    batched = ds.batched(batch_size, collation_fn=collate_fn)

    # Grab the first batch
    batch = next(iter(batched))

    print("\n✅ Got one batch!")
    print("Batch type:", type(batch))

    # Batch can be a dict or a tuple depending on the collate_fn
    if isinstance(batch, dict):
        print("Batch keys:", list(batch.keys()))
        for k, v in batch.items():
            try:
                shape = tuple(v.shape)
                print(f" - {k}: shape={shape}, dtype={getattr(v, 'dtype', type(v))}")
            except Exception:
                print(f" - {k}: type={type(v)}")
    else:
        print("Batch contents (not a dict). Type:", type(batch))
        try:
            print("Length:", len(batch))
            for i, item in enumerate(batch):
                try:
                    print(f" - item[{i}] shape={tuple(item.shape)} dtype={getattr(item, 'dtype', type(item))}")
                except Exception:
                    print(f" - item[{i}] type={type(item)}")
        except Exception:
            print("Could not get length. Batch is:", batch)

if __name__ == "__main__":
    main()
