import os

def main():
    # 1) HF token (Colab will have this in env already if you used the Setup cell)
    print("HF_TOKEN set:", os.environ.get("HF_TOKEN") is not None)

    # 2) Import the dataloader module
    from ups_challenge.dataloaders import base

    # 3) Print helpful info: what functions/classes exist in base.py
    names = [x for x in dir(base) if not x.startswith("_")]
    print("\nThings available in ups_challenge.dataloaders.base:")
    print(names)

    print("\nNext step: we will call the right dataset/loader function once we see what exists.")

if __name__ == "__main__":
    main()
